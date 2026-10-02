// Copyright 2026 Hermes authors.
// SPDX-License-Identifier: Apache-2.0

//! VFIO migration uAPI v2 for snapshots: quiescing devices on pause and
//! streaming device state into and out of snapshot files.
//!
//! Hermes snapshots a paused VM, so stop-and-copy is enough: the device is in
//! STOP whenever the VM is paused, its state is streamed in STOP_COPY, and a
//! destination device is loaded in RESUMING. PRE_COPY is not used.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use vfio_bindings::bindings::vfio::{
    VFIO_MIGRATION_P2P, VFIO_MIGRATION_STOP_COPY,
    vfio_device_mig_state_VFIO_DEVICE_STATE_ERROR as STATE_ERROR,
    vfio_device_mig_state_VFIO_DEVICE_STATE_RESUMING as STATE_RESUMING,
    vfio_device_mig_state_VFIO_DEVICE_STATE_RUNNING as STATE_RUNNING,
    vfio_device_mig_state_VFIO_DEVICE_STATE_RUNNING_P2P as STATE_RUNNING_P2P,
    vfio_device_mig_state_VFIO_DEVICE_STATE_STOP as STATE_STOP,
    vfio_device_mig_state_VFIO_DEVICE_STATE_STOP_COPY as STATE_STOP_COPY,
};

use super::{InternalVfioDevice, VfioError};
use crate::logger::{debug, error, info};

/// Chunk size for streaming device state. Bounded, so a 16 GiB vGPU state
/// never needs a 16 GiB buffer.
const STATE_CHUNK: usize = 1 << 20;

/// A device's migration capabilities, from `VFIO_DEVICE_FEATURE_MIGRATION`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VfioMigrationSupport {
    /// Raw `VFIO_MIGRATION_*` flags, or 0 if the device cannot migrate.
    pub flags: u64,
}

impl VfioMigrationSupport {
    /// Query the device.
    pub fn query(device: &InternalVfioDevice) -> Result<Self, VfioError> {
        Ok(Self {
            flags: device.query_migration_support()?.unwrap_or(0),
        })
    }

    /// Whether the device supports stop-and-copy migration.
    pub fn stop_copy(&self) -> bool {
        self.flags & u64::from(VFIO_MIGRATION_STOP_COPY) != 0
    }

    /// Whether the device supports the peer-to-peer quiesced state.
    pub fn p2p(&self) -> bool {
        self.flags & u64::from(VFIO_MIGRATION_P2P) != 0
    }
}

/// Human readable migration state, for logs and errors.
pub fn state_name(state: u32) -> &'static str {
    match state {
        STATE_ERROR => "ERROR",
        STATE_STOP => "STOP",
        STATE_RUNNING => "RUNNING",
        STATE_STOP_COPY => "STOP_COPY",
        STATE_RESUMING => "RESUMING",
        STATE_RUNNING_P2P => "RUNNING_P2P",
        _ => "PRE_COPY",
    }
}

/// Move the device to `state`. The kernel walks the intermediate arcs. On
/// failure the device may have entered ERROR, which only a reset leaves.
pub fn set_state(device: &InternalVfioDevice, state: u32) -> Result<(), VfioError> {
    debug!("VFIO migration state -> {}", state_name(state));
    if let Err(err) = device.set_migration_state(state) {
        let current = device.get_migration_state().ok();
        error!(
            "VFIO migration transition to {} failed: {err}; device is in {}",
            state_name(state),
            current.map_or("an unknown state", state_name)
        );
        if current == Some(STATE_ERROR) {
            // ERROR is left only through a reset, which returns to RUNNING.
            device.reset();
        }
        return Err(VfioError::MigrationTransition(state_name(state), err));
    }
    Ok(())
}

/// Quiesce a running device for a VM pause: RUNNING -> RUNNING_P2P -> STOP.
/// With several devices, callers first move every device to RUNNING_P2P
/// (`p2p_only`) and then all of them to STOP, so no device issues peer DMA
/// to a device that already stopped.
pub fn quiesce(
    device: &InternalVfioDevice,
    support: VfioMigrationSupport,
    p2p_only: bool,
) -> Result<(), VfioError> {
    if !support.stop_copy() {
        return Ok(());
    }
    if p2p_only {
        if support.p2p() {
            set_state(device, STATE_RUNNING_P2P)?;
        }
        return Ok(());
    }
    set_state(device, STATE_STOP)
}

/// Resume a quiesced device for a VM resume: STOP -> RUNNING_P2P -> RUNNING.
pub fn unquiesce(
    device: &InternalVfioDevice,
    support: VfioMigrationSupport,
    p2p_only: bool,
) -> Result<(), VfioError> {
    if !support.stop_copy() {
        return Ok(());
    }
    if p2p_only {
        if support.p2p() {
            set_state(device, STATE_RUNNING_P2P)?;
        }
        return Ok(());
    }
    set_state(device, STATE_RUNNING)
}

/// Stream the state of a stopped device into `path`, returning its size.
/// The device goes STOP -> STOP_COPY -> STOP.
pub fn save_state(device: &InternalVfioDevice, path: &Path, sync: bool) -> Result<u64, VfioError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(VfioError::StateFile)?;
    let mut span = crate::hermes_trace::Span::start("vfio.stop_copy");
    set_state(device, STATE_STOP_COPY)?;
    let copied = copy_out(device, &mut file);
    // Always leave STOP_COPY, even when the copy failed.
    let stopped = set_state(device, STATE_STOP);
    let size = copied?;
    stopped?;
    span.record("bytes", size);
    drop(span);
    if sync {
        let _span = crate::hermes_trace::Span::start("vfio.state_sync");
        file.sync_all().map_err(VfioError::StateFile)?;
    }
    info!("VFIO device state saved: {size} bytes");
    Ok(size)
}

fn copy_out(device: &InternalVfioDevice, file: &mut File) -> Result<u64, VfioError> {
    let mut buffer = vec![0u8; STATE_CHUNK];
    let mut total = 0u64;
    loop {
        let read = device.read_migration_data(&mut buffer)?;
        if read == 0 {
            return Ok(total);
        }
        file.write_all(&buffer[..read])
            .map_err(VfioError::StateFile)?;
        total += read as u64;
    }
}

/// Load the state in `path` into a freshly opened device. The device goes
/// RUNNING -> STOP -> RESUMING and is left in RESUMING, so the caller can
/// restore config space, BAR mappings and interrupts before calling
/// [`finish_load`].
pub fn begin_load(
    device: &InternalVfioDevice,
    path: &Path,
    expected_size: u64,
) -> Result<(), VfioError> {
    let mut file = File::open(path).map_err(VfioError::StateFile)?;
    let size = file.metadata().map_err(VfioError::StateFile)?.len();
    if size != expected_size {
        return Err(VfioError::StateSizeMismatch(expected_size, size));
    }
    let mut span = crate::hermes_trace::Span::start("vfio.resume_load");
    span.record("bytes", size);
    set_state(device, STATE_STOP)?;
    set_state(device, STATE_RESUMING)?;
    let mut buffer = vec![0u8; STATE_CHUNK];
    loop {
        let read = file.read(&mut buffer).map_err(VfioError::StateFile)?;
        if read == 0 {
            break;
        }
        device.write_migration_data(&buffer[..read])?;
    }
    Ok(())
}

/// Leave RESUMING for STOP; the VM's resume moves the device to RUNNING.
pub fn finish_load(device: &InternalVfioDevice) -> Result<(), VfioError> {
    set_state(device, STATE_STOP)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migration_support_flags() {
        let none = VfioMigrationSupport::default();
        assert!(!none.stop_copy());
        assert!(!none.p2p());
        let full = VfioMigrationSupport {
            flags: u64::from(VFIO_MIGRATION_STOP_COPY | VFIO_MIGRATION_P2P),
        };
        assert!(full.stop_copy());
        assert!(full.p2p());
    }

    #[test]
    fn test_state_names() {
        assert_eq!(state_name(STATE_STOP_COPY), "STOP_COPY");
        assert_eq!(state_name(STATE_ERROR), "ERROR");
    }
}
