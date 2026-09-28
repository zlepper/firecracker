// Copyright 2026 Hermes authors.
// SPDX-License-Identifier: Apache-2.0

//! Snapshot state of VFIO devices and their restore onto a new host device.
//!
//! The VM state file keeps what the VMM emulates: BAR placement, the MSI-X
//! table, the MSI capability, the INTx GSI and the guest's config space.
//! The device's own state (e.g. vGPU framebuffer and contexts) is streamed
//! into a separate file through the VFIO migration data fd.
//!
//! Snapshots are bitcode encoded, which is not self describing, so these
//! types avoid serde attributes that skip fields.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use vfio_bindings::bindings::vfio::{
    VFIO_PCI_CONFIG_REGION_INDEX, VFIO_PCI_INTX_IRQ_INDEX, VFIO_PCI_MSI_IRQ_INDEX,
    VFIO_PCI_MSIX_IRQ_INDEX,
};
use zerocopy::IntoBytes;

use super::interrupts::{VfioIntx, VfioIntxState, VfioIrqMode, VfioMsi, VfioMsiState};
use super::migration::{self, VfioMigrationSupport};
use super::{
    InternalVfioDevice, NUM_BAR_REGS, PCI_CONFIG_SPACE_REGS, VfioBars, VfioDevice, VfioError,
    VfioMsixState, VfioRegionInfo, vfio_calculate_bar_areas, vfio_create_bar_mappings_from_areas,
    vfio_get_pci_capabilities,
};
use crate::logger::info;
use crate::pci::PciSBDF;
use crate::pci::configuration::Bars;
use crate::pci::msix::{MsixCap, MsixConfig, MsixConfigState};
use crate::utils::u64_to_usize;
use crate::vmm_config::device_passthrough::DevicePassthroughConfig;
use crate::vstate::vm::KvmVm;

/// Config space offsets replayed onto a destination device.
const PCI_COMMAND: u64 = 0x04;
const PCI_BAR0: u64 = 0x10;
const PCI_INTERRUPT_LINE: u64 = 0x3c;

/// What identifies a device model. A snapshot only restores onto a device
/// whose regions and interrupts look the same.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VfioDeviceFingerprint {
    /// Config region size.
    pub config_size: u64,
    /// Vendor and device id (config dword 0).
    pub ids: u32,
    /// (flags, size) of BAR regions 0-5.
    pub regions: Vec<(u32, u64)>,
    /// INTx, MSI and MSI-X vector counts.
    pub irqs: Vec<u32>,
}

impl VfioDeviceFingerprint {
    pub(super) fn of(device: &InternalVfioDevice, config_size: u64) -> Self {
        let mut ids = 0u32;
        device.region_read(VFIO_PCI_CONFIG_REGION_INDEX, ids.as_mut_bytes(), 0);
        let regions = (0..u32::from(NUM_BAR_REGS))
            .map(|i| (device.get_region_flags(i), device.get_region_size(i)))
            .collect();
        let irqs = [
            VFIO_PCI_INTX_IRQ_INDEX,
            VFIO_PCI_MSI_IRQ_INDEX,
            VFIO_PCI_MSIX_IRQ_INDEX,
        ]
        .iter()
        .map(|index| device.get_irq_info(*index).map_or(0, |info| info.count))
        .collect();
        Self {
            config_size,
            ids,
            regions,
            irqs,
        }
    }
}

/// Saved MSI-X emulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VfioMsixSavedState {
    /// Config register holding the capability.
    pub register: u8,
    /// The capability.
    pub cap: MsixCap,
    /// Table, PBA and vectors.
    pub config: MsixConfigState,
}

/// Snapshot state of one VFIO device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VfioDeviceState {
    /// Device id.
    pub id: String,
    /// Host PCI address of an sbdf-sourced device.
    pub host_sbdf: Option<PciSBDF>,
    /// Group node of a group-sourced device.
    pub group_path: Option<PathBuf>,
    /// Device name within the group.
    pub device: Option<String>,
    /// Whether the device had to support migration.
    pub require_migration: bool,
    /// Guest PCI address.
    pub sbdf: PciSBDF,
    /// Device model identity.
    pub fingerprint: VfioDeviceFingerprint,
    /// BARs as the guest sees them.
    pub guest_bars: Bars,
    /// BARs as the VMM allocated them.
    pub vmm_bars: Bars,
    /// MSI-X emulation.
    pub msix: Option<VfioMsixSavedState>,
    /// MSI emulation.
    pub msi: Option<VfioMsiState>,
    /// INTx routing.
    pub intx: Option<VfioIntxState>,
    /// Armed interrupt mode.
    pub irq_mode: VfioIrqMode,
    /// The device's config space as the guest left it.
    pub config_space: Vec<u32>,
    /// Size of the device state file written at snapshot time.
    pub state_size: u64,
}

impl VfioDeviceState {
    /// The passthrough configuration this device was created with.
    pub fn passthrough_config(&self) -> DevicePassthroughConfig {
        DevicePassthroughConfig {
            id: self.id.clone(),
            sbdf: self.host_sbdf,
            group_path: self.group_path.clone(),
            device: self.device.clone(),
            require_migration: self.require_migration,
        }
    }
}

impl VfioDevice {
    /// Stream the device state of a stopped device into `path`. A device
    /// can only be snapshotted after this succeeded.
    pub fn save_device_state(&mut self, path: &Path, sync: bool) -> Result<(), VfioError> {
        if !self.migration.stop_copy() {
            return Err(VfioError::MigrationUnsupported(self.config.id.clone()));
        }
        self.saved_state_size = None;
        let size = migration::save_state(&self.device, path, sync)?;
        self.saved_state_size = Some(size);
        Ok(())
    }

    /// Whether the device state was streamed for the snapshot being taken.
    pub fn has_saved_state(&self) -> bool {
        self.saved_state_size.is_some()
    }

    /// Forget the streamed state once the snapshot is written.
    pub fn clear_saved_state(&mut self) {
        self.saved_state_size = None;
    }

    /// The VMM side of the device for the VM state file.
    pub fn state(&self) -> VfioDeviceState {
        let mut config_space = vec![0u32; u64_to_usize(self.config_size / 4)];
        self.device
            .region_read(VFIO_PCI_CONFIG_REGION_INDEX, config_space.as_mut_bytes(), 0);
        VfioDeviceState {
            id: self.config.id.clone(),
            host_sbdf: self.config.sbdf,
            group_path: self.config.group_path.clone(),
            device: self.config.device.clone(),
            require_migration: self.config.require_migration,
            sbdf: self.sbdf,
            fingerprint: self.fingerprint.clone(),
            guest_bars: self.bars.guest_bars,
            vmm_bars: self.bars.vmm_bars,
            msix: self.msix_state.as_ref().map(|msix| VfioMsixSavedState {
                register: msix.register,
                cap: msix.cap,
                config: msix.config.state(),
            }),
            msi: self.msi.as_ref().map(VfioMsi::state),
            intx: self.intx.as_ref().map(VfioIntx::state),
            irq_mode: self.irq_mode,
            config_space,
            state_size: self.saved_state_size.unwrap_or(0),
        }
    }

    /// Write `data` to the device's config space.
    fn write_config(&self, offset: u64, data: &[u8]) {
        if offset + data.len() as u64 <= self.config_size {
            self.device
                .region_write(VFIO_PCI_CONFIG_REGION_INDEX, data, offset);
        }
    }

    /// Replay the guest's config space programming onto a resuming device:
    /// BARs, the interrupt line, MSI and MSI-X control, then the command
    /// register, which enables decoding and bus mastering last.
    fn replay_config(&self, state: &VfioDeviceState) {
        for bar in 0..NUM_BAR_REGS {
            let mut value = 0u32;
            self.bars.guest_bars.read(bar, 0, value.as_mut_bytes());
            self.write_config(PCI_BAR0 + u64::from(bar) * 4, value.as_bytes());
        }
        let register = |offset: u64| {
            state
                .config_space
                .get(u64_to_usize(offset / 4))
                .copied()
                .unwrap_or(0)
                .to_le_bytes()
        };
        self.write_config(PCI_INTERRUPT_LINE, &register(PCI_INTERRUPT_LINE)[..1]);
        if let Some(msi) = &state.msi {
            // Control, address, data and mask, as emulated.
            self.write_config(u64::from(msi.cap_offset) + 2, &msi.bytes[2..]);
        }
        if let Some(msix) = self.msix_state.as_ref() {
            let msg_ctl = msix.config.as_msg_ctl();
            self.write_config(u64::from(msix.register) * 4 + 2, &msg_ctl.to_le_bytes());
        }
        self.write_config(PCI_COMMAND, &register(PCI_COMMAND)[..2]);
    }

    /// Quiesce for a VM pause (see [`migration::quiesce`]).
    pub fn quiesce(&self, p2p_only: bool) -> Result<(), VfioError> {
        migration::quiesce(&self.device, self.migration, p2p_only)
    }

    /// Resume after a VM pause (see [`migration::unquiesce`]).
    pub fn unquiesce(&self, p2p_only: bool) -> Result<(), VfioError> {
        migration::unquiesce(&self.device, self.migration, p2p_only)
    }

    /// Whether the device can be quiesced and snapshotted.
    pub fn migration_support(&self) -> VfioMigrationSupport {
        self.migration
    }
}

/// Open the device for `state` (with any override already applied), load
/// its state and restore the VMM emulation at the saved guest addresses.
/// The device is left in STOP; the VM's resume moves it to RUNNING. The
/// container must already map guest memory.
pub fn vfio_restore_device(
    container: &Arc<super::VfioContainer>,
    vm: &Arc<KvmVm>,
    state: &VfioDeviceState,
    state_path: &Path,
) -> Result<VfioDevice, VfioError> {
    let config = state.passthrough_config();
    let device = super::vfio_open_device(container, &config)?;
    device.reset();

    let config_size = device
        .get_region_size(VFIO_PCI_CONFIG_REGION_INDEX)
        .min(u64::from(PCI_CONFIG_SPACE_REGS) * 4);
    let fingerprint = VfioDeviceFingerprint::of(&device, config_size);
    if fingerprint != state.fingerprint {
        return Err(VfioError::FingerprintMismatch(state.id.clone()));
    }
    let migration_support = VfioMigrationSupport::query(&device)?;
    if !migration_support.stop_copy() {
        return Err(VfioError::MigrationUnsupported(state.id.clone()));
    }

    // Load the device state first: RESUMING keeps the device inert while the
    // emulation around it is rebuilt.
    migration::begin_load(&device, state_path, state.state_size)?;

    let mut config_space = [0_u32; PCI_CONFIG_SPACE_REGS as usize];
    device.region_read(
        VFIO_PCI_CONFIG_REGION_INDEX,
        &mut config_space.as_mut_bytes()[..u64_to_usize(config_size)],
        0,
    );
    let (_, _, masks) = vfio_get_pci_capabilities(&config_space);

    let msix_state = match &state.msix {
        Some(saved) => Some(VfioMsixState {
            register: saved.register,
            cap: saved.cap,
            config: MsixConfig::from_state(saved.config.clone(), vm.clone(), state.sbdf)?,
        }),
        None => None,
    };
    let msi = match &state.msi {
        Some(saved) => Some(VfioMsi::restore(vm, state.sbdf, saved)?),
        None => None,
    };
    let intx = match &state.intx {
        Some(saved) => {
            vm.resource_allocator()
                .gsi_legacy_allocator
                .allocate_id_at(saved.gsi)
                .map_err(VfioError::LegacyGsi)?;
            Some(VfioIntx::new(vm, saved.gsi)?)
        }
        None => None,
    };

    // The BARs keep the addresses the guest knows. The resource allocator was
    // restored with them still allocated, so adopt them instead of allocating.
    let bars = VfioBars {
        guest_bars: state.guest_bars,
        vmm_bars: state.vmm_bars,
        vm: vm.clone(),
    };
    let bar_region_infos: [VfioRegionInfo; NUM_BAR_REGS as usize] =
        std::array::from_fn(|i| super::vfio_region_info(&device, i));
    let (areas, emulated_areas) = vfio_calculate_bar_areas(
        &bars.vmm_bars,
        &bar_region_infos,
        msix_state.as_ref().map(|msix| &msix.cap),
    )?;
    let first_area_slot = vm
        .next_kvm_slot(u32::try_from(areas.len()).map_err(|_| VfioError::KvmSlot)?)
        .ok_or(VfioError::KvmSlot)?;
    let bar_mappings =
        vfio_create_bar_mappings_from_areas(vm.as_ref(), &areas, &device, first_area_slot)?;

    let vfio_device = VfioDevice {
        config,
        sbdf: state.sbdf,
        device,
        bars,
        bar_mappings,
        emulated_areas,
        msix_state,
        msi,
        intx,
        irq_mode: VfioIrqMode::None,
        masks,
        config_size,
        migration: migration_support,
        fingerprint,
        saved_state_size: None,
        vm: vm.clone(),
    };
    vfio_device.replay_config(state);
    info!(
        "[{}] VFIO device state loaded onto {:?}",
        vfio_device.config.id, vfio_device.config.device
    );
    Ok(vfio_device)
}

/// Finish a restore once KVM's routes are flushed: connect the interrupt
/// eventfds, arm the saved interrupt mode and leave RESUMING.
pub fn vfio_finish_restore(
    device: &mut VfioDevice,
    irq_mode: VfioIrqMode,
) -> Result<(), VfioError> {
    if let Some(msix) = device.msix_state.as_ref() {
        msix.config.enable_unmasked_vectors()?;
    }
    if let Some(msi) = device.msi.as_ref() {
        msi.update_routes()?;
    }
    let msix_fds = device.msix_state.as_ref().map(|msix| {
        msix.config
            .vectors
            .vectors
            .iter()
            .map(|v| &v.event_fd)
            .collect::<Vec<_>>()
    });
    super::interrupts::set_irq_mode(
        &device.device,
        &mut device.irq_mode,
        irq_mode,
        device.intx.as_ref(),
        device.msi.as_ref(),
        msix_fds,
    )?;
    migration::finish_load(&device.device)
}
