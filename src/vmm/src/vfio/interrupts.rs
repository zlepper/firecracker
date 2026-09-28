// Copyright 2026 Hermes authors.
// SPDX-License-Identifier: Apache-2.0

//! Interrupt delivery for VFIO devices: MSI-X (emulated table), MSI (emulated
//! capability) and INTx (level triggered, through a KVM resample irqfd).
//!
//! The guest selects a mode by programming the device's config space. The
//! VMM emulates the capability and arms the matching VFIO IRQ index with
//! `VFIO_DEVICE_SET_IRQS`. vfio-pci refuses to arm one index while another is
//! armed, so every transition first disarms the current mode.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use vfio_bindings::bindings::vfio::{
    VFIO_PCI_INTX_IRQ_INDEX, VFIO_PCI_MSI_IRQ_INDEX, VFIO_PCI_MSIX_IRQ_INDEX,
};
use vmm_sys_util::eventfd::EventFd;

use super::{InternalVfioDevice, VfioError};
use crate::logger::{debug, error};
use crate::pci::PciSBDF;
use crate::pci::msix::MsixTableEntry;
use crate::snapshot::Persist;
use crate::utils::u64_to_usize;
use crate::vstate::interrupts::MsixVectorGroup;
use crate::vstate::vm::KvmVm;

/// MSI message control: MSI enable.
const MSI_CTL_ENABLE: u16 = 1 << 0;
/// MSI message control: multiple message capable (read-only), bits 3:1.
const MSI_CTL_MMC_SHIFT: u16 = 1;
/// MSI message control: multiple message enable, bits 6:4.
const MSI_CTL_MME_SHIFT: u16 = 4;
const MSI_CTL_MME_MASK: u16 = 0x7 << MSI_CTL_MME_SHIFT;
/// MSI message control: 64 bit address capable (read-only).
const MSI_CTL_64BIT: u16 = 1 << 7;
/// MSI message control: per-vector masking capable (read-only).
const MSI_CTL_PER_VECTOR_MASK: u16 = 1 << 8;
/// Largest MSI capability: 64 bit address with per-vector masking.
const MSI_CAP_MAX_SIZE: usize = 24;

/// Which VFIO IRQ index is currently armed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VfioIrqMode {
    /// No interrupt is armed.
    None,
    /// Legacy level-triggered INTx.
    Intx,
    /// MSI with the given number of enabled vectors.
    Msi(u32),
    /// MSI-X with all table vectors.
    Msix,
}

/// Serializable MSI capability state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VfioMsiState {
    /// Config space offset of the capability.
    pub cap_offset: u8,
    /// Emulated capability bytes.
    pub bytes: Vec<u8>,
    /// GSIs of the vector group.
    pub gsis: Vec<u32>,
}

/// Emulated MSI capability (PCI Local Bus 3.0, 6.8.1).
///
/// The guest's address/data/mask writes never reach the device: vfio-pci
/// owns the physical MSI programming, and the VMM routes the eventfds that
/// VFIO signals to the vectors the guest configured.
#[derive(Debug)]
pub struct VfioMsi {
    cap_offset: u8,
    bytes: [u8; MSI_CAP_MAX_SIZE],
    size: usize,
    vectors: Arc<MsixVectorGroup>,
    sbdf: PciSBDF,
}

impl VfioMsi {
    /// Emulate the MSI capability found at `cap_offset`, whose message control
    /// word is `control`. Allocates one GSI per vector the device can request.
    pub fn new(
        vm: &Arc<KvmVm>,
        sbdf: PciSBDF,
        cap_offset: u8,
        control: u16,
    ) -> Result<Self, VfioError> {
        let capable = 1u16 << ((control >> MSI_CTL_MMC_SHIFT) & 0x7).min(5);
        let vectors = KvmVm::create_msix_group(vm.clone(), capable)?;
        let mut bytes = [0u8; MSI_CAP_MAX_SIZE];
        // Keep only the read-only capability bits: the guest starts with MSI
        // disabled and one message enabled.
        let control =
            control & ((0x7 << MSI_CTL_MMC_SHIFT) | MSI_CTL_64BIT | MSI_CTL_PER_VECTOR_MASK);
        bytes[2..4].copy_from_slice(&control.to_le_bytes());
        Ok(Self {
            cap_offset,
            bytes,
            size: Self::cap_size(control),
            vectors: Arc::new(vectors),
            sbdf,
        })
    }

    fn cap_size(control: u16) -> usize {
        match (
            control & MSI_CTL_64BIT != 0,
            control & MSI_CTL_PER_VECTOR_MASK != 0,
        ) {
            (false, false) => 10,
            (true, false) => 14,
            (false, true) => 20,
            (true, true) => 24,
        }
    }

    fn control(&self) -> u16 {
        u16::from_le_bytes([self.bytes[2], self.bytes[3]])
    }

    fn is_64bit(&self) -> bool {
        self.control() & MSI_CTL_64BIT != 0
    }

    fn u32_at(&self, offset: usize) -> u32 {
        u32::from_le_bytes([
            self.bytes[offset],
            self.bytes[offset + 1],
            self.bytes[offset + 2],
            self.bytes[offset + 3],
        ])
    }

    fn addr_lo(&self) -> u32 {
        self.u32_at(4)
    }

    fn addr_hi(&self) -> u32 {
        if self.is_64bit() { self.u32_at(8) } else { 0 }
    }

    fn data_offset(&self) -> usize {
        if self.is_64bit() { 12 } else { 8 }
    }

    fn data(&self) -> u16 {
        let offset = self.data_offset();
        u16::from_le_bytes([self.bytes[offset], self.bytes[offset + 1]])
    }

    fn mask_offset(&self) -> Option<usize> {
        (self.control() & MSI_CTL_PER_VECTOR_MASK != 0).then(|| self.data_offset() + 4)
    }

    fn mask_bits(&self) -> u32 {
        self.mask_offset().map_or(0, |offset| self.u32_at(offset))
    }

    /// Whether the guest enabled MSI.
    pub fn enabled(&self) -> bool {
        self.control() & MSI_CTL_ENABLE != 0
    }

    /// Number of vectors the guest enabled.
    pub fn enabled_vectors(&self) -> u32 {
        let requested = 1u32 << ((self.control() & MSI_CTL_MME_MASK) >> MSI_CTL_MME_SHIFT);
        requested.min(u32::from(self.vectors.num_vectors()))
    }

    /// Whether config space offset `offset` falls in this capability.
    pub fn contains(&self, offset: u64) -> bool {
        let start = u64::from(self.cap_offset);
        start <= offset && offset < start + self.size as u64
    }

    /// The config register `reg_idx`, with the capability's bytes emulated.
    pub fn read_register(&self, reg_idx: u16, device_value: u32) -> u32 {
        let mut bytes = device_value.to_le_bytes();
        for (i, byte) in bytes.iter_mut().enumerate() {
            let offset = u64::from(reg_idx) * 4 + i as u64;
            // The first two bytes (id and next pointer) come from the device.
            if self.contains(offset) && offset >= u64::from(self.cap_offset) + 2 {
                *byte = self.bytes[u64_to_usize(offset - u64::from(self.cap_offset))];
            }
        }
        u32::from_le_bytes(bytes)
    }

    /// Apply a guest write, returning whether anything changed.
    pub fn write(&mut self, config_offset: u64, data: &[u8]) -> bool {
        let before = self.bytes;
        let writable = self.writable_mask();
        for (i, value) in data.iter().enumerate() {
            let offset = config_offset + i as u64;
            if !self.contains(offset) {
                continue;
            }
            let index = u64_to_usize(offset - u64::from(self.cap_offset));
            let mask = writable[index];
            self.bytes[index] = (self.bytes[index] & !mask) | (value & mask);
        }
        before != self.bytes
    }

    fn writable_mask(&self) -> [u8; MSI_CAP_MAX_SIZE] {
        let mut mask = [0u8; MSI_CAP_MAX_SIZE];
        // Control: MSI enable and multiple message enable.
        mask[2] = 0x71;
        // Address low (dword aligned).
        mask[4..8].copy_from_slice(&0xffff_fffcu32.to_le_bytes());
        if self.is_64bit() {
            mask[8..12].copy_from_slice(&[0xff; 4]);
        }
        let data = self.data_offset();
        mask[data..data + 2].copy_from_slice(&[0xff; 2]);
        if let Some(offset) = self.mask_offset() {
            mask[offset..offset + 4].copy_from_slice(&[0xff; 4]);
        }
        mask
    }

    /// The eventfds of the enabled vectors, in vector order.
    pub fn eventfds(&self) -> Vec<&EventFd> {
        self.vectors.vectors[..self.enabled_vectors() as usize]
            .iter()
            .map(|v| &v.event_fd)
            .collect()
    }

    /// Program KVM's routes for the guest's current MSI configuration and
    /// connect the eventfds of unmasked vectors.
    pub fn update_routes(&self) -> Result<(), VfioError> {
        let enabled = if self.enabled() {
            self.enabled_vectors()
        } else {
            0
        };
        let masks = self.mask_bits();
        let vm = &self.vectors.vm;
        for (index, vector) in self.vectors.vectors.iter().enumerate() {
            let index = u32::try_from(index).unwrap_or(u32::MAX);
            let masked = index >= enabled || masks & (1 << index) != 0;
            let entry = MsixTableEntry {
                msg_addr_lo: self.addr_lo(),
                msg_addr_hi: self.addr_hi(),
                // Multiple messages vary the low bits of the data.
                msg_data: u32::from(self.data()) | index,
                vector_ctl: u32::from(masked),
            };
            if masked {
                vector.disable(&vm.common.fd)?;
            }
            vm.register_msi(vector, &entry, self.sbdf)
                .map_err(|e| VfioError::MsixConfig(e.into()))?;
        }
        vm.set_gsi_routes()?;
        for (index, vector) in self.vectors.vectors.iter().enumerate() {
            let index = u32::try_from(index).unwrap_or(u32::MAX);
            if index < enabled && masks & (1 << index) == 0 {
                vector.enable(&vm.common.fd)?;
            }
        }
        Ok(())
    }

    /// Serializable state.
    pub fn state(&self) -> VfioMsiState {
        VfioMsiState {
            cap_offset: self.cap_offset,
            bytes: self.bytes[..self.size].to_vec(),
            gsis: self.vectors.save(),
        }
    }

    /// Restore from state, re-reserving the saved GSIs.
    pub fn restore(
        vm: &Arc<KvmVm>,
        sbdf: PciSBDF,
        state: &VfioMsiState,
    ) -> Result<Self, VfioError> {
        let vectors = MsixVectorGroup::restore(vm.clone(), &state.gsis)?;
        let mut bytes = [0u8; MSI_CAP_MAX_SIZE];
        let len = state.bytes.len().min(MSI_CAP_MAX_SIZE);
        bytes[..len].copy_from_slice(&state.bytes[..len]);
        let control = u16::from_le_bytes([bytes[2], bytes[3]]);
        Ok(Self {
            cap_offset: state.cap_offset,
            bytes,
            size: Self::cap_size(control),
            vectors: Arc::new(vectors),
            sbdf,
        })
    }
}

/// Serializable INTx state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VfioIntxState {
    /// Legacy GSI (IOAPIC pin) of the device's INTA.
    pub gsi: u32,
}

/// Level-triggered INTx through a KVM resample irqfd: VFIO masks the line
/// when it fires and the guest's EOI unmasks it through `resample`.
#[derive(Debug)]
pub struct VfioIntx {
    gsi: u32,
    trigger: EventFd,
    resample: EventFd,
    vm: Arc<KvmVm>,
}

impl VfioIntx {
    /// Route legacy GSI `gsi` to the device's INTx.
    pub fn new(vm: &Arc<KvmVm>, gsi: u32) -> Result<Self, VfioError> {
        let trigger = EventFd::new(libc::EFD_NONBLOCK).map_err(VfioError::EventFd)?;
        let resample = EventFd::new(libc::EFD_NONBLOCK).map_err(VfioError::EventFd)?;
        vm.register_irq_with_resample(&trigger, &resample, gsi)
            .map_err(VfioError::Irqfd)?;
        Ok(Self {
            gsi,
            trigger,
            resample,
            vm: vm.clone(),
        })
    }

    /// The legacy GSI.
    pub fn gsi(&self) -> u32 {
        self.gsi
    }

    /// Serializable state.
    pub fn state(&self) -> VfioIntxState {
        VfioIntxState { gsi: self.gsi }
    }
}

impl Drop for VfioIntx {
    fn drop(&mut self) {
        if let Err(e) = self.vm.unregister_irq(&self.trigger, self.gsi) {
            error!("Failed to unregister VFIO INTx irqfd {}: {e}", self.gsi);
        }
        let mut allocator = self.vm.resource_allocator();
        if allocator.gsi_legacy_allocator.free_id(self.gsi).is_err() {
            error!("Failed to free VFIO INTx GSI {}", self.gsi);
        }
    }
}

/// Arm `mode` on `device`, disarming whatever was armed before.
pub fn set_irq_mode(
    device: &InternalVfioDevice,
    current: &mut VfioIrqMode,
    mode: VfioIrqMode,
    intx: Option<&VfioIntx>,
    msi: Option<&VfioMsi>,
    msix_fds: Option<Vec<&EventFd>>,
) -> Result<(), VfioError> {
    if *current == mode {
        return Ok(());
    }
    debug!("VFIO IRQ mode {current:?} -> {mode:?}");
    match *current {
        VfioIrqMode::None => {}
        VfioIrqMode::Intx => device.disable_irq(VFIO_PCI_INTX_IRQ_INDEX)?,
        VfioIrqMode::Msi(_) => device.disable_irq(VFIO_PCI_MSI_IRQ_INDEX)?,
        VfioIrqMode::Msix => device.disable_irq(VFIO_PCI_MSIX_IRQ_INDEX)?,
    }
    *current = VfioIrqMode::None;
    match mode {
        VfioIrqMode::None => {}
        VfioIrqMode::Intx => {
            let intx = intx.ok_or(VfioError::IrqModeUnavailable)?;
            device.enable_irq(VFIO_PCI_INTX_IRQ_INDEX, vec![&intx.trigger])?;
            device.set_irq_resample_fd(VFIO_PCI_INTX_IRQ_INDEX, vec![&intx.resample])?;
        }
        VfioIrqMode::Msi(_) => {
            let msi = msi.ok_or(VfioError::IrqModeUnavailable)?;
            device.enable_irq(VFIO_PCI_MSI_IRQ_INDEX, msi.eventfds())?;
        }
        VfioIrqMode::Msix => {
            let fds = msix_fds.ok_or(VfioError::IrqModeUnavailable)?;
            device.enable_irq(VFIO_PCI_MSIX_IRQ_INDEX, fds)?;
        }
    }
    *current = mode;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::tests::default_vmm;

    fn msi(control: u16) -> VfioMsi {
        let vmm = default_vmm();
        let vm = vmm.vm.as_kvm().unwrap().clone();
        VfioMsi::new(&vm, PciSBDF::new(0, 0, 3, 0), 0x40, control).unwrap()
    }

    #[test]
    fn test_msi_capability_layouts() {
        assert_eq!(VfioMsi::cap_size(0), 10);
        assert_eq!(VfioMsi::cap_size(MSI_CTL_64BIT), 14);
        assert_eq!(VfioMsi::cap_size(MSI_CTL_PER_VECTOR_MASK), 20);
        assert_eq!(
            VfioMsi::cap_size(MSI_CTL_64BIT | MSI_CTL_PER_VECTOR_MASK),
            24
        );
    }

    #[test]
    fn test_msi_write_respects_read_only_bits() {
        // 64 bit, 4 vectors capable.
        let mut msi = msi(MSI_CTL_64BIT | (2 << MSI_CTL_MMC_SHIFT));
        assert!(!msi.enabled());
        // The guest tries to clear the 64 bit flag while enabling 2 vectors.
        assert!(msi.write(0x42, &[0x11, 0x00]));
        assert!(msi.enabled());
        assert_eq!(msi.enabled_vectors(), 2);
        assert!(msi.is_64bit());
        // Address and data land in the 64 bit layout.
        msi.write(0x44, &0xfee0_0003u32.to_le_bytes());
        msi.write(0x48, &0x1u32.to_le_bytes());
        msi.write(0x4c, &0x4021u16.to_le_bytes());
        assert_eq!(msi.addr_lo(), 0xfee0_0000);
        assert_eq!(msi.addr_hi(), 1);
        assert_eq!(msi.data(), 0x4021);
        // Writes past the capability are ignored.
        assert!(!msi.write(0x50, &[0xff; 4]));
    }

    #[test]
    fn test_msi_read_merges_emulated_bytes() {
        let mut msi = msi(0);
        msi.write(0x44, &0xfee0_1000u32.to_le_bytes());
        // Capability id and next pointer stay the device's.
        assert_eq!(msi.read_register(0x10, 0xdead_5005) & 0xffff, 0x5005);
        assert_eq!(msi.read_register(0x11, 0), 0xfee0_1000);
    }

    #[test]
    fn test_msi_enabled_vectors_are_bounded_by_capability() {
        let mut msi = msi(1 << MSI_CTL_MMC_SHIFT);
        // Ask for 32 vectors while only 2 are capable.
        msi.write(0x42, &[0x51, 0x00]);
        assert_eq!(msi.enabled_vectors(), 2);
        assert_eq!(msi.eventfds().len(), 2);
    }

    #[test]
    fn test_msi_state_round_trip() {
        let vmm = default_vmm();
        let vm = vmm.vm.as_kvm().unwrap().clone();
        let mut msi = VfioMsi::new(&vm, PciSBDF::new(0, 0, 3, 0), 0x40, MSI_CTL_64BIT).unwrap();
        msi.write(0x42, &[0x01, 0x00]);
        msi.write(0x44, &0xfee0_2000u32.to_le_bytes());
        let state = msi.state();
        drop(msi);
        let restored = VfioMsi::restore(&vm, PciSBDF::new(0, 0, 3, 0), &state).unwrap();
        assert_eq!(restored.state(), state);
        assert!(restored.enabled());
    }
}
