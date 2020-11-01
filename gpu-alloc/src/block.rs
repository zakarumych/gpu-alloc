use {
    crate::{align_down, align_up, error::MapError},
    core::{
        convert::TryFrom as _,
        ptr::{copy_nonoverlapping, NonNull},
    },
    gpu_alloc_types::{MappedMemoryRange, MemoryDevice, MemoryPropertyFlags},
};

#[derive(Debug)]
pub struct MemoryBlock<M> {
    pub(crate) memory_type: u32,
    pub(crate) props: MemoryPropertyFlags,
    pub(crate) memory: M,
    pub(crate) offset: u64,
    pub(crate) size: u64,
    pub(crate) map_mask: u64,
    pub(crate) mapped: bool,
    pub(crate) flavor: MemoryBlockFlavor,
}

#[derive(Debug)]
pub(crate) enum MemoryBlockFlavor {
    Dedicated,
    Linear {
        chunk: u64,
        ptr: Option<NonNull<u8>>,
    },
    Buddy {
        chunk: usize,
        index: usize,
        ptr: Option<NonNull<u8>>,
    },
}

impl<M> MemoryBlock<M> {
    /// Returns reference to parent memory object.
    #[inline(always)]
    pub fn memory(&self) -> &M {
        &self.memory
    }

    /// Returns offset in bytes from start of memory object to start of this block.
    #[inline(always)]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Returns size of this memory block.
    #[inline(always)]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Returns memory property flags for parent memory object.
    #[inline(always)]
    pub fn props(&self) -> MemoryPropertyFlags {
        self.props
    }

    /// Returns index of type of parent memory object.
    #[inline(always)]
    pub fn memory_type(&self) -> u32 {
        self.memory_type
    }

    /// Returns pointer to mapped memory range of this block.
    /// This blocks becomes mapped.
    ///
    /// The user of returned pointer must guarantee that any previously submitted command that writes to this range has completed
    /// before the host reads from or writes to that range,
    /// and that any previously submitted command that reads from that range has completed
    /// before the host writes to that region.
    /// If the device memory was allocated without the `HOST_COHERENT` property flag set,
    /// these guarantees must be made for an extended range:
    /// the user must round down the start of the range to the nearest multiple of `non_coherent_atom_size`,
    /// and round the end of the range up to the nearest multiple of `non_coherent_atom_size`.
    ///
    /// # Panics
    ///
    /// This function panics if block is currently mapped.
    ///
    /// # Safety
    ///
    /// `block` must have been allocated from specified `device`.
    #[inline(always)]
    pub unsafe fn map(
        &mut self,
        device: &impl MemoryDevice<M>,
        offset: u64,
        size: usize,
    ) -> Result<NonNull<u8>, MapError> {
        self.assert_unmapped();

        let ptr = self.map_memory_internal(device, offset, size)?;
        self.mapped = true;
        Ok(NonNull::new_unchecked(ptr))
    }

    /// Unmaps memory range of this block that was previously mapped with `Block::map`.
    /// This block becomes unmapped.
    ///
    /// # Panics
    ///
    /// This function panics if this block is not currently mapped.
    ///
    /// # Safety
    ///
    /// `block` must have been allocated from specified `device`.
    #[inline(always)]
    pub unsafe fn unmap(&mut self, device: &impl MemoryDevice<M>) {
        self.assert_mapped();
        self.unmap_memory_internal(device);
        self.mapped = false;
    }

    /// Transiently maps block memory range and copies specified data
    /// to the mapped memory range.
    ///
    /// # Panics
    ///
    /// This function panics if block is currently mapped.
    ///
    /// # Safety
    ///
    /// `block` must have been allocated from specified `device`.
    /// The caller must guarantee that any previously submitted command that reads or writes to this range has completed.
    #[inline(always)]
    pub unsafe fn write_bytes(
        &self,
        device: &impl MemoryDevice<M>,
        offset: u64,
        data: &[u8],
    ) -> Result<(), MapError> {
        self.assert_unmapped();

        let size = data.len();
        let ptr = self.map_memory_internal(device, offset, size)?;

        copy_nonoverlapping(data.as_ptr(), ptr, size);
        let result = if !self.coherent() {
            let aligned_offset = align_down(offset, self.map_mask);
            let size = align_up(data.len() as u64, self.map_mask).unwrap();

            device.flush_memory_ranges(&[MappedMemoryRange {
                memory: &self.memory,
                offset: aligned_offset,
                size,
            }])
        } else {
            Ok(())
        };

        self.unmap_memory_internal(device);
        result.map_err(Into::into)
    }

    /// Transiently maps block memory range and copies specified data
    /// from the mapped memory range.
    ///
    /// # Panics
    ///
    /// This function panics if block is currently mapped.
    ///
    /// # Safety
    ///
    /// `block` must have been allocated from specified `device`.
    /// The caller must guarantee that any previously submitted command that reads to this range has completed.
    #[inline(always)]
    pub unsafe fn read_bytes(
        &self,
        device: &impl MemoryDevice<M>,
        offset: u64,
        data: &mut [u8],
    ) -> Result<(), MapError> {
        self.assert_unmapped();

        #[cfg(feature = "tracing")]
        if !self.cached() {
            tracing::warn!("Reading from non-cached memory may be slow. Consider allocating HOST_CACHED memory block for host reads.")
        }

        let size = data.len();
        let ptr = self.map_memory_internal(device, offset, size)?;
        let result = if !self.coherent() {
            let aligned_offset = align_down(offset, self.map_mask);
            let size = align_up(data.len() as u64, self.map_mask).unwrap();

            device.invalidate_memory_ranges(&[MappedMemoryRange {
                memory: &self.memory,
                offset: aligned_offset,
                size,
            }])
        } else {
            Ok(())
        };
        if result.is_ok() {
            copy_nonoverlapping(ptr, data.as_mut_ptr(), size);
        }

        self.unmap_memory_internal(device);
        result.map_err(Into::into)
    }

    unsafe fn map_memory_internal(
        &self,
        device: &impl MemoryDevice<M>,
        offset: u64,
        size: usize,
    ) -> Result<*mut u8, MapError> {
        let size_u64 = u64::try_from(size).expect("`size` doesn't fit device address space");
        let size = align_up(size_u64, self.map_mask)
            .expect("aligned `size` doesn't fit device address space");

        let aligned_offset = align_down(offset, self.map_mask);

        assert!(offset < self.size, "`offset` is out of memory block bounds");
        assert!(
            size_u64 <= self.size - offset,
            "`offset + size` is out of memory block bounds"
        );

        let ptr = match self.flavor {
            MemoryBlockFlavor::Dedicated => {
                let aligned_size = offset + size - aligned_offset;
                let ptr =
                    device.map_memory(&self.memory, self.offset + aligned_offset, aligned_size)?;

                let offset_align_shift = offset - aligned_offset;
                let offset_align_shift = isize::try_from(offset_align_shift)
                    .expect("`non_coherent_atom_size` is too large");

                ptr.as_ptr().offset(offset_align_shift)
            }
            MemoryBlockFlavor::Linear { ptr: Some(ptr), .. }
            | MemoryBlockFlavor::Buddy { ptr: Some(ptr), .. } => {
                let offset_isize = isize::try_from(offset)
                    .expect("Buddy and linear block should fit host address space");
                ptr.as_ptr().offset(offset_isize)
            }
            _ => return Err(MapError::NonHostVisible),
        };

        Ok(ptr)
    }

    unsafe fn unmap_memory_internal(&self, device: &impl MemoryDevice<M>) {
        match self.flavor {
            MemoryBlockFlavor::Dedicated => {
                device.unmap_memory(&self.memory);
            }
            MemoryBlockFlavor::Linear { .. } => {}
            MemoryBlockFlavor::Buddy { .. } => {}
        }
    }

    fn assert_mapped(&self) {
        assert!(self.mapped, "Memory block is not mapped");
    }

    fn assert_unmapped(&self) {
        assert!(!self.mapped, "Memory block is already mapped");
    }

    fn coherent(&self) -> bool {
        self.props.contains(MemoryPropertyFlags::HOST_COHERENT)
    }

    #[cfg(feature = "tracing")]
    fn cached(&self) -> bool {
        self.props.contains(MemoryPropertyFlags::HOST_CACHED)
    }
}
