use {
    crate::{align_down_usize, align_up_usize, error::MapError, greater},
    core::ptr::{copy_nonoverlapping, NonNull},
    galloc_types::{MemoryDevice, MemoryPropertyFlags},
};

pub struct MemoryBlock<M> {
    pub(crate) memory_type: u32,
    pub(crate) props: MemoryPropertyFlags,
    pub(crate) memory: M,
    pub(crate) offset: u64,
    pub(crate) size: u64,
    pub(crate) map_mask: usize,
    pub(crate) mapped: bool,
    pub(crate) flavor: MemoryBlockFlavor,
}

pub(crate) enum MemoryBlockFlavor {
    Dedicated,
    Linear {
        chunk: u64,
        ptr: Option<NonNull<u8>>,
    },
    Buddy {
        chunk: usize,
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

    #[inline(always)]
    pub unsafe fn map(
        &mut self,
        device: &impl MemoryDevice<M>,
        offset: usize,
        size: usize,
    ) -> Result<NonNull<u8>, MapError> {
        let ptr = self.map_memory_internal(device, offset, size)?;
        self.mapped = true;
        Ok(NonNull::new_unchecked(ptr))
    }

    #[inline(always)]
    pub unsafe fn unmap(&mut self, device: &impl MemoryDevice<M>) {
        self.unmap_memory_internal(device);
        self.mapped = false;
    }

    #[inline(always)]
    pub unsafe fn write(
        &self,
        device: &impl MemoryDevice<M>,
        offset: usize,
        data: &[u8],
    ) -> Result<(), MapError> {
        let size = data.len();
        let ptr = self.map_memory_internal(device, offset, size)?;
        copy_nonoverlapping(data.as_ptr(), ptr, size);
        Ok(())
    }

    #[inline(always)]
    pub unsafe fn read(
        &self,
        device: &impl MemoryDevice<M>,
        offset: usize,
        data: &mut [u8],
    ) -> Result<(), MapError> {
        let size = data.len();
        let ptr = self.map_memory_internal(device, offset, size)?;
        copy_nonoverlapping(ptr, data.as_mut_ptr(), size);
        Ok(())
    }

    unsafe fn map_memory_internal(
        &self,
        device: &impl MemoryDevice<M>,
        offset: usize,
        size: usize,
    ) -> Result<*mut u8, MapError> {
        assert!(!self.mapped, "Memory block is already mapped");

        let size = align_up_usize(size, self.map_mask)
            .expect("Requested memory range is out of bounds of this block");

        assert!(
            offset
                .checked_add(size)
                .map_or(false, |end| !greater(end, self.size)),
            "Requested memory range is out of bounds of this block"
        );

        let ptr = match self.flavor {
            MemoryBlockFlavor::Dedicated => {
                let aligned_offset = align_down_usize(offset, self.map_mask);
                let aligned_size = offset + size - aligned_offset;
                let ptr = device.map_memory(
                    &self.memory,
                    self.offset + aligned_offset as u64,
                    aligned_size as u64,
                )?;

                ptr.as_ptr().add(aligned_offset - offset)
            }
            MemoryBlockFlavor::Linear { ptr: Some(ptr), .. }
            | MemoryBlockFlavor::Buddy { ptr: Some(ptr), .. } => ptr.as_ptr().add(offset),
            _ => return Err(MapError::NonHostVisible),
        };

        Ok(ptr)
    }

    unsafe fn unmap_memory_internal(&self, device: &impl MemoryDevice<M>) {
        debug_assert!(self.mapped);
        match self.flavor {
            MemoryBlockFlavor::Dedicated => {
                device.unmap_memory(&self.memory);
            }
            MemoryBlockFlavor::Linear { .. } => {}
            MemoryBlockFlavor::Buddy { .. } => {}
        }
    }
}
