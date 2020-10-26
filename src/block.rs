use crate::types::MemoryPropertyFlags;

pub struct MemoryBlock<M> {
    pub(crate) memory_type: u32,
    pub(crate) props: MemoryPropertyFlags,
    pub(crate) shared_memory_object: M,
    pub(crate) offset: u64,
    pub(crate) size: u64,
    pub(crate) flavor: MemoryBlockFlavor,
}

pub(crate) enum MemoryBlockFlavor {
    Buddy,
    Linear,
    Dedicated,
}

impl<M> MemoryBlock<M> {
    /// Returns reference to parent memory object.
    pub fn memory(&self) -> &M {
        &self.shared_memory_object
    }

    /// Returns offset in bytes from start of memory object to start of this block.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Returns size of this memory block.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Returns memory property flags for parent memory object.
    pub fn props(&self) -> MemoryPropertyFlags {
        self.props
    }

    /// Returns index of type of parent memory object.
    pub fn memory_type(&self) -> u32 {
        self.memory_type
    }
}
