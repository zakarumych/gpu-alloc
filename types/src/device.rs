use {
    crate::types::{MemoryHeap, MemoryType},
    core::ptr::NonNull,
};

#[derive(Debug)]
pub enum OutOfMemory {
    OutOfDeviceMemory,
    OutOfHostMemory,
}

#[derive(Debug)]
pub enum DeviceMapError {
    OutOfDeviceMemory,
    OutOfHostMemory,
    MapFailed,
}

#[derive(Debug)]
pub struct MappedMemoryRange<'a, M> {
    pub memory: &'a M,
    pub offset: u64,
    pub size: u64,
}

/// Properties of the device that will be used for allocating memory objects.
#[derive(Debug)]
pub struct DeviceProperties<T: AsRef<[MemoryType]>, H: AsRef<[MemoryHeap]>> {
    /// Array of memory types provided by the device.
    pub memory_types: T,

    /// Array of memory heaps provided by the device.
    pub memory_heaps: H,

    /// Maximum number of valid memory allocations that can exist simultaneously within the device.
    pub max_memory_allocation_count: u32,

    /// Maximum size for single allocation supported by the device.
    pub max_memory_allocation_size: u64,

    /// Atom size for host mappable non-coherent memory.
    pub non_coherent_atom_size: u64,
}

impl<T, H> DeviceProperties<T, H>
where
    T: AsRef<[MemoryType]>,
    H: AsRef<[MemoryHeap]>,
{
    pub fn by_ref(&self) -> DeviceProperties<&[MemoryType], &[MemoryHeap]> {
        DeviceProperties {
            memory_types: self.memory_types.as_ref(),
            memory_heaps: self.memory_heaps.as_ref(),
            max_memory_allocation_count: self.max_memory_allocation_count,
            max_memory_allocation_size: self.max_memory_allocation_size,
            non_coherent_atom_size: self.non_coherent_atom_size,
        }
    }
}

/// Abstract device that allocated memory to sub-allocate.
pub trait MemoryDevice<M> {
    /// Allocate new memory object from device.
    /// This function may be expensive and even limit maximum number of memory
    /// objects allocated.
    /// Which is the reason for sub-allocation this crate provides.
    ///
    /// # Safety
    ///
    /// `memory_type` must be valid index for memory type associated with this device.
    /// Retreiving this information is implementation specific.
    unsafe fn allocate_memory(&self, size: u64, memory_type: u32) -> Result<M, OutOfMemory>;

    /// Deallocate memory object.
    /// All clones of specified memory handle become invalid.
    ///
    /// # Safety
    ///
    /// Memory object must have been allocated from this device.
    unsafe fn deallocate_memory(&self, memory: M);

    /// Map region of device memory to host memory space.
    ///
    /// # Safety
    ///
    /// * Memory object must have been allocated from this device.
    /// * Memory object must not be already mapped.
    /// * Memory must be allocated from type with `HOST_VISIBLE` property.
    /// * `offset + size` must not overflow.
    /// * `offset + size` must not be larger than memory object size specified when
    ///   memory object was allocated from this device.
    unsafe fn map_memory(
        &self,
        memory: &M,
        offset: u64,
        size: u64,
    ) -> Result<NonNull<u8>, DeviceMapError>;

    /// Unmap previously mapped memory region.
    ///
    /// # Safety
    ///
    /// * Memory object must have been allocated from this device.
    /// * Memory object must be mapped
    unsafe fn unmap_memory(&self, memory: &M);

    /// Invalidates ranges of memory mapped regions.
    ///
    /// # Safety
    ///
    /// * Memory objects must have been allocated from this device.
    /// * `offset` and `size` in each element of `ranges` must specify
    ///   subregion of currently mapped memory region
    /// * if `memory` in some element of `ranges` does not contain `HOST_COHERENT` property
    ///   then `offset` and `size` of that element must be multiple of `non_coherent_atom_size`.
    unsafe fn invalidate_memory_ranges(
        &self,
        ranges: &[MappedMemoryRange<'_, M>],
    ) -> Result<(), OutOfMemory>;

    /// Flushes ranges of memory mapped regions.
    ///
    /// # Safety
    ///
    /// * Memory objects must have been allocated from this device.
    /// * `offset` and `size` in each element of `ranges` must specify
    ///   subregion of currently mapped memory region
    /// * if `memory` in some element of `ranges` does not contain `HOST_COHERENT` property
    ///   then `offset` and `size` of that element must be multiple of `non_coherent_atom_size`.
    unsafe fn flush_memory_ranges(
        &self,
        ranges: &[MappedMemoryRange<'_, M>],
    ) -> Result<(), OutOfMemory>;
}
