use crate::error::AllocationError;

/// Abstract device that allocated memory to sub-allocate.
pub trait MemoryDevice {
    /// Memory object served by this device.
    type Memory: Clone;

    /// Allocate new memory object from device.
    /// This function may be expensive and even limit maximum number of memory
    /// objects allocated.
    /// Which is the reason for sub-allocation this crate provides.
    fn allocate_memory(&self, size: u64, memory_type: u32)
        -> Result<Self::Memory, AllocationError>;

    /// Deallocate memory object.
    fn deallocate_memory(&self, memory: Self::Memory);
}
