use {
    crate::{align_up, error::AllocationError, heap::Heap},
    alloc::vec::Vec,
    core::{convert::TryFrom as _, ptr::NonNull},
    galloc_types::{DeviceMapError, MemoryDevice, MemoryPropertyFlags},
};

#[cfg(feature = "tracing")]
use core::fmt::Debug as MemoryBounds;

#[cfg(not(feature = "tracing"))]
use core::any::Any as MemoryBounds;

#[derive(Debug)]
pub(crate) struct BuddyBlock<M> {
    pub memory: M,
    pub ptr: Option<NonNull<u8>>,
    pub offset: u64,
    pub size: u64,
    pub chunk: usize,
}

#[derive(Debug)]
struct Size {
    freelist: Vec<(usize, u64)>,
}

#[derive(Debug)]
struct Chunk<M> {
    memory: M,
    ptr: Option<NonNull<u8>>,
    _size: u64,
}

#[derive(Debug)]
pub(crate) struct BuddyAllocator<M> {
    minimal_size: u64,
    chunks: Vec<Chunk<M>>,
    sizes: Vec<Size>,
    memory_type: u32,
    props: MemoryPropertyFlags,
    atom_mask: u64,
}

impl<M> BuddyAllocator<M>
where
    M: MemoryBounds + 'static,
{
    pub fn new(
        minimal_size: u64,
        initial_dedicated_size: u64,
        memory_type: u32,
        props: MemoryPropertyFlags,
        atom_mask: u64,
    ) -> Self {
        assert!(
            minimal_size.is_power_of_two(),
            "Minimal allocation size of buddy allocator must be power of two"
        );
        assert!(
            initial_dedicated_size.is_power_of_two(),
            "Dedicated allocation size of buddy allocator must be power of two"
        );

        let initial_sizes = (initial_dedicated_size
            .trailing_zeros()
            .saturating_sub(minimal_size.trailing_zeros())) as usize;

        BuddyAllocator {
            minimal_size,
            chunks: Vec::new(),
            sizes: (0..initial_sizes)
                .map(|_| Size {
                    freelist: Vec::new(),
                })
                .collect(),
            memory_type,
            props,
            atom_mask: atom_mask | (minimal_size - 1),
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, device)))]
    pub unsafe fn alloc(
        &mut self,
        device: &impl MemoryDevice<M>,
        size: u64,
        align_mask: u64,
        heap: &mut Heap,
        allocations_remains: &mut u32,
    ) -> Result<BuddyBlock<M>, AllocationError>
    where
        M: Clone,
    {
        let align_mask = align_mask | self.atom_mask;

        let size = align_up(size, align_mask)
            .and_then(|size| size.checked_next_power_of_two())
            .ok_or(AllocationError::OutOfDeviceMemory)?;

        let size_index = size.trailing_zeros() - self.minimal_size.trailing_zeros();
        let size_index =
            usize::try_from(size_index).map_err(|_| AllocationError::OutOfDeviceMemory)?;

        if self.sizes.len() <= size_index {
            self.sizes.push(Size {
                freelist: Vec::new(),
            });
        }

        let mut candidate_size_index = size_index;
        let (free_chunk, free_offset, free_size) = loop {
            let candidate_size = &mut self.sizes[candidate_size_index];

            if let Some((free_chunk, free_offset)) = candidate_size.freelist.pop() {
                break (free_chunk, free_offset, candidate_size_index);
            }

            if self.sizes.len() == candidate_size_index + 1 {
                if *allocations_remains == 0 {
                    return Err(AllocationError::TooManyObjects);
                }

                let chunk_size = self.minimal_size << candidate_size_index;
                let memory = device.allocate_memory(chunk_size, self.memory_type)?;
                *allocations_remains -= 1;
                heap.alloc(chunk_size);

                let ptr = if self.host_visible() {
                    match device.map_memory(&memory, 0, chunk_size) {
                        Ok(ptr) => Some(ptr),
                        Err(DeviceMapError::OutOfDeviceMemory) => {
                            return Err(AllocationError::OutOfDeviceMemory)
                        }
                        Err(DeviceMapError::MapFailed) | Err(DeviceMapError::OutOfHostMemory) => {
                            return Err(AllocationError::OutOfHostMemory)
                        }
                    }
                } else {
                    None
                };

                self.chunks.push(Chunk {
                    memory,
                    ptr,
                    _size: chunk_size,
                });

                break (self.chunks.len() - 1, 0, candidate_size_index);
            }

            candidate_size_index += 1;
        };

        for back_size_index in (size_index + 1..free_size).rev() {
            let size = self.minimal_size << back_size_index;
            self.sizes[back_size_index]
                .freelist
                .push((free_chunk, free_offset + size));
        }

        let free_chunk_entry = &self.chunks[free_chunk];

        Ok(BuddyBlock {
            memory: free_chunk_entry.memory.clone(),
            ptr: free_chunk_entry
                .ptr
                .map(|ptr| NonNull::new_unchecked(ptr.as_ptr().add(free_offset as usize))),
            offset: free_offset,
            size,
            chunk: free_chunk,
        })
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, _device)))]
    pub unsafe fn dealloc(
        &mut self,
        _device: &impl MemoryDevice<M>,
        block: BuddyBlock<M>,
        _heap: &mut Heap,
        _allocations_remains: &mut u32,
    ) {
        let size_index =
            (block.size.trailing_zeros() - self.minimal_size.trailing_zeros()) as usize;

        self.sizes[size_index]
            .freelist
            .push((block.chunk, block.offset));
    }

    fn host_visible(&self) -> bool {
        self.props.contains(MemoryPropertyFlags::HOST_VISIBLE)
    }
}
