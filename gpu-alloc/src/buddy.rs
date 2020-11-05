use {
    crate::{align_up, error::AllocationError, heap::Heap, slab::Slab},
    alloc::vec::Vec,
    core::{convert::TryFrom as _, hint::unreachable_unchecked, mem::replace, ptr::NonNull},
    gpu_alloc_types::{AllocationFlags, DeviceMapError, MemoryDevice, MemoryPropertyFlags},
};

#[cfg(feature = "tracing")]
use core::fmt::Debug as MemoryBounds;

#[cfg(not(feature = "tracing"))]
use core::any::Any as MemoryBounds;

#[derive(Debug)]
pub(crate) struct BuddyBlock<M> {
    pub memory: M,
    pub ptr: Option<NonNull<u8>>,
    pub size: u64,
    pub chunk: usize,
    pub offset: u64,
    pub index: usize,
}

#[derive(Clone, Copy)]
enum PairState {
    Exhausted,
    Ready {
        ready: Side,
        next: usize,
        prev: usize,
    },
}

impl PairState {
    unsafe fn replace_next(&mut self, value: usize) -> usize {
        match self {
            PairState::Exhausted => unreachable_unchecked(),
            PairState::Ready { next, .. } => replace(next, value),
        }
    }

    unsafe fn replace_prev(&mut self, value: usize) -> usize {
        match self {
            PairState::Exhausted => unreachable_unchecked(),
            PairState::Ready { prev, .. } => replace(prev, value),
        }
    }
}

#[derive(Clone, Copy)]
enum Side {
    Left,
    Right,
}
use Side::*;
impl Side {
    fn bit(&self) -> u8 {
        match self {
            Left => 0,
            Right => 1,
        }
    }
}

struct PairEntry {
    state: PairState,
    chunk: usize,
    offset: u64,
    parent: Option<usize>,
}

struct SizeBlockEntry {
    chunk: usize,
    offset: u64,
    index: usize,
}

struct Size {
    ready: usize,
    pairs: Slab<PairEntry>,
}

enum Release {
    None,
    Parent(usize),
    Chunk(usize),
}

impl Size {
    fn new() -> Self {
        Size {
            pairs: Slab::new(),
            ready: 0,
        }
    }

    fn add_pair_and_acquire_left(
        &mut self,
        chunk: usize,
        offset: u64,
        parent: Option<usize>,
    ) -> SizeBlockEntry {
        let index = self.pairs.insert(PairEntry {
            state: PairState::Exhausted,
            chunk,
            offset,
            parent,
        });

        // if self.ready >= self.pairs.len() {
        self.ready = index;
        let entry = unsafe { self.pairs.get_unchecked_mut(index) };
        entry.state = PairState::Ready {
            next: index,
            prev: index,
            ready: Right, // Left is allocated.
        };
        // } else {
        //     unsafe { unreachable_unchecked() }

        // let next = self.ready;
        // let next_entry = unsafe { self.pairs.get_unchecked_mut(next) };
        // let prev = unsafe { next_entry.state.replace_prev(index) };

        // let prev_entry = unsafe { self.pairs.get_unchecked_mut(prev) };
        // let prev_next = unsafe { prev_entry.state.replace_next(index) };
        // debug_assert_eq!(prev_next, next);

        // let entry = unsafe { self.pairs.get_unchecked_mut(index) };
        // entry.state = PairState::Ready {
        //     next,
        //     prev,
        //     ready: Right, // Left is allocated.
        // }
        // }

        SizeBlockEntry {
            chunk,
            offset,
            index,
        }
    }

    fn acquire(&mut self, size: u64) -> Option<SizeBlockEntry> {
        if self.ready >= self.pairs.len() {
            return None;
        }

        let entry = unsafe { self.pairs.get_unchecked_mut(self.ready) };
        let chunk = entry.chunk;
        let offset = entry.offset;

        let bit = match entry.state {
            PairState::Exhausted => unsafe { unreachable_unchecked() },
            PairState::Ready { ready, next, prev } => {
                entry.state = PairState::Exhausted;

                let prev_entry = unsafe { self.pairs.get_unchecked_mut(prev) };
                let prev_next = unsafe { prev_entry.state.replace_next(next) };
                debug_assert_eq!(prev_next, self.ready);

                let next_entry = unsafe { self.pairs.get_unchecked_mut(next) };
                let next_prev = unsafe { next_entry.state.replace_prev(prev) };
                debug_assert_eq!(next_prev, self.ready);
                ready.bit()
            }
        };

        Some(SizeBlockEntry {
            chunk,
            offset: offset + bit as u64 * size,
            index: (self.ready << 1) | bit as usize,
        })
    }

    fn release(&mut self, index: usize) -> Release {
        let side = match index & 1 {
            0 => Left,
            1 => Right,
            _ => unsafe { unreachable_unchecked() },
        };
        let index = index >> 1;

        let len = self.pairs.len();
        let entry = self.pairs.get_mut(index);

        let chunk = entry.chunk;
        let offset = entry.offset;
        let parent = entry.parent;

        match (entry.state, side) {
            (PairState::Exhausted, side) => {
                if self.ready == len {
                    entry.state = PairState::Ready {
                        ready: side,
                        next: index,
                        prev: index,
                    };
                    self.ready = index;
                } else {
                    debug_assert!(self.ready < len);

                    let next = self.ready;
                    let next_entry = unsafe { self.pairs.get_unchecked_mut(next) };
                    let prev = unsafe { next_entry.state.replace_prev(index) };

                    let prev_entry = unsafe { self.pairs.get_unchecked_mut(prev) };
                    let prev_next = unsafe { prev_entry.state.replace_next(index) };
                    debug_assert_eq!(prev_next, next);

                    let entry = unsafe { self.pairs.get_unchecked_mut(index) };
                    entry.state = PairState::Ready {
                        ready: side,
                        next,
                        prev,
                    };
                }
                Release::None
            }
            (PairState::Ready { ready: Left, .. }, Left)
            | (PairState::Ready { ready: Right, .. }, Right) => {
                panic!("Attempt to dealloate already free block")
            }

            (
                PairState::Ready {
                    ready: Left,
                    next,
                    prev,
                },
                Side::Right,
            )
            | (
                PairState::Ready {
                    ready: Right,
                    next,
                    prev,
                },
                Side::Left,
            ) => {
                let prev_entry = unsafe { self.pairs.get_unchecked_mut(prev) };
                let prev_next = unsafe { prev_entry.state.replace_next(next) };
                debug_assert_eq!(prev_next, index);

                let next_entry = unsafe { self.pairs.get_unchecked_mut(next) };
                let next_prev = unsafe { next_entry.state.replace_prev(prev) };
                debug_assert_eq!(next_prev, index);

                match parent {
                    Some(parent) => Release::Parent(parent),
                    None => {
                        debug_assert_eq!(offset, 0);
                        Release::Chunk(chunk)
                    }
                }
            }
        }
    }
}

struct Chunk<M> {
    memory: M,
    ptr: Option<NonNull<u8>>,
    size: u64,
}

pub(crate) struct BuddyAllocator<M> {
    minimal_size: u64,
    chunks: Slab<Chunk<M>>,
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
            chunks: Slab::new(),
            sizes: (0..initial_sizes).map(|_| Size::new()).collect(),
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
        flags: AllocationFlags,
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
            self.sizes.push(Size::new());
        }

        let host_visible = self.host_visible();

        let mut candidate_size_index = size_index;
        let (mut entry, entry_size_index) = loop {
            let sizes_len = self.sizes.len();

            let candidate_size_entry = &mut self.sizes[candidate_size_index];
            let candidate_size = self.minimal_size << candidate_size_index;

            if let Some(entry) = candidate_size_entry.acquire(candidate_size) {
                break (entry, candidate_size_index);
            }

            if sizes_len == candidate_size_index + 1 {
                // That's size of device allocation.
                if *allocations_remains == 0 {
                    return Err(AllocationError::TooManyObjects);
                }

                let chunk_size = self.minimal_size << (candidate_size_index + 1);
                let memory = device.allocate_memory(chunk_size, self.memory_type, flags)?;
                *allocations_remains -= 1;
                heap.alloc(chunk_size);

                let ptr = if host_visible {
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

                let chunk = self.chunks.insert(Chunk {
                    memory,
                    ptr,
                    size: chunk_size,
                });

                let entry = candidate_size_entry.add_pair_and_acquire_left(chunk, 0, None);

                break (entry, candidate_size_index);
            }

            candidate_size_index += 1;
        };

        for size_index in (size_index..entry_size_index).rev() {
            let size_entry = &mut self.sizes[size_index];
            entry =
                size_entry.add_pair_and_acquire_left(entry.chunk, entry.offset, Some(entry.index));
        }

        let chunk_entry = self.chunks.get_unchecked(entry.chunk);

        Ok(BuddyBlock {
            memory: chunk_entry.memory.clone(),
            ptr: chunk_entry
                .ptr
                .map(|ptr| NonNull::new_unchecked(ptr.as_ptr().add(entry.offset as usize))),
            size,
            offset: entry.offset,
            chunk: entry.chunk,
            index: entry.index,
        })
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, device)))]
    pub unsafe fn dealloc(
        &mut self,
        device: &impl MemoryDevice<M>,
        block: BuddyBlock<M>,
        heap: &mut Heap,
        allocations_remains: &mut u32,
    ) {
        debug_assert!(block.size.is_power_of_two());

        let size_index =
            (block.size.trailing_zeros() - self.minimal_size.trailing_zeros()) as usize;

        let mut release_index = block.index;
        let mut release_size_index = size_index;

        loop {
            match self.sizes[release_size_index].release(release_index) {
                Release::Parent(parent) => {
                    release_size_index += 1;
                    release_index = parent;
                }
                Release::Chunk(chunk) => {
                    debug_assert_eq!(chunk, block.chunk);
                    let chunk = self.chunks.remove(chunk);
                    debug_assert_eq!(chunk.size, self.minimal_size << (release_size_index + 1));
                    device.deallocate_memory(chunk.memory);
                    *allocations_remains += 1;
                    heap.dealloc(chunk.size);

                    return;
                }
                Release::None => return,
            }
        }
    }

    fn host_visible(&self) -> bool {
        self.props.contains(MemoryPropertyFlags::HOST_VISIBLE)
    }
}
