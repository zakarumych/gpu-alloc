use {
    crate::{align_up, error::AllocationError, heap::Heap, usage::UsageFlags},
    alloc::collections::VecDeque,
    core::{convert::TryFrom as _, ptr::NonNull},
    gpu_alloc_types::{DeviceMapError, MemoryDevice, MemoryPropertyFlags},
};

#[cfg(feature = "tracing")]
use core::fmt::Debug as MemoryBounds;

#[cfg(not(feature = "tracing"))]
use core::any::Any as MemoryBounds;

#[derive(Debug)]
pub(crate) struct LinearBlock<M> {
    pub memory: M,
    pub ptr: Option<NonNull<u8>>,
    pub offset: u64,
    pub size: u64,
    pub chunk: u64,
}

#[derive(Debug)]
struct Chunk<M> {
    memory: M,
    offset: u64,
    allocated: u64,
    ptr: Option<NonNull<u8>>,
}

impl<M> Chunk<M> {
    fn exhaust(self) -> ExhaustedChunk<M> {
        ExhaustedChunk {
            memory: self.memory,
            allocated: self.allocated,
        }
    }
}

#[derive(Debug)]
struct ExhaustedChunk<M> {
    memory: M,
    allocated: u64,
}

#[derive(Debug)]
struct Chunks<M> {
    ready: Option<Chunk<M>>,
    exhausted: VecDeque<Option<ExhaustedChunk<M>>>,
    offset: u64,
}

#[derive(Debug)]
pub(crate) struct LinearAllocator<M> {
    chunks: Chunks<M>,
    chunks_unmapped: Chunks<M>,
    chunk_size: u64,
    memory_type: u32,
    props: MemoryPropertyFlags,
    atom_mask: u64,
}

impl<M> LinearAllocator<M>
where
    M: MemoryBounds + 'static,
{
    pub fn new(
        chunk_size: u64,
        memory_type: u32,
        props: MemoryPropertyFlags,
        atom_mask: u64,
    ) -> Self {
        LinearAllocator {
            chunks: Chunks {
                ready: None,
                exhausted: VecDeque::new(),
                offset: 0,
            },
            chunks_unmapped: Chunks {
                ready: None,
                exhausted: VecDeque::new(),
                offset: 0,
            },
            chunk_size: min(chunk_size, isize::max_value()),
            memory_type,
            props,
            atom_mask,
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, device)))]
    pub unsafe fn alloc(
        &mut self,
        device: &impl MemoryDevice<M>,
        size: u64,
        align_mask: u64,
        usage: UsageFlags,
        heap: &mut Heap,
        allocations_remains: &mut u32,
    ) -> Result<LinearBlock<M>, AllocationError>
    where
        M: Clone,
    {
        debug_assert!(
            size <= self.chunk_size,
            "GpuAllocator must not request allocations (size {}) greater than chunks size ({})",
            size,
            self.chunk_size
        );

        let align_mask = align_mask | self.atom_mask;

        if self.host_visible() {
            if !usage.contains(UsageFlags::HOST_ACCESS) {
                match &mut self.chunks_unmapped.ready {
                    Some(ready) if fits(self.chunk_size, ready.allocated, size, align_mask) => {
                        let chunks_offset = self.chunks_unmapped.offset;
                        let exhausted = self.chunks_unmapped.exhausted.len() as u64;
                        return Ok(Self::alloc_from_chunk(
                            ready,
                            self.chunk_size,
                            chunks_offset,
                            exhausted,
                            size,
                            align_mask,
                        ));
                    }
                    ready => {
                        self.chunks_unmapped
                            .exhausted
                            .extend(Some(ready.take().map(Chunk::exhaust)));
                    }
                }
            }

            match &mut self.chunks.ready {
                Some(ready) if fits(self.chunk_size, ready.allocated, size, align_mask) => {
                    let chunks_offset = self.chunks.offset;
                    let exhausted = self.chunks.exhausted.len() as u64;
                    Ok(Self::alloc_from_chunk(
                        ready,
                        self.chunk_size,
                        chunks_offset,
                        exhausted,
                        size,
                        align_mask,
                    ))
                }

                ready => {
                    self.chunks
                        .exhausted
                        .extend(Some(ready.take().map(Chunk::exhaust)));

                    if *allocations_remains == 0 {
                        return Err(AllocationError::TooManyObjects);
                    }

                    let memory = device.allocate_memory(self.chunk_size, self.memory_type)?;

                    *allocations_remains -= 1;
                    heap.alloc(self.chunk_size);

                    match device.map_memory(&memory, 0, self.chunk_size) {
                        Ok(ptr) => {
                            let ready = ready.get_or_insert(Chunk {
                                ptr: Some(ptr),
                                memory,
                                allocated: 0,
                                offset: 0,
                            });

                            let chunks_offset = self.chunks.offset;
                            let exhausted = self.chunks.exhausted.len() as u64;
                            Ok(Self::alloc_from_chunk(
                                ready,
                                self.chunk_size,
                                chunks_offset,
                                exhausted,
                                size,
                                align_mask,
                            ))
                        }
                        Err(DeviceMapError::MapFailed) => {
                            if !usage.contains(UsageFlags::HOST_ACCESS) {
                                #[cfg(feature = "tracing")]
                                tracing::warn!("Failed to map host-visible memory in linear allocator. This request does not require host-access");

                                debug_assert!(self.chunks_unmapped.ready.is_none());
                                let ready = self.chunks_unmapped.ready.get_or_insert(Chunk {
                                    ptr: None,
                                    memory,
                                    allocated: 0,
                                    offset: 0,
                                });

                                let chunks_offset = self.chunks_unmapped.offset;
                                let exhausted = self.chunks_unmapped.exhausted.len() as u64;
                                Ok(Self::alloc_from_chunk(
                                    ready,
                                    self.chunk_size,
                                    chunks_offset,
                                    exhausted,
                                    size,
                                    align_mask,
                                ))
                            } else {
                                #[cfg(feature = "tracing")]
                                tracing::error!(
                                    "Failed to map host-visible memory in linear allocator"
                                );
                                device.deallocate_memory(memory);
                                *allocations_remains += 1;
                                heap.dealloc(self.chunk_size);

                                Err(AllocationError::OutOfHostMemory)
                            }
                        }
                        Err(DeviceMapError::OutOfDeviceMemory) => {
                            Err(AllocationError::OutOfDeviceMemory)
                        }
                        Err(DeviceMapError::OutOfHostMemory) => {
                            Err(AllocationError::OutOfHostMemory)
                        }
                    }
                }
            }
        } else {
            debug_assert!(
                !usage.contains(UsageFlags::HOST_ACCESS),
                "GpuAllocator must not try to alloc non-host-visible memory for request with `HOST_ACCESS` usage",
            );

            match &mut self.chunks_unmapped.ready {
                Some(ready) if fits(self.chunk_size, ready.allocated, size, align_mask) => {
                    let chunks_offset = self.chunks_unmapped.offset;
                    let exhausted = self.chunks_unmapped.exhausted.len() as u64;
                    Ok(Self::alloc_from_chunk(
                        ready,
                        self.chunk_size,
                        chunks_offset,
                        exhausted,
                        size,
                        align_mask,
                    ))
                }
                ready => {
                    self.chunks_unmapped
                        .exhausted
                        .extend(Some(ready.take().map(Chunk::exhaust)));

                    let memory = device.allocate_memory(self.chunk_size, self.memory_type)?;
                    *allocations_remains -= 1;
                    heap.alloc(self.chunk_size);

                    let ready = self.chunks_unmapped.ready.get_or_insert(Chunk {
                        ptr: None,
                        memory,
                        allocated: 0,
                        offset: 0,
                    });

                    let chunks_offset = self.chunks_unmapped.offset;
                    let exhausted = self.chunks_unmapped.exhausted.len() as u64;
                    Ok(Self::alloc_from_chunk(
                        ready,
                        self.chunk_size,
                        chunks_offset,
                        exhausted,
                        size,
                        align_mask,
                    ))
                }
            }
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, device)))]
    pub unsafe fn dealloc(
        &mut self,
        device: &impl MemoryDevice<M>,
        block: LinearBlock<M>,
        heap: &mut Heap,
        allocations_remains: &mut u32,
    ) {
        if block.ptr.is_some() {
            Self::dealloc_from_chunks(
                device,
                &mut self.chunks,
                block.chunk,
                self.chunk_size,
                heap,
                allocations_remains,
            );
        } else {
            Self::dealloc_from_chunks(
                device,
                &mut self.chunks_unmapped,
                block.chunk,
                self.chunk_size,
                heap,
                allocations_remains,
            );
        }
    }

    unsafe fn dealloc_from_chunks(
        device: &impl MemoryDevice<M>,
        chunks: &mut Chunks<M>,
        chunk: u64,
        chunk_size: u64,
        heap: &mut Heap,
        allocations_remains: &mut u32,
    ) {
        debug_assert!(chunk > chunks.offset, "Chunk index is less than chunk offset in this allocator. Probably incorrect allocator instance");
        let chunk_offset = chunk - chunks.offset;
        match usize::try_from(chunk_offset) {
            Ok(chunk_offset) => {
                if chunk_offset > chunks.exhausted.len() {
                    panic!("Chunk index is out of bounds. Probably incorrect allocator instance")
                }

                if chunk_offset == chunks.exhausted.len() {
                    let chunk = chunks.ready.as_mut().expect(
                        "Chunk index is out of bounds. Probably incorrect allocator instance",
                    );
                    chunk.allocated -= 1;
                } else {
                    let chunk = &mut chunks.exhausted[chunk_offset].as_mut().expect("Chunk index points to deallocated chunk. Probably incorrect allocator instance");
                    chunk.allocated -= 1;

                    if chunk.allocated == 0 {
                        let memory = chunks.exhausted[chunk_offset].take().unwrap().memory;
                        device.deallocate_memory(memory);
                        *allocations_remains += 1;
                        heap.dealloc(chunk_size);

                        if chunk_offset == 0 {
                            while let Some(None) = chunks.exhausted.get(0) {
                                chunks.exhausted.pop_front();
                                chunks.offset += 1;
                            }
                        }
                    }
                }
            }
            Err(_) => {
                panic!("Chunk index does not fit `usize`. Probably incorrect allocator instance")
            }
        }
    }

    fn host_visible(&self) -> bool {
        self.props.contains(MemoryPropertyFlags::HOST_VISIBLE)
    }

    unsafe fn alloc_from_chunk(
        chunk: &mut Chunk<M>,
        chunk_size: u64,
        chunks_offset: u64,
        exhausted: u64,
        size: u64,
        align_mask: u64,
    ) -> LinearBlock<M>
    where
        M: Clone,
    {
        debug_assert!(
            fits(chunk_size, chunk.allocated, size, align_mask),
            "Must be checked in caller"
        );

        let offset =
            align_up(chunk.offset, align_mask).expect("Chunk must be checked to fit allocation");

        chunk.offset = offset + size;
        chunk.allocated += 1;
        LinearBlock {
            memory: chunk.memory.clone(),
            ptr: chunk
                .ptr
                .map(|ptr| NonNull::new_unchecked(ptr.as_ptr().add(size as usize))),
            offset,
            size,
            chunk: chunks_offset + exhausted,
        }
    }
}

fn fits(chunk_size: u64, chunk_allocated: u64, size: u64, align_mask: u64) -> bool {
    align_up(chunk_allocated, align_mask)
        .and_then(|aligned| aligned.checked_add(size))
        .map_or(false, |size| size < chunk_size)
}

fn min<L, R>(l: L, r: R) -> L
where
    R: core::convert::TryInto<L>,
    L: Ord,
{
    match r.try_into() {
        Ok(r) => core::cmp::min(l, r),
        Err(_) => l,
    }
}
