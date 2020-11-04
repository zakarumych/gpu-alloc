use {
    crate::{
        block::{MemoryBlock, MemoryBlockFlavor},
        buddy::{BuddyAllocator, BuddyBlock},
        config::Config,
        error::AllocationError,
        heap::Heap,
        linear::{LinearAllocator, LinearBlock},
        usage::{MemoryForUsage, UsageFlags},
        Dedicated, Request, Strategy,
    },
    alloc::boxed::Box,
    core::convert::TryFrom as _,
    gpu_alloc_types::{
        DeviceProperties, MemoryDevice, MemoryPropertyFlags, MemoryType, OutOfMemory,
    },
};

#[cfg(feature = "tracing")]
use core::fmt::Debug as MemoryBounds;

#[cfg(not(feature = "tracing"))]
use core::any::Any as MemoryBounds;

pub struct GpuAllocator<M> {
    dedicated_treshold: u64,
    preferred_dedicated_treshold: u64,
    transient_dedicated_treshold: u64,
    max_memory_allocation_size: u64,
    memory_for_usage: MemoryForUsage,
    memory_types: Box<[MemoryType]>,
    memory_heaps: Box<[Heap]>,
    allocations_remains: u32,
    non_coherent_atom_mask: u64,
    linear_chunk: u64,
    minimal_buddy_size: u64,
    initial_buddy_dedicated_size: u64,

    linear_allocators: Box<[Option<LinearAllocator<M>>]>,
    buddy_allocators: Box<[Option<BuddyAllocator<M>>]>,
}

impl<M> GpuAllocator<M>
where
    M: MemoryBounds + 'static,
{
    /// Creates  new instance of `GpuAllocator`.
    /// Provided `DeviceProperties` should match propertices of `MemoryDevice` that will be used
    /// with created `GpuAllocator` instance.
    #[cfg_attr(feature = "tracing", tracing::instrument)]
    pub fn new(config: Config, props: DeviceProperties<'_>) -> Self {
        assert!(
            props.non_coherent_atom_size.is_power_of_two(),
            "`non_coherent_atom_size` must be power of two"
        );

        assert!(
            isize::try_from(props.non_coherent_atom_size).is_ok(),
            "`non_coherent_atom_size` must fit host address space"
        );

        GpuAllocator {
            dedicated_treshold: config
                .dedicated_treshold
                .max(props.max_memory_allocation_size),
            preferred_dedicated_treshold: config
                .preferred_dedicated_treshold
                .min(config.dedicated_treshold)
                .max(props.max_memory_allocation_size),

            transient_dedicated_treshold: config
                .transient_dedicated_treshold
                .max(config.dedicated_treshold)
                .max(props.max_memory_allocation_size),

            max_memory_allocation_size: props.max_memory_allocation_size,

            memory_for_usage: MemoryForUsage::new(props.memory_types.as_ref()),

            memory_types: props.memory_types.as_ref().iter().copied().collect(),
            memory_heaps: props
                .memory_heaps
                .as_ref()
                .iter()
                .map(|heap| Heap::new(heap.size))
                .collect(),

            allocations_remains: props.max_memory_allocation_count,
            non_coherent_atom_mask: props.non_coherent_atom_size - 1,

            linear_chunk: config.linear_chunk,
            minimal_buddy_size: config.minimal_buddy_size,
            initial_buddy_dedicated_size: config.initial_buddy_dedicated_size,

            linear_allocators: props.memory_types.as_ref().iter().map(|_| None).collect(),
            buddy_allocators: props.memory_types.as_ref().iter().map(|_| None).collect(),
        }
    }

    /// Allocates memory block from specified `device` according to the `request`.
    ///
    /// # Safety
    ///
    /// * `device` must be one with `DeviceProperties` that were provided to create this `GpuAllocator` instance.
    /// * Same `device` instance must be used for all interactions with one `GpuAllocator` instance
    ///   and memory blocks allocated from it.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, device)))]
    pub unsafe fn alloc(
        &mut self,
        device: &impl MemoryDevice<M>,
        request: Request,
    ) -> Result<MemoryBlock<M>, AllocationError>
    where
        M: Clone,
    {
        self.alloc_internal(device, request, None)
    }

    /// Allocates memory block from specified `device` according to the `request`.
    /// This function allows user to force specific allocation strategy.
    /// Improper use can lead to suboptimal performance or too large overhead.
    /// Prefer `GpuAllocator::alloc` if doubt.
    ///
    /// # Safety
    ///
    /// * `device` must be one with `DeviceProperties` that were provided to create this `GpuAllocator` instance.
    /// * Same `device` instance must be used for all interactions with one `GpuAllocator` instance
    ///   and memory blocks allocated from it.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, device)))]
    pub unsafe fn alloc_with_strategy(
        &mut self,
        device: &impl MemoryDevice<M>,
        request: Request,
        strategy: Strategy,
    ) -> Result<MemoryBlock<M>, AllocationError>
    where
        M: Clone,
    {
        self.alloc_internal(device, request, Some(strategy))
    }

    pub unsafe fn alloc_internal(
        &mut self,
        device: &impl MemoryDevice<M>,
        mut request: Request,
        strategy: Option<Strategy>,
    ) -> Result<MemoryBlock<M>, AllocationError>
    where
        M: Clone,
    {
        request.usage = with_implicit_usage_flags(request.usage);

        if request.size > self.max_memory_allocation_size {
            return Err(AllocationError::OutOfDeviceMemory);
        }

        if 0 == self.memory_for_usage.mask(request.usage) & request.memory_types {
            #[cfg(feature = "tracing")]
            tracing::error!(
                "Cannot serve request {:?}, no memory among bitset `{}` support usage {:?}",
                request,
                request.memory_types,
                request.usage
            );

            return Err(AllocationError::NoCompatibleMemoryTypes);
        }

        let strategy = strategy.unwrap_or_else(|| match request.dedicated {
            Dedicated::Required => Strategy::Dedicated,
            Dedicated::Preferred
                if request.size >= self.preferred_dedicated_treshold
                    && self.allocations_remains > 0 =>
            {
                Strategy::Dedicated
            }
            _ if request.usage.contains(UsageFlags::TRANSIENT) => {
                if request.size > self.transient_dedicated_treshold && self.allocations_remains > 0
                {
                    Strategy::Dedicated
                } else {
                    Strategy::Linear
                }
            }
            _ if request.size > self.dedicated_treshold && self.allocations_remains > 0 => {
                Strategy::Dedicated
            }
            _ => Strategy::Buddy,
        });

        if let Strategy::Dedicated = strategy {
            if self.allocations_remains == 0 {
                return Err(AllocationError::TooManyObjects);
            }
        }

        for &index in self.memory_for_usage.types(request.usage) {
            let memory_type = &self.memory_types[index as usize];
            let heap = memory_type.heap;
            let heap = &mut self.memory_heaps[heap as usize];

            let map_mask = if host_visible_non_coherent(memory_type.props) {
                self.non_coherent_atom_mask
            } else {
                0
            };

            match strategy {
                Strategy::Dedicated => {
                    if !heap.budget() >= request.size {
                        continue;
                    }

                    if self.allocations_remains == 0 {
                        return Err(AllocationError::TooManyObjects);
                    }

                    #[cfg(feature = "tracing")]
                    tracing::debug!(
                        "Allocating memory object `{}@{:?}`",
                        request.size,
                        memory_type
                    );

                    match device.allocate_memory(request.size, index) {
                        Ok(memory) => {
                            self.allocations_remains -= 1;
                            heap.alloc(request.size);

                            return Ok(MemoryBlock {
                                memory_type: index,
                                props: memory_type.props,
                                memory,
                                offset: 0,
                                size: request.size,
                                map_mask,
                                mapped: false,
                                flavor: MemoryBlockFlavor::Dedicated,
                            });
                        }
                        Err(OutOfMemory::OutOfDeviceMemory) => continue,
                        Err(OutOfMemory::OutOfHostMemory) => {
                            return Err(AllocationError::OutOfHostMemory)
                        }
                    }
                }
                Strategy::Linear => {
                    let allocator = match &mut self.linear_allocators[index as usize] {
                        Some(allocator) => allocator,
                        slot => {
                            let memory_type = &self.memory_types[index as usize];
                            slot.get_or_insert(LinearAllocator::new(
                                self.linear_chunk.min(heap.size() / 32),
                                index,
                                memory_type.props,
                                if host_visible_non_coherent(memory_type.props) {
                                    self.non_coherent_atom_mask
                                } else {
                                    0
                                },
                            ))
                        }
                    };
                    let result = allocator.alloc(
                        device,
                        request.size,
                        request.align_mask,
                        request.usage,
                        heap,
                        &mut self.allocations_remains,
                    );

                    match result {
                        Ok(block) => {
                            return Ok(MemoryBlock {
                                memory_type: index,
                                props: memory_type.props,
                                memory: block.memory,
                                offset: block.offset,
                                size: block.size,
                                map_mask,
                                mapped: false,
                                flavor: MemoryBlockFlavor::Linear {
                                    chunk: block.chunk,
                                    ptr: block.ptr,
                                },
                            })
                        }
                        Err(AllocationError::OutOfDeviceMemory) => continue,
                        Err(err) => return Err(err),
                    }
                }
                Strategy::Buddy => {
                    let allocator = match &mut self.buddy_allocators[index as usize] {
                        Some(allocator) => allocator,
                        slot => {
                            let memory_type = &self.memory_types[index as usize];
                            slot.get_or_insert(BuddyAllocator::new(
                                self.minimal_buddy_size.min(heap.size() / 1024),
                                self.initial_buddy_dedicated_size.min(heap.size() / 32),
                                index,
                                memory_type.props,
                                if host_visible_non_coherent(memory_type.props) {
                                    self.non_coherent_atom_mask
                                } else {
                                    0
                                },
                            ))
                        }
                    };
                    let result = allocator.alloc(
                        device,
                        request.size,
                        request.align_mask,
                        heap,
                        &mut self.allocations_remains,
                    );

                    match result {
                        Ok(block) => {
                            return Ok(MemoryBlock {
                                memory_type: index,
                                props: memory_type.props,
                                memory: block.memory,
                                offset: block.offset,
                                size: block.size,
                                map_mask,
                                mapped: false,
                                flavor: MemoryBlockFlavor::Buddy {
                                    chunk: block.chunk,
                                    ptr: block.ptr,
                                    index: block.index,
                                },
                            })
                        }
                        Err(AllocationError::OutOfDeviceMemory) => continue,
                        Err(err) => return Err(err),
                    }
                }
            }
        }

        Err(AllocationError::OutOfDeviceMemory)
    }

    /// Deallocates memory block previously allocated from this `GpuAllocator` instance.
    ///
    /// # Safety
    ///
    /// * Memory block must have been allocated by this `GpuAllocator` instance
    /// * `device` must be one with `DeviceProperties` that were provided to create this `GpuAllocator` instance
    /// * Same `device` instance must be used for all interactions with one `GpuAllocator` instance
    ///   and memory blocks allocated from it
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, device)))]
    pub unsafe fn dealloc(&mut self, device: &impl MemoryDevice<M>, block: MemoryBlock<M>) {
        match block.flavor {
            MemoryBlockFlavor::Dedicated => {
                device.deallocate_memory(block.memory);
                self.allocations_remains += 1;
                let heap = self.memory_types[block.memory_type as usize].heap;
                self.memory_heaps[heap as usize].dealloc(block.size);
            }
            MemoryBlockFlavor::Linear { chunk, ptr } => {
                let memory_type = block.memory_type;
                let heap = self.memory_types[memory_type as usize].heap;
                let heap = &mut self.memory_heaps[heap as usize];

                let allocator = self.linear_allocators[memory_type as usize]
                    .as_mut()
                    .expect("Allocator should exist");

                allocator.dealloc(
                    device,
                    LinearBlock {
                        memory: block.memory,
                        offset: block.offset,
                        size: block.size,
                        ptr,
                        chunk,
                    },
                    heap,
                    &mut self.allocations_remains,
                );
            }
            MemoryBlockFlavor::Buddy { chunk, ptr, index } => {
                let memory_type = block.memory_type;
                let heap = self.memory_types[memory_type as usize].heap;
                let heap = &mut self.memory_heaps[heap as usize];

                let allocator = self.buddy_allocators[memory_type as usize]
                    .as_mut()
                    .expect("Allocator should exist");

                allocator.dealloc(
                    device,
                    BuddyBlock {
                        memory: block.memory,
                        offset: block.offset,
                        size: block.size,
                        index,
                        ptr,
                        chunk,
                    },
                    heap,
                    &mut self.allocations_remains,
                );
            }
        }
    }
}

fn host_visible_non_coherent(props: MemoryPropertyFlags) -> bool {
    (props & (MemoryPropertyFlags::HOST_COHERENT | MemoryPropertyFlags::HOST_VISIBLE))
        == MemoryPropertyFlags::HOST_VISIBLE
}

fn with_implicit_usage_flags(usage: UsageFlags) -> UsageFlags {
    if usage.is_empty() {
        UsageFlags::FAST_DEVICE_ACCESS
    } else if usage.intersects(UsageFlags::DOWNLOAD | UsageFlags::UPLOAD) {
        usage | UsageFlags::HOST_ACCESS
    } else {
        usage
    }
}
