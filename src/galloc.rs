use {
    crate::{
        block::{MemoryBlock, MemoryBlockFlavor},
        config::Config,
        device::MemoryDevice,
        error::AllocationError,
        types::{MemoryHeap, MemoryType},
        usage::{MemoryForUsage, UsageFlags},
        Dedicated, Request,
    },
    core::marker::PhantomData,
};

pub struct GpuAllocator<D: MemoryDevice> {
    dedicated_treshold: u64,
    preferred_dedicated_treshold: u64,
    transient_dedicated_treshold: u64,
    memory_for_usage: MemoryForUsage,
    memory_types: Box<[MemoryType]>,
    memory_heaps: Box<[Heap]>,
    marker: PhantomData<D>,
}

impl<D> GpuAllocator<D>
where
    D: MemoryDevice,
{
    #[cfg_attr(feature = "tracing", tracing::instrument)]
    pub fn new(config: Config, memory_types: &[MemoryType], memory_heaps: &[MemoryHeap]) -> Self {
        GpuAllocator {
            dedicated_treshold: config.dedicated_treshold,
            preferred_dedicated_treshold: config
                .preferred_dedicated_treshold
                .min(config.dedicated_treshold),

            transient_dedicated_treshold: config
                .transient_dedicated_treshold
                .max(config.dedicated_treshold),

            memory_for_usage: MemoryForUsage::new(memory_types),

            memory_types: memory_types.iter().copied().collect(),
            memory_heaps: memory_heaps
                .iter()
                .map(|heap| Heap {
                    size: heap.size,
                    used: 0,
                    allocated: 0,
                    deallocated: 0,
                })
                .collect(),

            marker: PhantomData,
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, device)))]
    pub fn alloc(
        &mut self,
        device: &D,
        request: Request,
    ) -> Result<MemoryBlock<D::Memory>, AllocationError> {
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

        let flavor = match request.dedicated {
            Dedicated::Required => MemoryBlockFlavor::Dedicated,
            Dedicated::Preferred if request.size >= self.preferred_dedicated_treshold => {
                MemoryBlockFlavor::Dedicated
            }
            _ if request.usage.contains(UsageFlags::TRANSIENT) => {
                if request.size > self.transient_dedicated_treshold {
                    MemoryBlockFlavor::Dedicated
                } else {
                    MemoryBlockFlavor::Linear
                }
            }
            _ if request.size > self.dedicated_treshold => MemoryBlockFlavor::Dedicated,
            _ => MemoryBlockFlavor::Buddy,
        };

        for &memory_type in self.memory_for_usage.types(request.usage) {
            match flavor {
                MemoryBlockFlavor::Dedicated => {
                    let heap = self.memory_types[memory_type as usize].heap;
                    let heap = &mut self.memory_heaps[heap as usize];
                    if !heap.budget() >= request.size {
                        continue;
                    }

                    #[cfg(feature = "tracing")]
                    tracing::debug!(
                        "Allocating memory object `{}@{}`",
                        request.size,
                        memory_type
                    );

                    let memory = device.allocate_memory(request.size, memory_type)?;
                    heap.alloc(request.size);

                    return Ok(MemoryBlock {
                        memory_type,
                        props: self.memory_types[memory_type as usize].props,
                        shared_memory_object: memory,
                        offset: 0,
                        size: request.size,
                        flavor: MemoryBlockFlavor::Dedicated,
                    });
                }
                MemoryBlockFlavor::Linear => todo!(),
                MemoryBlockFlavor::Buddy => todo!(),
            }
        }

        Err(AllocationError::OutOfDeviceMemory)
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, device)))]
    pub fn dealloc(&mut self, device: &D, block: MemoryBlock<D::Memory>) {
        match block.flavor {
            MemoryBlockFlavor::Dedicated => {
                device.deallocate_memory(block.shared_memory_object);
                let heap = self.memory_types[block.memory_type as usize].heap;
                self.memory_heaps[heap as usize].dealloc(block.size);
            }
            MemoryBlockFlavor::Linear => unreachable!(),
            MemoryBlockFlavor::Buddy => unreachable!(),
        }
    }
}

#[derive(Debug)]
struct Heap {
    size: u64,
    used: u64,
    allocated: u128,
    deallocated: u128,
}

impl Heap {
    fn budget(&mut self) -> u64 {
        self.size - self.used
    }

    fn alloc(&mut self, size: u64) {
        self.used += size;
        self.allocated += u128::from(size);
    }

    fn dealloc(&mut self, size: u64) {
        self.used -= size;
        self.deallocated += u128::from(size);
    }
}
