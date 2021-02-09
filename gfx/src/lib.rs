//!
//! # Gfx backend for `gpu-alloc`
//!  
//! # Usage example
//!
//! ```ignore
//!
//! use {
//!     eyre::eyre,
//!     gfx_backend_vulkan::Instance,
//!     gfx_hal::{
//!         adapter::{Gpu, PhysicalDevice as _},
//!         queue::QueueFamily as _,
//!         Features, Instance as _,
//!     },
//!     gpu_alloc::{Config, GpuAllocator, Request, UsageFlags},
//!     gpu_alloc_gfx::{gfx_device_properties, GfxMemoryDevice},
//! };
//!
//! fn main() -> eyre::Result<()> {
//!     color_eyre::install()?;
//!
//!     let instance =
//!         Instance::create("gpu_alloc-example", 1).map_err(|_| eyre!("Unsupported backend"))?;
//!
//!     let adapters = instance.enumerate_adapters();
//!
//!     let adapter = adapters
//!         .iter()
//!         .min_by_key(|a| {
//!             use gfx_hal::adapter::DeviceType::*;
//!             match a.info.device_type {
//!                 Other => 4,
//!                 IntegratedGpu => 1,
//!                 DiscreteGpu => 0,
//!                 VirtualGpu => 2,
//!                 Cpu => 3,
//!             }
//!         })
//!         .ok_or_else(|| eyre!("No adapters found"))?;
//!
//!     let queue_family = adapter
//!         .queue_families
//!         .iter()
//!         .min_by_key(|qf| {
//!             use gfx_hal::queue::QueueType::*;
//!             match qf.queue_type() {
//!                 General => 0,
//!                 Graphics => 1,
//!                 Compute => 3,
//!                 Transfer => 4,
//!             }
//!         })
//!         .ok_or_else(|| eyre!("No queue families found"))?;
//!
//!     let props = gfx_device_properties(adapter);
//!
//!     let Gpu { device, .. } = unsafe {
//!         adapter
//!             .physical_device
//!             .open(&[(queue_family, &[1.0])], Features::empty())
//!     }?;
//!
//!     let config = Config::i_am_potato();
//!
//!     let mut allocator = GpuAllocator::new(config, props);
//!
//!     let mut block = unsafe {
//!         allocator.alloc(
//!             GfxMemoryDevice::wrap(&device),
//!             Request {
//!                 size: 10,
//!                 align_mask: 1,
//!                 usage: UsageFlags::HOST_ACCESS,
//!                 memory_types: !0,
//!             },
//!         )
//!     }?;
//!
//!     unsafe {
//!         block.write_bytes(
//!             GfxMemoryDevice::wrap(&device),
//!             0,
//!             &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
//!         )
//!     }?;
//!
//!     unsafe { allocator.dealloc(GfxMemoryDevice::wrap(&device), block) }
//!
//!     Ok(())
//! }
//!

use {
    gfx_hal::{
        adapter::{Adapter, PhysicalDevice as _},
        device::{AllocationError, Device, MapError, OutOfMemory as GfxOutOfMemory},
        memory::{Properties, Segment},
        Backend, MemoryTypeId,
    },
    gpu_alloc_types::{
        AllocationFlags, DeviceMapError, DeviceProperties, MappedMemoryRange, MemoryDevice,
        MemoryHeap, MemoryPropertyFlags, MemoryType, OutOfMemory,
    },
    std::{convert::TryFrom as _, ptr::NonNull},
};

#[repr(transparent)]
pub struct GfxMemoryDevice<B: Backend> {
    device: B::Device,
}

impl<B> GfxMemoryDevice<B>
where
    B: Backend,
{
    pub fn wrap<D>(device: &D) -> &Self
    where
        D: Device<B>,
        B: Backend<Device = D>,
    {
        unsafe {
            // Safe because `Self` is `repr(transparent)`
            // with only non-zero-sized field being `D`.
            &*(device as *const D as *const Self)
        }
    }
}

impl<B> MemoryDevice<B::Memory> for GfxMemoryDevice<B>
where
    B: Backend,
{
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn allocate_memory(
        &self,
        size: u64,
        memory_type: u32,
        flags: AllocationFlags,
    ) -> Result<B::Memory, OutOfMemory> {
        debug_assert!(flags.is_empty(), "No allocation flags supported");

        let memory_type =
            MemoryTypeId(usize::try_from(memory_type).expect("memory_type out of bound"));

        match self.device.allocate_memory(memory_type, size) {
            Ok(memory) => Ok(memory),
            Err(AllocationError::OutOfMemory(GfxOutOfMemory::Device)) => {
                Err(OutOfMemory::OutOfDeviceMemory)
            }
            Err(AllocationError::OutOfMemory(GfxOutOfMemory::Host)) => {
                Err(OutOfMemory::OutOfHostMemory)
            }
            Err(AllocationError::TooManyObjects) => panic!("Too many objects"),
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn deallocate_memory(&self, memory: B::Memory) {
        self.device.free_memory(memory);
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn map_memory(
        &self,
        memory: &mut B::Memory,
        offset: u64,
        size: u64,
    ) -> Result<NonNull<u8>, DeviceMapError> {
        let result = self.device.map_memory(
            memory,
            Segment {
                offset,
                size: Some(size),
            },
        );

        match result {
            Ok(ptr) => Ok(NonNull::new(ptr).expect("Pointer to memory mapping must not be null")),
            Err(MapError::OutOfMemory(GfxOutOfMemory::Device)) => {
                Err(DeviceMapError::OutOfDeviceMemory)
            }
            Err(MapError::OutOfMemory(GfxOutOfMemory::Host)) => {
                Err(DeviceMapError::OutOfHostMemory)
            }
            Err(MapError::OutOfBounds) => panic!("Memory mapping out of bounds"),
            Err(MapError::MappingFailed) => Err(DeviceMapError::MapFailed),
            Err(MapError::Access) => panic!("Attempt to map non-host-visible memory"),
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn unmap_memory(&self, memory: &mut B::Memory) {
        self.device.unmap_memory(memory);
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn invalidate_memory_ranges(
        &self,
        ranges: &[MappedMemoryRange<'_, B::Memory>],
    ) -> Result<(), OutOfMemory> {
        self.device
            .invalidate_mapped_memory_ranges(ranges.iter().map(|range| {
                (
                    &*range.memory,
                    Segment {
                        offset: range.offset,
                        size: Some(range.size),
                    },
                )
            }))
            .map_err(|err| match err {
                GfxOutOfMemory::Device => OutOfMemory::OutOfDeviceMemory,
                GfxOutOfMemory::Host => OutOfMemory::OutOfHostMemory,
            })
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn flush_memory_ranges(
        &self,
        ranges: &[MappedMemoryRange<'_, B::Memory>],
    ) -> Result<(), OutOfMemory> {
        self.device
            .flush_mapped_memory_ranges(ranges.iter().map(|range| {
                (
                    &*range.memory,
                    Segment {
                        offset: range.offset,
                        size: Some(range.size),
                    },
                )
            }))
            .map_err(|err| match err {
                GfxOutOfMemory::Device => OutOfMemory::OutOfDeviceMemory,
                GfxOutOfMemory::Host => OutOfMemory::OutOfHostMemory,
            })
    }
}

/// Returns `DeviceProperties` from gfx's `Adapter`, required to create `GpuAllocator`.
pub fn gfx_device_properties<B>(adapter: &Adapter<B>) -> DeviceProperties<'static>
where
    B: Backend,
{
    let limits = adapter.physical_device.limits();
    let memory_properties = adapter.physical_device.memory_properties();
    DeviceProperties {
        max_memory_allocation_count: u32::try_from(limits.max_memory_allocation_count)
            .unwrap_or(u32::max_value()),
        max_memory_allocation_size: u64::max_value(),
        non_coherent_atom_size: u64::try_from(limits.non_coherent_atom_size)
            .unwrap_or(u64::max_value()),
        memory_types: memory_properties
            .memory_types
            .iter()
            .map(|memory_type| MemoryType {
                props: memory_properties_from_gfx(memory_type.properties),
                heap: u32::try_from(memory_type.heap_index)
                    .expect("Memory heap index should fit `u32`"),
            })
            .collect(),
        memory_heaps: memory_properties
            .memory_heaps
            .iter()
            .map(|&memory_heap| MemoryHeap {
                size: memory_heap.size,
            })
            .collect(),
        buffer_device_address: false,
    }
}

pub fn memory_properties_from_gfx(props: Properties) -> MemoryPropertyFlags {
    let mut result = MemoryPropertyFlags::empty();
    if props.contains(Properties::DEVICE_LOCAL) {
        result |= MemoryPropertyFlags::DEVICE_LOCAL;
    }
    if props.contains(Properties::CPU_VISIBLE) {
        result |= MemoryPropertyFlags::HOST_VISIBLE;
    }
    if props.contains(Properties::COHERENT) {
        result |= MemoryPropertyFlags::HOST_COHERENT;
    }
    if props.contains(Properties::CPU_CACHED) {
        result |= MemoryPropertyFlags::HOST_CACHED;
    }
    if props.contains(Properties::LAZILY_ALLOCATED) {
        result |= MemoryPropertyFlags::LAZILY_ALLOCATED;
    }
    result
}

pub fn memory_properties_to_gfx(props: MemoryPropertyFlags) -> Properties {
    let mut result = Properties::empty();
    if props.contains(MemoryPropertyFlags::DEVICE_LOCAL) {
        result |= Properties::DEVICE_LOCAL;
    }
    if props.contains(MemoryPropertyFlags::HOST_VISIBLE) {
        result |= Properties::CPU_VISIBLE;
    }
    if props.contains(MemoryPropertyFlags::HOST_COHERENT) {
        result |= Properties::COHERENT;
    }
    if props.contains(MemoryPropertyFlags::HOST_CACHED) {
        result |= Properties::CPU_CACHED;
    }
    if props.contains(MemoryPropertyFlags::LAZILY_ALLOCATED) {
        result |= Properties::LAZILY_ALLOCATED;
    }
    result
}
