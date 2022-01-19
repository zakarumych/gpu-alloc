//!
//! # Erupt backend for `gpu-alloc`
//!  
//! # Usage example
//!
//! ```ignore
//! use {
//!     erupt::{vk1_0, DeviceLoader, EntryLoader, InstanceLoader},
//!     gpu_alloc::{Config, GpuAllocator, Request, UsageFlags},
//!     gpu_alloc_erupt::{device_properties, EruptMemoryDevice},
//!     std::ffi::CStr,
//! };
//!
//! fn main() -> eyre::Result<()> {
//!     color_eyre::install()?;
//!
//!     let entry = EntryLoader::new()?;
//!
//!     let instance = unsafe {
//!         InstanceLoader::new(
//!             &entry,
//!             &vk1_0::InstanceCreateInfo::default()
//!                 .into_builder()
//!                 .application_info(
//!                     &vk1_0::ApplicationInfo::default()
//!                         .into_builder()
//!                         .engine_name(CStr::from_bytes_with_nul(b"GpuAlloc\0").unwrap())
//!                         .engine_version(1)
//!                         .application_name(CStr::from_bytes_with_nul(b"GpuAllocApp\0").unwrap())
//!                         .application_version(1)
//!                         .api_version(entry.instance_version()),
//!                 ),
//!             None,
//!         )
//!     }?;
//!
//!     let physical_devices = unsafe { instance.enumerate_physical_devices(None) }.result()?;
//!     let physical_device = physical_devices[0];
//!
//!     let props = unsafe { device_properties(&instance, physical_device) }?;
//!
//!     let device = unsafe {
//!         DeviceLoader::new(
//!             &instance,
//!             physical_device,
//!             &vk1_0::DeviceCreateInfoBuilder::new().queue_create_infos(&[
//!                 vk1_0::DeviceQueueCreateInfoBuilder::new()
//!                     .queue_family_index(0)
//!                     .queue_priorities(&[0f32]),
//!             ]),
//!             None,
//!         )
//!     }?;
//!
//!     let config = Config::i_am_potato();
//!
//!     let mut allocator = GpuAllocator::new(config, props);
//!
//!     let mut block = unsafe {
//!         allocator.alloc(
//!             EruptMemoryDevice::wrap(&device),
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
//!             EruptMemoryDevice::wrap(&device),
//!             0,
//!             &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
//!         )
//!     }?;
//!
//!     unsafe { allocator.dealloc(EruptMemoryDevice::wrap(&device), block) }
//!
//!     Ok(())
//! }
//! ```
//!

use std::ptr::NonNull;

use erupt::{vk::MemoryMapFlags, vk1_0, vk1_1, DeviceLoader, ExtendableFrom, InstanceLoader};
use gpu_alloc_types::{
    AllocationFlags, DeviceMapError, DeviceProperties, MappedMemoryRange, MemoryDevice, MemoryHeap,
    MemoryPropertyFlags, MemoryType, OutOfMemory,
};
use tinyvec::TinyVec;

#[repr(transparent)]
pub struct EruptMemoryDevice {
    device: DeviceLoader,
}

impl EruptMemoryDevice {
    pub fn wrap(device: &DeviceLoader) -> &Self {
        unsafe {
            // Safe because `Self` is `repr(transparent)`
            // with only field being `DeviceLoader`.
            &*(device as *const DeviceLoader as *const Self)
        }
    }
}

impl MemoryDevice<vk1_0::DeviceMemory> for EruptMemoryDevice {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn allocate_memory(
        &self,
        size: u64,
        memory_type: u32,
        flags: AllocationFlags,
    ) -> Result<vk1_0::DeviceMemory, OutOfMemory> {
        assert!((flags & !(AllocationFlags::DEVICE_ADDRESS)).is_empty());

        let mut info = vk1_0::MemoryAllocateInfoBuilder::new()
            .allocation_size(size)
            .memory_type_index(memory_type);

        let mut info_flags;

        if flags.contains(AllocationFlags::DEVICE_ADDRESS) {
            info_flags = vk1_1::MemoryAllocateFlagsInfoBuilder::new()
                .flags(vk1_1::MemoryAllocateFlags::DEVICE_ADDRESS);
            info = info.extend_from(&mut info_flags);
        }

        match self.device.allocate_memory(&info, None).result() {
            Ok(memory) => Ok(memory),
            Err(vk1_0::Result::ERROR_OUT_OF_DEVICE_MEMORY) => Err(OutOfMemory::OutOfDeviceMemory),
            Err(vk1_0::Result::ERROR_OUT_OF_HOST_MEMORY) => Err(OutOfMemory::OutOfHostMemory),
            Err(vk1_0::Result::ERROR_TOO_MANY_OBJECTS) => panic!("Too many objects"),
            Err(err) => panic!("Unexpected Vulkan error: `{}`", err),
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn deallocate_memory(&self, memory: vk1_0::DeviceMemory) {
        self.device.free_memory(memory, None);
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn map_memory(
        &self,
        memory: &mut vk1_0::DeviceMemory,
        offset: u64,
        size: u64,
    ) -> Result<NonNull<u8>, DeviceMapError> {
        match self
            .device
            .map_memory(*memory, offset, size, MemoryMapFlags::empty())
            .result()
        {
            Ok(ptr) => {
                Ok(NonNull::new(ptr as *mut u8)
                    .expect("Pointer to memory mapping must not be null"))
            }
            Err(vk1_0::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                Err(DeviceMapError::OutOfDeviceMemory)
            }
            Err(vk1_0::Result::ERROR_OUT_OF_HOST_MEMORY) => Err(DeviceMapError::OutOfHostMemory),
            Err(vk1_0::Result::ERROR_MEMORY_MAP_FAILED) => Err(DeviceMapError::MapFailed),
            Err(err) => panic!("Unexpected Vulkan error: `{}`", err),
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn unmap_memory(&self, memory: &mut vk1_0::DeviceMemory) {
        self.device.unmap_memory(*memory);
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn invalidate_memory_ranges(
        &self,
        ranges: &[MappedMemoryRange<'_, vk1_0::DeviceMemory>],
    ) -> Result<(), OutOfMemory> {
        self.device
            .invalidate_mapped_memory_ranges(
                &ranges
                    .iter()
                    .map(|range| {
                        vk1_0::MappedMemoryRangeBuilder::new()
                            .memory(*range.memory)
                            .offset(range.offset)
                            .size(range.size)
                    })
                    .collect::<TinyVec<[_; 4]>>(),
            )
            .result()
            .map_err(|err| match err {
                vk1_0::Result::ERROR_OUT_OF_DEVICE_MEMORY => OutOfMemory::OutOfDeviceMemory,
                vk1_0::Result::ERROR_OUT_OF_HOST_MEMORY => OutOfMemory::OutOfHostMemory,
                err => panic!("Unexpected Vulkan error: `{}`", err),
            })
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn flush_memory_ranges(
        &self,
        ranges: &[MappedMemoryRange<'_, vk1_0::DeviceMemory>],
    ) -> Result<(), OutOfMemory> {
        self.device
            .flush_mapped_memory_ranges(
                &ranges
                    .iter()
                    .map(|range| {
                        vk1_0::MappedMemoryRangeBuilder::new()
                            .memory(*range.memory)
                            .offset(range.offset)
                            .size(range.size)
                    })
                    .collect::<TinyVec<[_; 4]>>(),
            )
            .result()
            .map_err(|err| match err {
                vk1_0::Result::ERROR_OUT_OF_DEVICE_MEMORY => OutOfMemory::OutOfDeviceMemory,
                vk1_0::Result::ERROR_OUT_OF_HOST_MEMORY => OutOfMemory::OutOfHostMemory,
                err => panic!("Unexpected Vulkan error: `{}`", err),
            })
    }
}

/// Returns `DeviceProperties` from erupt's `InstanceLoader` for specified `PhysicalDevice`, required to create `GpuAllocator`.
///
/// # Safety
///
/// `physical_device` must be queried from `Instance` associated with this `instance`.
/// Even if returned properties' field `buffer_device_address` is set to true,
/// feature `PhysicalDeviceBufferDeviceAddressFeatures::buffer_derive_address`  must be enabled explicitly on device creation
/// and extension "VK_KHR_buffer_device_address" for Vulkan prior 1.2.
/// Otherwise the field must be set to false before passing to `GpuAllocator::new`.
pub unsafe fn device_properties(
    instance: &InstanceLoader,
    physical_device: vk1_0::PhysicalDevice,
) -> Result<DeviceProperties<'static>, vk1_0::Result> {
    use {
        erupt::{
            extensions::khr_buffer_device_address::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            vk1_1::PhysicalDeviceFeatures2, vk1_2::PhysicalDeviceBufferDeviceAddressFeatures,
        },
        std::ffi::CStr,
    };

    let limits = instance
        .get_physical_device_properties(physical_device)
        .limits;

    let memory_properties = instance.get_physical_device_memory_properties(physical_device);

    let buffer_device_address = if instance.enabled().vk1_1
        || instance.enabled().khr_get_physical_device_properties2
    {
        let mut bda_features_available = instance.enabled().vk1_2;

        if !bda_features_available {
            let extensions = instance
                .enumerate_device_extension_properties(physical_device, None, None)
                .result()?;

            bda_features_available = extensions.iter().any(|ext| {
                let name = CStr::from_bytes_with_nul({
                    std::slice::from_raw_parts(
                        ext.extension_name.as_ptr() as *const _,
                        ext.extension_name.len(),
                    )
                });
                if let Ok(name) = name {
                    name == { CStr::from_ptr(KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) }
                } else {
                    false
                }
            });
        }

        if bda_features_available {
            let features = PhysicalDeviceFeatures2::default().into_builder();
            let mut bda_features = PhysicalDeviceBufferDeviceAddressFeatures::default();
            let features = features.extend_from(&mut bda_features);
            instance
                .get_physical_device_features2(physical_device, Some(features.build_dangling()));
            bda_features.buffer_device_address != 0
        } else {
            false
        }
    } else {
        false
    };

    Ok(DeviceProperties {
        max_memory_allocation_count: limits.max_memory_allocation_count,
        max_memory_allocation_size: u64::max_value(), // FIXME: Can query this information if instance is v1.1
        non_coherent_atom_size: limits.non_coherent_atom_size,
        memory_types: memory_properties.memory_types
            [..memory_properties.memory_type_count as usize]
            .iter()
            .map(|memory_type| MemoryType {
                props: memory_properties_from_erupt(memory_type.property_flags),
                heap: memory_type.heap_index,
            })
            .collect(),
        memory_heaps: memory_properties.memory_heaps
            [..memory_properties.memory_heap_count as usize]
            .iter()
            .map(|&memory_heap| MemoryHeap {
                size: memory_heap.size,
            })
            .collect(),
        buffer_device_address,
    })
}

pub fn memory_properties_from_erupt(props: vk1_0::MemoryPropertyFlags) -> MemoryPropertyFlags {
    let mut result = MemoryPropertyFlags::empty();
    if props.contains(vk1_0::MemoryPropertyFlags::DEVICE_LOCAL) {
        result |= MemoryPropertyFlags::DEVICE_LOCAL;
    }
    if props.contains(vk1_0::MemoryPropertyFlags::HOST_VISIBLE) {
        result |= MemoryPropertyFlags::HOST_VISIBLE;
    }
    if props.contains(vk1_0::MemoryPropertyFlags::HOST_COHERENT) {
        result |= MemoryPropertyFlags::HOST_COHERENT;
    }
    if props.contains(vk1_0::MemoryPropertyFlags::HOST_CACHED) {
        result |= MemoryPropertyFlags::HOST_CACHED;
    }
    if props.contains(vk1_0::MemoryPropertyFlags::LAZILY_ALLOCATED) {
        result |= MemoryPropertyFlags::LAZILY_ALLOCATED;
    }
    result
}

pub fn memory_properties_to_erupt(props: MemoryPropertyFlags) -> vk1_0::MemoryPropertyFlags {
    let mut result = vk1_0::MemoryPropertyFlags::empty();
    if props.contains(MemoryPropertyFlags::DEVICE_LOCAL) {
        result |= vk1_0::MemoryPropertyFlags::DEVICE_LOCAL;
    }
    if props.contains(MemoryPropertyFlags::HOST_VISIBLE) {
        result |= vk1_0::MemoryPropertyFlags::HOST_VISIBLE;
    }
    if props.contains(MemoryPropertyFlags::HOST_COHERENT) {
        result |= vk1_0::MemoryPropertyFlags::HOST_COHERENT;
    }
    if props.contains(MemoryPropertyFlags::HOST_CACHED) {
        result |= vk1_0::MemoryPropertyFlags::HOST_CACHED;
    }
    if props.contains(MemoryPropertyFlags::LAZILY_ALLOCATED) {
        result |= vk1_0::MemoryPropertyFlags::LAZILY_ALLOCATED;
    }
    result
}
