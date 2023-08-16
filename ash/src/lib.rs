//!
//! # Ash backend for `gpu-alloc`
//!  
//! # Usage example
//!
//! ```ignore
//! use {
//!     ash::{vk, DefaultEntryLoader, DeviceLoader, InstanceLoader},
//!     gpu_alloc::{Config, GpuAllocator, Request, UsageFlags},
//!     gpu_alloc_ash::{device_properties, AshMemoryDevice},
//!     std::ffi::CStr,
//! };
//!
//! fn main() -> eyre::Result<()> {
//!     color_eyre::install()?;
//!
//!     let entry = DefaultEntryLoader::new()?;
//!
//!     let instance = InstanceLoader::new(
//!         &entry,
//!         &vk::InstanceCreateInfo::default()
//!             .into_builder()
//!             .application_info(
//!                 &vk::ApplicationInfo::default()
//!                     .into_builder()
//!                     .engine_name(CStr::from_bytes_with_nul(b"GpuAlloc\0").unwrap())
//!                     .engine_version(1)
//!                     .application_name(CStr::from_bytes_with_nul(b"GpuAllocApp\0").unwrap())
//!                     .application_version(1)
//!                     .api_version(entry.instance_version()),
//!             ),
//!         None,
//!     )?;
//!
//!     let physical_devices = unsafe { instance.enumerate_physical_devices(None) }.result()?;
//!     let physical_device = physical_devices[0];
//!
//!     let props = unsafe { device_properties(&instance, physical_device) }?;
//!
//!     let device = DeviceLoader::new(
//!         &instance,
//!         physical_device,
//!         &vk::DeviceCreateInfoBuilder::new().queue_create_infos(&[
//!             vk::DeviceQueueCreateInfoBuilder::new()
//!                 .queue_family_index(0)
//!                 .queue_priorities(&[0f32]),
//!         ]),
//!         None,
//!     )?;
//!
//!     let config = Config::i_am_potato();
//!
//!     let mut allocator = GpuAllocator::new(config, props);
//!
//!     let mut block = unsafe {
//!         allocator.alloc(
//!             AshMemoryDevice::wrap(&device),
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
//!             AshMemoryDevice::wrap(&device),
//!             0,
//!             &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
//!         )
//!     }?;
//!
//!     unsafe { allocator.dealloc(AshMemoryDevice::wrap(&device), block) }
//!
//!     // the `ash::Device` also implements `AsRef<AshMemoryDevice>`
//!     // you can pass a reference of `ash::Device` directly as argument
//!     let mut block = unsafe {
//!         allocator.alloc(
//!             &device,
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
//!             &device,
//!             0,
//!             &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
//!         )
//!     }?;
//!
//!     unsafe { allocator.dealloc(&device, block) }
//!
//!     Ok(())
//! }
//! ```
//!

use {
    ash::{vk, Device, Instance},
    gpu_alloc_types::{
        AllocationFlags, DeviceMapError, DeviceProperties, MappedMemoryRange, MemoryDevice,
        MemoryHeap, MemoryPropertyFlags, MemoryType, OutOfMemory,
    },
    std::ptr::NonNull,
    tinyvec::TinyVec,
};

#[repr(transparent)]
pub struct AshMemoryDevice {
    device: Device,
}

impl AshMemoryDevice {
    pub fn wrap(device: &Device) -> &Self {
        unsafe {
            // Safe because `Self` is `repr(transparent)`
            // with only field being `DeviceLoader`.
            &*(device as *const Device as *const Self)
        }
    }
}

impl AsRef<AshMemoryDevice> for Device {
    #[inline(always)]
    fn as_ref(&self) -> &AshMemoryDevice {
        AshMemoryDevice::wrap(self)
    }
}

// AsRef does not have a blanket implementation. need to add this impl so that
// old user code (i.e. explicit wrap) still compiles without any change
impl AsRef<AshMemoryDevice> for AshMemoryDevice {
    #[inline(always)]
    fn as_ref(&self) -> &AshMemoryDevice {
        self
    }
}

impl MemoryDevice<vk::DeviceMemory> for AshMemoryDevice {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn allocate_memory(
        &self,
        size: u64,
        memory_type: u32,
        flags: AllocationFlags,
    ) -> Result<vk::DeviceMemory, OutOfMemory> {
        assert!((flags & !(AllocationFlags::DEVICE_ADDRESS)).is_empty());

        let mut info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size)
            .memory_type_index(memory_type);

        let mut info_flags;

        if flags.contains(AllocationFlags::DEVICE_ADDRESS) {
            info_flags = vk::MemoryAllocateFlagsInfo::builder()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            info = info.push_next(&mut info_flags);
        }

        match self.device.allocate_memory(&info, None) {
            Ok(memory) => Ok(memory),
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => Err(OutOfMemory::OutOfDeviceMemory),
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => Err(OutOfMemory::OutOfHostMemory),
            Err(vk::Result::ERROR_TOO_MANY_OBJECTS) => panic!("Too many objects"),
            Err(err) => panic!("Unexpected Vulkan error: `{}`", err),
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn deallocate_memory(&self, memory: vk::DeviceMemory) {
        self.device.free_memory(memory, None);
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn map_memory(
        &self,
        memory: &mut vk::DeviceMemory,
        offset: u64,
        size: u64,
    ) -> Result<NonNull<u8>, DeviceMapError> {
        match self
            .device
            .map_memory(*memory, offset, size, vk::MemoryMapFlags::empty())
        {
            Ok(ptr) => {
                Ok(NonNull::new(ptr as *mut u8)
                    .expect("Pointer to memory mapping must not be null"))
            }
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => Err(DeviceMapError::OutOfDeviceMemory),
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => Err(DeviceMapError::OutOfHostMemory),
            Err(vk::Result::ERROR_MEMORY_MAP_FAILED) => Err(DeviceMapError::MapFailed),
            Err(err) => panic!("Unexpected Vulkan error: `{}`", err),
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn unmap_memory(&self, memory: &mut vk::DeviceMemory) {
        self.device.unmap_memory(*memory);
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn invalidate_memory_ranges(
        &self,
        ranges: &[MappedMemoryRange<'_, vk::DeviceMemory>],
    ) -> Result<(), OutOfMemory> {
        self.device
            .invalidate_mapped_memory_ranges(
                &ranges
                    .iter()
                    .map(|range| {
                        vk::MappedMemoryRange::builder()
                            .memory(*range.memory)
                            .offset(range.offset)
                            .size(range.size)
                            .build()
                    })
                    .collect::<TinyVec<[_; 4]>>(),
            )
            .map_err(|err| match err {
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => OutOfMemory::OutOfDeviceMemory,
                vk::Result::ERROR_OUT_OF_HOST_MEMORY => OutOfMemory::OutOfHostMemory,
                err => panic!("Unexpected Vulkan error: `{}`", err),
            })
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn flush_memory_ranges(
        &self,
        ranges: &[MappedMemoryRange<'_, vk::DeviceMemory>],
    ) -> Result<(), OutOfMemory> {
        self.device
            .flush_mapped_memory_ranges(
                &ranges
                    .iter()
                    .map(|range| {
                        vk::MappedMemoryRange::builder()
                            .memory(*range.memory)
                            .offset(range.offset)
                            .size(range.size)
                            .build()
                    })
                    .collect::<TinyVec<[_; 4]>>(),
            )
            .map_err(|err| match err {
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => OutOfMemory::OutOfDeviceMemory,
                vk::Result::ERROR_OUT_OF_HOST_MEMORY => OutOfMemory::OutOfHostMemory,
                err => panic!("Unexpected Vulkan error: `{}`", err),
            })
    }
}

/// Returns `DeviceProperties` from ash's `InstanceLoader` for specified `PhysicalDevice`, required to create `GpuAllocator`.
///
/// # Safety
///
/// `physical_device` must be queried from `Instance` associated with this `instance`.
/// Even if returned properties' field `buffer_device_address` is set to true,
/// feature `PhysicalDeviceBufferDeviceAddressFeatures::buffer_derive_address`  must be enabled explicitly on device creation
/// and extension "VK_KHR_buffer_device_address" for Vulkan prior 1.2.
/// Otherwise the field must be set to false before passing to `GpuAllocator::new`.
pub unsafe fn device_properties(
    instance: &Instance,
    version: u32,
    physical_device: vk::PhysicalDevice,
) -> Result<DeviceProperties<'static>, vk::Result> {
    use ash::vk::PhysicalDeviceFeatures2;

    let limits = instance
        .get_physical_device_properties(physical_device)
        .limits;

    let memory_properties = instance.get_physical_device_memory_properties(physical_device);

    let buffer_device_address =
        if vk::api_version_major(version) >= 1 && vk::api_version_minor(version) >= 2 {
            let mut features = PhysicalDeviceFeatures2::builder();
            let mut bda_features = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
            features.p_next =
                &mut bda_features as *mut vk::PhysicalDeviceBufferDeviceAddressFeatures as *mut _;
            instance.get_physical_device_features2(physical_device, &mut features);
            bda_features.buffer_device_address != 0
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
                props: memory_properties_from_ash(memory_type.property_flags),
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

pub fn memory_properties_from_ash(props: vk::MemoryPropertyFlags) -> MemoryPropertyFlags {
    let mut result = MemoryPropertyFlags::empty();
    if props.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
        result |= MemoryPropertyFlags::DEVICE_LOCAL;
    }
    if props.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
        result |= MemoryPropertyFlags::HOST_VISIBLE;
    }
    if props.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
        result |= MemoryPropertyFlags::HOST_COHERENT;
    }
    if props.contains(vk::MemoryPropertyFlags::HOST_CACHED) {
        result |= MemoryPropertyFlags::HOST_CACHED;
    }
    if props.contains(vk::MemoryPropertyFlags::LAZILY_ALLOCATED) {
        result |= MemoryPropertyFlags::LAZILY_ALLOCATED;
    }
    result
}

pub fn memory_properties_to_ash(props: MemoryPropertyFlags) -> vk::MemoryPropertyFlags {
    let mut result = vk::MemoryPropertyFlags::empty();
    if props.contains(MemoryPropertyFlags::DEVICE_LOCAL) {
        result |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
    }
    if props.contains(MemoryPropertyFlags::HOST_VISIBLE) {
        result |= vk::MemoryPropertyFlags::HOST_VISIBLE;
    }
    if props.contains(MemoryPropertyFlags::HOST_COHERENT) {
        result |= vk::MemoryPropertyFlags::HOST_COHERENT;
    }
    if props.contains(MemoryPropertyFlags::HOST_CACHED) {
        result |= vk::MemoryPropertyFlags::HOST_CACHED;
    }
    if props.contains(MemoryPropertyFlags::LAZILY_ALLOCATED) {
        result |= vk::MemoryPropertyFlags::LAZILY_ALLOCATED;
    }
    result
}
