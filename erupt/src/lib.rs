use {
    erupt::{vk1_0, DeviceLoader, InstanceLoader},
    gpu_alloc_types::{
        DeviceMapError, DeviceProperties, MappedMemoryRange, MemoryDevice, MemoryHeap,
        MemoryPropertyFlags, MemoryType, OutOfMemory,
    },
    std::ptr::NonNull,
    tinyvec::TinyVec,
};

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
    ) -> Result<vk1_0::DeviceMemory, OutOfMemory> {
        match self
            .device
            .allocate_memory(
                &vk1_0::MemoryAllocateInfoBuilder::new()
                    .allocation_size(size)
                    .memory_type_index(memory_type),
                None,
                None,
            )
            .result()
        {
            Ok(memory) => Ok(memory),
            Err(vk1_0::Result::ERROR_OUT_OF_DEVICE_MEMORY) => Err(OutOfMemory::OutOfDeviceMemory),
            Err(vk1_0::Result::ERROR_OUT_OF_HOST_MEMORY) => Err(OutOfMemory::OutOfHostMemory),
            Err(vk1_0::Result::ERROR_TOO_MANY_OBJECTS) => panic!("Too many objects"),
            Err(err) => panic!("Unexpected Vulkan error: `{}`", err),
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn deallocate_memory(&self, memory: vk1_0::DeviceMemory) {
        self.device.free_memory(Some(memory), None);
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn map_memory(
        &self,
        memory: &vk1_0::DeviceMemory,
        offset: u64,
        size: u64,
    ) -> Result<NonNull<u8>, DeviceMapError> {
        let mut ptr = core::ptr::null_mut();

        match self
            .device
            .map_memory(*memory, offset, size, None, &mut ptr)
            .result()
        {
            Ok(()) => {
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
    unsafe fn unmap_memory(&self, memory: &vk1_0::DeviceMemory) {
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
pub unsafe fn device_properties(
    instance: &InstanceLoader,
    physical_device: vk1_0::PhysicalDevice,
) -> DeviceProperties<'static> {
    let limits = instance
        .get_physical_device_properties(physical_device, None)
        .limits;
    let memory_properties = instance.get_physical_device_memory_properties(physical_device, None);
    DeviceProperties {
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
    }
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
