use {
    erupt::{vk1_0, DeviceLoader, EntryLoader, InstanceLoader},
    gpu_alloc::{Config, GpuAllocator, Request, UsageFlags},
    gpu_alloc_erupt::{device_properties, EruptMemoryDevice},
    std::ffi::CStr,
};

fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let entry = EntryLoader::new()?;

    let instance = unsafe {
        InstanceLoader::new(
            &entry,
            &vk1_0::InstanceCreateInfo::default()
                .into_builder()
                .application_info(
                    &vk1_0::ApplicationInfo::default()
                        .into_builder()
                        .engine_name(CStr::from_bytes_with_nul(b"GpuAlloc\0").unwrap())
                        .engine_version(1)
                        .application_name(CStr::from_bytes_with_nul(b"GpuAllocApp\0").unwrap())
                        .application_version(1)
                        .api_version(entry.instance_version()),
                ),
        )
    }?;

    let physical_devices = unsafe { instance.enumerate_physical_devices(None) }.result()?;
    let physical_device = physical_devices[0];

    let props = unsafe { device_properties(&instance, physical_device) }?;

    let device = unsafe {
        DeviceLoader::new(
            &instance,
            physical_device,
            &vk1_0::DeviceCreateInfoBuilder::new().queue_create_infos(&[
                vk1_0::DeviceQueueCreateInfoBuilder::new()
                    .queue_family_index(0)
                    .queue_priorities(&[0f32]),
            ]),
        )
    }?;

    let config = Config::i_am_potato();

    let mut allocator = GpuAllocator::new(config, props);

    let mut block = unsafe {
        allocator.alloc(
            EruptMemoryDevice::wrap(&device),
            Request {
                size: 10,
                align_mask: 1,
                usage: UsageFlags::HOST_ACCESS,
                memory_types: !0,
            },
        )
    }?;

    unsafe {
        block.write_bytes(
            EruptMemoryDevice::wrap(&device),
            0,
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        )
    }?;

    unsafe { allocator.dealloc(EruptMemoryDevice::wrap(&device), block) }

    Ok(())
}
