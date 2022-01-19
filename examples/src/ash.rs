use {
    ash::{vk, Entry},
    gpu_alloc::{Config, GpuAllocator, Request, UsageFlags},
    gpu_alloc_ash::{device_properties, AshMemoryDevice},
    std::ffi::CStr,
};

fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let entry = unsafe { Entry::load() }?;

    let version = entry
        .try_enumerate_instance_version()?
        .unwrap_or(vk::make_api_version(0, 1, 0, 0));

    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::builder().application_info(
                &vk::ApplicationInfo::builder()
                    .engine_name(CStr::from_bytes_with_nul(b"GpuAlloc\0").unwrap())
                    .engine_version(1)
                    .application_name(CStr::from_bytes_with_nul(b"GpuAllocApp\0").unwrap())
                    .application_version(1)
                    .api_version(version),
            ),
            None,
        )
    }?;

    let physical_devices = unsafe { instance.enumerate_physical_devices() }?;
    let physical_device = physical_devices[0];

    let props = unsafe { device_properties(&instance, version, physical_device) }?;

    let device = unsafe {
        instance.create_device(
            physical_device,
            &vk::DeviceCreateInfo::builder().queue_create_infos(&[
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(0)
                    .queue_priorities(&[0f32])
                    .build(),
            ]),
            None,
        )
    }?;

    let config = Config::i_am_potato();

    let mut allocator = GpuAllocator::new(config, props);

    let mut block = unsafe {
        allocator.alloc(
            AshMemoryDevice::wrap(&device),
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
            AshMemoryDevice::wrap(&device),
            0,
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        )
    }?;

    unsafe { allocator.dealloc(AshMemoryDevice::wrap(&device), block) }

    Ok(())
}
