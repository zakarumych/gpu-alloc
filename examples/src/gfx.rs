use {
    eyre::eyre,
    gfx_backend_vulkan::Instance,
    gfx_hal::{
        adapter::{Gpu, PhysicalDevice as _},
        queue::QueueFamily as _,
        Features, Instance as _,
    },
    gpu_alloc::{Config, GpuAllocator, Request, UsageFlags},
    gpu_alloc_gfx::{gfx_device_properties, GfxMemoryDevice},
};

fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let instance =
        Instance::create("gpu_alloc-example", 1).map_err(|_| eyre!("Unsupported backend"))?;

    let adapters = instance.enumerate_adapters();

    let adapter = adapters
        .iter()
        .min_by_key(|a| {
            use gfx_hal::adapter::DeviceType::*;
            match a.info.device_type {
                Other => 4,
                IntegratedGpu => 1,
                DiscreteGpu => 0,
                VirtualGpu => 2,
                Cpu => 3,
            }
        })
        .ok_or_else(|| eyre!("No adapters found"))?;

    let queue_family = adapter
        .queue_families
        .iter()
        .min_by_key(|qf| {
            use gfx_hal::queue::QueueType::*;
            match qf.queue_type() {
                General => 0,
                Graphics => 1,
                Compute => 3,
                Transfer => 4,
            }
        })
        .ok_or_else(|| eyre!("No queue families found"))?;

    let props = gfx_device_properties(adapter);

    let Gpu { device, .. } = unsafe {
        adapter
            .physical_device
            .open(&[(queue_family, &[1.0])], Features::empty())
    }?;

    let config = Config::i_am_potato();

    let mut allocator = GpuAllocator::new(config, props);

    let mut block = unsafe {
        allocator.alloc(
            GfxMemoryDevice::wrap(&device),
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
            GfxMemoryDevice::wrap(&device),
            0,
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        )
    }?;

    unsafe { allocator.dealloc(GfxMemoryDevice::wrap(&device), block) }

    Ok(())
}
