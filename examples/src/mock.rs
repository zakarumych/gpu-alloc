use {
    gpu_alloc::{
        Config, DeviceProperties, GpuAllocator, MemoryHeap, MemoryPropertyFlags, MemoryType,
        Request, UsageFlags,
    },
    gpu_alloc_mock::MockMemoryDevice,
    std::borrow::Cow,
};

fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let device = MockMemoryDevice::new(DeviceProperties {
        memory_types: Cow::Borrowed(&[MemoryType {
            heap: 0,
            props: MemoryPropertyFlags::HOST_VISIBLE,
        }]),
        memory_heaps: Cow::Borrowed(&[MemoryHeap { size: 1024 * 1024 }]),
        max_memory_allocation_count: 32,
        max_memory_allocation_size: 1024 * 1024,
        non_coherent_atom_size: 8,
        buffer_device_address: false,
    });

    let config = Config::i_am_potato();

    let mut allocator = GpuAllocator::new(config, device.props());

    let mut block = unsafe {
        allocator.alloc(
            &device,
            Request {
                size: 10,
                align_mask: 1,
                usage: UsageFlags::HOST_ACCESS,
                memory_types: !0,
            },
        )
    }?;

    let another_block = unsafe {
        allocator.alloc(
            &device,
            Request {
                size: 10,
                align_mask: 1,
                usage: UsageFlags::HOST_ACCESS,
                memory_types: !0,
            },
        )
    }?;

    unsafe { block.write_bytes(&device, 0, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) }?;

    unsafe { allocator.dealloc(&device, block) }
    unsafe { allocator.dealloc(&device, another_block) }

    Ok(())
}
