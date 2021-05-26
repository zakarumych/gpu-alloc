use {
    gpu_alloc::{
        Config, DeviceProperties, GpuAllocator, MemoryHeap, MemoryPropertyFlags, MemoryType,
        Request, UsageFlags,
    },
    gpu_alloc_mock::MockMemoryDevice,
    std::{borrow::Cow, collections::VecDeque},
    tracing_subscriber::layer::SubscriberExt as _,
};

fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    tracing::subscriber::set_global_default(
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .pretty()
            .finish()
            .with(tracing_error::ErrorLayer::default()),
    )?;

    let device = MockMemoryDevice::new(DeviceProperties {
        memory_types: Cow::Borrowed(&[MemoryType {
            heap: 0,
            props: MemoryPropertyFlags::HOST_VISIBLE,
        }]),
        memory_heaps: Cow::Borrowed(&[MemoryHeap {
            size: 32 * 1024 * 1024,
        }]),
        max_memory_allocation_count: 5,
        max_memory_allocation_size: 1024 * 1024,
        non_coherent_atom_size: 8,
        buffer_device_address: false,
    });

    let config = Config::i_am_potato();

    let mut allocator = GpuAllocator::new(config, device.props());

    let mut blocks = VecDeque::new();

    for _ in 0..1_000_000 {
        if blocks.len() >= 1024 {
            while blocks.len() > 700 {
                let block = blocks.pop_front().unwrap();

                unsafe {
                    allocator.dealloc(&device, block);
                }
            }
        }

        let block = unsafe {
            allocator.alloc(
                &device,
                Request {
                    size: 128,
                    align_mask: 0,
                    usage: UsageFlags::HOST_ACCESS | UsageFlags::TRANSIENT,
                    memory_types: !0,
                },
            )
        }?;

        blocks.push_back(block);
    }

    while let Some(block) = blocks.pop_front() {
        unsafe {
            allocator.dealloc(&device, block);
        }
    }

    // assert_eq!(device.total_allocations(), 2);

    tracing::warn!(
        "Total memory object allocations: {}",
        device.total_allocations()
    );

    unsafe {
        allocator.cleanup(&device);
    }

    Ok(())
}
