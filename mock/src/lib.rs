use {
    galloc_types::{
        DeviceAllocationError, DeviceMapError, DeviceProperties, MappedMemoryRange, MemoryDevice,
        MemoryHeap, MemoryPropertyFlags, MemoryType,
    },
    slab::Slab,
    std::{
        cell::{Cell, RefCell, UnsafeCell},
        convert::TryFrom as _,
        mem::transmute,
        ptr::NonNull,
    },
};

struct MemoryMapping {
    content: Box<UnsafeCell<[u8]>>,
    offset: u64,
}

struct MockMemory {
    memory_type: u32,
    size: u64,
    mapped: Option<MemoryMapping>,
}

pub struct MockMemoryDevice {
    memory_types: Box<[MemoryType]>,
    memory_heaps: Box<[MemoryHeap]>,
    max_memory_allocation_count: u32,
    max_memory_allocation_size: u64,
    non_coherent_atom_size: u64,

    allocations_remains: Cell<u32>,
    memory_heaps_remaining_capacity: Box<[Cell<u64>]>,
    allocations: RefCell<Slab<MockMemory>>,
}

impl MockMemoryDevice {
    pub fn new(props: DeviceProperties<'_>) -> Self {
        MockMemoryDevice {
            memory_types: props.memory_types.iter().copied().collect(),
            memory_heaps: props.memory_heaps.iter().copied().collect(),
            max_memory_allocation_count: props.max_memory_allocation_count,
            max_memory_allocation_size: props.max_memory_allocation_size,
            non_coherent_atom_size: props.non_coherent_atom_size,

            allocations_remains: Cell::new(props.max_memory_allocation_count),
            memory_heaps_remaining_capacity: props
                .memory_heaps
                .iter()
                .map(|heap| Cell::new(heap.size))
                .collect(),
            allocations: RefCell::new(Slab::new()),
        }
    }

    pub fn properties(&self) -> DeviceProperties<'_> {
        DeviceProperties {
            memory_types: &self.memory_types,
            memory_heaps: &self.memory_heaps,
            max_memory_allocation_count: self.max_memory_allocation_count,
            max_memory_allocation_size: self.max_memory_allocation_size,
            non_coherent_atom_size: self.non_coherent_atom_size,
        }
    }
}

impl MemoryDevice<usize> for MockMemoryDevice {
    #[tracing::instrument(skip(self))]
    unsafe fn allocate_memory(
        &self,
        size: u64,
        memory_type: u32,
    ) -> Result<usize, DeviceAllocationError> {
        assert!(
            size <= self.max_memory_allocation_size,
            "Allocation size exceeds limit"
        );

        let allocations_remains = self.allocations_remains.get();
        assert!(
            allocations_remains > 0,
            "Allocator should not try to allocate too many objects"
        );
        self.allocations_remains.set(allocations_remains - 1);

        let heap = &self.memory_heaps_remaining_capacity
            [self.memory_types[memory_type as usize].heap as usize];
        if heap.get() < size {
            return Err(DeviceAllocationError::OutOfDeviceMemory);
        }
        heap.set(heap.get() - size);

        tracing::info!("Memory object allocated");
        Ok(self.allocations.borrow_mut().insert(MockMemory {
            memory_type,
            size,
            mapped: None,
        }))
    }

    #[tracing::instrument(skip(self))]
    unsafe fn deallocate_memory(&self, memory: usize) {
        let memory = self.allocations.borrow_mut().remove(memory);
        self.allocations_remains
            .set(self.allocations_remains.get() + 1);
        let heap = &self.memory_heaps_remaining_capacity
            [self.memory_types[memory.memory_type as usize].heap as usize];
        heap.set(heap.get() + memory.size);
        tracing::info!("Memory object deallocated");
    }

    #[tracing::instrument(skip(self))]
    unsafe fn map_memory(
        &self,
        memory: &usize,
        offset: u64,
        size: u64,
    ) -> Result<NonNull<u8>, DeviceMapError> {
        assert_ne!(size, 0, "Mapping size must be larger than 0");

        let mut allocations = self.allocations.borrow_mut();
        let memory = allocations
            .get_mut(*memory)
            .expect("Non-existing memory object");

        assert!(
            self.memory_types[memory.memory_type as usize]
                .props
                .contains(MemoryPropertyFlags::HOST_VISIBLE),
            "Attempt to map non-host-visible memory"
        );

        assert!(memory.mapped.is_none(), "Already mapped");

        assert!(
            offset < memory.size,
            "offset must be less than the size of memory"
        );
        assert_ne!(size, 0, "Mapping size must be greater than 0");
        assert!(
            size < memory.size - offset,
            "size must be less than or equal to the size of the memory minus offset"
        );

        let size_usize = usize::try_from(size).map_err(|_| DeviceMapError::OutOfHostMemory)?;
        let mapping = memory.mapped.get_or_insert(MemoryMapping {
            content: transmute(vec![0; size_usize].into_boxed_slice()),
            offset,
        });

        tracing::info!("Memory object mapped");
        Ok(NonNull::from(&mut (&mut *mapping.content.get())[0]))
    }

    unsafe fn unmap_memory(&self, memory: &usize) {
        let mut allocations = self.allocations.borrow_mut();
        let memory = allocations
            .get_mut(*memory)
            .expect("Non-existing memory object");
        assert!(memory.mapped.take().is_some(), "Was not mapped");
    }

    unsafe fn invalidate_memory_ranges(&self, ranges: &[MappedMemoryRange<'_, usize>]) {
        for range in ranges {
            let mut allocations = self.allocations.borrow_mut();
            let memory = allocations
                .get_mut(*range.memory)
                .expect("Non-existing memory object");

            let mapped = memory.mapped.as_ref().expect("Not mapped");

            let coherent = self.memory_types[memory.memory_type as usize]
                .props
                .contains(MemoryPropertyFlags::HOST_COHERENT);

            if coherent {
                tracing::warn!("Invalidating host-coherent memory");
            }

            let mapped_size = (&*mapped.content.get()).len() as u64;

            assert!(
                range.offset >= mapped.offset,
                "range `offset` specifies range before mapped region"
            );
            assert!(
                range.offset - mapped.offset <= mapped_size,
                "range `offset` specifies range after mapped region"
            );
            assert!(
                range.size < mapped_size - (range.offset - mapped.offset),
                "range `size` specifies range after mapped region"
            );
            assert_eq!(
                range.offset % self.non_coherent_atom_size,
                0,
                "`offset` must be a multiple of `non_coherent_atom_size`"
            );
            assert!(
                range.size % self.non_coherent_atom_size == 0
                    || range.offset + range.size == memory.size,
                "`size` must either be a multiple of `non_coherent_atom_size`, or `offset + size` must equal the size of memory"
            );
        }
    }

    unsafe fn flush_memory_ranges(&self, ranges: &[MappedMemoryRange<'_, usize>]) {
        for range in ranges {
            let mut allocations = self.allocations.borrow_mut();
            let memory = allocations
                .get_mut(*range.memory)
                .expect("Non-existing memory object");

            let mapped = memory.mapped.as_ref().expect("Not mapped");

            let coherent = self.memory_types[memory.memory_type as usize]
                .props
                .contains(MemoryPropertyFlags::HOST_COHERENT);

            if coherent {
                tracing::warn!("Invalidating host-coherent memory");
            }

            let mapped_size = (&*mapped.content.get()).len() as u64;

            assert!(
                range.offset >= mapped.offset,
                "`offset` specifies range before mapped region"
            );
            assert!(
                range.offset - mapped.offset <= mapped_size,
                "`offset` specifies range after mapped region"
            );
            assert!(
                range.size < mapped_size - (range.offset - mapped.offset),
                "`size` specifies range after mapped region"
            );
            assert_eq!(
                range.offset % self.non_coherent_atom_size,
                0,
                "`offset` must be a multiple of `non_coherent_atom_size`"
            );
            assert!(
                range.size % self.non_coherent_atom_size == 0
                    || range.offset + range.size == memory.size,
                "`size` must either be a multiple of `non_coherent_atom_size`, or `offset + size` must equal the size of memory"
            );
        }
    }
}
