use {
    core::fmt::{self, Display},
    gpu_alloc_types::{DeviceMapError, OutOfMemory},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AllocationError {
    OutOfDeviceMemory,
    OutOfHostMemory,
    NoCompatibleMemoryTypes,
    TooManyObjects,
}

impl From<OutOfMemory> for AllocationError {
    fn from(err: OutOfMemory) -> Self {
        match err {
            OutOfMemory::OutOfDeviceMemory => AllocationError::OutOfDeviceMemory,
            OutOfMemory::OutOfHostMemory => AllocationError::OutOfHostMemory,
        }
    }
}

impl Display for AllocationError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocationError::OutOfDeviceMemory => fmt.write_str("Device memory exhausted"),
            AllocationError::OutOfHostMemory => fmt.write_str("Host memory exhausted"),
            AllocationError::NoCompatibleMemoryTypes => fmt.write_str(
                "No compatible memory types from requested mask support requested usage",
            ),
            AllocationError::TooManyObjects => {
                fmt.write_str("Reached limit on memory objects count")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AllocationError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MapError {
    OutOfDeviceMemory,
    OutOfHostMemory,
    NonHostVisible,
    MapFailed,
}

impl From<DeviceMapError> for MapError {
    fn from(err: DeviceMapError) -> Self {
        match err {
            DeviceMapError::OutOfDeviceMemory => MapError::OutOfDeviceMemory,
            DeviceMapError::OutOfHostMemory => MapError::OutOfHostMemory,
            DeviceMapError::MapFailed => MapError::MapFailed,
        }
    }
}

impl From<OutOfMemory> for MapError {
    fn from(err: OutOfMemory) -> Self {
        match err {
            OutOfMemory::OutOfDeviceMemory => MapError::OutOfDeviceMemory,
            OutOfMemory::OutOfHostMemory => MapError::OutOfHostMemory,
        }
    }
}

impl Display for MapError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MapError::OutOfDeviceMemory => fmt.write_str("Device memory exhausted"),
            MapError::OutOfHostMemory => fmt.write_str("Host memory exhausted"),
            MapError::MapFailed => fmt.write_str("Failed to map memory object"),
            MapError::NonHostVisible => fmt.write_str("Impossible to map non-host-visible memory"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MapError {}
