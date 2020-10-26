use core::fmt::{self, Display};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AllocationError {
    OutOfDeviceMemory,
    OutOfHostMemory,
    NoCompatibleMemoryTypes,
}

impl Display for AllocationError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfDeviceMemory => fmt.write_str("Device memory exhausted"),
            Self::OutOfHostMemory => fmt.write_str("Host memory exhausted"),
            Self::NoCompatibleMemoryTypes => fmt.write_str(
                "No compatible memory types from requested mask support requested usage",
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AllocationError {}
