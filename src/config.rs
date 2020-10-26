/// Configuration for [`GpuAllocator`]
///
/// [`GpuAllocator`]: type.GpuAllocator
#[cfg_attr(feature = "serde", serde::Serialize, serde::Deserialize)]
pub struct Config {
    /// Size in bytes of request that will be served by dedicated memory object.
    /// This value should be large enough to not exhaust memory object limit
    /// and not use slow memory object allocation when it is not necessary.
    pub dedicated_treshold: u64,

    /// Size in bytes of request that will be served by dedicated memory object if preferred.
    /// This value should be large enough to not exhaust memory object limit
    /// and not use slow memory object allocation when it is not necessary.
    ///
    /// This won't make much sense if this value is larger than `dedicated_treshold`.
    pub preferred_dedicated_treshold: u64,

    /// Size in bytes of transient memory request that will be served by dedicated memory object.
    /// This value should be large enough to not exhaust memory object limit
    /// and not use slow memory object allocation when it is not necessary.
    ///
    /// This won't make much sense if this value is lesser than `dedicated_treshold`.
    pub transient_dedicated_treshold: u64,
}

impl Config {
    /// Returns default configuration.
    /// This is not `Default` implementation to discourage usage outside of
    /// prototyping.
    /// Proper configuration should depend on hardware and intended usage.
    /// But those values can be used as starting point.
    /// Note that they can simply not work for some platforms with lesser
    /// memory capacity than today's "modern" GPU (year 2020).
    pub fn i_am_prototyping() -> Self {
        // Assume that today's modern GPU is made of 1024 potatoes.
        let potato = Config::i_am_potato();

        Config {
            dedicated_treshold: potato.dedicated_treshold * 1024,
            preferred_dedicated_treshold: potato.preferred_dedicated_treshold * 1024,
            transient_dedicated_treshold: potato.transient_dedicated_treshold * 1024,
        }
    }

    /// Returns default configuration for potato.
    pub fn i_am_potato() -> Self {
        Config {
            dedicated_treshold: 32 * 1024,
            preferred_dedicated_treshold: 1024,
            transient_dedicated_treshold: 128 * 1024,
        }
    }
}
