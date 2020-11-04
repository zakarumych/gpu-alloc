//!
//! Implementation agnostic memory allocator for Vulkan like APIs.
//!
//! # Usage
//!
//! Start with fetching `DeviceProperties` from `gfx-alloc-<backend>` crate for the backend of choice.\
//! Then create `GpuAllocator` instance and use it for all device memory allocations.\
//! `GpuAllocator` will take care for all necessary bookkeeping like memory object count limit,
//! heap budget and memory mapping.
//!
//! There are many `unsafe` functions in this crate, but most of them has only one unchecked safety requirement -
//! only one `device` should be used with `GpuAllocator` instance and all blocks allocated with it.
//! Other safety requirements should be as easy to fullfill.
//!
//! #
//!
//! ### Note
//!
//! Backend supporting crates should not depend on this crate.\
//! Instead they should depend on `gpu-alloc-types` which is much more stable,
//! allowing to upgrade `gpu-alloc` version without `gfx-alloc-<backend>` upgrade.
//!

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod allocator;
mod block;
mod buddy;
mod config;
mod error;
mod heap;
mod linear;
mod slab;
mod usage;

pub use {
    self::{allocator::*, block::MemoryBlock, config::*, error::*, usage::*},
    gpu_alloc_types::*,
};

/// Possible requirements for dedicated memory object allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dedicated {
    /// No requirements.
    /// Should be used for most cases.
    Indifferent,

    /// Hint for allocator that dedicated memory object is preferred.
    /// Should be used if it is known that resource placed in dedicated memory object would allow for better performance.
    /// Implementation is allowed to return block to shared memory object.
    Preferred,

    /// Specifies that dedicated memory block MUST be returned.
    /// Should be used only if resource has to be bound to dedicated memory object.
    Required,
}

impl Default for Dedicated {
    fn default() -> Self {
        Dedicated::Indifferent
    }
}

/// Memory request for allocator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Request {
    /// Minimal size of memory block required.
    /// Returned block may have larger size,
    /// use `MemoryBlock::size` to learn actual size of returned block.
    pub size: u64,

    /// Minimal alignment mask required.
    /// Returnd block may have larger alignment,
    /// use `MemoryBlock::align` to learn actual alignment of returned block.
    pub align_mask: u64,

    /// Intended memory usage.
    /// Returned block may support additional usages,
    /// use `MemoryBlock::props` to learn memory properties of returned block.
    pub usage: UsageFlags,

    /// Bitset for memory types.
    /// Returned block will be from memory type corresponding to one of set bits,
    /// use `MemoryBlock::memory_type` to learn memory type index of returned block.
    pub memory_types: u32,

    /// Specifies if dedicated memory object is required, preferred or not.
    pub dedicated: Dedicated,
}

/// Specifies allocation strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Strategy {
    /// Allocation directly from device.
    /// Use with caution.
    /// Very slow.
    /// Count of allocations is limited.
    Dedicated,

    /// Linear allocation suitable for transient use case.
    /// Minimal overhead when used properly.
    /// Huge overhead if allocated memory block outlives
    /// other blocks allocated from the same chunk.
    Linear,

    /// General purpose allocator with moderate overhead.
    /// Splits bigger blocks in halves to satisfy smaller requests.
    /// Deallocated memory is immediately reusable.
    /// Deallocated twin blocks merge back into larger block.
    Buddy,
}

/// Aligns `value` up to `align_maks`
/// Returns smallest integer not lesser than `value` aligned by `align_mask`.
/// Returns `None` on overflow.
pub(crate) fn align_up(value: u64, align_mask: u64) -> Option<u64> {
    Some(value.checked_add(align_mask)? & !align_mask)
}

/// Align `value` down to `align_maks`
/// Returns largest integer not bigger than `value` aligned by `align_mask`.
pub(crate) fn align_down(value: u64, align_mask: u64) -> u64 {
    value & !align_mask
}
