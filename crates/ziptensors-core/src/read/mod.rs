//! File: read/mod.rs
//! Purpose: reader API: opening a container, reading members by offset or tensor name, and
//! restoring files.
//!
//! Main responsibilities: the module list and the public surface of the reader.
//!
//! Key structures: [`Container`], [`OpenOptions`], [`MemberInfo`], [`Member`], [`ReadRequest`],
//! [`RestoreReport`], [`RestoreAllReport`], and the opt-in [`ReadCache`] with [`CachedMember`],
//! [`CacheOptions`], [`CacheStats`].

mod batch;
mod cache;
#[cfg(test)]
mod cache_tests;
mod container;
mod member;
mod restore;
mod scratch;
#[cfg(test)]
pub(crate) mod testkit;
#[cfg(test)]
mod tests;

pub use cache::{CacheOptions, CacheStats, CachedMember, ReadCache};
pub use container::{Container, MemberInfo, OpenOptions};
pub use member::{Member, ReadRequest};
pub use restore::{RestoreAllReport, RestoreReport};
