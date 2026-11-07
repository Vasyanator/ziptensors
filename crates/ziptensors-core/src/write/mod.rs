//! File: write/mod.rs
//! Purpose: writer API: turning source files into a container, applying the mode policy, blob
//! deduplication, delta blobs against a named base, verified commit and member copy between
//! containers.
//!
//! Main responsibilities: the module list and the public surface of the writer.
//!
//! Key structures: [`EncodeOptions`], [`PackBuilder`], [`EncodeReport`], [`MemberReport`],
//! [`LayerBytes`], [`ClassBytes`], [`VerifyReport`].
//!
//! Key functions: [`encode_file`], [`layer_report`].

mod builder;
mod commit;
mod dedup;
mod ingest;
mod merge;
mod pairing;
mod policy;
mod report;
#[cfg(test)]
mod tests;

pub use builder::{EncodeOptions, PackBuilder, encode_file};
pub use report::{ClassBytes, EncodeReport, LayerBytes, MemberReport, VerifyReport, layer_report};
