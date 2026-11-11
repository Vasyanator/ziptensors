//! File: write/mod.rs
//! Purpose: writer API: turning source files into a container, applying the mode policy, blob
//! deduplication, delta blobs against a named base, verified commit and member copy between
//! containers, and the walk of input trees into member names.
//!
//! Main responsibilities: the module list, the public surface of the writer, and the crate-private
//! surface the `edit` module builds on (a builder over an existing container, its seed, the append
//! sink, inputs that are container members, bases named by catalog).
//!
//! Key structures: [`EncodeOptions`], [`PackBuilder`], [`EncodeReport`], [`MemberReport`],
//! [`LayerBytes`], [`ClassBytes`], [`VerifyReport`], [`inputs::Inputs`].
//!
//! Key functions: [`encode_file`], [`layer_report`], [`inputs::collect_inputs`],
//! [`inputs::check_output`].

mod builder;
mod commit;
mod dedup;
mod ingest;
pub mod inputs;
mod merge;
mod pairing;
mod policy;
mod report;
#[cfg(test)]
mod tests;

pub use builder::{EncodeOptions, PackBuilder, encode_file};
pub(crate) use builder::{Commit, DeltaSpec, NameRule, Seed, SourceStamp};
pub(crate) use commit::Sink;
#[cfg(test)]
pub(crate) use commit::{SinkFault, SinkFaultMode};
pub(crate) use pairing::BaseCatalog;
pub use report::{ClassBytes, EncodeReport, LayerBytes, MemberReport, VerifyReport, layer_report};
