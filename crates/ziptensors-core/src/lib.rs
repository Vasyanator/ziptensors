//! File: lib.rs
//! Purpose: crate root of `ziptensors-core`, the library behind every ziptensors frontend.
//!
//! Main responsibilities: declares the modules, the crate-wide lint policy and the public
//! re-exports of the shared types.
//!
//! Key structures: [`Error`], [`ErrorCode`], [`Chain`], [`Step`], [`Layout`], [`View`], [`Dtype`],
//! [`Profile`], [`Mode`], [`Verify`], [`TypedRange`], [`TensorInfo`], [`RangePlan`],
//! [`RandomAccess`], [`FileSource`], [`Container`], [`Member`], [`OpenOptions`], [`MemberInfo`],
//! [`ReadRequest`], [`RestoreReport`], [`RestoreAllReport`], [`ReadCache`], [`CachedMember`],
//! [`CacheOptions`], [`CacheStats`], [`EncodeOptions`], [`PackBuilder`], [`EncodeReport`],
//! [`MemberReport`], [`LayerBytes`], [`ClassBytes`], [`VerifyReport`], [`Inputs`], [`EditOp`],
//! [`EditOptions`], [`EditReport`], [`Strategy`], [`RecoverOptions`], [`RecoverReport`], [`ScanBudget`],
//! [`Scan`], [`CompressTreeOptions`], [`DecompressTreeOptions`], [`TreeReport`], [`FileOutcome`],
//! [`FileStatus`], [`TreeView`], [`ViewEntry`], [`ViewOptions`].
//!
//! Key functions: [`encode_file`], [`layer_report`], [`collect_inputs`], [`check_member_names`],
//! [`edit()`], [`recover`],
//! [`find_last_valid_generation`], [`compress_tree`], [`decompress_tree`], [`scan_view`].
//!
//! Implementation notes: the format is defined by `spec/ziptensors-format.md`. `unsafe` is denied
//! here and permitted only for the AVX2 kernels of `codec`, which lift the denial locally.

#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![deny(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod chain;
pub mod codec;
pub mod edit;
pub mod error;
pub mod format;
pub mod io;
pub mod profile;
pub mod read;
pub mod recover;
pub mod tree;
pub mod types;
pub mod write;

pub use chain::{
    BlockGeometry, Chain, DeltaOp, ElementWidth, FieldSpan, FieldTable, Layout, RawStep, Step, View,
};
pub use edit::{EditOp, EditOptions, EditReport, Strategy, edit};
pub use error::{Error, ErrorCode};
pub use io::{FileSource, RandomAccess};
pub use read::{
    CacheOptions, CacheStats, CachedMember, Container, Member, MemberInfo, OpenOptions, ReadCache,
    ReadRequest, RestoreAllReport, RestoreReport,
};
pub use recover::{
    Candidate, RecoverOptions, RecoverReport, Scan, ScanBudget, ScanOutcome,
    find_last_valid_generation, recover,
};
pub use tree::{
    CompressTreeOptions, DecompressTreeOptions, FileOutcome, FileStatus, TreeReport, TreeView,
    ViewEntry, ViewOptions, compress_tree, decompress_tree, scan_view,
};
pub use types::{Dtype, Mode, Profile, RangePlan, TensorInfo, TypedRange, Verify};
pub use write::inputs::{Inputs, check_member_names, collect_inputs};
pub use write::{
    ClassBytes, EncodeOptions, EncodeReport, LayerBytes, MemberReport, PackBuilder, VerifyReport,
    encode_file, layer_report,
};
