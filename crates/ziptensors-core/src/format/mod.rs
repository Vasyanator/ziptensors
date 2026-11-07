//! File: format/mod.rs
//! Purpose: on-disk layout of the container: header, footer, index records and structural
//! validation. Knows nothing of codecs, profiles or I/O; ids, `Chain` and `View` belong to
//! `crate::chain`.
//!
//! Main responsibilities: the module list and the public surface of the format layer.
//!
//! Key structures: [`Header`], [`Footer`], [`Index`], [`IndexBuilder`], [`Limits`].
//!
//! Key functions: [`parse_header`], [`parse_footer`], [`validate`], [`check_decode_budget`],
//! [`check_restore_name`].

mod builder;
mod footer;
mod header;
mod index;
mod limits;
mod records;
#[cfg(test)]
mod testkit;
mod validate;
mod wire;

pub use builder::IndexBuilder;
pub use footer::{END_MAGIC, FOOTER_LEN, Footer, GENERATION_INITIAL, parse_footer};
pub use header::{FORMAT_MAJOR, FORMAT_MINOR, HEADER_LEN, Header, MAGIC, parse_header};
pub use index::Index;
pub use limits::{
    DECODE_WORK_FLOOR, DEFAULT_MAX_DECODE_AMPLIFICATION, DEFAULT_MAX_INDEX_BYTES,
    DEFAULT_MAX_NAME_WORK_BYTES, Limits, MAX_INDEX_RAW_LEN,
};
pub use records::{
    BlobRecord, ChainRecord, ChunkRecord, DELTA_PARAMS_LEN, DeltaParams, ExtentRecord,
    MemberRecord, PARAMS_ALIGN, StrRef, TensorRecord,
};
pub use validate::{MAX_NDIM, check_decode_budget, check_restore_name, validate};
