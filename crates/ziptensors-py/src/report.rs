//! File: report.rs
//! Purpose: what an encode, a pack and a restore return to Python.
//!
//! Main responsibilities: read-only classes mirroring the report types of `ziptensors-core`, with
//! their conversions. Sums and meanings stay in the core.
//!
//! Key structures: [`LayerBytes`], [`ClassBytes`], [`MemberReport`], [`VerifyReport`],
//! [`EncodeReport`], [`RestoreReport`].
//!
//! Implementation notes: fields use the core's names. Times are seconds, as `float`. Dtypes and
//! profiles are names, by `names.rs`. `Clone` exists only because a getter hands out a copy.

use pyo3::prelude::*;
use ziptensors_core as zt;

use crate::names::{dtype_name, profile_name};

/// Bytes of one member or class, by what became of them.
#[pyclass(frozen, get_all, module = "ziptensors")]
#[derive(Debug, Clone)]
pub(crate) struct LayerBytes {
    /// Source bytes.
    raw: u64,
    /// Bytes not stored because a widening view reproduces them.
    widen_saved: u64,
    /// Canonical bytes not stored because a blob with the same identity exists.
    dedup_saved: u64,
    /// Canonical bytes the streams of the new blobs saved.
    entropy_saved: u64,
    /// Bytes the stored form adds over canonical bytes, on blobs that did not compress.
    overhead: u64,
    /// Stored chunk bytes of the new blobs.
    stored: u64,
}

impl From<zt::LayerBytes> for LayerBytes {
    fn from(layers: zt::LayerBytes) -> Self {
        Self {
            raw: layers.raw,
            widen_saved: layers.widen_saved,
            dedup_saved: layers.dedup_saved,
            entropy_saved: layers.entropy_saved,
            overhead: layers.overhead,
            stored: layers.stored,
        }
    }
}

/// The bytes of one class of source data.
#[pyclass(frozen, get_all, module = "ziptensors")]
#[derive(Debug, Clone)]
pub(crate) struct ClassBytes {
    /// Registry name of the dtype; None for skeleton fragments and opaque files.
    dtype: Option<String>,
    /// Where the bytes went.
    layers: LayerBytes,
}

impl From<zt::ClassBytes> for ClassBytes {
    fn from(class: zt::ClassBytes) -> Self {
        Self { dtype: class.dtype.map(dtype_name), layers: class.layers.into() }
    }
}

/// What adding one member did.
#[pyclass(frozen, get_all, module = "ziptensors")]
#[derive(Debug, Clone)]
pub(crate) struct MemberReport {
    /// Member name.
    name: String,
    /// Length of the member in bytes.
    len: u64,
    /// BLAKE3 of the member's source content, 32 bytes.
    blake3: [u8; 32],
    /// Recognized source format.
    profile: String,
    /// Rows of the tensor table.
    tensor_count: u32,
    /// Extents of the member.
    extent_count: u32,
    /// Blobs this member added.
    blobs_added: u32,
    /// Extents that went to a blob that already existed.
    blobs_reused: u32,
    /// Ranges streamed narrowed, found not exact, and streamed again.
    widen_restreams: u32,
    /// Blobs this member added that are delta blobs, stored against a base. Counted in `blobs_added`.
    delta_blobs: u32,
    /// Bytes this member compressed into the file.
    compressed_bytes: u64,
    /// Stored bytes copied from another container with no recompression.
    copied_bytes: u64,
    /// Where the bytes of this member went.
    layers: LayerBytes,
}

impl From<zt::MemberReport> for MemberReport {
    fn from(member: zt::MemberReport) -> Self {
        Self {
            name: member.name,
            len: member.len,
            blake3: member.blake3,
            profile: profile_name(member.profile),
            tensor_count: member.tensor_count,
            extent_count: member.extent_count,
            blobs_added: member.blobs_added,
            blobs_reused: member.blobs_reused,
            widen_restreams: member.widen_restreams,
            delta_blobs: member.delta_blobs,
            compressed_bytes: member.compressed_bytes,
            copied_bytes: member.copied_bytes,
            layers: member.layers.into(),
        }
    }
}

/// What the full decode before commit did.
#[pyclass(frozen, get_all, module = "ziptensors")]
#[derive(Debug, Clone)]
pub(crate) struct VerifyReport {
    /// Bytes decoded and hashed: the sum of member lengths.
    bytes: u64,
    /// Wall time of the decode in seconds.
    seconds: f64,
}

impl From<zt::VerifyReport> for VerifyReport {
    fn from(verify: zt::VerifyReport) -> Self {
        Self { bytes: verify.bytes, seconds: verify.elapsed.as_secs_f64() }
    }
}

/// What a finished container holds and what it cost.
#[pyclass(frozen, get_all, module = "ziptensors")]
#[derive(Debug)]
pub(crate) struct EncodeReport {
    /// Length of the output file in bytes.
    output_len: u64,
    /// Stored bytes of the index.
    index_stored_len: u64,
    /// The members, in file order.
    members: Vec<MemberReport>,
    /// The sum of `classes`.
    total: LayerBytes,
    /// Bytes per class, skeleton first, then by dtype id.
    classes: Vec<ClassBytes>,
    /// The verification before commit.
    verify: VerifyReport,
}

impl From<zt::EncodeReport> for EncodeReport {
    fn from(report: zt::EncodeReport) -> Self {
        Self {
            output_len: report.output_len,
            index_stored_len: report.index_stored_len,
            members: report.members.into_iter().map(MemberReport::from).collect(),
            total: report.total.into(),
            classes: report.classes.into_iter().map(ClassBytes::from).collect(),
            verify: report.verify.into(),
        }
    }
}

/// What a restore produced.
#[pyclass(frozen, get_all, module = "ziptensors")]
#[derive(Debug)]
pub(crate) struct RestoreReport {
    /// Bytes written.
    bytes: u64,
    /// BLAKE3 of those bytes, equal to the hash of the member, 32 bytes.
    blake3: [u8; 32],
}

impl From<zt::RestoreReport> for RestoreReport {
    fn from(report: zt::RestoreReport) -> Self {
        Self { bytes: report.bytes, blake3: report.blake3 }
    }
}
