//! File: profile/mod.rs
//! Purpose: recognizes source file formats by content and produces a `RangePlan`. Knows nothing
//! of the container format; every result is a hint.
//!
//! Main responsibilities:
//! - dispatch to one parser by magic or structure, never by file name;
//! - the rule that a malformed input degrades to an opaque plan and only real I/O failure is `Err`.
//!
//! Key functions: [`scan`], and the re-exported [`normalize`] and [`width_trial`].
//!
//! Key structures: [`Stop`], the internal outcome that separates "give up" from "I/O failed".
//!
//! Implementation notes: a parser's output picks codecs and names tensors and is never trusted for
//! restore. `scan` returns what the parser saw, so a caller must pass the plan through
//! [`normalize`] before using it. All parsers read through [`reader::Reader`], which checks every
//! length against the source before reading or allocating.

mod gguf;
mod json;
mod onnx;
mod plan;
mod pt_zip;
mod reader;
mod safetensors;
mod width_trial;

#[cfg(test)]
mod tests;

pub use plan::{MIN_RANGE_BYTES, normalize};
pub use width_trial::width_trial;

use crate::error::Error;
use crate::io::RandomAccess;
use crate::types::{Profile, RangePlan};

use reader::Reader;

/// Most entries a parser accepts in `ranges` or `tensors`. Far above any real model, whose count is
/// in the thousands, and low enough that the records fit the spec's default 256 MiB index cap.
const MAX_TENSORS: usize = 1 << 18;

/// Longest tensor or entry name a parser reads, in bytes.
const MAX_NAME_BYTES: usize = 64 * 1024;

/// Bytes of the file start that the magic checks look at.
const HEAD_BYTES: u64 = 9;

/// Why a parser stopped. Only `Io` leaves this module.
#[derive(Debug)]
enum Stop {
    /// The bytes are not what the parser needs. Degrades to an opaque plan.
    Malformed(&'static str),
    /// A read of the source failed.
    Io(Error),
}

impl From<Error> for Stop {
    fn from(error: Error) -> Self {
        Self::Io(error)
    }
}

/// Result of a parser or of one of its steps.
type ParseResult<T> = Result<T, Stop>;

/// Finds the tensor data of `src` and describes it, choosing the parser from the content alone.
///
/// Returns the parser's view of the file: ranges may overlap, leave the file or be short, so the
/// caller passes the plan through [`normalize`]. Input the parser cannot follow, whatever is wrong
/// with it, gives a plan with [`Profile::Opaque`], no ranges and no tensors. Reads are bounded and
/// no allocation follows a size the file declares unless the bytes are present.
///
/// # Errors
/// Returns the error of a failed read of `src`, [`Error::Io`] in practice. Malformed content is
/// never an error.
pub fn scan(src: &dyn RandomAccess) -> Result<RangePlan, Error> {
    let len = src.len();
    tracing::debug!(len, "scanning source for tensor data");
    let plan = match dispatch(src) {
        Ok(plan) => plan,
        Err(Stop::Malformed(reason)) => {
            tracing::info!(len, reason, "source treated as opaque");
            opaque_plan()
        }
        Err(Stop::Io(error)) => {
            tracing::error!(len, %error, "reading the source failed during scan");
            return Err(error);
        }
    };
    tracing::debug!(
        len,
        profile = ?plan.profile,
        ranges = plan.ranges.len(),
        tensors = plan.tensors.len(),
        "scan finished"
    );
    Ok(plan)
}

/// Picks and runs a parser. A file nothing recognizes is an opaque plan and not a failure.
fn dispatch(src: &dyn RandomAccess) -> ParseResult<RangePlan> {
    let mut reader = Reader::new(src);
    let head = reader.take(reader.len().min(HEAD_BYTES), HEAD_BYTES)?;
    reader.seek(0)?;
    if head.starts_with(gguf::MAGIC) {
        gguf::parse(&mut reader)
    } else if head.starts_with(pt_zip::LOCAL_MAGIC) {
        pt_zip::parse(&mut reader)
    } else if safetensors::matches(&head, reader.len()) {
        safetensors::parse(&mut reader)
    } else if onnx::matches(&mut reader)? {
        onnx::parse(&mut reader)
    } else {
        Ok(opaque_plan())
    }
}

/// The plan of a file with no recognized structure.
fn opaque_plan() -> RangePlan {
    RangePlan { profile: Profile::Opaque, ranges: Vec::new(), tensors: Vec::new() }
}
