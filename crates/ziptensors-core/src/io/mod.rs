//! File: io/mod.rs
//! Purpose: file access below the reader, the writer and the profile parsers: the byte source
//! abstraction, and the temporary file both commits go through.
//!
//! Key structures: re-exports [`RandomAccess`] and [`FileSource`] from `source.rs`; the crate-private
//! `TempFile` from `temp.rs`, and its public [`TEMP_FILE_SUFFIX`].

mod source;
mod temp;

pub use source::{FileSource, RandomAccess};
pub use temp::TEMP_FILE_SUFFIX;
pub(crate) use temp::{TempFile, same_file};
