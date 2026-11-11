//! File: io/mod.rs
//! Purpose: file access below the reader, the writer and the profile parsers: the byte source
//! abstraction, views over a source, and the temporary file the replace commit goes through.
//!
//! Key structures: re-exports [`RandomAccess`] and [`FileSource`] from `source.rs`, [`SpliceSource`]
//! and [`PrefixSource`] from `splice.rs`; the crate-private `TempFile` from `temp.rs`, and its public
//! [`TEMP_FILE_SUFFIX`].

mod source;
mod splice;
mod temp;

pub use source::{FileSource, RandomAccess};
pub use splice::{PrefixSource, SpliceSource};
pub use temp::TEMP_FILE_SUFFIX;
pub(crate) use temp::{TempFile, same_file};
