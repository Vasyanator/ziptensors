//! File: error.rs
//! Purpose: the single public error type of `ziptensors-core`.
//!
//! Main responsibilities:
//! - name every failure the library can report, with the context a caller needs to act on it.
//!
//! Key structures: [`Error`], [`ErrorCode`].
//!
//! Implementation notes: the set of variants is a frozen contract shared by the C ABI, the Python
//! module and the CLI. Corruption of any kind surfaces here as a value; nothing in the library
//! panics on file contents. `Error` is `#[non_exhaustive]`, so other crates cannot match it
//! exhaustively: they use [`Error::code`], whose exhaustive match is what forces every new variant
//! to get a new [`ErrorCode`].

use std::path::PathBuf;

/// Everything `ziptensors-core` can fail with.
///
/// Conditions that have no variant of their own map as follows: a damaged header or footer is
/// [`Error::MalformedIndex`] with `section` set to `"header"` or `"footer"`, an unknown section or
/// blob kind is [`Error::MalformedIndex`] naming the section, and an unknown step, view or stream
/// codec is [`Error::UnknownCodec`].
///
/// A new variant needs a new [`ErrorCode`] and an arm in [`Error::code`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An operating system I/O call failed. `op` names the operation, `path` the file involved.
    #[error("I/O error during {op} on {}: {source}", path.display())]
    Io {
        /// Operation that failed, such as `"open"` or `"read_exact_at"`.
        op: &'static str,
        /// File the operation was applied to.
        path: PathBuf,
        /// Underlying OS error.
        source: std::io::Error,
    },

    /// The input does not start with the ziptensors magic, or is too short to hold one.
    #[error("not a ziptensors file")]
    NotZiptensors,

    /// The header carries a major version this library does not read.
    #[error("unsupported format version {major}.{minor}")]
    UnsupportedVersion {
        /// Major version found in the file.
        major: u16,
        /// Minor version found in the file.
        minor: u16,
    },

    /// `required_features` has bits this library does not implement.
    #[error("unsupported required feature bits {bits:#018x}")]
    UnsupportedFeature {
        /// The bits that are set and not understood.
        bits: u64,
    },

    /// A transform step, widening view or stream codec id is not known to this library.
    #[error("unknown codec id {id:#06x}")]
    UnknownCodec {
        /// The rejected id. Stream codec bytes are widened to `u16`.
        id: u16,
    },

    /// The header, footer or index violates the format. `section` is a section tag or one of
    /// `"header"`, `"footer"`, `"index"`.
    #[error("malformed index in {section}: {detail}")]
    MalformedIndex {
        /// Where the violation was found.
        section: &'static str,
        /// What is wrong, for humans.
        detail: String,
    },

    /// A declared size exceeds a caller limit. Raised before the data is read or allocated.
    #[error("{what} declares {declared} bytes, over the limit of {limit}")]
    BudgetExceeded {
        /// The quantity that was checked, such as `"index"` or `"total decoded size"`.
        what: &'static str,
        /// Size the file declares.
        declared: u64,
        /// Limit in force.
        limit: u64,
    },

    /// A chunk cannot be decoded: bad stream words, truncated stream, wrong decoded length.
    #[error("corrupt chunk {chunk} of blob {blob}: {detail}")]
    ChunkCorrupt {
        /// Index of the blob record.
        blob: u32,
        /// Chunk index within the blob.
        chunk: u32,
        /// What is wrong, for humans.
        detail: String,
    },

    /// A chunk decoded, but its XXH3-64 differs from the index record.
    #[error("checksum mismatch in chunk {chunk} of blob {blob}: expected {expected:#018x}, got {actual:#018x}")]
    ChecksumMismatch {
        /// Index of the blob record.
        blob: u32,
        /// Chunk index within the blob.
        chunk: u32,
        /// Hash stored in the index.
        expected: u64,
        /// Hash of the decoded bytes.
        actual: u64,
    },

    /// A read of `len` bytes at `offset` does not fit in `file_len` bytes.
    #[error("read of {len} bytes at offset {offset} is outside the {file_len} byte file")]
    OutOfBounds {
        /// Requested start.
        offset: u64,
        /// Requested length.
        len: u64,
        /// Length of the file or member that was addressed.
        file_len: u64,
    },

    /// A destination buffer does not have the length the operation requires.
    #[error("destination holds {actual} bytes, {expected} required")]
    DestinationSize {
        /// Length the operation needs.
        expected: u64,
        /// Length the caller passed.
        actual: u64,
    },

    /// The member has no tensor with this name.
    #[error("tensor not found: {name}")]
    TensorNotFound {
        /// Requested name.
        name: String,
    },

    /// The container has no such member, or `single` was used on a container that does not hold
    /// exactly one.
    #[error("member not found: {name}")]
    MemberNotFound {
        /// Requested name.
        name: String,
    },

    /// A member name is not safe to join to an output directory. Nothing was written for it.
    #[error("unsafe member name: {name:?}")]
    UnsafeMemberName {
        /// The rejected name, as stored in the file.
        name: String,
    },

    /// The restored bytes of a member do not hash to the value in its record.
    #[error("verification of member {member} failed: expected BLAKE3 {expected}, got {actual}")]
    VerifyFailed {
        /// Member name.
        member: String,
        /// Hash stored in the member record.
        expected: blake3::Hash,
        /// Hash of the bytes produced.
        actual: blake3::Hash,
    },

    /// An input file changed while it was being encoded. No output was committed.
    #[error("input changed while encoding: {}", path.display())]
    InputChanged {
        /// The input file.
        path: PathBuf,
    },

    /// An input is a symbolic link, and links are neither followed nor stored, spec section 2.
    ///
    /// A variant of its own and not [`Error::Io`], since nothing went wrong: a frontend walking a
    /// directory skips this one deliberately, while it must report every real I/O failure.
    #[error("input is a symbolic link: {}", path.display())]
    SymlinkInput {
        /// The link.
        path: PathBuf,
    },

    /// An output path is the file being read. Writing it would replace the input by what comes out of
    /// it, say a pack by one of its members. Compared by canonical path and, where the platform gives
    /// one, by file identity, so that a link to the same file is refused too.
    #[error("output is the input file: {}", path.display())]
    OutputIsInput {
        /// The output path.
        path: PathBuf,
    },

    /// The zstd library reported an error outside of chunk decoding, for example on the index.
    #[error("zstd error: {detail}")]
    Zstd {
        /// Message from the library.
        detail: String,
    },

    /// A commit renamed its new file to `path`, which therefore holds the complete, verified output,
    /// but the directory could not be synced afterwards: until the system syncs it on its own, a
    /// power loss may bring back what was at `path` before. Every other error of a commit leaves
    /// `path` untouched; this one alone means the output was written. Raised on Unix only.
    #[error(
        "{} was written, but syncing its directory failed, so the write may not survive a power loss: {source}",
        path.display()
    )]
    CommittedNotDurable {
        /// The output that now holds the new file.
        path: PathBuf,
        /// Error of opening or syncing the directory.
        source: std::io::Error,
    },
}

/// Stable numeric identity of an [`Error`] variant, for the C ABI and other frontends.
///
/// Each value is part of the public ABI: never change or reuse one, and give a new variant the next
/// free number. 0 is left unused, because the C ABI uses it for success. This enum is deliberately
/// not `#[non_exhaustive]`, so that a frontend's exhaustive match breaks when a code is added.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum ErrorCode {
    /// [`Error::Io`].
    Io = 1,
    /// [`Error::NotZiptensors`].
    NotZiptensors = 2,
    /// [`Error::UnsupportedVersion`].
    UnsupportedVersion = 3,
    /// [`Error::UnsupportedFeature`].
    UnsupportedFeature = 4,
    /// [`Error::UnknownCodec`].
    UnknownCodec = 5,
    /// [`Error::MalformedIndex`].
    MalformedIndex = 6,
    /// [`Error::BudgetExceeded`].
    BudgetExceeded = 7,
    /// [`Error::ChunkCorrupt`].
    ChunkCorrupt = 8,
    /// [`Error::ChecksumMismatch`].
    ChecksumMismatch = 9,
    /// [`Error::OutOfBounds`].
    OutOfBounds = 10,
    /// [`Error::DestinationSize`].
    DestinationSize = 11,
    /// [`Error::TensorNotFound`].
    TensorNotFound = 12,
    /// [`Error::MemberNotFound`].
    MemberNotFound = 13,
    /// [`Error::UnsafeMemberName`].
    UnsafeMemberName = 14,
    /// [`Error::VerifyFailed`].
    VerifyFailed = 15,
    /// [`Error::InputChanged`].
    InputChanged = 16,
    /// [`Error::Zstd`].
    Zstd = 17,
    /// [`Error::SymlinkInput`].
    SymlinkInput = 18,
    /// [`Error::OutputIsInput`].
    OutputIsInput = 19,
    /// [`Error::CommittedNotDurable`].
    CommittedNotDurable = 20,
}

impl ErrorCode {
    /// Returns the numeric value.
    #[must_use]
    pub fn as_i32(self) -> i32 {
        // Lossless: the enum is `repr(i32)` and has no fields.
        self as i32
    }
}

impl Error {
    /// Returns the stable code of this error's variant.
    #[must_use]
    pub fn code(&self) -> ErrorCode {
        match self {
            Self::Io { .. } => ErrorCode::Io,
            Self::NotZiptensors => ErrorCode::NotZiptensors,
            Self::UnsupportedVersion { .. } => ErrorCode::UnsupportedVersion,
            Self::UnsupportedFeature { .. } => ErrorCode::UnsupportedFeature,
            Self::UnknownCodec { .. } => ErrorCode::UnknownCodec,
            Self::MalformedIndex { .. } => ErrorCode::MalformedIndex,
            Self::BudgetExceeded { .. } => ErrorCode::BudgetExceeded,
            Self::ChunkCorrupt { .. } => ErrorCode::ChunkCorrupt,
            Self::ChecksumMismatch { .. } => ErrorCode::ChecksumMismatch,
            Self::OutOfBounds { .. } => ErrorCode::OutOfBounds,
            Self::DestinationSize { .. } => ErrorCode::DestinationSize,
            Self::TensorNotFound { .. } => ErrorCode::TensorNotFound,
            Self::MemberNotFound { .. } => ErrorCode::MemberNotFound,
            Self::UnsafeMemberName { .. } => ErrorCode::UnsafeMemberName,
            Self::VerifyFailed { .. } => ErrorCode::VerifyFailed,
            Self::InputChanged { .. } => ErrorCode::InputChanged,
            Self::Zstd { .. } => ErrorCode::Zstd,
            Self::SymlinkInput { .. } => ErrorCode::SymlinkInput,
            Self::OutputIsInput { .. } => ErrorCode::OutputIsInput,
            Self::CommittedNotDurable { .. } => ErrorCode::CommittedNotDurable,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Error, ErrorCode};
    use std::collections::HashSet;
    use std::error::Error as _;
    use std::path::PathBuf;

    #[test]
    fn error_crosses_threads() {
        fn assert_send_sync<T: Send + Sync + 'static>() {}
        assert_send_sync::<Error>();
    }

    /// One value of every variant with its expected code and pinned numeric value.
    fn every_variant() -> Vec<(Error, ErrorCode, i32)> {
        let hash = blake3::Hash::from_bytes([0; 32]);
        let text = String::new;
        vec![
            (
                Error::Io {
                    op: "open",
                    path: PathBuf::new(),
                    source: std::io::Error::from(std::io::ErrorKind::NotFound),
                },
                ErrorCode::Io,
                1,
            ),
            (Error::NotZiptensors, ErrorCode::NotZiptensors, 2),
            (Error::UnsupportedVersion { major: 1, minor: 0 }, ErrorCode::UnsupportedVersion, 3),
            (Error::UnsupportedFeature { bits: 1 }, ErrorCode::UnsupportedFeature, 4),
            (Error::UnknownCodec { id: 0 }, ErrorCode::UnknownCodec, 5),
            (Error::MalformedIndex { section: "", detail: text() }, ErrorCode::MalformedIndex, 6),
            (
                Error::BudgetExceeded { what: "", declared: 1, limit: 0 },
                ErrorCode::BudgetExceeded,
                7,
            ),
            (Error::ChunkCorrupt { blob: 0, chunk: 0, detail: text() }, ErrorCode::ChunkCorrupt, 8),
            (
                Error::ChecksumMismatch { blob: 0, chunk: 0, expected: 0, actual: 1 },
                ErrorCode::ChecksumMismatch,
                9,
            ),
            (Error::OutOfBounds { offset: 0, len: 1, file_len: 0 }, ErrorCode::OutOfBounds, 10),
            (Error::DestinationSize { expected: 1, actual: 0 }, ErrorCode::DestinationSize, 11),
            (Error::TensorNotFound { name: text() }, ErrorCode::TensorNotFound, 12),
            (Error::MemberNotFound { name: text() }, ErrorCode::MemberNotFound, 13),
            (Error::UnsafeMemberName { name: text() }, ErrorCode::UnsafeMemberName, 14),
            (
                Error::VerifyFailed { member: text(), expected: hash, actual: hash },
                ErrorCode::VerifyFailed,
                15,
            ),
            (Error::InputChanged { path: PathBuf::new() }, ErrorCode::InputChanged, 16),
            (Error::Zstd { detail: text() }, ErrorCode::Zstd, 17),
            (Error::SymlinkInput { path: PathBuf::new() }, ErrorCode::SymlinkInput, 18),
            (Error::OutputIsInput { path: PathBuf::new() }, ErrorCode::OutputIsInput, 19),
            (
                Error::CommittedNotDurable {
                    path: PathBuf::new(),
                    source: std::io::Error::from(std::io::ErrorKind::PermissionDenied),
                },
                ErrorCode::CommittedNotDurable,
                20,
            ),
        ]
    }

    #[test]
    fn every_variant_has_its_pinned_code() {
        for (error, code, value) in every_variant() {
            assert_eq!(error.code(), code, "{error:?}");
            assert_eq!(code.as_i32(), value, "{code:?}");
        }
    }

    #[test]
    fn codes_are_unique_and_never_zero() {
        let variants = every_variant();
        let values: HashSet<i32> = variants.iter().map(|(_, code, _)| code.as_i32()).collect();
        assert_eq!(values.len(), variants.len());
        assert!(!values.contains(&0));
    }

    #[test]
    fn io_keeps_source_and_names_op_and_path() {
        let err = Error::Io {
            op: "open",
            path: PathBuf::from("model.ziptensors"),
            source: std::io::Error::from(std::io::ErrorKind::NotFound),
        };
        let text = err.to_string();
        assert!(text.contains("open"), "{text}");
        assert!(text.contains("model.ziptensors"), "{text}");
        assert!(err.source().is_some());
    }

    #[test]
    fn committed_not_durable_says_the_output_was_written() {
        let err = Error::CommittedNotDurable {
            path: PathBuf::from("model.ziptensors"),
            source: std::io::Error::other("disk on fire"),
        };
        let text = err.to_string();
        assert!(text.starts_with("model.ziptensors was written"), "{text}");
        assert!(text.contains("power loss"), "{text}");
        assert!(text.contains("disk on fire"), "{text}");
        assert!(err.source().is_some());
    }

    #[test]
    fn verify_failed_prints_hex_digests() {
        let err = Error::VerifyFailed {
            member: "a".to_owned(),
            expected: blake3::Hash::from_bytes([0xAB; 32]),
            actual: blake3::Hash::from_bytes([0x01; 32]),
        };
        let text = err.to_string();
        assert!(text.contains(&"ab".repeat(32)), "{text}");
        assert!(text.contains(&"01".repeat(32)), "{text}");
    }
}
