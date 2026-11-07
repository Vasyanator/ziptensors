//! File: status.rs
//! Purpose: the status code of every fallible entry point, and the crate's internal failure type.
//!
//! Main responsibilities:
//! - [`ZtStatus`], the C view of a result, with the exhaustive mapping from `ErrorCode`;
//! - [`Failure`], which pairs a status with the text `zt_last_error` shows.
//!
//! Key structures: [`ZtStatus`], [`Failure`].
//!
//! Implementation notes: positive values are `ErrorCode` values of core, negative ones belong to this
//! crate, so core can take the next free positive number without a collision. The literals exist
//! because cbindgen does not evaluate `ErrorCode::Io as i32`; a test pins each one to its code.

use ziptensors_core::{Error, ErrorCode};

/// Result of an entry point. `ZT_STATUS_OK` is 0, positive values name a failure of the library, and
/// negative ones a call that was refused at the ABI. Values never change and are never reused. After
/// any value but `ZT_STATUS_OK`, `zt_last_error` has the details.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ZtStatus {
    /// Success.
    Ok = 0,
    /// A file operation failed.
    Io = 1,
    /// The file is not a ziptensors container.
    NotZiptensors = 2,
    /// The container has a format version this library does not read.
    UnsupportedVersion = 3,
    /// The container requires a feature this library does not implement.
    UnsupportedFeature = 4,
    /// The container uses a transform, view or stream codec this library does not know.
    UnknownCodec = 5,
    /// The header, footer or index of the container violates the format.
    MalformedIndex = 6,
    /// A declared size is over a limit of `zt_open_options`.
    BudgetExceeded = 7,
    /// A chunk cannot be decoded.
    ChunkCorrupt = 8,
    /// A chunk decoded to bytes that do not match its hash.
    ChecksumMismatch = 9,
    /// The requested range does not lie in the member.
    OutOfBounds = 10,
    /// The destination does not have the length the operation requires.
    DestinationSize = 11,
    /// The member has no tensor of this name.
    TensorNotFound = 12,
    /// The container has no such member, or not exactly one when a single member was asked for.
    MemberNotFound = 13,
    /// A member name is not safe to join to an output directory.
    UnsafeMemberName = 14,
    /// The bytes of a member do not hash to the value in its record.
    VerifyFailed = 15,
    /// An input file changed while it was being encoded. No output was committed.
    InputChanged = 16,
    /// The zstd library reported an error.
    Zstd = 17,
    /// An input is a symbolic link. Links are not followed.
    SymlinkInput = 18,
    /// An output path is the file being read.
    OutputIsInput = 19,
    /// The output WAS written: the new file replaced the output path, but its directory could not be
    /// synced afterwards, so a power loss may bring back the previous file. Every other failure of a
    /// call that writes a file leaves its output untouched. Unix only.
    CommittedNotDurable = 20,
    /// A required pointer was NULL.
    NullArgument = -1,
    /// A string argument is not valid UTF-8.
    InvalidUtf8 = -2,
    /// An argument has a value the call does not accept: an unknown enum value, a tensor index past
    /// the end, a length over `PTRDIFF_MAX`.
    InvalidArgument = -3,
    /// `struct_size` of an options struct is not the size this library knows: the caller was built
    /// against another header.
    StructSize = -4,
    /// The handle cannot do this any more: a `zt_pack` that was finished, or that panicked.
    InvalidState = -5,
    /// The library panicked. The call did not complete, and handles it used should be closed.
    Panic = -6,
}

impl ZtStatus {
    /// The status of a core error code. Exhaustive on purpose: a new `ErrorCode` stops this crate from
    /// building until it has a status.
    #[must_use]
    pub(crate) fn from_code(code: ErrorCode) -> Self {
        match code {
            ErrorCode::Io => Self::Io,
            ErrorCode::NotZiptensors => Self::NotZiptensors,
            ErrorCode::UnsupportedVersion => Self::UnsupportedVersion,
            ErrorCode::UnsupportedFeature => Self::UnsupportedFeature,
            ErrorCode::UnknownCodec => Self::UnknownCodec,
            ErrorCode::MalformedIndex => Self::MalformedIndex,
            ErrorCode::BudgetExceeded => Self::BudgetExceeded,
            ErrorCode::ChunkCorrupt => Self::ChunkCorrupt,
            ErrorCode::ChecksumMismatch => Self::ChecksumMismatch,
            ErrorCode::OutOfBounds => Self::OutOfBounds,
            ErrorCode::DestinationSize => Self::DestinationSize,
            ErrorCode::TensorNotFound => Self::TensorNotFound,
            ErrorCode::MemberNotFound => Self::MemberNotFound,
            ErrorCode::UnsafeMemberName => Self::UnsafeMemberName,
            ErrorCode::VerifyFailed => Self::VerifyFailed,
            ErrorCode::InputChanged => Self::InputChanged,
            ErrorCode::Zstd => Self::Zstd,
            ErrorCode::SymlinkInput => Self::SymlinkInput,
            ErrorCode::OutputIsInput => Self::OutputIsInput,
            ErrorCode::CommittedNotDurable => Self::CommittedNotDurable,
        }
    }
}

/// Why an entry point did not succeed. Its text is what `zt_last_error` returns.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Failure {
    /// The library failed.
    #[error(transparent)]
    Core(#[from] Error),

    /// A required pointer was NULL.
    #[error("argument `{argument}` is NULL")]
    Null {
        /// C name of the parameter.
        argument: &'static str,
    },

    /// A string is not valid UTF-8.
    #[error("argument `{argument}` is not valid UTF-8")]
    Utf8 {
        /// C name of the parameter.
        argument: &'static str,
    },

    /// An argument has a value the call does not accept.
    #[error("{detail}")]
    Invalid {
        /// What is wrong, naming the parameter.
        detail: String,
    },

    /// `struct_size` of an options struct differs from the size of this build.
    #[error("`{argument}` has struct_size {actual}, this library expects {expected}")]
    StructSize {
        /// C name of the parameter.
        argument: &'static str,
        /// Size in this build.
        expected: usize,
        /// Size the caller stated.
        actual: usize,
    },

    /// The handle is past the point where this call is possible.
    #[error("{detail}")]
    State {
        /// What happened to the handle.
        detail: &'static str,
    },
}

impl Failure {
    /// The status C sees.
    #[must_use]
    pub(crate) fn status(&self) -> ZtStatus {
        match self {
            Self::Core(error) => ZtStatus::from_code(error.code()),
            Self::Null { .. } => ZtStatus::NullArgument,
            Self::Utf8 { .. } => ZtStatus::InvalidUtf8,
            Self::Invalid { .. } => ZtStatus::InvalidArgument,
            Self::StructSize { .. } => ZtStatus::StructSize,
            Self::State { .. } => ZtStatus::InvalidState,
        }
    }
}
