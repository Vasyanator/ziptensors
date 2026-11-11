//! File: errors.rs
//! Purpose: the Python exceptions of the module and the translation of a core `Error` into them.
//!
//! Main responsibilities:
//! - `ZiptensorsError`, the base class, and one subclass per family of `ErrorCode`, plus
//!   `NotDurableError` under `ZiptensorsIoError`, the one failure after which the output exists;
//! - [`to_py_err`] and [`raised`], the one path from a core error to a raised exception.
//!
//! Key functions: [`to_py_err`], [`raised`].
//!
//! Implementation notes: the mapping matches on `ErrorCode`, which is exhaustive by design, because
//! `Error` is `#[non_exhaustive]`. A new code therefore fails to compile here until it has a family.
//! The core logs its failures; an exception carries the message and the stable numeric code.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use ziptensors_core::{Error, ErrorCode};

create_exception!(
    ziptensors,
    ZiptensorsError,
    PyException,
    "Base class of every error of this module. `code` is the stable numeric ErrorCode of the failure."
);
create_exception!(
    ziptensors,
    ZiptensorsIoError,
    ZiptensorsError,
    "An operating system file operation failed, or a file holds a valid container followed by bytes that could not be removed; recovery truncates such a file."
);
create_exception!(
    ziptensors,
    NotDurableError,
    ZiptensorsIoError,
    "The output WAS written and is complete, but its directory could not be synced, so a power loss may bring back the previous file. Every other error of a write leaves the output untouched."
);
create_exception!(
    ziptensors,
    FormatError,
    ZiptensorsError,
    "The file is not a ziptensors container, its header, footer or index violates the format, or recovery found no valid container in it."
);
create_exception!(
    ziptensors,
    UnsupportedError,
    ZiptensorsError,
    "The file needs a format version, feature or codec this library does not implement."
);
create_exception!(
    ziptensors,
    BudgetError,
    ZiptensorsError,
    "A size declared by the file exceeds a limit. Raised before the data is read."
);
create_exception!(
    ziptensors,
    CorruptionError,
    ZiptensorsError,
    "Stored data does not decode, or does not match its checksum or hash, or an in-place edit failed after it began moving data and damaged the file."
);
create_exception!(
    ziptensors,
    RangeError,
    ZiptensorsError,
    "A read range lies outside the member, or a buffer has the wrong size."
);
create_exception!(
    ziptensors,
    NotFoundError,
    ZiptensorsError,
    "The container has no such member, or the member no such tensor."
);
create_exception!(
    ziptensors,
    InputError,
    ZiptensorsError,
    "An input of an encode was refused: an unsafe name, a symbolic link, or a file that changed; or an edit request cannot be applied as asked, or a tree operation found names that collide, and nothing was written; or two inputs of a write would get one member name."
);

/// Builds the exception of `error`: the subclass of its family, its message, and `code`.
///
/// Never fails: if the `code` attribute cannot be set, which takes a failure of the interpreter
/// itself, that failure is what the caller raises.
pub(crate) fn to_py_err(py: Python<'_>, error: &Error) -> PyErr {
    let code = error.code();
    let message = error.to_string();
    let err = match code {
        ErrorCode::Io | ErrorCode::TrailingBytes => ZiptensorsIoError::new_err(message),
        ErrorCode::CommittedNotDurable => NotDurableError::new_err(message),
        ErrorCode::NotZiptensors | ErrorCode::MalformedIndex | ErrorCode::Unrecoverable => {
            FormatError::new_err(message)
        }
        ErrorCode::UnsupportedVersion
        | ErrorCode::UnsupportedFeature
        | ErrorCode::UnknownCodec => UnsupportedError::new_err(message),
        ErrorCode::BudgetExceeded => BudgetError::new_err(message),
        ErrorCode::ChunkCorrupt
        | ErrorCode::ChecksumMismatch
        | ErrorCode::VerifyFailed
        | ErrorCode::Zstd
        | ErrorCode::DamagedInPlace => CorruptionError::new_err(message),
        ErrorCode::OutOfBounds | ErrorCode::DestinationSize => RangeError::new_err(message),
        ErrorCode::TensorNotFound | ErrorCode::MemberNotFound => NotFoundError::new_err(message),
        ErrorCode::UnsafeMemberName
        | ErrorCode::InputChanged
        | ErrorCode::SymlinkInput
        | ErrorCode::OutputIsInput
        | ErrorCode::EditRefused
        | ErrorCode::TreeConflict
        | ErrorCode::DuplicateMember => InputError::new_err(message),
    };
    match err.value(py).setattr("code", code.as_i32()) {
        Ok(()) => err,
        Err(attribute_error) => attribute_error,
    }
}

/// Turns the result of a core call into the result of a Python call.
///
/// # Errors
/// The exception of [`to_py_err`] if `result` is an error.
pub(crate) fn raised<T>(py: Python<'_>, result: Result<T, Error>) -> PyResult<T> {
    result.map_err(|error| to_py_err(py, &error))
}
