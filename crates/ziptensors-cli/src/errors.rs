//! File: errors.rs
//! Purpose: how a failed command tells the user: a human message and what to check on stderr, and the
//! technical detail as a structured log.
//!
//! Key functions: [`report`].
//!
//! Implementation notes: the hint is keyed on `ErrorCode` and not on `Error`, which is
//! `#[non_exhaustive]`, so the exhaustive match here breaks when the core adds a failure.

use std::io::{self, Write};

use ziptensors_core::{Error, ErrorCode};

/// What to check for a failure of class `code`.
fn hint(code: ErrorCode) -> &'static str {
    match code {
        ErrorCode::Io => "check the path, its permissions and the free disk space",
        ErrorCode::NotZiptensors => "the file is not a ziptensors container",
        ErrorCode::UnsupportedVersion | ErrorCode::UnsupportedFeature | ErrorCode::UnknownCodec => {
            "the file needs a newer ziptensors"
        }
        ErrorCode::MalformedIndex
        | ErrorCode::ChunkCorrupt
        | ErrorCode::ChecksumMismatch
        | ErrorCode::OutOfBounds
        | ErrorCode::VerifyFailed
        | ErrorCode::Zstd => "the file is damaged or was not written by a correct writer",
        ErrorCode::BudgetExceeded => "the file declares more than this reader accepts",
        ErrorCode::DestinationSize | ErrorCode::TensorNotFound => {
            "check the tensor name and the size of the destination"
        }
        ErrorCode::MemberNotFound => "`ziptensors inspect` lists the members of the container",
        ErrorCode::UnsafeMemberName => {
            "the name could escape the output directory or cannot be restored, so nothing was written for it"
        }
        ErrorCode::InputChanged => "an input was modified during the run: run it again when it is stable",
        ErrorCode::SymlinkInput => "links are not followed: pass the file they point to",
        ErrorCode::OutputIsInput => "choose an output path that is not the file being read",
        ErrorCode::CommittedNotDurable => {
            "the output WAS written and is complete, but it may not survive a power loss: run `sync`, or check the directory's file system"
        }
    }
}

/// The causes of `error`, outermost first, joined with `: `. A cause whose text is already in the message
/// is left out: `Error::Io` prints its OS error and also returns it as its source.
fn message(error: &anyhow::Error) -> String {
    let mut text = String::new();
    for cause in error.chain() {
        let part = cause.to_string();
        if text.contains(&part) {
            continue;
        }
        if !text.is_empty() {
            text.push_str(": ");
        }
        text.push_str(&part);
    }
    text
}

/// Reports a failed command: `error: <what failed>: <why>` and, for a failure of the core, a `hint:`
/// line on stderr, plus an `ERROR` log with the numeric code and the error's fields.
pub(crate) fn report(error: &anyhow::Error) {
    let failure = error.chain().find_map(|cause| cause.downcast_ref::<Error>());
    let code = failure.map(Error::code);
    tracing::error!(
        code = ?code,
        code_value = code.map(ErrorCode::as_i32),
        detail = ?failure,
        "command failed"
    );
    let mut stderr = io::stderr();
    // Stderr is the last channel there is: if these writes fail, nothing can carry the news, and the
    // exit code still does.
    let _ = writeln!(stderr, "error: {}", message(error));
    if let Some(code) = code {
        let _ = writeln!(stderr, "hint: {}", hint(code));
    }
}

#[cfg(test)]
mod tests {
    use super::{hint, message};
    use anyhow::Context;
    use std::io;
    use std::path::PathBuf;
    use ziptensors_core::Error;

    #[test]
    fn a_commit_without_durability_says_the_output_was_written() {
        let failed: Result<(), Error> = Err(Error::CommittedNotDurable {
            path: PathBuf::from("model.ziptensors"),
            source: io::Error::other("disk on fire"),
        });
        let error = failed.context("cannot pack model.ziptensors").expect_err("failed");
        let text = message(&error);
        assert!(text.contains("model.ziptensors was written"), "{text}");
        assert_eq!(text.matches("disk on fire").count(), 1, "{text}");
        let code = error.downcast_ref::<Error>().map(Error::code).expect("a core error");
        assert!(hint(code).contains("WAS written"));
    }

    #[test]
    fn the_message_names_each_cause_once() {
        let failed: Result<(), Error> = Err(Error::Io {
            op: "open",
            path: PathBuf::from("model.bin"),
            source: io::Error::other("disk on fire"),
        });
        let error = failed.context("cannot pack model.bin").expect_err("failed");
        let text = message(&error);
        assert_eq!(text.matches("disk on fire").count(), 1, "{text}");
        assert!(text.starts_with("cannot pack model.bin: I/O error during open"), "{text}");
    }
}
