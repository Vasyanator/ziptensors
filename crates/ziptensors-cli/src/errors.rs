//! File: errors.rs
//! Purpose: how a failed command tells the user: a human message and what to check on stderr, and the
//! technical detail as a structured log.
//!
//! Key structures: [`Hint`], [`ContainerContext`].
//!
//! Key functions: [`report`].
//!
//! Implementation notes: the hint is keyed on `ErrorCode` and not on `Error`, which is
//! `#[non_exhaustive]`, so the exhaustive match here breaks when the core adds a failure. A command
//! that knows better than the code (a `recover` that cannot open its file for writing) attaches a
//! [`Hint`] as context, which replaces the hint of the code and is left out of the message. A hint
//! that names a command on the container says [`CONTAINER_PLACEHOLDER`]; when the failure carries
//! a [`ContainerContext`], the path the user gave replaces it, quoted for the shell.

use std::fmt::{self, Display, Formatter};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use ziptensors_core::{Error, ErrorCode};

/// A hint chosen by the command, attached with `anyhow::Context::context`. It replaces the hint of the
/// `ErrorCode` and is not repeated in the `error:` line.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Hint(pub(crate) &'static str);

impl Display for Hint {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

/// Where a hint names the container a command should be run on. Replaced by the path of a
/// [`ContainerContext`] when the failure carries one, kept where no path is known.
pub(crate) const CONTAINER_PLACEHOLDER: &str = "<CONTAINER>";

/// The context of a failure about one container the user named: prints `<what> <path>` (`cannot
/// open container model.ziptensors`) in the `error:` line, and gives the hint the path to suggest
/// a command on.
#[derive(Debug, Clone)]
pub(crate) struct ContainerContext {
    /// What failed, such as `"cannot open container"`.
    pub(crate) what: &'static str,
    /// The container path as the user gave it.
    pub(crate) path: PathBuf,
}

impl Display for ContainerContext {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.what, self.path.display())
    }
}

/// `path` as one shell word: as it is when it holds only characters no shell treats specially,
/// else quoted (single quotes on Unix, double quotes on Windows, whose paths cannot hold one).
/// `None` for a path that is not UTF-8, which no hint can spell.
fn shell_word(path: &Path) -> Option<String> {
    let text = path.to_str()?;
    let plain = |c: char| c.is_ascii_alphanumeric() || "_-./:+,@%=".contains(c);
    if !text.is_empty() && text.chars().all(plain) {
        return Some(text.to_owned());
    }
    Some(if cfg!(windows) {
        format!("\"{text}\"")
    } else {
        format!("'{}'", text.replace('\'', r"'\''"))
    })
}

/// `text` with [`CONTAINER_PLACEHOLDER`] replaced by `container` as a shell word, where there is
/// one and it can be spelled.
fn fill_container(text: &str, container: Option<&Path>) -> String {
    match container.and_then(shell_word) {
        Some(word) => text.replace(CONTAINER_PLACEHOLDER, &word),
        None => text.to_owned(),
    }
}

/// What to check for a failure of class `code`.
fn hint(code: ErrorCode) -> &'static str {
    match code {
        ErrorCode::Io => "check the path, its permissions and the free disk space",
        ErrorCode::NotZiptensors => "the file is not a ziptensors container",
        ErrorCode::UnsupportedVersion | ErrorCode::UnsupportedFeature | ErrorCode::UnknownCodec => {
            "the file needs a newer ziptensors"
        }
        // What a torn append leaves (a footer or index of the newest generation that does not
        // hold), so `recover` may bring back the generation before it.
        ErrorCode::MalformedIndex | ErrorCode::Zstd => {
            "the file is damaged or was not written by a correct writer; if a write to it was interrupted, \
             `ziptensors recover --dry-run <CONTAINER>` shows which earlier state it can go back to"
        }
        ErrorCode::ChunkCorrupt
        | ErrorCode::ChecksumMismatch
        | ErrorCode::OutOfBounds
        | ErrorCode::VerifyFailed => "the file is damaged or was not written by a correct writer",
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
        ErrorCode::DamagedInPlace => {
            "the file is damaged: restore it from a backup, or pack it again from the original files"
        }
        ErrorCode::EditRefused => {
            "nothing was written: change the request, for example choose another edit strategy"
        }
        ErrorCode::TrailingBytes => {
            "close every program that has the file open, then run `ziptensors recover` on it \
             (`--dry-run` first shows what it would cut)"
        }
        ErrorCode::Unrecoverable => "no valid container was found: restore the file from a backup",
        ErrorCode::TreeConflict => {
            "nothing was written: rename or remove the files named, so that every file gets a name of its own"
        }
        ErrorCode::DuplicateMember => {
            "two inputs would be stored under one member name: rename one of them, or leave one out"
        }
    }
}

/// The causes of `error`, outermost first, joined with `: `. A cause whose text is already in the message
/// is left out: `Error::Io` prints its OS error and also returns it as its source.
fn message(error: &anyhow::Error) -> String {
    let hint = error.downcast_ref::<Hint>().map(ToString::to_string);
    let mut text = String::new();
    for cause in error.chain() {
        let part = cause.to_string();
        if text.contains(&part) || hint.as_ref() == Some(&part) {
            continue;
        }
        if !text.is_empty() {
            text.push_str(": ");
        }
        text.push_str(&part);
    }
    text
}

/// Reports a failed command: `error: <what failed>: <why>` and, for a failure of the core or one with
/// a [`Hint`], a `hint:` line on stderr, plus an `ERROR` log with the numeric code and the error's
/// fields.
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
    let chosen = error.downcast_ref::<Hint>().map(|chosen| chosen.0);
    let container = error.downcast_ref::<ContainerContext>().map(|context| context.path.as_path());
    if let Some(text) = chosen.or_else(|| code.map(hint)) {
        let _ = writeln!(stderr, "hint: {}", fill_container(text, container));
    }
}

#[cfg(test)]
mod tests {
    use super::{ContainerContext, Hint, fill_container, hint, message};
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

    #[test]
    fn a_chosen_hint_is_found_under_more_context_and_left_out_of_the_message() {
        let failed: Result<(), Error> = Err(Error::Io {
            op: "open",
            path: PathBuf::from("model.ziptensors"),
            source: io::Error::other("read-only file system"),
        });
        let error = failed
            .context(Hint("use --dry-run"))
            .context("cannot recover model.ziptensors")
            .expect_err("failed");
        assert_eq!(error.downcast_ref::<Hint>().map(|chosen| chosen.0), Some("use --dry-run"));
        let text = message(&error);
        assert!(!text.contains("--dry-run"), "{text}");
        assert!(text.starts_with("cannot recover model.ziptensors: I/O error during open"), "{text}");
    }

    #[test]
    fn a_hint_names_the_container_the_user_gave_quoted_for_the_shell() {
        let text = hint(ziptensors_core::ErrorCode::MalformedIndex);
        assert!(text.contains("recover --dry-run <CONTAINER>"), "{text}");
        assert_eq!(fill_container("run x <CONTAINER>", None), "run x <CONTAINER>");
        assert_eq!(fill_container("run x <CONTAINER>", Some(PathBuf::from("m/a.zt").as_path())), "run x m/a.zt");
        let spaced = fill_container("run x <CONTAINER>", Some(PathBuf::from("my model's/a b.zt").as_path()));
        if cfg!(windows) {
            assert_eq!(spaced, "run x \"my model's/a b.zt\"");
        } else {
            assert_eq!(spaced, r"run x 'my model'\''s/a b.zt'");
        }

        let failed: Result<(), Error> =
            Err(Error::MalformedIndex { section: "footer", detail: "end magic is wrong".to_owned() });
        let error = failed
            .context(ContainerContext { what: "cannot open container", path: PathBuf::from("m.zt") })
            .context("cannot verify")
            .expect_err("failed");
        let found = error.downcast_ref::<ContainerContext>().map(|context| context.path.clone());
        assert_eq!(found, Some(PathBuf::from("m.zt")));
        assert!(message(&error).starts_with("cannot verify: cannot open container m.zt: malformed footer"));
    }

    #[test]
    fn every_edit_and_recovery_code_has_a_hint() {
        use ziptensors_core::ErrorCode;
        for code in [
            ErrorCode::DamagedInPlace,
            ErrorCode::EditRefused,
            ErrorCode::TrailingBytes,
            ErrorCode::Unrecoverable,
            ErrorCode::TreeConflict,
            ErrorCode::DuplicateMember,
        ] {
            assert!(!hint(code).is_empty(), "{code:?}");
        }
        assert!(hint(ErrorCode::DuplicateMember).contains("member name"));
        assert!(hint(ErrorCode::TrailingBytes).contains("recover"));
        // A torn or damaged index points to the recovery dry run; damaged data cannot be recovered.
        for code in [ErrorCode::MalformedIndex, ErrorCode::Zstd] {
            assert!(hint(code).contains("ziptensors recover --dry-run"), "{code:?}");
        }
        assert!(!hint(ErrorCode::ChunkCorrupt).contains("recover"));
    }
}
