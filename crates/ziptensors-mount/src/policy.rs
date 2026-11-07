//! File: policy.rs
//! Purpose: which member names a filesystem can show, per operating system.
//!
//! Main responsibilities:
//! - [`NamePolicy`]: the checks a backend's filesystem puts on one path part and on a whole path, and
//!   the key under which two parts are the same name;
//! - [`PosixPolicy`] for Linux and macOS, [`WindowsPolicy`] for Windows;
//! - [`NameProblem`], why a name was refused.
//!
//! Key structures: [`NamePolicy`], [`PosixPolicy`], [`WindowsPolicy`], [`NameProblem`].
//!
//! Implementation notes: pure validation, no OS call, so both policies build and are tested on every
//! target. A member name is `/` separated (spec), and every policy runs the core's
//! `format::check_restore_name` on the whole name first, so nothing a restore refuses can be mounted.

use std::borrow::Cow;
use std::fmt;

use ziptensors_core::format::check_restore_name;

/// Longest path part of POSIX filesystems (`NAME_MAX`), in bytes.
const POSIX_NAME_MAX_BYTES: usize = 255;

/// Longest path part of NTFS and of `WinFsp`, in UTF-16 code units.
const WINDOWS_NAME_MAX_UNITS: usize = 255;

/// Longest member name under classic Windows paths: `MAX_PATH` is 260 units including the drive root
/// `X:\` and the terminating NUL, so 256 units remain for the path under the root.
const WINDOWS_CLASSIC_PATH_MAX_UNITS: usize = 256;

/// Longest member name with long paths enabled: 32767 units, less the `X:\` root and the NUL.
const WINDOWS_LONG_PATH_MAX_UNITS: usize = 32_763;

/// Device names Windows resolves in every directory, whatever the extension after them. Compared
/// without case. The list of Microsoft's "Naming Files, Paths, and Namespaces": `COM0` and `LPT0` and
/// the superscript digits included, although current kernels open `COM0` as a file. Refusing costs
/// nothing; a name some Windows API refuses would.
const WINDOWS_RESERVED: &[&str] = &[
    "CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$", "COM0", "COM1", "COM2", "COM3", "COM4", "COM5",
    "COM6", "COM7", "COM8", "COM9", "COM\u{b9}", "COM\u{b2}", "COM\u{b3}", "LPT0", "LPT1", "LPT2",
    "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9", "LPT\u{b9}", "LPT\u{b2}", "LPT\u{b3}",
];

/// Characters no Windows file name may hold, besides those below U+0020.
const WINDOWS_FORBIDDEN: &[char] = &['<', '>', ':', '"', '|', '?', '*', '\\'];

/// Why a name cannot be shown by a filesystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NameProblem {
    /// `format::check_restore_name` refuses the whole name: empty, NUL, `:`, absolute, or a `..` part.
    Unsafe,
    /// A part is empty: two `/` in a row, or a trailing `/`.
    EmptyPart,
    /// A part is `.` or `..`.
    DotPart,
    /// A part is longer than the filesystem allows, in its own unit: bytes for POSIX, UTF-16 units for
    /// Windows.
    PartTooLong {
        /// Length of the part.
        len: usize,
        /// Longest allowed.
        limit: usize,
    },
    /// The whole name is longer than the filesystem allows, in UTF-16 units.
    PathTooLong {
        /// Length of the name.
        len: usize,
        /// Longest allowed.
        limit: usize,
    },
    /// A part holds a character the filesystem refuses.
    ForbiddenChar(char),
    /// A part is a device name such as `CON` or `com1.txt`.
    Reserved,
    /// A part ends with `.` or a space, which Windows strips.
    TrailingDotOrSpace,
    /// Two members map to the same path under the policy's notion of equal names, here the other one.
    Collision {
        /// The member already at that path.
        with: String,
    },
    /// The name is a file and also the directory of other members, here one of them.
    FileIsDirectory {
        /// A member under the directory of the same name.
        under: String,
    },
}

impl fmt::Display for NameProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsafe => f.write_str("it could escape the mount or cannot be restored"),
            Self::EmptyPart => f.write_str("it has an empty path part"),
            Self::DotPart => f.write_str("it has a `.` or `..` path part"),
            Self::PartTooLong { len, limit } => {
                write!(f, "a path part is {len} long, over the limit of {limit}")
            }
            Self::PathTooLong { len, limit } => {
                write!(f, "it is {len} UTF-16 units long, over the limit of {limit}")
            }
            Self::ForbiddenChar(c) => write!(f, "it holds the character {c:?}"),
            Self::Reserved => f.write_str("a path part is a reserved device name"),
            Self::TrailingDotOrSpace => f.write_str("a path part ends with a dot or a space"),
            Self::Collision { with } => write!(f, "it is the same path as member {with:?}"),
            Self::FileIsDirectory { under } => {
                write!(f, "it is a file and also the directory of member {under:?}")
            }
        }
    }
}

/// The names a filesystem can show, and when two names are the same.
///
/// Implementations are stateless validators: no I/O, no allocation unless a key differs from the part.
pub trait NamePolicy: fmt::Debug + Send + Sync {
    /// Short name of the policy, for logs.
    fn label(&self) -> &'static str;

    /// Checks one part of a member name: the text between two `/`.
    ///
    /// # Errors
    /// The first [`NameProblem`] found in `part`.
    fn check_part(&self, part: &str) -> Result<(), NameProblem>;

    /// Checks a whole member name. The default checks only what every policy shares:
    /// `format::check_restore_name`.
    ///
    /// # Errors
    /// [`NameProblem::Unsafe`], or a length problem of the implementation.
    fn check_path(&self, path: &str) -> Result<(), NameProblem> {
        check_restore_name(path).map_err(|_| NameProblem::Unsafe)
    }

    /// The key under which two parts of one directory are the same entry. POSIX: the part itself.
    fn key<'a>(&self, part: &'a str) -> Cow<'a, str>;
}

/// Linux and macOS: any byte but `/` and NUL, parts up to 255 bytes, case-sensitive.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PosixPolicy;

impl NamePolicy for PosixPolicy {
    fn label(&self) -> &'static str {
        "posix"
    }

    fn check_part(&self, part: &str) -> Result<(), NameProblem> {
        common_part(part)?;
        if part.len() > POSIX_NAME_MAX_BYTES {
            return Err(NameProblem::PartTooLong { len: part.len(), limit: POSIX_NAME_MAX_BYTES });
        }
        Ok(())
    }

    fn key<'a>(&self, part: &'a str) -> Cow<'a, str> {
        Cow::Borrowed(part)
    }
}

/// Windows: the rules of `CreateFileW` on NTFS. Case-insensitive, so the key folds case.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WindowsPolicy {
    /// Whether the system accepts paths over `MAX_PATH` (the `LongPathsEnabled` setting and an aware
    /// program). Off by default: most programs that load models are not long-path aware.
    pub long_paths: bool,
}

impl WindowsPolicy {
    /// Longest member name in UTF-16 units, see [`WindowsPolicy::long_paths`].
    #[must_use]
    pub fn path_limit(&self) -> usize {
        if self.long_paths { WINDOWS_LONG_PATH_MAX_UNITS } else { WINDOWS_CLASSIC_PATH_MAX_UNITS }
    }
}

impl NamePolicy for WindowsPolicy {
    fn label(&self) -> &'static str {
        "windows"
    }

    fn check_part(&self, part: &str) -> Result<(), NameProblem> {
        common_part(part)?;
        if let Some(c) = part.chars().find(|c| *c < ' ' || WINDOWS_FORBIDDEN.contains(c)) {
            return Err(NameProblem::ForbiddenChar(c));
        }
        if part.ends_with(['.', ' ']) {
            return Err(NameProblem::TrailingDotOrSpace);
        }
        let units = part.encode_utf16().count();
        if units > WINDOWS_NAME_MAX_UNITS {
            return Err(NameProblem::PartTooLong { len: units, limit: WINDOWS_NAME_MAX_UNITS });
        }
        // `nul.txt` and `NUL .bin` open the device: what counts is the text before the first dot, with
        // trailing spaces dropped.
        let stem = part.split('.').next().unwrap_or(part).trim_end_matches(' ');
        if WINDOWS_RESERVED.iter().any(|reserved| reserved.eq_ignore_ascii_case(stem)) {
            return Err(NameProblem::Reserved);
        }
        Ok(())
    }

    fn check_path(&self, path: &str) -> Result<(), NameProblem> {
        check_restore_name(path).map_err(|_| NameProblem::Unsafe)?;
        let units = path.encode_utf16().count();
        let limit = self.path_limit();
        if units > limit {
            return Err(NameProblem::PathTooLong { len: units, limit });
        }
        Ok(())
    }

    /// Simple upper-case mapping per character, from the Unicode tables of this Rust. NTFS compares
    /// through the frozen `$UpCase` table of the volume, also one to one; a character whose upper case
    /// is several characters (`ß`) is kept as it is, as NTFS does. The Unicode tables fold more than
    /// `$UpCase` (`ı` and `I`, `ſ` and `S`, Georgian Mtavruli, Cherokee): such a pair is a collision
    /// here and refuses the mount, where Windows would show two files. No case is known where this
    /// folds less, which would serve two members as one.
    fn key<'a>(&self, part: &'a str) -> Cow<'a, str> {
        let folded: String = part.chars().map(simple_upper).collect();
        if folded == part { Cow::Borrowed(part) } else { Cow::Owned(folded) }
    }
}

/// What every policy refuses in a part.
fn common_part(part: &str) -> Result<(), NameProblem> {
    if part.is_empty() {
        return Err(NameProblem::EmptyPart);
    }
    if part == "." || part == ".." {
        return Err(NameProblem::DotPart);
    }
    if part.contains('\0') {
        return Err(NameProblem::ForbiddenChar('\0'));
    }
    Ok(())
}

/// `c` in upper case when that is one character, else `c`.
fn simple_upper(c: char) -> char {
    let mut upper = c.to_uppercase();
    match (upper.next(), upper.next()) {
        (Some(single), None) => single,
        _ => c,
    }
}

#[cfg(test)]
mod tests {
    use super::{NamePolicy, NameProblem, PosixPolicy, WindowsPolicy};

    #[test]
    fn posix_refuses_empty_dot_and_long_parts() {
        let policy = PosixPolicy;
        assert_eq!(policy.check_part(""), Err(NameProblem::EmptyPart));
        assert_eq!(policy.check_part("."), Err(NameProblem::DotPart));
        assert_eq!(policy.check_part(".."), Err(NameProblem::DotPart));
        assert_eq!(policy.check_part("a\0b"), Err(NameProblem::ForbiddenChar('\0')));
        assert_eq!(
            policy.check_part(&"x".repeat(256)),
            Err(NameProblem::PartTooLong { len: 256, limit: 255 })
        );
        // The limit is in bytes: 128 two-byte characters are 256 bytes.
        assert!(matches!(policy.check_part(&"é".repeat(128)), Err(NameProblem::PartTooLong { .. })));
        assert_eq!(policy.check_part(&"x".repeat(255)), Ok(()));
    }

    #[test]
    fn posix_accepts_what_windows_refuses() {
        let policy = PosixPolicy;
        for part in ["CON", "a:b", "a\\b", "trailing.", "trailing ", "q?", "Model.BIN", ".hidden"] {
            assert_eq!(policy.check_part(part), Ok(()), "{part}");
        }
        assert_eq!(policy.key("Model.BIN"), "Model.BIN");
    }

    #[test]
    fn every_policy_refuses_what_a_restore_refuses() {
        for name in ["", "/abs", "\\abs", "a/../b", "..", "a:b", "a\0b"] {
            assert_eq!(PosixPolicy.check_path(name), Err(NameProblem::Unsafe), "{name:?}");
            assert_eq!(WindowsPolicy::default().check_path(name), Err(NameProblem::Unsafe), "{name:?}");
        }
        assert_eq!(PosixPolicy.check_path("a/b/c.bin"), Ok(()));
    }

    #[test]
    fn windows_refuses_forbidden_characters() {
        let policy = WindowsPolicy::default();
        for c in ['<', '>', '"', '|', '?', '*', '\\', '\u{1}', '\u{1f}', '\t'] {
            let part = format!("a{c}b");
            assert_eq!(policy.check_part(&part), Err(NameProblem::ForbiddenChar(c)), "{part:?}");
        }
        assert_eq!(policy.check_part("a:b"), Err(NameProblem::ForbiddenChar(':')));
    }

    #[test]
    fn windows_refuses_reserved_device_names_with_any_extension_and_case() {
        let policy = WindowsPolicy::default();
        for part in [
            "CON", "con", "Nul.txt", "aux.tar.gz", "COM1", "lpt9.bin", "PRN .x", "CONIN$", "COM\u{b9}", "COM0",
            "lpt0.bin",
        ] {
            assert_eq!(policy.check_part(part), Err(NameProblem::Reserved), "{part}");
        }
        for part in ["CONSOLE", "COM10", "LPT00", "xcon", "con_x.bin", "COM"] {
            assert_eq!(policy.check_part(part), Ok(()), "{part}");
        }
    }

    #[test]
    fn windows_refuses_a_trailing_dot_or_space() {
        let policy = WindowsPolicy::default();
        assert_eq!(policy.check_part("a."), Err(NameProblem::TrailingDotOrSpace));
        assert_eq!(policy.check_part("a "), Err(NameProblem::TrailingDotOrSpace));
        assert_eq!(policy.check_part(".a"), Ok(()));
        assert_eq!(policy.check_part(" a"), Ok(()));
    }

    #[test]
    fn windows_limits_are_in_utf16_units() {
        let policy = WindowsPolicy::default();
        // 255 characters outside the BMP are 510 units.
        assert_eq!(
            policy.check_part(&"\u{1f600}".repeat(128)),
            Err(NameProblem::PartTooLong { len: 256, limit: 255 })
        );
        // 255 two-byte characters are 510 bytes but 255 units.
        assert_eq!(policy.check_part(&"é".repeat(255)), Ok(()));
        let long = format!("{}/{}", "a".repeat(199), "b".repeat(56));
        assert_eq!(policy.check_path(&long), Ok(()));
        let longer = format!("{long}c");
        assert_eq!(policy.check_path(&longer), Err(NameProblem::PathTooLong { len: 257, limit: 256 }));
        assert_eq!(WindowsPolicy { long_paths: true }.check_path(&longer), Ok(()));
    }

    #[test]
    fn windows_keys_fold_case_one_to_one() {
        let policy = WindowsPolicy::default();
        assert_eq!(policy.key("Model.BIN"), policy.key("model.bin"));
        assert_eq!(policy.key("ÄÖ"), policy.key("äö"));
        assert_eq!(policy.key("straße"), "STRAßE");
        assert_ne!(policy.key("straße"), policy.key("STRASSE"));
    }

    #[test]
    fn windows_keys_fold_more_than_ntfs_never_less() {
        // `$UpCase` keeps dotless i and long s; folding them anyway refuses a mount, never merges files.
        let policy = WindowsPolicy::default();
        assert_eq!(policy.key("\u{131}.bin"), policy.key("I.bin"));
        assert_eq!(policy.key("\u{17f}.bin"), policy.key("s.bin"));
    }
}
