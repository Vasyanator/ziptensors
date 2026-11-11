//! File: options.rs
//! Purpose: the option structs of the C ABI and their translation to core options.
//!
//! Main responsibilities: [`ZtOpenOptions`], [`ZtEncodeOptions`], [`ZtEditOptions`],
//! [`ZtRecoverOptions`], the enums their fields hold, and the `struct_size` rule, for option structs
//! read and for structs the library fills in alike.
//!
//! Key structures: [`ZtOpenOptions`], [`ZtEncodeOptions`], [`ZtEditOptions`], [`ZtRecoverOptions`],
//! [`ZtMode`], [`ZtVerify`], [`ZtStrategy`].
//!
//! Key functions: [`open_options`], [`encode_options`], [`edit_options`], [`recover_options`],
//! [`check_out_size`].
//!
//! Implementation notes: enum fields are `i32` in the structs and validated here. C can store any
//! integer in such a field, and a Rust enum holding a value outside its variants is undefined behavior
//! the moment it is read; flag fields are `i32` too and only 0 and 1 are taken. All-zero fields mean
//! defaults, so `{ .struct_size = sizeof o }` is a complete options struct and C needs no constants for
//! the core defaults. That is why `max_total_decoded = 0` means "no budget" here, while the Rust, Python
//! and command line surfaces take 0 literally: released ABI 1 gave 0 that meaning, and a zero budget
//! refuses every non-empty container, so nothing useful is lost.

use ziptensors_core::{
    EditOptions, EncodeOptions, Mode, OpenOptions, RecoverOptions, Strategy, Verify,
};

use crate::status::Failure;

/// Writer effort, for `zt_encode_options.mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ZtMode {
    /// The default of the entry point: balanced for `zt_encode_file`, max for `zt_pack_new`.
    Default = 0,
    /// Plane split and per-stream entropy coding only; quantized data is stored.
    Fast = 1,
    /// Adds widening views and field split of quantized data.
    Balanced = 2,
    /// Adds the bit rotation trial per blob and blob deduplication.
    Max = 3,
}

impl ZtMode {
    /// The mode with this value.
    fn from_raw(raw: i32) -> Result<Self, Failure> {
        [Self::Default, Self::Fast, Self::Balanced, Self::Max]
            .into_iter()
            .find(|mode| *mode as i32 == raw)
            .ok_or_else(|| Failure::Invalid { detail: format!("unknown zt_mode value {raw}") })
    }
}

/// What a reader checks, for `zt_open_options.verify`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ZtVerify {
    /// Check the hash of every decoded chunk. The default.
    Chunk = 0,
    /// Skip chunk hashes. Exact decoded lengths are still enforced.
    Off = 1,
}

/// How `zt_edit` replaces the container, for `zt_edit_options.strategy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ZtStrategy {
    /// Writes the whole result to a temporary file beside the container, proves it and renames it over
    /// the container. Needs room for a second copy; the container is intact after any failure. Drops
    /// the dead data of earlier edits. The default.
    Rewrite = 0,
    /// Writes only the new data, a new index and a new footer after the end of the container. Needs
    /// room only for what is new; the old data stays in the file as dead bytes. A failure cuts the file
    /// back to its old length, or returns `ZT_STATUS_TRAILING_BYTES` if that cut fails (see
    /// `zt_recover`). Refused with `repack` and with no operation.
    Append = 1,
    /// RISKY. Rewrites the container inside its own file, with no temporary file and no extra space. A
    /// crash, power loss or failed write after the first write leaves a damaged container that may not
    /// be recoverable (`ZT_STATUS_DAMAGED_IN_PLACE`): keep a backup or the original inputs. Nothing may
    /// read the file meanwhile. Refused with `repack` and with a `delta_base` that is a kept member.
    InPlace = 2,
}

impl ZtStrategy {
    /// The core strategy with this value.
    fn from_raw(raw: i32) -> Result<Strategy, Failure> {
        let all = [Self::Rewrite, Self::Append, Self::InPlace];
        let found = all.into_iter().find(|strategy| *strategy as i32 == raw);
        match found {
            Some(Self::Rewrite) => Ok(Strategy::Rewrite),
            Some(Self::Append) => Ok(Strategy::Append),
            Some(Self::InPlace) => Ok(Strategy::InPlace),
            None => Err(Failure::Invalid { detail: format!("unknown zt_strategy value {raw}") }),
        }
    }
}

/// A flag field of an options struct: 0 is off, 1 is on, anything else is refused, since a value that
/// is neither would have to be guessed at.
fn flag(raw: i32, field: &'static str) -> Result<bool, Failure> {
    match raw {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(Failure::Invalid { detail: format!("`{field}` must be 0 or 1, not {other}") }),
    }
}

/// The core mode of a `zt_mode` value, with `defaults` for `ZT_MODE_DEFAULT`.
fn core_mode(raw: i32, default: Mode) -> Result<Mode, Failure> {
    Ok(match ZtMode::from_raw(raw)? {
        ZtMode::Default => default,
        ZtMode::Fast => Mode::Fast,
        ZtMode::Balanced => Mode::Balanced,
        ZtMode::Max => Mode::Max,
    })
}

impl ZtVerify {
    /// The core level with this value.
    fn from_raw(raw: i32) -> Result<Verify, Failure> {
        let found = [Self::Chunk, Self::Off].into_iter().find(|verify| *verify as i32 == raw);
        match found {
            Some(Self::Chunk) => Ok(Verify::Chunk),
            Some(Self::Off) => Ok(Verify::Off),
            None => Err(Failure::Invalid { detail: format!("unknown zt_verify value {raw}") }),
        }
    }
}

/// Options of `zt_open`. Set `struct_size` to `sizeof(zt_open_options)`. A field left 0 takes its
/// default.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ZtOpenOptions {
    /// `sizeof(zt_open_options)` of the header the caller was built with.
    pub struct_size: usize,
    /// Decode threads, and the size of the scratch pool. 0 means one per logical CPU, 1 means
    /// everything runs inline on the calling thread.
    pub threads: usize,
    /// Cap on the decompressed index in bytes. 0 means 256 MiB; values over 1 GiB act as 1 GiB.
    pub max_index_bytes: u64,
    /// Budget on the sum of member lengths in bytes; a container over it is refused at open.
    /// NOTE: 0 means NO budget here, unlike the Rust, Python and command line interfaces, where 0 is a
    /// budget of zero bytes. Set it for untrusted input.
    pub max_total_decoded: u64,
    /// Bound on the container's decode work as a multiple of the sum of member lengths, on top of one
    /// floor of 256 MiB.
    /// 0 means 16.
    pub max_decode_amplification: u64,
    /// One of `zt_verify`.
    pub verify: i32,
}

/// Options of `zt_encode_file` and `zt_pack_new`. Set `struct_size` to `sizeof(zt_encode_options)`. A
/// field left 0 takes its default.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ZtEncodeOptions {
    /// `sizeof(zt_encode_options)` of the header the caller was built with.
    pub struct_size: usize,
    /// Compression threads, also the threads of the final verification. 0 means one per logical CPU.
    /// The output bytes do not depend on this.
    pub threads: usize,
    /// One of `zt_mode`.
    pub mode: i32,
}

/// Options of `zt_edit`. Set `struct_size` to `sizeof(zt_edit_options)`. A field left 0 takes its
/// default: `ZT_STRATEGY_REWRITE`, no repack, `ZT_MODE_MAX` for new files, one thread per CPU, no
/// decode budget.
///
/// NOTE on `max_total_decoded`: 0 means NO budget, unlike the Rust, Python and command line
/// interfaces, where 0 is a budget of zero bytes.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ZtEditOptions {
    /// `sizeof(zt_edit_options)` of the header the caller was built with.
    pub struct_size: usize,
    /// One of `zt_strategy`.
    pub strategy: i32,
    /// 1 re-encodes every kept member with `mode`, deduplication and delta pairing redone over the
    /// whole result; 0 copies kept members as they are stored. Only with `ZT_STRATEGY_REWRITE`: the
    /// other strategies return `ZT_STATUS_EDIT_REFUSED`. Other values are refused.
    pub repack: i32,
    /// One of `zt_mode`: the mode of new files, and of every member under `repack`.
    /// `ZT_MODE_DEFAULT` means `ZT_MODE_MAX`.
    pub mode: i32,
    /// Compression, verification and decode threads. 0 means one per logical CPU.
    pub threads: usize,
    /// Budget on the sum of member lengths of the container being edited, in bytes; a container over
    /// it is refused before anything is written. 0 means NO budget (see the note above).
    pub max_total_decoded: u64,
}

/// Options of `zt_recover`. Set `struct_size` to `sizeof(zt_recover_options)`. A field left 0 takes
/// its default: write the cut, verify the generation found, at most 64 candidates, no bound on the
/// bytes scanned, no decode budget.
///
/// NOTE on `max_total_decoded`: 0 means NO budget, unlike the Rust, Python and command line
/// interfaces, where 0 is a budget of zero bytes.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ZtRecoverOptions {
    /// `sizeof(zt_recover_options)` of the header the caller was built with.
    pub struct_size: usize,
    /// 1 reports what would be done and writes nothing, opening the file read only; 0 cuts the file.
    /// Other values are refused.
    pub dry_run: i32,
    /// 0, the default, decodes every member of the generation found and checks its BLAKE3 before
    /// anything is cut, and passes over a generation that fails. 1 skips that and takes the latest
    /// generation that passes the structural checks: only for a file known never to have been edited
    /// in place. Other values are refused.
    pub skip_verify: i32,
    /// Most candidate footers given a full check. 0 means 64.
    pub max_candidates: u32,
    /// Most distinct bytes the backward scan reads; the overlap of its blocks counts once. 0 means no
    /// bound.
    pub max_scan_bytes: u64,
    /// Budget on the sum of member lengths of each candidate, in bytes. 0 means NO budget (see the note
    /// above).
    pub max_total_decoded: u64,
}

/// Borrows an options struct, or `None` for NULL, after checking its `struct_size`.
///
/// Only the exact size of this build is taken. A smaller struct has no bytes for the later fields, and
/// a larger one asks for options this library would silently ignore. A later ABI version may accept
/// the sizes of earlier ones.
///
/// # Safety
/// `raw` is NULL, or points at a struct that starts with its `usize` size, and stays valid and
/// unchanged for `'a`. `T` must have that first field.
unsafe fn sized<'a, T>(raw: *const T, argument: &'static str) -> Result<Option<&'a T>, Failure> {
    if raw.is_null() {
        return Ok(None);
    }
    // SAFETY: every struct of any ABI version starts with `struct_size`, by this function's contract.
    // Only this field is read before the size is known.
    let actual = unsafe { raw.cast::<usize>().read() };
    let expected = size_of::<T>();
    if actual != expected {
        return Err(Failure::StructSize { argument, expected, actual });
    }
    // SAFETY: the caller states the struct has the size of `T`, and validity is this function's
    // contract. Every field type of the option structs accepts any bit pattern.
    Ok(Some(unsafe { &*raw }))
}

/// Translates `zt_open_options`. NULL means all defaults.
///
/// # Safety
/// As [`sized`].
pub(crate) unsafe fn open_options(raw: *const ZtOpenOptions) -> Result<OpenOptions, Failure> {
    let mut options = OpenOptions::default();
    // SAFETY: passed on from this function's contract.
    let Some(raw) = (unsafe { sized(raw, "options") })? else {
        return Ok(options);
    };
    options.threads = raw.threads;
    options.verify = ZtVerify::from_raw(raw.verify)?;
    if raw.max_index_bytes != 0 {
        options.max_index_bytes = raw.max_index_bytes;
    }
    if raw.max_total_decoded != 0 {
        options.max_total_decoded = raw.max_total_decoded;
    }
    if raw.max_decode_amplification != 0 {
        options.max_decode_amplification = raw.max_decode_amplification;
    }
    Ok(options)
}

/// Translates `zt_encode_options`. NULL and `ZT_MODE_DEFAULT` take `defaults`.
///
/// # Safety
/// As [`sized`].
pub(crate) unsafe fn encode_options(
    raw: *const ZtEncodeOptions,
    defaults: EncodeOptions,
) -> Result<EncodeOptions, Failure> {
    // SAFETY: passed on from this function's contract.
    let Some(raw) = (unsafe { sized(raw, "options") })? else {
        return Ok(defaults);
    };
    let mode = core_mode(raw.mode, defaults.mode)?;
    Ok(EncodeOptions { mode, threads: raw.threads })
}

/// Translates `zt_edit_options`. NULL means all defaults.
///
/// # Safety
/// As [`sized`].
pub(crate) unsafe fn edit_options(raw: *const ZtEditOptions) -> Result<EditOptions, Failure> {
    let mut options = EditOptions::default();
    // SAFETY: passed on from this function's contract.
    let Some(raw) = (unsafe { sized(raw, "options") })? else {
        return Ok(options);
    };
    options.strategy = ZtStrategy::from_raw(raw.strategy)?;
    options.repack = flag(raw.repack, "repack")?;
    options.mode = core_mode(raw.mode, options.mode)?;
    options.threads = raw.threads;
    options.open.threads = raw.threads;
    if raw.max_total_decoded != 0 {
        options.open.max_total_decoded = raw.max_total_decoded;
    }
    Ok(options)
}

/// Translates `zt_recover_options`. NULL means all defaults.
///
/// # Safety
/// As [`sized`].
pub(crate) unsafe fn recover_options(
    raw: *const ZtRecoverOptions,
) -> Result<RecoverOptions, Failure> {
    let mut options = RecoverOptions::default();
    // SAFETY: passed on from this function's contract.
    let Some(raw) = (unsafe { sized(raw, "options") })? else {
        return Ok(options);
    };
    options.dry_run = flag(raw.dry_run, "dry_run")?;
    options.verify = !flag(raw.skip_verify, "skip_verify")?;
    if raw.max_candidates != 0 {
        options.budget.max_candidates = raw.max_candidates;
    }
    if raw.max_scan_bytes != 0 {
        options.budget.max_scan_bytes = raw.max_scan_bytes;
    }
    if raw.max_total_decoded != 0 {
        options.open.max_total_decoded = raw.max_total_decoded;
    }
    Ok(options)
}

/// Checks a struct the library fills in: `out` is not NULL, and its `struct_size`, which the caller
/// sets, is the size of this build. Only that first field is read; nothing is written.
///
/// # Safety
/// `out` is NULL, or points at a struct that starts with its `usize` size, readable for that field.
/// `T` must have that first field.
pub(crate) unsafe fn check_out_size<T>(out: *mut T, argument: &'static str) -> Result<(), Failure> {
    if out.is_null() {
        return Err(Failure::Null { argument });
    }
    // SAFETY: not NULL, and the first field is readable by this function's contract.
    let actual = unsafe { out.cast::<usize>().read() };
    let expected = size_of::<T>();
    if actual != expected {
        return Err(Failure::StructSize { argument, expected, actual });
    }
    Ok(())
}
