//! File: options.rs
//! Purpose: the option structs of the C ABI and their translation to core options.
//!
//! Main responsibilities: [`ZtOpenOptions`], [`ZtEncodeOptions`], the enums their fields hold, and the
//! `struct_size` rule.
//!
//! Key structures: [`ZtOpenOptions`], [`ZtEncodeOptions`], [`ZtMode`], [`ZtVerify`].
//!
//! Key functions: [`open_options`], [`encode_options`].
//!
//! Implementation notes: enum fields are `i32` in the structs and validated here. C can store any
//! integer in such a field, and a Rust enum holding a value outside its variants is undefined behavior
//! the moment it is read. All-zero fields mean defaults, so `{ .struct_size = sizeof o }` is a complete
//! options struct and C needs no constants for the core defaults.

use ziptensors_core::{EncodeOptions, Mode, OpenOptions, Verify};

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
    /// Budget on the sum of member lengths in bytes; a container over it is refused at open. 0 means no
    /// budget.
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
    let mode = match ZtMode::from_raw(raw.mode)? {
        ZtMode::Default => defaults.mode,
        ZtMode::Fast => Mode::Fast,
        ZtMode::Balanced => Mode::Balanced,
        ZtMode::Max => Mode::Max,
    };
    Ok(EncodeOptions { mode, threads: raw.threads })
}
