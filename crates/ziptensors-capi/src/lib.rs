//! File: lib.rs
//! Purpose: crate root of the C ABI over `ziptensors-core`. See `MODULE_README.md` in this
//! directory for the contract.
//!
//! Main responsibilities: the module list, the crate-wide lint policy and the ABI version.
//!
//! Key structures: [`ZtStatus`], [`ZtContainer`], [`ZtMember`], [`ZtPack`].
//!
//! Key functions: [`zt_abi_version`]; the other entry points live in `read.rs`, `write.rs` and
//! `barrier.rs`.
//!
//! Implementation notes: `include/ziptensors.h` is generated from the declarations of this crate,
//! doc comments included, so these comments are written for a C reader.

#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(unsafe_code, reason = "an FFI edge: every entry point takes raw pointers from C")]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod args;
mod barrier;
mod dtype;
mod options;
mod read;
mod status;
#[cfg(test)]
mod tests;
mod write;

pub use barrier::zt_last_error;
pub use dtype::*;
pub use options::{ZtEncodeOptions, ZtMode, ZtOpenOptions, ZtVerify};
pub use read::{
    ZtContainer, ZtMember, ZtTensorDesc, zt_close, zt_decode_tensor_into, zt_member_close,
    zt_member_count, zt_member_len, zt_member_name, zt_member_open, zt_open, zt_read_at,
    zt_restore_file, zt_tensor_count, zt_tensor_info,
};
pub use status::ZtStatus;
pub use write::{ZtPack, zt_encode_file, zt_pack_add_file, zt_pack_finish, zt_pack_free, zt_pack_new};

/// The version of this ABI. Raised when a declaration of this header changes in a way that breaks
/// callers built against an older one.
pub const ZT_ABI_VERSION: u32 = 1;

/// Returns the ABI version of the loaded library. Compare it with `ZT_ABI_VERSION` of the header
/// the caller was built with. Returns 0 only if the call itself panicked.
#[must_use]
#[unsafe(no_mangle)]
pub extern "C" fn zt_abi_version() -> u32 {
    barrier::guarded_value("zt_abi_version", 0, || ZT_ABI_VERSION)
}
