//! File: names.rs
//! Purpose: the strings Python sees for the core's dtype and profile enums.
//!
//! Key functions: [`dtype_name`], [`profile_name`].
//!
//! Implementation notes: names are those of the registries in sections 5.3 and 5.4 of
//! `spec/ziptensors-format.md`; core owns the dtype names. Both registries are hints, so an id unknown to this build is a legal
//! value with a name of its own, never an error.

use ziptensors_core::{Dtype, Profile};

/// The registry name of `dtype`, from [`Dtype::name`], or `UNKNOWN(0x....)` with the wire id. Id 0
/// means undetermined.
pub(crate) fn dtype_name(dtype: Dtype) -> String {
    dtype.name().map_or_else(|| format!("UNKNOWN({:#06x})", dtype.to_wire()), str::to_owned)
}

/// The registry name of `profile`, or `unknown(n)` with the wire id.
pub(crate) fn profile_name(profile: Profile) -> String {
    let name = match profile {
        Profile::Opaque => "opaque",
        Profile::Safetensors => "safetensors",
        Profile::Gguf => "gguf",
        Profile::Onnx => "onnx",
        Profile::PtZip => "pt_zip",
        Profile::Unknown(id) => return format!("unknown({id})"),
    };
    name.to_owned()
}
