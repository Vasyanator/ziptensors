//! File: lib.rs
//! Purpose: crate root of the Python extension module `ziptensors` over `ziptensors-core`.
//!
//! Main responsibilities: the module list, the crate-wide lint policy, and the table of what the
//! Python module exports. See `MODULE_README.md` in this directory for the contract.
//!
//! Key structures: `ziptensors_module`, the `#[pymodule]` itself.
//!
//! Implementation notes: the library target keeps its default name, so it cannot collide with the
//! `ziptensors` binary. Only `#[pymodule(name)]` and the maturin `module-name` say `ziptensors`.

#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![deny(unsafe_op_in_unsafe_fn)]

mod buffer;
mod errors;
mod names;
mod reader;
mod report;
mod writer;

/// Lossless tensor containers with random access: reading, encoding and packing.
#[pyo3::pymodule(name = "ziptensors")]
mod ziptensors_module {
    #[pymodule_export]
    use crate::errors::{
        BudgetError, CorruptionError, FormatError, InputError, NotDurableError, NotFoundError,
        RangeError, UnsupportedError, ZiptensorsError, ZiptensorsIoError,
    };
    #[pymodule_export]
    use crate::reader::{Container, Member, MemberInfo, Tensor};
    #[pymodule_export]
    use crate::report::{
        ClassBytes, EncodeReport, LayerBytes, MemberReport, RestoreReport, VerifyReport,
    };
    #[pymodule_export]
    use crate::writer::{encode_file, pack};
}
