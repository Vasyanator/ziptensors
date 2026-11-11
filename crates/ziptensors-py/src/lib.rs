//! File: lib.rs
//! Purpose: crate root of the Python extension module `ziptensors` over `ziptensors-core`.
//!
//! Main responsibilities: the module list, the crate-wide lint policy, and the table of what the
//! Python module exports. See `MODULE_README.md` in this directory for the contract.
//!
//! Key structures: `ziptensors_module`, the `#[pymodule]` itself, and its init, which sets
//! `__version__` from the crate version.
//!
//! Implementation notes: the library target keeps its default name, so it cannot collide with the
//! `ziptensors` binary. Only `#[pymodule(name)]` and the maturin `module-name` say `ziptensors`.

#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![deny(unsafe_op_in_unsafe_fn)]

mod buffer;
mod edit;
mod errors;
mod logging;
mod mount;
mod names;
mod reader;
mod report;
mod tree;
mod writer;

/// Lossless tensor containers with random access: reading, encoding, packing, editing, recovery,
/// per-file tree compression and read-only mounts (Linux and Windows).
#[pyo3::pymodule(name = "ziptensors")]
mod ziptensors_module {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::edit::{AddOp, RemoveOp, ReplaceOp, compact, edit, recover};
    #[pymodule_export]
    use crate::errors::{
        BudgetError, CorruptionError, FormatError, InputError, NotDurableError, NotFoundError,
        RangeError, UnsupportedError, ZiptensorsError, ZiptensorsIoError,
    };
    #[pymodule_export]
    use crate::logging::enable_stderr_logging;
    #[pymodule_export]
    use crate::mount::{Mount, MountError, mount, mount_overlay};
    #[pymodule_export]
    use crate::reader::{Container, Member, MemberInfo, Tensor};
    #[pymodule_export]
    use crate::report::{
        ClassBytes, EditReport, EncodeReport, Inputs, LayerBytes, MemberReport, RecoverReport,
        RestoreAllReport, RestoreReport, VerifyReport,
    };
    #[pymodule_export]
    use crate::tree::{FileOutcome, TreeReport, TreeTotals, compress_tree, decompress_tree};
    #[pymodule_export]
    use crate::writer::{PackBuilder, collect_inputs, encode_file, pack};

    /// Sets `__version__`, the version of the crate this module was built from.
    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add("__version__", env!("CARGO_PKG_VERSION"))
    }
}
