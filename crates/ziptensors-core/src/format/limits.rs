//! File: format/limits.rs
//! Purpose: the caller-set bounds that parsing and validation enforce against hostile input.
//!
//! Key structures: [`Limits`].
//!
//! Implementation notes: the 1 GiB ceiling on the index is a rule of the format, so
//! [`Limits::index_cap`] applies it whatever the caller configured.

/// Default reader cap on `index_raw_len`: 256 MiB (spec section 3.2).
pub const DEFAULT_MAX_INDEX_BYTES: u64 = 256 << 20;
/// Largest `index_raw_len` any file may have, whatever the configuration: 1 GiB (spec section 3.2).
pub const MAX_INDEX_RAW_LEN: u64 = 1 << 30;
/// Default of [`Limits::max_name_work_bytes`]: 256 MiB, about fifty times the names of a pack of 45k
/// tensors, and the size of the default index cap, since open keeps a copy of every name.
pub const DEFAULT_MAX_NAME_WORK_BYTES: u64 = 256 << 20;
/// Default of [`Limits::max_decode_amplification`].
pub const DEFAULT_MAX_DECODE_AMPLIFICATION: u64 = 16;
/// Decode work a whole container may spend on top of its amplification allowance: 256 MiB, once, not
/// per member. It keeps the rule out of the way of small files, for which any layout is cheap.
pub const DECODE_WORK_FLOOR: u64 = 256 << 20;

/// Bounds on what a file may make a reader do. `Default` gives the spec's 256 MiB index cap and no
/// decode budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Cap on `index_raw_len` in bytes. Values over [`MAX_INDEX_RAW_LEN`] act as that ceiling.
    pub max_index_bytes: u64,
    /// Caller budget on the sum of `file_len` over the members to be decoded, in bytes.
    pub max_total_decoded: u64,
    /// Bound on the sum of `name_len` over all `MEMB` and `TENS` records, in bytes.
    ///
    /// Not a format rule but a reader defence. Many records may name overlapping ranges of one
    /// large `STRS` heap, so checking names costs records × heap unless the sum is bounded. It bounds
    /// memory too: a reader's tensor table copies every name, so this many bytes are held at open.
    pub max_name_work_bytes: u64,
    /// Bound on the decode work of a container, as a multiple of the sum of its member lengths;
    /// [`DECODE_WORK_FLOOR`] comes on top, once. Decode work sums, over the extents of all members, the
    /// raw bytes of every chunk the extent touches.
    ///
    /// A reader defence, spec section 4. A partial chunk decodes whole and nothing is cached, so
    /// many small extents over one chunk cost extents × chunk while the members stay small. Per
    /// container, because a floor per member would multiply by the member count.
    pub max_decode_amplification: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_index_bytes: DEFAULT_MAX_INDEX_BYTES,
            max_total_decoded: u64::MAX,
            max_name_work_bytes: DEFAULT_MAX_NAME_WORK_BYTES,
            max_decode_amplification: DEFAULT_MAX_DECODE_AMPLIFICATION,
        }
    }
}

impl Limits {
    /// The cap in force on `index_raw_len`: the configured value, never above 1 GiB.
    #[must_use]
    pub fn index_cap(&self) -> u64 {
        self.max_index_bytes.min(MAX_INDEX_RAW_LEN)
    }
}

#[cfg(test)]
mod tests {
    use super::{Limits, MAX_INDEX_RAW_LEN};

    #[test]
    fn default_cap_is_256_mib() {
        assert_eq!(Limits::default().index_cap(), 268_435_456);
        assert_eq!(MAX_INDEX_RAW_LEN, 1_073_741_824);
    }

    #[test]
    fn configuration_cannot_raise_the_cap_above_one_gib() {
        let limits = Limits { max_index_bytes: u64::MAX, ..Limits::default() };
        assert_eq!(limits.index_cap(), MAX_INDEX_RAW_LEN);
        let limits = Limits { max_index_bytes: 1000, ..Limits::default() };
        assert_eq!(limits.index_cap(), 1000);
    }
}
