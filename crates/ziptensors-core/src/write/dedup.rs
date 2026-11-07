//! File: write/dedup.rs
//! Purpose: the pack's own record of its blobs: identity for deduplication, spec section 8, and the
//! shape a delta blob needs of its base, spec section 7a.
//!
//! Key structures: [`BlobTable`], [`BlobShape`].
//!
//! Implementation notes: identity is BLAKE3 of canonical bytes, after narrowing and before `rotl1` and
//! the layout, so the chain of a blob is no part of it, and a view on an extent is what tells users of
//! one blob apart. Identity covers whole blobs: there is no chunk or sub-tensor identity. The shapes
//! are kept here because `IndexBuilder` gives no records back: a writer that wants to rest a delta
//! blob on an existing one must know that one's kind, `chunk_size` and length. One identity entry
//! (36 bytes of payload) and one shape (16 bytes) per blob, so a few MiB for the 45k blobs of the
//! acceptance set.

use std::collections::HashMap;

use crate::chain::BLOB_KIND_DELTA;
use crate::error::Error;

/// A `BLOB.blake3` value. All zero means "not computed", and such a blob is never matched.
pub(super) type Identity = [u8; 32];

/// What the writer must know of an existing blob to use it as a delta base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BlobShape {
    /// `BLOB.kind`.
    pub(super) kind: u8,
    /// `BLOB.chunk_size`.
    pub(super) chunk_size: u32,
    /// `BLOB.raw_len`.
    pub(super) raw_len: u64,
}

/// Every blob of this pack, by id, and the blobs with an identity, by identity.
#[derive(Debug, Default)]
pub(super) struct BlobTable {
    by_identity: HashMap<Identity, u32>,
    /// Indexed by blob id.
    shapes: Vec<BlobShape>,
    /// Blobs of kind 1, which decide required feature bit 0.
    delta_count: u32,
}

impl BlobTable {
    /// The blob with this identity, if there is one. The all-zero value matches nothing.
    #[must_use]
    pub(super) fn find(&self, identity: &Identity) -> Option<u32> {
        if *identity == [0; 32] {
            return None;
        }
        self.by_identity.get(identity).copied()
    }

    /// The shape of blob `id`, if the pack has it.
    #[must_use]
    pub(super) fn shape(&self, id: u32) -> Option<BlobShape> {
        self.shapes.get(usize::try_from(id).ok()?).copied()
    }

    /// Number of blobs of kind 1 recorded. Required feature bit 0 is set exactly when it is not 0.
    #[must_use]
    pub(super) fn delta_count(&self) -> u32 {
        self.delta_count
    }

    /// Records blob `id`, which must be the next id: blobs are recorded in id order, as they get
    /// their ids. `identity` is entered unless it is the all-zero "not computed"; the first blob with
    /// an identity stays its owner.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] for `BLOB` if `id` is not the next id, which would mean the
    /// writer gave out ids out of order.
    pub(super) fn record(
        &mut self,
        id: u32,
        identity: Identity,
        shape: BlobShape,
    ) -> Result<(), Error> {
        if usize::try_from(id).ok() != Some(self.shapes.len()) {
            return Err(Error::MalformedIndex {
                section: "BLOB",
                detail: format!("blob {id} recorded as blob {}", self.shapes.len()),
            });
        }
        if identity != [0; 32] {
            self.by_identity.entry(identity).or_insert(id);
        }
        if shape.kind == BLOB_KIND_DELTA {
            // Only ever compared with 0, so saturating at 2^32 - 1 blobs changes nothing.
            self.delta_count = self.delta_count.saturating_add(1);
        }
        self.shapes.push(shape);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{BlobShape, BlobTable};
    use crate::chain::{BLOB_KIND_CHUNKED, BLOB_KIND_DELTA};

    const PLAIN: BlobShape = BlobShape { kind: BLOB_KIND_CHUNKED, chunk_size: 4, raw_len: 8 };

    #[test]
    fn a_recorded_identity_is_found() {
        let mut table = BlobTable::default();
        assert_eq!(table.find(&[1; 32]), None);
        table.record(0, [1; 32], PLAIN).expect("next id");
        assert_eq!(table.find(&[1; 32]), Some(0));
        assert_eq!(table.find(&[2; 32]), None);
        assert_eq!(table.shape(0), Some(PLAIN));
        assert_eq!(table.shape(1), None);
    }

    #[test]
    fn the_first_owner_stays() {
        let mut table = BlobTable::default();
        table.record(0, [1; 32], PLAIN).expect("next id");
        table.record(1, [1; 32], PLAIN).expect("next id");
        assert_eq!(table.find(&[1; 32]), Some(0));
    }

    #[test]
    fn all_zero_means_not_computed_and_never_matches() {
        let mut table = BlobTable::default();
        table.record(0, [0; 32], PLAIN).expect("next id");
        assert_eq!(table.find(&[0; 32]), None);
        assert_eq!(table.shape(0), Some(PLAIN));
    }

    #[test]
    fn ids_out_of_order_are_refused_and_delta_blobs_counted() {
        let mut table = BlobTable::default();
        assert!(table.record(1, [1; 32], PLAIN).is_err());
        table.record(0, [1; 32], PLAIN).expect("next id");
        assert_eq!(table.delta_count(), 0);
        let delta = BlobShape { kind: BLOB_KIND_DELTA, ..PLAIN };
        table.record(1, [2; 32], delta).expect("next id");
        assert_eq!(table.delta_count(), 1);
        assert!(table.record(1, [3; 32], PLAIN).is_err());
    }
}
