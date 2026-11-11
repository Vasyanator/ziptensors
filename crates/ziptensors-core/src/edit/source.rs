//! File: edit/source.rs
//! Purpose: a member of an open container as a [`RandomAccess`] input of the writer.
//!
//! Key structures: [`MemberSource`].
//!
//! Implementation notes: reads go through [`Member::read_at`], which decodes every chunk a read
//! touches and checks it as the container's `OpenOptions::verify` says. The writer reads its inputs
//! sequentially in pieces of at most a chunk, so a repack decodes each stored chunk at most twice. A
//! container's bytes are fixed while it is open, which is what `SourceStamp::Member` relies on.

use crate::error::Error;
use crate::io::RandomAccess;
use crate::read::Member;

/// The decoded bytes of one container member. Its length is the member's length.
#[derive(Debug, Clone)]
pub(super) struct MemberSource {
    member: Member,
}

impl MemberSource {
    /// A source over `member`. Holds its container open.
    pub(super) fn new(member: Member) -> Self {
        Self { member }
    }
}

impl RandomAccess for MemberSource {
    fn len(&self) -> u64 {
        self.member.len()
    }

    fn read_exact_at(&self, off: u64, buf: &mut [u8]) -> Result<(), Error> {
        self.member.read_at(off, buf)
    }
}
