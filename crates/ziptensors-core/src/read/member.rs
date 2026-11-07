//! File: read/member.rs
//! Purpose: one member of a container: reads by offset and by tensor name.
//!
//! Main responsibilities: [`Member`] with `read_at`, `decode_tensor_into`, `read_many`, the
//! member's tensor table, and the chunk geometry of a member range (`chunk_aligned_end`,
//! `chunk_window`) for the streaming reads of `restore.rs` and the windows of `cache.rs`.
//!
//! Key structures: [`Member`], [`ReadRequest`].
//!
//! Implementation notes: this file checks a caller's arguments and turns them into segments;
//! `batch.rs` does the work. A tensor is a named range and nothing more: it goes through `read_at`.

use std::ops::Range;

use crate::error::Error;
use crate::format::{ExtentRecord, MemberRecord};
use crate::types::TensorInfo;

use super::batch::{BlobView, Segment, Segments, broken, read_segments, to_usize};
use super::container::{Container, Shared};

/// One range to read with [`Member::read_many`].
#[derive(Debug)]
pub struct ReadRequest<'a> {
    /// Start in the member, in bytes.
    pub offset: u64,
    /// Filled completely, or the call fails.
    pub dst: &'a mut [u8],
}

/// A handle on one member. Holds the container alive. Cloning is cheap, and all methods take `&self`
/// and may be called from any number of threads.
///
/// On `Err` from a read, the destination holds unspecified bytes. A read of a part of a chunk decodes
/// the whole chunk; nothing is cached between calls. [`super::ReadCache`] is the opt-in cache.
#[derive(Debug, Clone)]
pub struct Member {
    pub(super) container: Container,
    record: MemberRecord,
    extents: Range<usize>,
    tensors: Range<usize>,
}

impl Member {
    /// The handle on member `id`.
    pub(super) fn new(container: Container, id: u32) -> Result<Self, Error> {
        let index = &container.shared.index;
        let record = *index
            .members()
            .get(to_usize(u64::from(id))?)
            .ok_or_else(|| Error::MemberNotFound { name: format!("#{id}") })?;
        // Validation proved both ranges, and checking them here keeps the accessors below total.
        let first_extent = to_usize(record.first_extent)?;
        let extent_count = index.member_extents(&record)?.len();
        let first_tensor = to_usize(u64::from(record.first_tensor))?;
        let tensor_count = index.member_tensors(&record)?.len();
        let tensors = first_tensor..first_tensor + tensor_count;
        if tensors.end > container.shared.tensors.len() {
            return Err(broken("the tensor table is shorter than the index"));
        }
        let extents = first_extent..first_extent + extent_count;
        Ok(Self { container, record, extents, tensors })
    }

    fn shared(&self) -> &Shared {
        &self.container.shared
    }

    /// The record of this member. For `write::merge`.
    pub(crate) fn record(&self) -> &MemberRecord {
        &self.record
    }

    /// The extents of this member, in member order.
    fn extents(&self) -> &[ExtentRecord] {
        &self.shared().index.extents()[self.extents.clone()]
    }

    /// Length of the virtual file in bytes.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.record.file_len
    }

    /// Whether the virtual file has no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.record.file_len == 0
    }

    /// BLAKE3 of the whole content, as the file states it.
    #[must_use]
    pub fn blake3(&self) -> [u8; 32] {
        self.record.blake3
    }

    /// The name of this member.
    ///
    /// # Errors
    /// Returns [`Error::MalformedIndex`] if the name is broken, which validation rules out.
    pub fn name(&self) -> Result<&str, Error> {
        self.shared().index.string(self.record.name)
    }

    /// The tensor table of this member. Metadata: a tensor range never decides restored bytes.
    #[must_use]
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.shared().tensors[self.tensors.clone()]
    }

    /// Fills all of `dst` from byte `offset` of the member.
    ///
    /// # Errors
    /// - [`Error::OutOfBounds`] if the range does not lie in the member, before any I/O.
    /// - [`Error::ChunkCorrupt`], [`Error::ChecksumMismatch`] with blob and chunk ids,
    ///   [`Error::UnknownCodec`] for a stream codec, [`Error::Zstd`], [`Error::Io`].
    pub fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), Error> {
        self.check_range(offset, dst.len())?;
        read_segments(self.shared(), Segments::new(self.shared(), self.extents(), offset, dst))
    }

    /// Fills `dst` with the bytes of the tensor called `name`.
    ///
    /// # Errors
    /// [`Error::TensorNotFound`] if there is no such tensor, [`Error::DestinationSize`] unless
    /// `dst.len()` is the tensor's length, and any error of [`Member::read_at`].
    pub fn decode_tensor_into(&self, name: &str, dst: &mut [u8]) -> Result<(), Error> {
        let tensor = self
            .tensors()
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| Error::TensorNotFound { name: name.to_owned() })?;
        // Lossless: `usize` is at most 64 bits on every supported target.
        if dst.len() as u64 != tensor.len {
            return Err(Error::DestinationSize { expected: tensor.len, actual: dst.len() as u64 });
        }
        self.read_at(tensor.offset, dst)
    }

    /// Fills every request, visiting the file in order of pack offset so that a disk moves one way.
    /// Requests are cut along extents first, so a fragmented range is sorted piece by piece.
    /// `reqs` keeps its order. Costs one small record per extent touched, on top of [`Member::read_at`].
    ///
    /// # Errors
    /// As [`Member::read_at`]. Every range is checked before any I/O.
    pub fn read_many(&self, reqs: &mut [ReadRequest<'_>]) -> Result<(), Error> {
        for req in reqs.iter() {
            self.check_range(req.offset, req.dst.len())?;
        }
        let shared = self.shared();
        let mut keyed: Vec<(u64, Segment<'_, '_>)> = Vec::with_capacity(reqs.len());
        for req in reqs.iter_mut() {
            for segment in Segments::new(shared, self.extents(), req.offset, &mut *req.dst) {
                let segment = segment?;
                keyed.push((segment.first_file_offset()?, segment));
            }
        }
        keyed.sort_by_key(|(file_offset, _)| *file_offset);
        read_segments(shared, keyed.into_iter().map(|(_, segment)| Ok(segment)))
    }

    /// Checks a range against the length of the member.
    ///
    /// # Errors
    /// Returns [`Error::OutOfBounds`] unless `offset..offset + len` lies in the member.
    pub(super) fn check_range(&self, offset: u64, len: usize) -> Result<(), Error> {
        // Lossless: `usize` is at most 64 bits on every supported target.
        let len = len as u64;
        let file_len = self.record.file_len;
        if offset.checked_add(len).is_none_or(|end| end > file_len) {
            return Err(Error::OutOfBounds { offset, len, file_len });
        }
        Ok(())
    }

    /// A good end for a sequential read of the range `offset..end`: `end` moved back to a chunk
    /// boundary if that leaves a non-empty range, so the next read starts on a whole chunk and does
    /// not decode this chunk a second time. Never moves past the start of the extent holding `end`.
    pub(super) fn chunk_aligned_end(&self, offset: u64, end: u64) -> Result<u64, Error> {
        let extents = self.extents();
        let after = extents.partition_point(|extent| extent.virt_offset <= end);
        let Some(extent) = after.checked_sub(1).and_then(|at| extents.get(at)) else {
            return Ok(end);
        };
        let skip = end - extent.virt_offset;
        if skip >= extent.virt_len {
            return Ok(end);
        }
        let factor = extent.view()?.length_factor();
        let position = extent
            .blob_offset
            .checked_add(skip / factor)
            .ok_or_else(|| broken("an extent overflows its blob"))?;
        let chunk_size = BlobView::new(self.shared(), extent.blob_id)?.chunk_size();
        let within = position.checked_rem(chunk_size).unwrap_or(0);
        let back = within.saturating_mul(factor).saturating_add(skip % factor).min(skip);
        Ok(Some(end - back).filter(|aligned| *aligned > offset).unwrap_or(end))
    }

    /// The window holding byte `offset`: the member bytes that one chunk of one extent's blob serves,
    /// so the part of that chunk the extent shows, viewed. Windows partition the member, depend only on
    /// the index, and are at most `chunk_size` × view factor long. Under an odd `chunk_size` a widened
    /// element may straddle two windows; [`Member::read_at`] serves such a split like any other range.
    ///
    /// # Errors
    /// Returns [`Error::OutOfBounds`] if `offset` is not in the member, and [`Error::MalformedIndex`] or
    /// [`Error::UnknownCodec`] for a broken extent, which validation rules out.
    pub(super) fn chunk_window(&self, offset: u64) -> Result<Range<u64>, Error> {
        self.check_range(offset, 1)?;
        let extents = self.extents();
        let after = extents.partition_point(|extent| extent.virt_offset <= offset);
        let partition = || broken("extents do not partition the member");
        let extent = after.checked_sub(1).and_then(|at| extents.get(at)).ok_or_else(partition)?;
        let skip = offset.checked_sub(extent.virt_offset).filter(|skip| *skip < extent.virt_len);
        let skip = skip.ok_or_else(partition)?;
        let factor = extent.view()?.length_factor();
        let chunk_size = BlobView::new(self.shared(), extent.blob_id)?.chunk_size();
        let overflow = || broken("an extent overflows its blob");
        let position = extent.blob_offset.checked_add(skip / factor).ok_or_else(overflow)?;
        let within = position.checked_rem(chunk_size).ok_or_else(|| broken("a chunk size is 0"))?;
        let chunk_start = position - within;
        // Narrow bytes of the extent before the chunk, and up to its end. `position` lies in the
        // extent, so the chunk ends after the extent's start and the subtraction cannot underflow.
        let first = chunk_start.saturating_sub(extent.blob_offset);
        let last = chunk_start
            .checked_add(chunk_size)
            .and_then(|chunk_end| chunk_end.checked_sub(extent.blob_offset))
            .ok_or_else(overflow)?;
        let start = first.checked_mul(factor).and_then(|first| extent.virt_offset.checked_add(first));
        let len = last.saturating_mul(factor).min(extent.virt_len);
        let end = extent.virt_offset.checked_add(len);
        Ok(start.ok_or_else(overflow)?..end.ok_or_else(overflow)?)
    }
}
