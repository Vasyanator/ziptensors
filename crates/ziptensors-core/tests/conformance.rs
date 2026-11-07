//! File: tests/conformance.rs
//! Purpose: the accept and reject list of format version 0.3, for a third party to read: small files
//! written byte by byte from `spec/ziptensors-format.md`, each with the behaviour a reader must show.
//!
//! Main responsibilities:
//! - [`accept_cases`]: valid files, each with the exact member bytes it must give, derived by hand;
//!   delta blobs of version 0.3 (spec section 7a) in [`accept_delta`];
//! - [`reject_cases`]: invalid files, each citing the spec rule it breaks and where it must fail;
//!   invariant 13 in [`reject_delta`];
//! - [`restore_name_cases`]: invariant 11, which is about strings and not files.
//!
//! Key structures: `File` (a whole file as plain fields), `Section`, `Accept`, `Reject`, `Want`.
//!
//! Implementation notes: nothing here uses `format::IndexBuilder`, the header or footer code, or the
//! chunk encoder, so a shared bug cannot hide. `File::to_bytes` writes every field itself, and the
//! zstd frames are written by hand too, as raw and RLE blocks, which needs no compressor. The only
//! outside code is XXH3 and BLAKE3. Offsets in the `poke` calls are the offsets of the record tables
//! of spec section 4.1. The reference reader is driven through its public `Container` alone.
//!
//! An `ErrorCode` is this implementation's name for a refusal. The conformance requirement on
//! another reader is the rejection, at or before the stage named. Most delta cases refuse with
//! `MalformedIndex` for one of many rules, so they also name words of this implementation's message
//! (`Reject::because`): a case cannot then pass by breaking an earlier rule by accident.

use std::sync::Arc;

use blake3::hash as blake3_hash;
use xxhash_rust::xxh3::xxh3_64;
use ziptensors_core::format::check_restore_name;
use ziptensors_core::{Container, Dtype, Error, ErrorCode, OpenOptions, RandomAccess};

/// `planes(2)`: step id, `param0`, `param1`.
const PLANES_2: (u16, u16, u32) = (0x0101, 2, 0);
/// `whole`.
const WHOLE: (u16, u16, u32) = (0x0100, 0, 0);
/// `rotl1(2)`.
const ROTL1_2: (u16, u16, u32) = (0x0200, 2, 0);
/// `fields(1)`, the `Q8_0` table.
const FIELDS_Q8_0: (u16, u16, u32) = (0x0102, 1, 0);
/// `delta_xor(2)`, version 0.3.
const DELTA_XOR_2: (u16, u16, u32) = (0x0400, 2, 0);
/// Step id of `f16 to f32`.
const VIEW_F16: u16 = 0x0301;
/// Step id of `bf16 to f32`.
const VIEW_BF16: u16 = 0x0302;
/// The content of the base file: 16 bytes.
const CONTENT: &[u8; 16] = b"0123456789abcdef";
/// Size of a zstd raw block's largest payload.
const MAX_ZSTD_BLOCK: usize = 128 << 10;

/// The smallest conforming file: no records at all, every section present and empty, the index a
/// frame of one empty raw block. Spelled out so that a reader of any language can start here.
///
/// ```text
///   0  magic 895A54530D0A1A0A, major 0, minor 1, header_len 64, features 0 0, reserved, header_xxh3
///  64  zstd: magic 28B52FFD, descriptor E0, content size 296 (8 bytes), then raw block header
///      (last, raw, 296 bytes) 41 09 00, then the 296 bytes of the index: section_count 9, 0, and 32
///      bytes per section: tag, flags 1, offset 296, len 0, count 0
/// 376  index_offset 64, stored_len 312, raw_len 296, index_xxh3, generation 1, 0, footer_xxh3,
///      "ZTS-END\0"
/// ```
const EMPTY_CONTAINER_HEX: &str = concat!(
    "895a54530d0a1a0a0000010040000000000000000000000000000000000000000000000000000000",
    "00000000000000000000000000000000a732af19117e7c1228b52ffde02801000000000000410900",
    "09000000000000005354525301000000280100000000000000000000000000000000000000000000",
    "4348414e010000002801000000000000000000000000000000000000000000004348535401000000",
    "280100000000000000000000000000000000000000000000424c4f42010000002801000000000000",
    "0000000000000000000000000000000043484e4b0100000028010000000000000000000000000000",
    "00000000000000004d454d4201000000280100000000000000000000000000000000000000000000",
    "4558544e0100000028010000000000000000000000000000000000000000000054454e5301000000",
    "28010000000000000000000000000000000000000000000053484150010000002801000000000000",
    "00000000000000000000000000000000400000000000000038010000000000002801000000000000",
    "2916079423111b2e0100000000000000000000000000000042ec1a097308a8545a54532d454e4400",
);

fn put_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// A slice length as `u64`. Lossless: `usize` is at most 64 bits on every supported target.
fn to_u64(len: usize) -> u64 {
    len as u64
}

/// A short length as `u32`.
fn to_u32(len: usize) -> u32 {
    u32::try_from(len).expect("a short length")
}

/// A zstd frame of raw blocks holding `content`: single segment, 8-byte content size. Never smaller
/// than `content`, which is fine for an index and is what section 6.1 forbids for a stream.
fn raw_frame(content: &[u8]) -> Vec<u8> {
    let mut frame = vec![0x28, 0xB5, 0x2F, 0xFD, 0xE0];
    put_u64(&mut frame, to_u64(content.len()));
    let mut blocks = content.chunks(MAX_ZSTD_BLOCK).peekable();
    if blocks.peek().is_none() {
        frame.extend_from_slice(&[1, 0, 0]);
    }
    while let Some(block) = blocks.next() {
        let last = u32::from(blocks.peek().is_none());
        let header = (to_u32(block.len()) << 3) | last;
        frame.extend_from_slice(&header.to_le_bytes()[..3]);
        frame.extend_from_slice(block);
    }
    frame
}

/// A zstd frame of one RLE block: `count` copies of `byte`. `descriptor` is the frame header after
/// the magic, so that a case can break it.
fn rle_frame(descriptor: &[u8], byte: u8, count: u32) -> Vec<u8> {
    let mut frame = vec![0x28, 0xB5, 0x2F, 0xFD];
    frame.extend_from_slice(descriptor);
    let header = (count << 3) | 0b011;
    frame.extend_from_slice(&header.to_le_bytes()[..3]);
    frame.push(byte);
    frame
}

/// The ordinary hand-written frame of 100 bytes `x`: single segment, content size 100 in one byte.
fn rle_100() -> Vec<u8> {
    rle_frame(&[0x20, 100], b'x', 100)
}

/// One stream word of a chunk prologue: `codec` in the high 8 bits, `stored_len` below.
fn word(codec: u32, stored_len: u32) -> [u8; 4] {
    ((codec << 24) | stored_len).to_le_bytes()
}

/// A stored chunk of one stream.
fn one_stream(codec: u32, stream: &[u8]) -> Vec<u8> {
    [word(codec, to_u32(stream.len())).as_slice(), stream].concat()
}

/// One entry of the section table, and its body.
#[derive(Debug, Clone)]
struct Section {
    tag: [u8; 4],
    flags: u32,
    body: Vec<u8>,
    count: u64,
    /// Replaces the offset this section would get.
    offset: Option<u64>,
    /// Replaces `body.len()` in the table.
    len: Option<u64>,
    /// Bytes put between the previous body and this one, after alignment.
    gap: Vec<u8>,
}

impl Section {
    fn new(tag: [u8; 4], flags: u32) -> Self {
        Self { tag, flags, body: Vec::new(), count: 0, offset: None, len: None, gap: Vec::new() }
    }
}

/// A whole file, field by field. [`File::to_bytes`] applies no rule at all.
#[derive(Debug, Clone)]
struct File {
    magic: [u8; 8],
    major: u16,
    minor: u16,
    header_len: u32,
    required: u64,
    optional: u64,
    header_reserved: [u8; 24],
    header_xxh3_xor: u64,
    /// Everything between the header and the stored index.
    data: Vec<u8>,
    table_zero: u32,
    sections: Vec<Section>,
    /// Replaces the stored index.
    stored_index: Option<Vec<u8>>,
    /// Replaces `section_count`.
    section_count: Option<u32>,
    /// Added to `index_offset`.
    index_offset_delta: i64,
    /// Replaces `index_raw_len`.
    index_raw_len: Option<u64>,
    /// Replaces `index_stored_len`.
    index_stored_len: Option<u64>,
    index_xxh3_xor: u64,
    generation: u64,
    footer_reserved: u64,
    footer_xxh3_xor: u64,
    end_magic: [u8; 8],
}

/// The tags of the mandatory sections, in the order of the table in spec section 4.
const MANDATORY: [&[u8; 4]; 9] =
    [b"STRS", b"CHAN", b"CHST", b"BLOB", b"CHNK", b"MEMB", b"EXTN", b"TENS", b"SHAP"];

impl File {
    /// The empty container.
    fn empty() -> Self {
        Self {
            magic: [0x89, 0x5A, 0x54, 0x53, 0x0D, 0x0A, 0x1A, 0x0A],
            major: 0,
            minor: 1,
            header_len: 64,
            required: 0,
            optional: 0,
            header_reserved: [0; 24],
            header_xxh3_xor: 0,
            data: Vec::new(),
            table_zero: 0,
            sections: MANDATORY.iter().map(|tag| Section::new(**tag, 1)).collect(),
            stored_index: None,
            section_count: None,
            index_offset_delta: 0,
            index_raw_len: None,
            index_stored_len: None,
            index_xxh3_xor: 0,
            generation: 1,
            footer_reserved: 0,
            footer_xxh3_xor: 0,
            end_magic: *b"ZTS-END\0",
        }
    }

    /// The base of nearly every case: one member `a.bin` of [`CONTENT`], one `whole` chunk with
    /// stream codec 0, one tensor `t` over all of it.
    fn base() -> Self {
        let mut file = Self::empty();
        let chain = file.chain(&[WHOLE]);
        let chunk = file.chunk(&one_stream(0, CONTENT), CONTENT);
        let blob = file.blob(16, chunk, 1, 16, chain);
        file.extent(0, 16, blob, 0, 0);
        file.shape(&[16]);
        let name = file.name("t");
        file.tensor(name, 0, 16, 0, 1, 2);
        file.member("a.bin", 16, *blake3_hash(CONTENT).as_bytes(), 0, 1, 0, 1, 0);
        file
    }

    fn section(&mut self, tag: &[u8; 4]) -> &mut Section {
        self.sections.iter_mut().find(|section| section.tag == *tag).expect("a section of the file")
    }

    /// Appends one record to a section and returns its index.
    fn record(&mut self, tag: &[u8; 4], bytes: &[u8]) -> u64 {
        let section = self.section(tag);
        section.body.extend_from_slice(bytes);
        section.count += 1;
        section.count - 1
    }

    /// Adds a string to `STRS` and returns `(off, len)`.
    fn name(&mut self, text: &str) -> (u32, u32) {
        let section = self.section(b"STRS");
        let off = to_u32(section.body.len());
        section.body.extend_from_slice(text.as_bytes());
        section.count = to_u64(section.body.len());
        (off, to_u32(text.len()))
    }

    /// Adds a chain of raw steps and returns its `chain_id`.
    fn chain(&mut self, steps: &[(u16, u16, u32)]) -> u32 {
        let first_step = to_u32(self.section(b"CHST").body.len() / 8);
        for (codec_id, param0, param1) in steps {
            let mut record = Vec::new();
            put_u16(&mut record, *codec_id);
            put_u16(&mut record, *param0);
            put_u32(&mut record, *param1);
            self.record(b"CHST", &record);
        }
        let mut record = Vec::new();
        put_u32(&mut record, first_step);
        put_u16(&mut record, u16::try_from(steps.len()).expect("few steps"));
        put_u16(&mut record, 0);
        to_u32(usize::try_from(self.record(b"CHAN", &record)).expect("few chains"))
    }

    /// Appends `stored` to the data region, adds its chunk record with the hash of `raw`, and returns
    /// the chunk's index.
    fn chunk(&mut self, stored: &[u8], raw: &[u8]) -> u64 {
        let file_offset = 64 + to_u64(self.data.len());
        self.data.extend_from_slice(stored);
        self.chunk_at(file_offset, xxh3_64(raw), to_u32(stored.len()))
    }

    /// Adds a chunk record as given.
    fn chunk_at(&mut self, file_offset: u64, xxh3_decoded: u64, stored_len: u32) -> u64 {
        let mut record = Vec::new();
        put_u64(&mut record, file_offset);
        put_u64(&mut record, xxh3_decoded);
        put_u32(&mut record, stored_len);
        put_u32(&mut record, 0);
        self.record(b"CHNK", &record)
    }

    /// Adds a blob of kind 0 with no identity and dtype U8, and returns its `blob_id`.
    fn blob(&mut self, raw_len: u64, first_chunk: u64, chunk_count: u32, chunk_size: u32, chain_id: u32) -> u32 {
        let mut record = vec![0; 32];
        put_u64(&mut record, raw_len);
        put_u64(&mut record, first_chunk);
        put_u32(&mut record, chunk_count);
        put_u32(&mut record, chunk_size);
        put_u32(&mut record, chain_id);
        record.extend_from_slice(&[0, 0]);
        put_u16(&mut record, 2);
        record.resize(80, 0);
        to_u32(usize::try_from(self.record(b"BLOB", &record)).expect("few blobs"))
    }

    /// Adds an extent.
    fn extent(&mut self, virt_offset: u64, virt_len: u64, blob_id: u32, view: u16, blob_offset: u64) {
        let mut record = Vec::new();
        put_u64(&mut record, virt_offset);
        put_u64(&mut record, virt_len);
        put_u32(&mut record, blob_id);
        put_u16(&mut record, view);
        put_u16(&mut record, 0);
        put_u64(&mut record, blob_offset);
        self.record(b"EXTN", &record);
    }

    /// Adds dimensions to `SHAP`.
    fn shape(&mut self, dims: &[u64]) {
        for dim in dims {
            self.record(b"SHAP", &dim.to_le_bytes());
        }
    }

    /// Adds a tensor row.
    fn tensor(&mut self, name: (u32, u32), virt_offset: u64, virt_len: u64, shape_off: u32, ndim: u16, dtype: u16) {
        let mut record = Vec::new();
        put_u32(&mut record, name.0);
        put_u32(&mut record, name.1);
        put_u64(&mut record, virt_offset);
        put_u64(&mut record, virt_len);
        put_u32(&mut record, shape_off);
        put_u16(&mut record, ndim);
        put_u16(&mut record, dtype);
        self.record(b"TENS", &record);
    }

    /// Adds a member. The arguments are the fields of the `MEMB` record, in its order.
    #[allow(clippy::too_many_arguments, reason = "one argument per field of a wire record")]
    fn member(
        &mut self,
        name: &str,
        file_len: u64,
        blake3: [u8; 32],
        first_extent: u64,
        extent_count: u32,
        first_tensor: u32,
        tensor_count: u32,
        profile: u8,
    ) {
        let (off, len) = self.name(name);
        let mut record = Vec::new();
        put_u32(&mut record, off);
        put_u32(&mut record, len);
        put_u64(&mut record, file_len);
        record.extend_from_slice(&blake3);
        put_u64(&mut record, first_extent);
        put_u32(&mut record, extent_count);
        put_u32(&mut record, first_tensor);
        put_u32(&mut record, tensor_count);
        record.push(profile);
        record.extend_from_slice(&[0; 3]);
        self.record(b"MEMB", &record);
    }

    /// Overwrites `bytes` at `offset` of record `id` of a fixed-record section of `size` bytes.
    fn poke(&mut self, tag: &[u8; 4], size: usize, id: usize, offset: usize, bytes: &[u8]) {
        let start = id * size + offset;
        self.section(tag).body[start..start + bytes.len()].copy_from_slice(bytes);
    }

    /// Appends a delta entry of spec section 7a to `PARM` at the next multiple of 8, adding the section
    /// with its required bit if the file has none, and returns its `params_off`.
    fn params(&mut self, base_blob: u32, base_offset: u64) -> u64 {
        if !self.sections.iter().any(|section| section.tag == *b"PARM") {
            self.sections.push(Section::new(*b"PARM", 1));
        }
        let section = self.section(b"PARM");
        section.body.resize(section.body.len().next_multiple_of(8), 0);
        let off = to_u64(section.body.len());
        put_u32(&mut section.body, base_blob);
        put_u32(&mut section.body, 0);
        put_u64(&mut section.body, base_offset);
        section.count = to_u64(section.body.len());
        off
    }

    /// Makes blob `blob` a delta blob, kind 1, whose 16-byte entry is at `params_off` of `PARM`.
    fn make_delta(&mut self, blob: u32, params_off: u64) {
        let id = usize::try_from(blob).expect("few blobs");
        self.poke(b"BLOB", 80, id, 60, &[1]);
        self.poke(b"BLOB", 80, id, 64, &params_off.to_le_bytes());
        self.poke(b"BLOB", 80, id, 72, &16_u32.to_le_bytes());
    }

    /// Drops a section from the table.
    fn remove(&mut self, tag: &[u8; 4]) {
        self.sections.retain(|section| section.tag != *tag);
    }

    /// The decompressed index.
    fn raw_index(&self) -> Vec<u8> {
        let table_len = 8 + 32 * self.sections.len();
        let mut bodies = vec![0; table_len];
        let mut entries = Vec::new();
        for section in &self.sections {
            bodies.resize(bodies.len().next_multiple_of(8), 0);
            bodies.extend_from_slice(&section.gap);
            bodies.resize(bodies.len().next_multiple_of(8), 0);
            let at = to_u64(bodies.len());
            bodies.extend_from_slice(&section.body);
            entries.extend_from_slice(&section.tag);
            put_u32(&mut entries, section.flags);
            put_u64(&mut entries, section.offset.unwrap_or(at));
            put_u64(&mut entries, section.len.unwrap_or(to_u64(section.body.len())));
            put_u64(&mut entries, section.count);
        }
        let count = self.section_count.unwrap_or(to_u32(self.sections.len()));
        bodies[..4].copy_from_slice(&count.to_le_bytes());
        bodies[4..8].copy_from_slice(&self.table_zero.to_le_bytes());
        bodies[8..table_len].copy_from_slice(&entries);
        bodies
    }

    /// The file. Hash fields are computed, then spoiled as the `*_xor` fields say.
    fn to_bytes(&self) -> Vec<u8> {
        let mut file = self.magic.to_vec();
        put_u16(&mut file, self.major);
        put_u16(&mut file, self.minor);
        put_u32(&mut file, self.header_len);
        put_u64(&mut file, self.required);
        put_u64(&mut file, self.optional);
        file.extend_from_slice(&self.header_reserved);
        let hash = xxh3_64(&file);
        put_u64(&mut file, hash ^ self.header_xxh3_xor);
        file.extend_from_slice(&self.data);
        let raw_index = self.raw_index();
        let stored = self.stored_index.clone().unwrap_or_else(|| raw_frame(&raw_index));
        let index_offset = to_u64(file.len());
        file.extend_from_slice(&stored);
        let mut footer = Vec::new();
        put_u64(&mut footer, index_offset.wrapping_add_signed(self.index_offset_delta));
        put_u64(&mut footer, self.index_stored_len.unwrap_or(to_u64(stored.len())));
        put_u64(&mut footer, self.index_raw_len.unwrap_or(to_u64(raw_index.len())));
        put_u64(&mut footer, xxh3_64(&raw_index) ^ self.index_xxh3_xor);
        put_u64(&mut footer, self.generation);
        put_u64(&mut footer, self.footer_reserved);
        let hash = xxh3_64(&footer);
        put_u64(&mut footer, hash ^ self.footer_xxh3_xor);
        footer.extend_from_slice(&self.end_magic);
        file.extend_from_slice(&footer);
        file
    }
}

/// A byte vector as a source.
#[derive(Debug)]
struct Memory(Vec<u8>);

impl RandomAccess for Memory {
    fn len(&self) -> u64 {
        to_u64(self.0.len())
    }

    fn read_exact_at(&self, off: u64, buf: &mut [u8]) -> Result<(), Error> {
        let bytes = usize::try_from(off).ok().and_then(|start| self.0.get(start..start + buf.len()));
        let error = || Error::OutOfBounds { offset: off, len: to_u64(buf.len()), file_len: self.len() };
        buf.copy_from_slice(bytes.ok_or_else(error)?);
        Ok(())
    }
}

fn open(bytes: Vec<u8>) -> Result<Container, Error> {
    let options = OpenOptions { threads: 1, ..OpenOptions::default() };
    Container::from_source(Arc::new(Memory(bytes)), &options)
}

/// A valid file and what it must give: each member's name and bytes.
#[derive(Debug)]
struct Accept {
    name: &'static str,
    file: File,
    members: Vec<(&'static str, Vec<u8>)>,
}

/// Where a refusal must happen.
#[derive(Debug, Clone, Copy)]
enum Want {
    /// `Container` refuses the file.
    Open(ErrorCode),
    /// The file opens, and a read of the whole first member fails.
    Read(ErrorCode),
    /// The file opens and reads, and a whole-member check fails.
    Verify(ErrorCode),
}

/// An invalid file and the rule it breaks.
#[derive(Debug)]
struct Reject {
    name: &'static str,
    /// Section or invariant of the spec.
    rule: &'static str,
    /// Words this implementation's error message must contain, where the code alone could come from
    /// another rule.
    because: Option<&'static str>,
    want: Want,
    file: File,
}

/// Builder of the accept list.
#[derive(Debug, Default)]
struct AcceptList(Vec<Accept>);

impl AcceptList {
    /// A case with one member, `a.bin`, of `content`.
    fn one(&mut self, name: &'static str, file: File, content: &[u8]) {
        self.0.push(Accept { name, file, members: vec![("a.bin", content.to_vec())] });
    }

    /// The base file after `edit`, with the base content.
    fn base(&mut self, name: &'static str, edit: impl FnOnce(&mut File)) {
        let mut file = File::base();
        edit(&mut file);
        self.one(name, file, CONTENT);
    }
}

/// A file with one member `a.bin` of `content`, whose one extent has `view` over a blob of `canonical`
/// in the chain `steps`, stored as `stored`.
fn single_blob(steps: &[(u16, u16, u32)], stored: &[u8], canonical: &[u8], view: u16, content: &[u8]) -> File {
    let mut file = File::empty();
    let chain = file.chain(steps);
    let chunk = file.chunk(stored, canonical);
    let raw_len = to_u64(canonical.len());
    let blob = file.blob(raw_len, chunk, 1, to_u32(canonical.len()), chain);
    file.extent(0, to_u64(content.len()), blob, view, 0);
    file.member("a.bin", to_u64(content.len()), *blake3_hash(content).as_bytes(), 0, 1, 0, 0, 0);
    file
}

/// Stored form of `planes(2)` over two elements: each stream is stored.
fn planes_chunk() -> (Vec<u8>, Vec<u8>) {
    let canonical = vec![0x34, 0x12, 0x78, 0x56];
    let stored = [
        word(0, 2).as_slice(),
        &word(0, 2),
        // Stream 0 is the low byte of each element, stream 1 the high byte.
        &[0x34, 0x78],
        &[0x12, 0x56],
    ]
    .concat();
    (stored, canonical)
}

/// The valid files. Member bytes are derived by hand from the spec, not computed by the reader.
fn accept_cases() -> Vec<Accept> {
    let mut list = AcceptList::default();
    list.0.push(Accept { name: "empty-container", file: File::empty(), members: Vec::new() });
    list.one("one-member-stored", File::base(), CONTENT);

    let mut file = File::empty();
    file.member("nothing", 0, *blake3_hash(b"").as_bytes(), 0, 0, 0, 0, 0);
    list.0.push(Accept { name: "empty-member-has-no-extents", file, members: vec![("nothing", Vec::new())] });

    let mut file = File::empty();
    let chain = file.chain(&[WHOLE]);
    let first = file.chunk(&one_stream(0, b"0123456789"), b"0123456789");
    file.chunk(&one_stream(0, b"abcdef"), b"abcdef");
    let blob = file.blob(16, first, 2, 10, chain);
    file.extent(0, 16, blob, 0, 0);
    file.member("a.bin", 16, *blake3_hash(CONTENT).as_bytes(), 0, 1, 0, 0, 0);
    list.one("two-chunks-last-is-short", file, CONTENT);

    let (stored, canonical) = planes_chunk();
    list.one("planes-2-stored", single_blob(&[PLANES_2], &stored, &canonical, 0, &canonical), &canonical);

    // Elements 0x8001 and 0x4000 rotate left by one bit to 0x0003 and 0x8000.
    let canonical = [0x01, 0x80, 0x00, 0x40];
    let stored = one_stream(0, &[0x03, 0x00, 0x00, 0x80]);
    list.one("rotl1-then-whole", single_blob(&[ROTL1_2, WHOLE], &stored, &canonical, 0, &canonical), &canonical);

    let canonical: Vec<u8> = (0..34).collect();
    let mut stored = [word(0, 1), word(0, 1), word(0, 32)].concat();
    stored.extend_from_slice(&canonical);
    list.one("fields-q8-0-stored", single_blob(&[FIELDS_Q8_0], &stored, &canonical, 0, &canonical), &canonical);

    let content = vec![b'x'; 100];
    list.one("zstd-rle-stream", single_blob(&[WHOLE], &one_stream(1, &rle_100()), &content, 0, &content), &content);
    // Window descriptor 0x70 is exponent 14, so 2^24 bytes, the largest a stream may have.
    let frame = rle_frame(&[0x80, 0x70, 100, 0, 0, 0], b'x', 100);
    list.one("zstd-window-at-the-limit", single_blob(&[WHOLE], &one_stream(1, &frame), &content, 0, &content), &content);

    // 1.0, the smallest subnormal, -0.0, infinity, a signalling NaN, by the table of spec section 7.
    let narrow: [u16; 5] = [0x3C00, 0x0001, 0x8000, 0x7C00, 0x7C01];
    let wide: [u32; 5] = [0x3F80_0000, 0x3380_0000, 0x8000_0000, 0x7F80_0000, 0x7F80_2000];
    let canonical: Vec<u8> = narrow.iter().flat_map(|element| element.to_le_bytes()).collect();
    let content: Vec<u8> = wide.iter().flat_map(|element| element.to_le_bytes()).collect();
    let stored = one_stream(0, &canonical);
    list.one("view-f16-to-f32", single_blob(&[WHOLE], &stored, &canonical, VIEW_F16, &content), &content);

    let canonical = [0x80, 0x3F];
    let content = [0x00, 0x00, 0x80, 0x3F];
    let stored = one_stream(0, &canonical);
    list.one("view-bf16-to-f32", single_blob(&[WHOLE], &stored, &canonical, VIEW_BF16, &content), &content);

    // Chunk size 3 cuts the element at bytes 2..4 across two chunks, which is valid.
    let canonical = [0x00, 0x3C, 0x00, 0x3C];
    let mut file = File::empty();
    let chain = file.chain(&[WHOLE]);
    let first = file.chunk(&one_stream(0, &canonical[..3]), &canonical[..3]);
    file.chunk(&one_stream(0, &canonical[3..]), &canonical[3..]);
    let blob = file.blob(4, first, 2, 3, chain);
    file.extent(0, 8, blob, VIEW_F16, 0);
    let content = [0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F];
    file.member("a.bin", 8, *blake3_hash(&content).as_bytes(), 0, 1, 0, 0, 0);
    list.one("view-element-straddles-chunks", file, &content);

    let mut file = File::base();
    let swapped = b"89abcdef01234567";
    file.extent(0, 8, 0, 0, 8);
    file.extent(8, 8, 0, 0, 0);
    file.member("swapped", 16, *blake3_hash(swapped).as_bytes(), 1, 2, 0, 0, 0);
    list.0.push(Accept {
        name: "extents-reorder-a-blob-for-a-second-member",
        file,
        members: vec![("a.bin", CONTENT.to_vec()), ("swapped", swapped.to_vec())],
    });

    let mut file = File::base();
    file.member("alias", 16, *blake3_hash(CONTENT).as_bytes(), 0, 1, 0, 1, 0);
    list.0.push(Accept {
        name: "members-share-extents-and-tensors",
        file,
        members: vec![("a.bin", CONTENT.to_vec()), ("alias", CONTENT.to_vec())],
    });

    list.base("identical-chunk-records", |file| {
        file.chunk_at(64, xxh3_64(CONTENT), 20);
    });
    list.base("unreferenced-bytes-in-the-data-region", |file| file.data.extend_from_slice(&[0xEE; 40]));
    list.base("unknown-optional-section-is-skipped", |file| {
        let mut section = Section::new(*b"XOPT", 0);
        section.body = vec![0xAA; 16];
        section.count = 1;
        file.sections.push(section);
    });
    list.base("parm-without-the-required-bit-is-skipped", |file| {
        let mut section = Section::new(*b"PARM", 0);
        section.body = vec![1, 2, 3];
        section.count = 3;
        file.sections.push(section);
    });
    list.base("sections-in-another-order", |file| file.sections.reverse());
    list.base("non-zero-bytes-between-bodies-are-ignored", |file| {
        file.section(b"MEMB").gap = vec![0xFF; 24];
    });
    list.base("a-later-minor-version", |file| file.minor = 9);
    list.base("minor-version-2", |file| file.minor = 2);
    list.base("dtype-registered-in-minor-2", |file| {
        file.poke(b"TENS", 32, 0, 30, &0x0047_u16.to_le_bytes());
        file.poke(b"BLOB", 80, 0, 62, &0x0034_u16.to_le_bytes());
    });
    list.base("unknown-optional-feature-bits", |file| file.optional = u64::MAX);
    list.base("generation-is-informational", |file| file.generation = u64::MAX);
    list.base("unknown-dtype-and-profile", |file| {
        file.poke(b"TENS", 32, 0, 30, &0x7777_u16.to_le_bytes());
        file.poke(b"BLOB", 80, 0, 62, &0x7777_u16.to_le_bytes());
        file.poke(b"MEMB", 72, 0, 68, &[0xEE]);
    });
    list.base("blob-identity-may-be-any-value", |file| file.poke(b"BLOB", 80, 0, 0, &[0x5A; 32]));
    list.base("tensor-of-16-huge-dimensions", |file| {
        file.shape(&[u64::MAX; 16]);
        file.poke(b"TENS", 32, 0, 24, &1_u32.to_le_bytes());
        file.poke(b"TENS", 32, 0, 28, &16_u16.to_le_bytes());
    });
    list.base("tensor-of-no-bytes-and-no-dimensions", |file| {
        file.poke(b"TENS", 32, 0, 16, &0_u64.to_le_bytes());
        file.poke(b"TENS", 32, 0, 28, &0_u16.to_le_bytes());
    });
    list.base("invalid-utf8-in-unreferenced-string-bytes", |file| {
        let section = file.section(b"STRS");
        section.body.extend_from_slice(&[0xFF, 0xFE]);
        section.count += 2;
    });
    list.base("index-frame-of-two-blocks", |file| {
        let section = file.section(b"STRS");
        section.body.resize(MAX_ZSTD_BLOCK + 1, b' ');
        section.count = to_u64(section.body.len());
    });
    list.base("an-update-appended-after-an-old-index", |file| {
        // The old index and footer of spec section 10 are just unreferenced bytes now.
        let old = File::base().to_bytes();
        let appended = &old[84..];
        file.data.extend_from_slice(appended);
        file.generation = 2;
    });
    accept_delta(&mut list);
    list.0
}

/// The target bytes of [`delta_file`]: base chunk 1, `89abcdef`, with bit 0 of byte 0 and bit 5 of
/// byte 4 flipped by the residual.
const DELTA_TARGET: &[u8; 8] = b"99abCdef";

/// The base of the delta cases (spec section 7a): version 0.3, required feature bit 0. Blob 0 is
/// [`CONTENT`] in two stored `whole` chunks of 8 bytes and has no extent. Blob 1, of kind 1, is 8
/// bytes over base chunk 1 (`base_offset` 8) in the chain `delta_xor(2)`, `whole`, its residual
/// stored, and its `PARM` entry at 0. Member `a.bin` is blob 1, [`DELTA_TARGET`].
fn delta_file() -> File {
    let mut file = File::empty();
    file.minor = 3;
    file.required = 1;
    let plain = file.chain(&[WHOLE]);
    let first = file.chunk(&one_stream(0, &CONTENT[..8]), &CONTENT[..8]);
    file.chunk(&one_stream(0, &CONTENT[8..]), &CONTENT[8..]);
    let base = file.blob(16, first, 2, 8, plain);
    let chain = file.chain(&[DELTA_XOR_2, WHOLE]);
    let residual = [0x01, 0, 0, 0, 0x20, 0, 0, 0];
    let chunk = file.chunk(&one_stream(0, &residual), DELTA_TARGET);
    let blob = file.blob(8, chunk, 1, 8, chain);
    let params = file.params(base, 8);
    file.make_delta(blob, params);
    file.extent(0, 8, blob, 0, 0);
    file.member("a.bin", 8, *blake3_hash(DELTA_TARGET).as_bytes(), 0, 1, 0, 0, 0);
    file
}

/// Valid files with delta blobs, version 0.3. Each target is derived by hand: residual XOR base bytes.
fn accept_delta(list: &mut AcceptList) {
    list.one("delta-xor-over-the-second-base-chunk", delta_file(), DELTA_TARGET);

    // `rotl1(2)` before `whole`: stored element 0x0003 rotates right to 0x8001, bytes 01 80, and
    // XOR "01" gives "1" and 0xB1. Chunk 1 stores zeros, so it is the base chunk unchanged.
    let mut file = delta_file();
    let chain = file.chain(&[DELTA_XOR_2, ROTL1_2, WHOLE]);
    let target = *b"1\xB1234567";
    let first = file.chunk(&one_stream(0, &[0x03, 0, 0, 0, 0, 0, 0, 0]), &target);
    file.chunk(&one_stream(0, &[0; 8]), &CONTENT[8..]);
    let blob = file.blob(16, first, 2, 8, chain);
    let params = file.params(0, 0);
    file.make_delta(blob, params);
    file.extent(0, 16, 0, 0, 0);
    file.extent(0, 16, blob, 0, 0);
    let content = [target.as_slice(), &CONTENT[8..]].concat();
    file.member("base.bin", 16, *blake3_hash(CONTENT).as_bytes(), 1, 1, 0, 0, 0);
    file.member("rotated.bin", 16, *blake3_hash(&content).as_bytes(), 2, 1, 0, 0, 0);
    list.0.push(Accept {
        name: "delta-rotl1-over-a-whole-base-that-a-member-reads-too",
        file,
        members: vec![("a.bin", DELTA_TARGET.to_vec()), ("base.bin", CONTENT.to_vec()), ("rotated.bin", content)],
    });

    // A delta blob of 4 bytes in chunks of 8 combines with the first 4 bytes of base chunk 1, "89ab":
    // the residual is zero, so the canonical bytes are 38 39 61 62, and `bf16 to f32` shifts each
    // element up by 16 bits.
    let mut file = delta_file();
    let chunk = file.chunk(&one_stream(0, &[0; 4]), b"89ab");
    let blob = file.blob(4, chunk, 1, 8, 1);
    let params = file.params(0, 8);
    file.make_delta(blob, params);
    file.extent(0, 8, blob, VIEW_BF16, 0);
    let content = [0x00, 0x00, 0x38, 0x39, 0x00, 0x00, 0x61, 0x62];
    file.member("wide", 8, *blake3_hash(&content).as_bytes(), 1, 1, 0, 0, 0);
    list.0.push(Accept {
        name: "delta-shorter-than-its-base-chunk-under-a-widening-view",
        file,
        members: vec![("a.bin", DELTA_TARGET.to_vec()), ("wide", content.to_vec())],
    });

    // The record of base chunk 0 again, so the residual is the base bytes and the target zero.
    let mut file = delta_file();
    let chunk = file.chunk_at(64, xxh3_64(&[0; 8]), 12);
    let blob = file.blob(8, chunk, 1, 8, 1);
    let params = file.params(0, 0);
    file.make_delta(blob, params);
    file.extent(0, 8, blob, 0, 0);
    file.member("zero", 8, *blake3_hash(&[0; 8]).as_bytes(), 1, 1, 0, 0, 0);
    list.0.push(Accept {
        name: "delta-chunk-is-the-stored-bytes-of-its-base-chunk",
        file,
        members: vec![("a.bin", DELTA_TARGET.to_vec()), ("zero", vec![0; 8])],
    });

    let mut file = delta_file();
    let section = file.section(b"PARM");
    section.body.splice(0..0, [0xEE; 8]);
    section.count = 24;
    file.poke(b"BLOB", 80, 1, 64, &8_u64.to_le_bytes());
    list.one("delta-parm-bytes-no-blob-refers-to-are-ignored", file, DELTA_TARGET);
}

/// Builder of the reject list. Each case is the base file after one edit.
#[derive(Debug, Default)]
struct RejectList(Vec<Reject>);

impl RejectList {
    /// `edit` applied to the base file must be refused with `code` at open.
    fn open(&mut self, name: &'static str, rule: &'static str, code: ErrorCode, edit: impl FnOnce(&mut File)) {
        let mut file = File::base();
        edit(&mut file);
        self.0.push(Reject { name, rule, because: None, want: Want::Open(code), file });
    }

    /// As [`Self::open`], for `MalformedIndex`.
    fn malformed(&mut self, name: &'static str, rule: &'static str, edit: impl FnOnce(&mut File)) {
        self.open(name, rule, ErrorCode::MalformedIndex, edit);
    }

    /// A file of `steps` over the base content must be refused with `code` at open.
    fn chain(&mut self, name: &'static str, code: ErrorCode, steps: &[(u16, u16, u32)]) {
        let file = single_blob(steps, &one_stream(0, CONTENT), CONTENT, 0, CONTENT);
        let rule = "5.1, 5.2, inv 5, inv 9";
        self.0.push(Reject { name, rule, because: None, want: Want::Open(code), file });
    }

    /// `edit` applied to [`delta_file`] must be refused as `want`, with `because` in the message.
    fn delta(&mut self, name: &'static str, want: Want, because: &'static str, edit: impl FnOnce(&mut File)) {
        let mut file = delta_file();
        edit(&mut file);
        self.0.push(Reject { name, rule: "7a, inv 13", because: Some(because), want, file });
    }

    /// A chunk of 100 raw bytes stored as `stored` must fail its first read with `code`.
    fn read(&mut self, name: &'static str, rule: &'static str, code: ErrorCode, stored: &[u8]) {
        let content = vec![b'x'; 100];
        let file = single_blob(&[WHOLE], stored, &content, 0, &content);
        self.0.push(Reject { name, rule, because: None, want: Want::Read(code), file });
    }
}

/// Header and footer: spec sections 3.1 to 3.3, invariant 1.
fn reject_header_and_footer(list: &mut RejectList) {
    use ErrorCode::{NotZiptensors, UnsupportedFeature, UnsupportedVersion};
    list.open("header-wrong-magic", "3.1, inv 1", NotZiptensors, |file| file.magic[1] = b'X');
    list.open("header-major-1", "3.3", UnsupportedVersion, |file| file.major = 1);
    list.malformed("header-len-65", "3.3, inv 1", |file| file.header_len = 65);
    list.malformed("header-len-0", "3.3, inv 1", |file| file.header_len = 0);
    list.malformed("header-reserved-byte", "3.1, inv 1", |file| file.header_reserved[23] = 1);
    list.malformed("header-hash", "3.1, inv 1", |file| file.header_xxh3_xor = 1);
    list.malformed("required-feature-delta-without-delta-blobs", "7a, inv 13", |file| file.required = 1);
    list.open("required-feature-derived", "3.3, 10", UnsupportedFeature, |file| file.required = 2);
    list.open("required-feature-multi-volume", "3.3, 10", UnsupportedFeature, |file| file.required = 4);
    list.open("required-feature-unassigned", "3.3", UnsupportedFeature, |file| file.required = 1 << 63);
    list.malformed("footer-hash", "3.2, inv 1", |file| file.footer_xxh3_xor = 1);
    list.malformed("footer-end-magic", "3.2, inv 1", |file| file.end_magic[7] = 1);
    list.malformed("footer-reserved", "3.2, inv 1", |file| file.footer_reserved = 1);
    list.malformed("footer-index-before-the-header-ends", "inv 1", |file| {
        file.index_offset_delta = -64;
    });
    list.malformed("footer-index-range-short-of-the-file", "inv 1", |file| {
        file.index_offset_delta = 1;
    });
    list.malformed("footer-index-range-past-the-file", "inv 1", |file| {
        file.index_stored_len = Some(u64::MAX);
    });
    list.open("footer-raw-len-over-the-cap", "3.2, inv 2", ErrorCode::BudgetExceeded, |file| {
        file.index_raw_len = Some((256 << 20) + 1);
    });
    list.open("footer-raw-len-over-one-gib", "3.2, inv 2", ErrorCode::BudgetExceeded, |file| {
        file.index_raw_len = Some((1 << 30) + 1);
    });
}

/// The stored index: spec section 4, invariants 1 and 2.
fn reject_stored_index(list: &mut RejectList) {
    let raw = File::base().raw_index();
    list.malformed("index-hash", "inv 1", |file| file.index_xxh3_xor = 1);
    list.malformed("index-not-a-zstd-frame", "4", |file| {
        file.stored_index = Some(raw.clone());
    });
    list.malformed("index-a-skippable-frame-first", "4", |file| {
        let frame = raw_frame(&raw);
        let skippable = [&[0x50, 0x2A, 0x4D, 0x18, 0, 0, 0, 0], frame.as_slice()].concat();
        file.stored_index = Some(skippable);
    });
    list.malformed("index-two-frames", "4", |file| {
        let frame = raw_frame(&raw);
        file.stored_index = Some([frame.as_slice(), raw_frame(&[]).as_slice()].concat());
    });
    list.malformed("index-frame-without-content-size", "4", |file| {
        // Descriptor 0: no content size, window byte follows: one raw block, last.
        let mut frame = vec![0x28, 0xB5, 0x2F, 0xFD, 0x00, 0x70];
        let header = (to_u32(raw.len()) << 3) | 1;
        frame.extend_from_slice(&header.to_le_bytes()[..3]);
        frame.extend_from_slice(&raw);
        file.stored_index = Some(frame);
    });
    list.malformed("index-frame-of-another-length", "4, inv 2", |file| {
        file.stored_index = Some(raw_frame(&raw[..raw.len() - 1]));
    });
    list.malformed("index-footer-length-differs-from-the-frame", "inv 2", |file| {
        file.index_raw_len = Some(to_u64(raw.len()) + 1);
    });
}

/// The section table: spec section 4, invariant 2.
fn reject_section_table(list: &mut RejectList) {
    list.malformed("table-count-past-the-index", "4", |file| file.section_count = Some(u32::MAX));
    list.malformed("table-zero-field", "4", |file| file.table_zero = 1);
    list.malformed("table-duplicate-tag", "4", |file| {
        let copy = Section::new(*b"SHAP", 1);
        file.sections.push(copy);
    });
    list.malformed("table-flag-bit-1", "4", |file| file.section(b"MEMB").flags = 3);
    list.malformed("table-body-unaligned", "4", |file| {
        let section = file.section(b"SHAP");
        let aligned = 8 + 32 * 9 + 4;
        section.offset = Some(aligned);
    });
    list.malformed("table-body-in-the-table", "4", |file| file.section(b"SHAP").offset = Some(0));
    list.malformed("table-bodies-overlap", "4", |file| {
        // After the table come STRS, CHAN and CHST; aim the last at the first.
        file.section(b"CHST").offset = Some(8 + 32 * 9);
    });
    list.malformed("table-body-past-the-end", "4", |file| {
        file.section(b"SHAP").offset = Some(1 << 40);
    });
    list.malformed("table-record-len-not-count-times-size", "4, inv 2", |file| {
        file.section(b"SHAP").count = 2;
    });
    list.malformed("table-record-count-overflows", "inv 2", |file| {
        file.section(b"SHAP").count = 1 << 61;
    });
    list.malformed("table-heap-len-not-count", "4", |file| file.section(b"STRS").count = 1);
    for (tag, name) in [
        (b"STRS", "table-missing-strs"),
        (b"CHAN", "table-missing-chan"),
        (b"CHST", "table-missing-chst"),
        (b"BLOB", "table-missing-blob"),
        (b"CHNK", "table-missing-chnk"),
        (b"MEMB", "table-missing-memb"),
        (b"EXTN", "table-missing-extn"),
        (b"TENS", "table-missing-tens"),
        (b"SHAP", "table-missing-shap"),
    ] {
        list.malformed(name, "4", |file| file.remove(tag));
    }
    list.malformed("table-mandatory-without-the-required-bit", "4", |file| {
        file.section(b"CHST").flags = 0;
    });
    list.malformed("table-unknown-required-section", "4, inv 9", |file| {
        file.sections.push(Section::new(*b"XREQ", 1));
    });
    list.malformed("table-parm-required", "4, 10", |file| {
        file.sections.push(Section::new(*b"PARM", 1));
    });
}

/// Records: spec section 4.1, invariants 3 to 8.
fn reject_records(list: &mut RejectList) {
    list.malformed("string-past-the-heap", "inv 3", |file| {
        file.poke(b"MEMB", 72, 0, 0, &1000_u32.to_le_bytes());
    });
    list.malformed("string-range-overflows", "inv 3", |file| {
        file.poke(b"MEMB", 72, 0, 0, &u32::MAX.to_le_bytes());
    });
    list.malformed("name-invalid-utf8", "inv 3", |file| file.section(b"STRS").body[0] = 0xFF);
    list.malformed("duplicate-member-names", "inv 3", |file| {
        file.member("a.bin", 16, [0; 32], 0, 1, 0, 0, 0);
    });
    list.malformed("duplicate-tensor-names-in-a-member", "inv 3", |file| {
        file.tensor((0, 1), 0, 1, 0, 0, 2);
        file.poke(b"MEMB", 72, 0, 64, &2_u32.to_le_bytes());
    });
    list.malformed("tensor-ndim-17", "inv 3", |file| {
        file.poke(b"TENS", 32, 0, 28, &17_u16.to_le_bytes());
    });
    list.malformed("tensor-shape-past-shap", "inv 8", |file| {
        file.poke(b"TENS", 32, 0, 28, &2_u16.to_le_bytes());
    });
    list.malformed("tensor-ends-past-the-member", "inv 8", |file| {
        file.poke(b"TENS", 32, 0, 16, &17_u64.to_le_bytes());
    });
    list.malformed("tensor-range-overflows", "inv 8, 1", |file| {
        file.poke(b"TENS", 32, 0, 8, &u64::MAX.to_le_bytes());
    });
    list.malformed("member-tensors-past-tens", "inv 3", |file| {
        file.poke(b"MEMB", 72, 0, 64, &2_u32.to_le_bytes());
    });
    list.malformed("member-zero-field", "4.1", |file| file.poke(b"MEMB", 72, 0, 71, &[1]));
    list.malformed("member-extents-past-extn", "inv 6", |file| {
        file.poke(b"MEMB", 72, 0, 56, &2_u32.to_le_bytes());
    });
    list.malformed("member-of-bytes-with-no-extents", "inv 6", |file| {
        file.poke(b"MEMB", 72, 0, 56, &0_u32.to_le_bytes());
    });
    list.malformed("member-longer-than-its-extents", "inv 6", |file| {
        file.poke(b"MEMB", 72, 0, 8, &17_u64.to_le_bytes());
    });
    list.malformed("chain-of-no-steps", "4.1", |file| {
        file.poke(b"CHAN", 8, 0, 4, &0_u16.to_le_bytes());
    });
    list.malformed("chain-of-nine-steps", "4.1", |file| {
        file.poke(b"CHAN", 8, 0, 4, &9_u16.to_le_bytes());
    });
    list.malformed("chain-steps-past-chst", "inv 3", |file| {
        file.poke(b"CHAN", 8, 0, 0, &5_u32.to_le_bytes());
    });
    list.malformed("chain-zero-field", "4.1", |file| file.poke(b"CHAN", 8, 0, 6, &[1]));
    list.malformed("blob-kind-delta-without-feature-bit-0", "4.1, 7a, inv 13", |file| file.poke(b"BLOB", 80, 0, 60, &[1]));
    list.malformed("blob-kind-derived", "4.1, 10, inv 7", |file| file.poke(b"BLOB", 80, 0, 60, &[2]));
    list.malformed("blob-flags", "4.1", |file| file.poke(b"BLOB", 80, 0, 61, &[1]));
    list.malformed("blob-zero-field", "4.1", |file| file.poke(b"BLOB", 80, 0, 79, &[1]));
    list.malformed("blob-params-off", "inv 7", |file| file.poke(b"BLOB", 80, 0, 64, &[1]));
    list.malformed("blob-params-len", "inv 7", |file| file.poke(b"BLOB", 80, 0, 72, &[1]));
    list.malformed("blob-chunk-size-0", "6.1, inv 5", |file| {
        file.poke(b"BLOB", 80, 0, 52, &0_u32.to_le_bytes());
    });
    list.malformed("blob-chunk-size-over-8-mib", "6.1, inv 5", |file| {
        file.poke(b"BLOB", 80, 0, 52, &8_388_609_u32.to_le_bytes());
    });
    list.malformed("blob-chunk-count-too-many", "inv 5", |file| {
        file.chunk_at(64, 0, 20);
        file.poke(b"BLOB", 80, 0, 48, &2_u32.to_le_bytes());
    });
    list.malformed("blob-chunk-count-zero", "inv 5", |file| {
        file.poke(b"BLOB", 80, 0, 48, &0_u32.to_le_bytes());
    });
    list.malformed("blob-chunks-past-chnk", "inv 5", |file| {
        file.poke(b"BLOB", 80, 0, 40, &1_u64.to_le_bytes());
    });
    list.malformed("blob-chain-out-of-range", "inv 5", |file| {
        file.poke(b"BLOB", 80, 0, 56, &1_u32.to_le_bytes());
    });
    list.malformed("chunk-zero-field", "4.1", |file| file.poke(b"CHNK", 24, 0, 23, &[1]));
    list.malformed("chunk-shorter-than-its-prologue", "inv 5", |file| {
        file.poke(b"CHNK", 24, 0, 16, &3_u32.to_le_bytes());
    });
    list.malformed("chunk-longer-than-prologue-plus-raw", "inv 5, 6.1", |file| {
        file.data.push(0);
        file.poke(b"CHNK", 24, 0, 16, &21_u32.to_le_bytes());
    });
    list.malformed("chunk-in-the-header", "inv 4", |file| {
        file.poke(b"CHNK", 24, 0, 0, &63_u64.to_le_bytes());
    });
    list.malformed("chunk-into-the-index", "inv 4", |file| {
        file.poke(b"CHNK", 24, 0, 0, &65_u64.to_le_bytes());
    });
    list.malformed("chunks-overlap-partly", "inv 4", |file| {
        file.data.extend_from_slice(&[0; 8]);
        file.chunk_at(70, 0, 20);
    });
    list.malformed("chunks-same-start-another-length", "inv 4", |file| {
        file.chunk_at(64, 0, 19);
    });
    list.malformed("extent-zero-field", "4.1", |file| file.poke(b"EXTN", 32, 0, 22, &[1]));
    list.malformed("extent-of-no-bytes", "inv 6", |file| {
        file.extent(16, 0, 0, 0, 0);
        file.poke(b"MEMB", 72, 0, 56, &2_u32.to_le_bytes());
    });
    list.malformed("extents-gap", "inv 6", |file| {
        file.poke(b"EXTN", 32, 0, 8, &8_u64.to_le_bytes());
        file.extent(9, 7, 0, 0, 8);
        file.poke(b"MEMB", 72, 0, 56, &2_u32.to_le_bytes());
    });
    list.malformed("extents-overlap", "inv 6", |file| {
        file.poke(b"EXTN", 32, 0, 8, &8_u64.to_le_bytes());
        file.extent(7, 9, 0, 0, 7);
        file.poke(b"MEMB", 72, 0, 56, &2_u32.to_le_bytes());
    });
    list.malformed("extents-unsorted", "inv 6", |file| {
        file.poke(b"EXTN", 32, 0, 0, &8_u64.to_le_bytes());
        file.poke(b"EXTN", 32, 0, 8, &8_u64.to_le_bytes());
        file.extent(0, 8, 0, 0, 0);
        file.poke(b"MEMB", 72, 0, 56, &2_u32.to_le_bytes());
    });
    list.malformed("extent-past-its-blob", "inv 7", |file| {
        file.poke(b"EXTN", 32, 0, 24, &1_u64.to_le_bytes());
    });
    list.malformed("extent-blob-out-of-range", "inv 7", |file| {
        file.poke(b"EXTN", 32, 0, 16, &1_u32.to_le_bytes());
    });
    list.malformed("extent-view-odd-blob-offset", "inv 7, 7", |file| {
        file.poke(b"EXTN", 32, 0, 8, &8_u64.to_le_bytes());
        file.poke(b"EXTN", 32, 0, 20, &VIEW_F16.to_le_bytes());
        file.poke(b"EXTN", 32, 0, 24, &1_u64.to_le_bytes());
        file.poke(b"MEMB", 72, 0, 8, &8_u64.to_le_bytes());
        file.poke(b"TENS", 32, 0, 16, &8_u64.to_le_bytes());
    });
    list.malformed("extent-view-length-not-a-multiple-of-4", "inv 7, 7", |file| {
        file.poke(b"EXTN", 32, 0, 8, &6_u64.to_le_bytes());
        file.poke(b"EXTN", 32, 0, 20, &VIEW_F16.to_le_bytes());
        file.poke(b"MEMB", 72, 0, 8, &6_u64.to_le_bytes());
        file.poke(b"TENS", 32, 0, 16, &6_u64.to_le_bytes());
    });
    list.malformed("extent-view-range-past-the-blob", "inv 7, 7", |file| {
        file.poke(b"EXTN", 32, 0, 8, &36_u64.to_le_bytes());
        file.poke(b"EXTN", 32, 0, 20, &VIEW_F16.to_le_bytes());
        file.poke(b"MEMB", 72, 0, 8, &36_u64.to_le_bytes());
    });
    list.open("extent-view-is-a-layout-id", "5.1, inv 9", ErrorCode::UnknownCodec, |file| {
        file.poke(b"EXTN", 32, 0, 20, &0x0100_u16.to_le_bytes());
    });
    list.open("extent-view-unknown", "5.1, inv 9", ErrorCode::UnknownCodec, |file| {
        file.poke(b"EXTN", 32, 0, 20, &0x0303_u16.to_le_bytes());
    });
}

/// Chains: spec sections 5.1 and 5.2.
fn reject_chains(list: &mut RejectList) {
    use ErrorCode::{MalformedIndex, UnknownCodec};
    // An id that is known with parameters outside the table is rejected too; this implementation
    // says `MalformedIndex` for that and keeps `UnknownCodec` for the id itself.
    let step = |codec_id| (codec_id, 0, 0);
    list.chain("step-0-is-invalid", UnknownCodec, &[step(0x0000)]);
    list.chain("step-a-view-id", UnknownCodec, &[step(VIEW_F16)]);
    list.chain("step-reserved-delta", UnknownCodec, &[step(0x0401)]);
    list.chain("step-reserved-derived", UnknownCodec, &[step(0x05FF)]);
    list.chain("step-private", UnknownCodec, &[step(0x8000)]);
    list.chain("step-unassigned", UnknownCodec, &[step(0x0999)]);
    list.chain("whole-with-a-parameter", MalformedIndex, &[(0x0100, 1, 0)]);
    list.chain("whole-with-param1", MalformedIndex, &[(0x0100, 0, 1)]);
    list.chain("planes-of-width-3", MalformedIndex, &[(0x0101, 3, 0)]);
    list.chain("fields-table-0", MalformedIndex, &[(0x0102, 0, 0)]);
    list.chain("fields-table-5", MalformedIndex, &[(0x0102, 5, 0)]);
    list.chain("rotl1-of-width-3", MalformedIndex, &[(0x0200, 3, 0), WHOLE]);
    list.chain("rotl1-with-no-layout", MalformedIndex, &[ROTL1_2]);
    list.chain("layout-then-rotl1", MalformedIndex, &[WHOLE, ROTL1_2]);
    list.chain("two-layouts", MalformedIndex, &[WHOLE, PLANES_2]);
    list.chain("two-rotl1", MalformedIndex, &[ROTL1_2, ROTL1_2, WHOLE]);
    list.chain("delta-xor-of-width-1", MalformedIndex, &[(0x0400, 1, 0), WHOLE]);
    list.chain("delta-xor-of-width-3", MalformedIndex, &[(0x0400, 3, 0), WHOLE]);
    list.chain("delta-xor-with-param1", MalformedIndex, &[(0x0400, 2, 1), WHOLE]);
    list.chain("delta-xor-with-no-layout", MalformedIndex, &[DELTA_XOR_2]);
    list.chain("delta-xor-after-rotl1", MalformedIndex, &[ROTL1_2, DELTA_XOR_2, WHOLE]);
    list.chain("delta-xor-after-the-layout", MalformedIndex, &[WHOLE, DELTA_XOR_2]);
    list.chain("two-delta-steps", MalformedIndex, &[DELTA_XOR_2, DELTA_XOR_2, WHOLE]);

    let stored = [word(0, 1).as_slice(), &word(0, 1), b"ab"].concat();
    let mut odd = single_blob(&[PLANES_2], &stored, b"ab", 0, b"ab");
    odd.poke(b"BLOB", 80, 0, 52, &3_u32.to_le_bytes());
    list.0.push(Reject {
        name: "chunk-size-not-a-multiple-of-the-chain-block",
        rule: "6.1, inv 5",
        because: None,
        want: Want::Open(MalformedIndex),
        file: odd,
    });
    let mut odd = single_blob(&[PLANES_2], &[0; 8], &[0; 4], 0, &[0; 3]);
    odd.poke(b"BLOB", 80, 0, 32, &3_u64.to_le_bytes());
    list.0.push(Reject {
        name: "raw-len-not-a-multiple-of-the-chain-block",
        rule: "inv 5",
        because: None,
        want: Want::Open(MalformedIndex),
        file: odd,
    });
}

/// Chunk contents, seen only when decoding: spec section 6, invariant 10.
fn reject_chunks(list: &mut RejectList) {
    use ErrorCode::{ChecksumMismatch, ChunkCorrupt, UnknownCodec, VerifyFailed};
    list.read("stream-codec-2", "6.1, inv 9", UnknownCodec, &one_stream(2, &[b'x'; 100]));
    list.read("stream-codec-255", "6.1, inv 9", UnknownCodec, &one_stream(255, &[b'x'; 100]));
    list.read("words-do-not-sum-to-the-chunk", "6.2, inv 10", ChunkCorrupt, &{
        let mut stored = word(0, 99).to_vec();
        stored.extend_from_slice(&[b'x'; 100]);
        stored
    });
    list.read("store-stream-shorter-than-raw", "6.1, inv 10", ChunkCorrupt, &one_stream(0, &[b'x'; 99]));
    list.read("zstd-empty-stream", "6.1", ChunkCorrupt, &one_stream(1, &[]));
    list.read("zstd-garbage", "6.1", ChunkCorrupt, &one_stream(1, &[0x55; 20]));
    list.read("zstd-skippable-frame", "6.1", ChunkCorrupt, &one_stream(1, &[0x50, 0x2A, 0x4D, 0x18, 0, 0, 0, 0]));
    list.read("zstd-no-content-size", "6.1", ChunkCorrupt, &one_stream(1, &rle_frame(&[0x00, 0x70], b'x', 100)));
    list.read("zstd-content-size-short", "6.1, inv 10", ChunkCorrupt, &one_stream(1, &rle_frame(&[0x20, 99], b'x', 99)));
    list.read("zstd-content-size-lies", "6.1, inv 10", ChunkCorrupt, &one_stream(1, &rle_frame(&[0x20, 100], b'x', 101)));
    list.read("zstd-frame-checksum", "6.1", ChunkCorrupt, &one_stream(1, &{
        let mut frame = rle_frame(&[0x24, 100], b'x', 100);
        frame.extend_from_slice(&[0; 4]);
        frame
    }));
    list.read("zstd-dictionary-id", "6.1", ChunkCorrupt, &one_stream(1, &rle_frame(&[0x21, 1, 100], b'x', 100)));
    list.read("zstd-reserved-bit", "6.1", ChunkCorrupt, &one_stream(1, &rle_frame(&[0x28, 100], b'x', 100)));
    list.read("zstd-unused-bit", "6.1", ChunkCorrupt, &one_stream(1, &rle_frame(&[0x30, 100], b'x', 100)));
    list.read("zstd-window-over-the-limit", "6.1", ChunkCorrupt, &one_stream(1, &rle_frame(&[0x80, 0x78, 100, 0, 0, 0], b'x', 100)));
    list.read("zstd-two-frames", "6.1", ChunkCorrupt, &one_stream(1, &{
        let half = rle_frame(&[0x20, 50], b'x', 50);
        [half.as_slice(), half.as_slice()].concat()
    }));

    // Stream 0 of two claims three bytes of a two-byte stream, though the words sum to the chunk.
    let stored = [word(1, 3).as_slice(), &word(0, 1), &[0, 0, 0, 0]].concat();
    let file = single_blob(&[PLANES_2], &stored, &[0; 4], 0, &[0; 4]);
    list.0.push(Reject {
        name: "stream-longer-than-its-raw-length",
        rule: "6.1, inv 10",
        because: None,
        want: Want::Read(ChunkCorrupt),
        file,
    });

    let mut file = File::base();
    file.poke(b"CHNK", 24, 0, 8, &0_u64.to_le_bytes());
    list.0.push(Reject {
        name: "chunk-hash",
        rule: "6.2, inv 10",
        because: None,
        want: Want::Read(ChecksumMismatch),
        file,
    });
    let mut file = File::base();
    file.poke(b"MEMB", 72, 0, 16, &[0x5A; 32]);
    list.0.push(Reject {
        name: "member-hash",
        rule: "9.1 step 6",
        because: None,
        want: Want::Verify(VerifyFailed),
        file,
    });
}

/// Delta blobs: spec section 7a, invariant 13. Each case is [`delta_file`] after one edit; blob 1 is
/// the delta blob, and its `PARM` entry holds `base_blob` at 0, a zero field at 4 and `base_offset`
/// at 8.
fn reject_delta(list: &mut RejectList) {
    use ErrorCode::{ChecksumMismatch, MalformedIndex};
    let open = Want::Open(MalformedIndex);
    let parm = |file: &mut File, at: usize, bytes: &[u8]| {
        file.section(b"PARM").body[at..at + bytes.len()].copy_from_slice(bytes);
    };
    list.delta("delta-base-is-the-blob-itself", open, "base blob 1 is not an earlier blob", |file| {
        parm(file, 0, &1_u32.to_le_bytes());
    });
    list.delta("delta-base-is-a-later-blob", open, "base blob 2 is not an earlier blob", |file| {
        file.blob(16, 0, 2, 8, 0);
        parm(file, 0, &2_u32.to_le_bytes());
    });
    list.delta("delta-base-out-of-range", open, "base blob 7 is not an earlier blob", |file| {
        parm(file, 0, &7_u32.to_le_bytes());
    });
    list.delta("delta-base-is-a-delta-blob", open, "base blob 1 is of kind 1", |file| {
        let chunk = file.chunk_at(64 + 24, xxh3_64(DELTA_TARGET), 12);
        let blob = file.blob(8, chunk, 1, 8, 1);
        let params = file.params(1, 0);
        file.make_delta(blob, params);
    });
    list.delta("delta-chunk-size-differs-from-the-base", open, "differs from 8 of base blob 0", |file| {
        file.poke(b"BLOB", 80, 1, 52, &16_u32.to_le_bytes());
    });
    list.delta("delta-base-offset-not-a-multiple-of-chunk-size", open, "no multiple of chunk_size", |file| {
        parm(file, 8, &4_u64.to_le_bytes());
    });
    list.delta("delta-past-the-end-of-its-base", open, "outside base blob 0", |file| {
        parm(file, 8, &16_u64.to_le_bytes());
    });
    list.delta("delta-base-offset-overflows", open, "outside base blob 0", |file| {
        parm(file, 8, &(u64::MAX - 7).to_le_bytes());
    });
    list.delta("delta-params-len-0", open, "not 0 at 0", |file| {
        file.poke(b"BLOB", 80, 1, 72, &0_u32.to_le_bytes());
    });
    list.delta("delta-params-len-24", open, "not 24 at 0", |file| {
        let section = file.section(b"PARM");
        section.body.resize(24, 0);
        section.count = 24;
        file.poke(b"BLOB", 80, 1, 72, &24_u32.to_le_bytes());
    });
    list.delta("delta-params-off-not-a-multiple-of-8", open, "not 16 at 4", |file| {
        let section = file.section(b"PARM");
        section.body.splice(0..0, [0; 4]);
        section.count = 20;
        file.poke(b"BLOB", 80, 1, 64, &4_u64.to_le_bytes());
    });
    list.delta("delta-params-past-parm", open, "range of 16 at 8", |file| {
        file.poke(b"BLOB", 80, 1, 64, &8_u64.to_le_bytes());
    });
    list.delta("delta-params-zero-field", open, "zero field of a delta entry", |file| parm(file, 4, &[1]));
    list.delta("delta-without-parm", open, "needs PARM with its required bit", |file| file.remove(b"PARM"));
    list.delta("delta-parm-without-the-required-bit", open, "needs PARM with its required bit", |file| {
        file.section(b"PARM").flags = 0;
    });
    list.delta("delta-blob-without-feature-bit-0", open, "without required feature bit 0", |file| {
        file.required = 0;
    });
    list.delta("feature-bit-0-with-the-delta-blob-made-kind-0", open, "no blob is a delta blob", |file| {
        file.poke(b"BLOB", 80, 1, 60, &[0]);
        file.poke(b"BLOB", 80, 1, 64, &[0; 12]);
    });
    list.delta("delta-blob-chain-without-a-delta-step", open, "chain has no delta step", |file| {
        file.poke(b"BLOB", 80, 1, 56, &0_u32.to_le_bytes());
    });
    list.delta("delta-step-in-the-chain-of-a-kind-0-blob", open, "kind 0 blob's chain has a delta step", |file| {
        file.poke(b"BLOB", 80, 0, 56, &1_u32.to_le_bytes());
    });
    list.delta("delta-base-chunk-hash", Want::Read(ChecksumMismatch), "chunk 1 of blob 0", |file| {
        file.poke(b"CHNK", 24, 1, 8, &0_u64.to_le_bytes());
    });
    list.delta("delta-chunk-hash", Want::Read(ChecksumMismatch), "chunk 0 of blob 1", |file| {
        file.poke(b"CHNK", 24, 2, 8, &0_u64.to_le_bytes());
    });
}

/// The invalid files.
fn reject_cases() -> Vec<Reject> {
    let mut list = RejectList::default();
    reject_header_and_footer(&mut list);
    reject_stored_index(&mut list);
    reject_section_table(&mut list);
    reject_records(&mut list);
    reject_chains(&mut list);
    reject_chunks(&mut list);
    reject_delta(&mut list);
    list.0
}

/// Member names and whether a restore into a directory may use them: invariant 11.
fn restore_name_cases() -> Vec<(&'static str, bool)> {
    vec![
        ("model.safetensors", true),
        ("dir/sub/model.bin", true),
        ("..hidden", true),
        ("a..b", true),
        ("...", true),
        ("", false),
        ("a\0b", false),
        ("/absolute", false),
        ("\\absolute", false),
        ("..", false),
        ("../up", false),
        ("a/../b", false),
        ("a/..", false),
        ("..\\up", false),
        ("a\\..\\b", false),
        ("C:drive", false),
        ("file:stream", false),
    ]
}

/// What went wrong with an accept case, or `None`.
fn accept_failure(case: &Accept) -> Option<String> {
    let container = match open(case.file.to_bytes()) {
        Ok(container) => container,
        Err(error) => return Some(format!("{error:?}")),
    };
    let listed: Vec<&str> = container.members().iter().map(|info| info.name.as_str()).collect();
    let expected: Vec<&str> = case.members.iter().map(|(name, _)| *name).collect();
    if listed != expected {
        return Some(format!("members {listed:?}"));
    }
    for (name, content) in &case.members {
        let outcome = container.member(name).and_then(|member| {
            let mut read = vec![0; content.len()];
            member.read_at(0, &mut read)?;
            member.verify()?;
            Ok(read)
        });
        match outcome {
            Ok(read) if read == *content => {}
            Ok(read) => return Some(format!("{name}: bytes {read:?}")),
            Err(error) => return Some(format!("{name}: {error:?}")),
        }
    }
    None
}

/// What went wrong with a reject case, or `None`.
fn reject_failure(case: &Reject) -> Option<String> {
    let opened = open(case.file.to_bytes());
    let found = |error: Error, want: ErrorCode| {
        let code = error.code();
        let message = error.to_string();
        if code != want {
            Some(format!("gave {code:?}, wanted {want:?}: {message}"))
        } else {
            case.because.filter(|words| !message.contains(words)).map(|words| {
                format!("refused for another reason, wanted {words:?}: {message}")
            })
        }
    };
    match case.want {
        Want::Open(want) => match opened {
            Ok(_) => Some("opened".to_owned()),
            Err(error) => found(error, want),
        },
        Want::Read(want) => match read_first_member(opened) {
            Ok(()) => Some("read".to_owned()),
            Err(error) => found(error, want),
        },
        Want::Verify(want) => match opened {
            Ok(container) => {
                let verified = container.member("a.bin").and_then(|member| {
                    member.read_at(0, &mut [0; 16])?;
                    member.verify()
                });
                match verified {
                    Ok(()) => Some("verified".to_owned()),
                    Err(error) => found(error, want),
                }
            }
            Err(error) => Some(format!("{error:?}")),
        },
    }
}

/// Reads all of the first member.
fn read_first_member(opened: Result<Container, Error>) -> Result<(), Error> {
    let container = opened?;
    let member = container.member("a.bin")?;
    member.read_at(0, &mut vec![0; usize::try_from(member.len()).unwrap_or(0)])
}

#[test]
fn every_valid_file_opens_and_gives_its_bytes() {
    let cases = accept_cases();
    assert!(cases.len() >= 37, "{}", cases.len());
    let failures: Vec<String> = cases
        .iter()
        .filter_map(|case| accept_failure(case).map(|text| format!("{}: {text}", case.name)))
        .collect();
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn every_invalid_file_is_refused_where_the_spec_says() {
    let cases = reject_cases();
    assert!(cases.len() >= 162, "{}", cases.len());
    let failures: Vec<String> = cases
        .iter()
        .filter_map(|case| {
            reject_failure(case).map(|text| format!("{} ({}): {text}", case.name, case.rule))
        })
        .collect();
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A tensor record keeps its dtype id as written, whichever minor version registered it: ids of
/// version 0.1, ids version 0.2 added, and ids still unregistered. A reader of either version gives
/// the same bytes for each.
#[test]
fn tensor_dtype_ids_read_back_unchanged() {
    for id in [0x0000, 0x0012, 0x0030, 0x0033, 0x0034, 0x003C, 0x0047, 0x0048, 0x7777] {
        let mut file = File::base();
        file.poke(b"TENS", 32, 0, 30, &u16::to_le_bytes(id));
        let container = open(file.to_bytes()).expect("opens");
        let member = container.member("a.bin").expect("the member");
        let tensor = member.tensors().first().expect("one tensor");
        assert_eq!(tensor.dtype.to_wire(), id, "{id:#06x}");
        assert_eq!(tensor.dtype, Dtype::from_wire(id), "{id:#06x}");
        let mut read = [0; 16];
        member.read_at(0, &mut read).expect("reads");
        assert_eq!(&read, CONTENT, "{id:#06x}");
    }
}

#[test]
fn the_base_of_the_reject_list_is_valid() {
    // The base of the reject list must itself be accepted, or every case would pass for nothing.
    let base = Accept { name: "base", file: File::base(), members: vec![("a.bin", CONTENT.to_vec())] };
    assert_eq!(accept_failure(&base), None);
}

#[test]
fn restore_names_follow_invariant_11() {
    for (name, safe) in restore_name_cases() {
        let result = check_restore_name(name);
        assert_eq!(result.is_ok(), safe, "{name:?}: {result:?}");
    }
}

#[test]
fn case_names_are_unique() {
    let accepted = accept_cases();
    let rejected = reject_cases();
    let mut names: Vec<&str> = accepted.iter().map(|case| case.name).collect();
    names.extend(rejected.iter().map(|case| case.name));
    let total = names.len();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), total);
}

/// Lower-case hex of `bytes`.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn the_empty_container_is_exactly_these_bytes() {
    let bytes = File::empty().to_bytes();
    assert_eq!(bytes.len(), 440);
    assert_eq!(hex(&bytes), EMPTY_CONTAINER_HEX);
    let container = open(bytes).expect("the smallest valid file");
    assert!(container.members().is_empty());
}
