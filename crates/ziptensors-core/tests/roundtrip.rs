//! File: tests/roundtrip.rs
//! Purpose: the writer through its public API: bit-exact roundtrips, sharing, rollback, commit.
//!
//! Main responsibilities:
//! - every dtype and source profile, restored bit-exact in all three modes;
//! - what a pack shares, read back from its index, which is how a reader would see it;
//! - `add_member_from` compared chunk by chunk with its source;
//! - delta blobs of a finetune against its base: pairing, the trial, dedup first, the header bit,
//!   restore of the delta member alone, copies that bring the base, chunk size refusals;
//! - rollback, determinism, the decode work of many small fragments, input re-checks.
//!
//! Key structures: `Scratch`, `Packed`.
//!
//! Implementation notes: inputs are generated and live in a unique temporary directory, which is
//! removed. Tensors are a little over two chunks, so that every blob has several chunks and a short
//! last one. The memory bound has its own binary, `tests/write_memory.rs`. Corruption before commit
//! needs a seam, so it is in `src/write/tests.rs`.

#[path = "common/fixtures.rs"]
mod fixtures;

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use fixtures::{
    Elem, Fixture, GgufOptions, OnnxModel, OnnxStorage, OnnxTensor, PtZipOptions, Rng, TensorSpec,
};
use ziptensors_core::chain::{BLOB_KIND_CHUNKED, BLOB_KIND_DELTA, FEATURE_DELTA, STREAM_ZSTD};
use ziptensors_core::codec::widen_into;
use ziptensors_core::format::{
    BlobRecord, FOOTER_LEN, Footer, HEADER_LEN, Index, Limits, check_decode_budget, parse_footer,
    parse_header,
};
use ziptensors_core::{
    Container, Dtype, ElementWidth, EncodeOptions, EncodeReport, Error, FieldTable, Layout, Member,
    Mode, OpenOptions, PackBuilder, Profile, View, encode_file, layer_report,
};

const MODES: [Mode; 3] = [Mode::Fast, Mode::Balanced, Mode::Max];
/// Bytes of a tensor that spans several chunks.
const TENSOR_BYTES: u64 = (2 << 20) + (1 << 19);
/// `F32` elements of the sample a writer takes from the start of a range.
const SAMPLE_ELEMENTS: usize = (64 << 10) / 4;
/// `0x0301`.
const VIEW_F16: u16 = 0x0301;
/// `0x0302`.
const VIEW_BF16: u16 = 0x0302;

/// A unique scratch directory, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let name = format!("ziptensors-roundtrip-{}-{id}", std::process::id());
        let path = std::env::temp_dir().join(name);
        fs::create_dir_all(&path).expect("scratch directory");
        Self(path)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    fn file(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.path(name);
        fs::write(&path, bytes).expect("scratch file");
        path
    }

    fn entries(&self) -> Vec<String> {
        let entries = fs::read_dir(&self.0).expect("readable");
        let mut names: Vec<String> = entries
            .map(|entry| entry.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("cannot remove {}: {error}", self.0.display());
        }
    }
}

fn options(mode: Mode) -> EncodeOptions {
    EncodeOptions { mode, threads: 3 }
}

/// The footer and the validated-by-the-reader index of a finished container, read as a third party
/// would: the public format layer, on the bytes of the file.
fn index_of(path: &Path) -> (Footer, Index) {
    let bytes = fs::read(path).expect("container");
    let at = bytes.len() - 64;
    let footer: [u8; 64] = bytes[at..].try_into().expect("64 bytes");
    let footer = parse_footer(&footer).expect("valid footer");
    let start = usize::try_from(footer.index_offset).expect("fits");
    let stored = &bytes[start..at];
    let raw_len = usize::try_from(footer.index_raw_len).expect("fits");
    let raw = zstd::bulk::decompress(stored, raw_len).expect("valid index frame");
    let index = Index::parse(&raw, &Limits::default()).expect("valid index");
    (footer, index)
}

/// What every finished container must satisfy, whatever is in it.
fn assert_container_invariants(path: &Path, report: &EncodeReport) {
    let (footer, index) = index_of(path);
    assert_eq!(report.output_len, fs::metadata(path).expect("exists").len());

    // No dead bytes: every rollback really cut the file. The data region is exactly the chunks.
    let stored: u64 = index.chunks().iter().map(|chunk| u64::from(chunk.stored_len)).sum();
    assert_eq!(footer.index_offset, HEADER_LEN + stored, "bytes left behind by a rollback");
    assert_eq!(report.output_len, HEADER_LEN + stored + footer.index_stored_len + FOOTER_LEN);
    assert_eq!(report.total.stored, stored);

    for class in &report.classes {
        let layers = class.layers;
        assert_eq!(
            layers.raw + layers.overhead,
            layers.widen_saved + layers.dedup_saved + layers.entropy_saved + layers.stored,
            "layers of {:?} do not balance",
            class.dtype
        );
    }

    // The decode work of spec section 4: under this writer it is at most the member itself.
    let member_bytes: u64 = index.members().iter().map(|member| member.file_len).sum();
    assert_eq!(report.verify.bytes, member_bytes);
    assert_eq!(report.total.raw, member_bytes);
    for member in index.members() {
        let work: u64 = index
            .member_extents(member)
            .expect("in range")
            .iter()
            .map(|extent| {
                let blob = &index.blobs()[extent.blob_id as usize];
                assert_eq!(extent.blob_offset, 0, "an extent starts inside a blob");
                let factor = extent.view().expect("valid").length_factor();
                assert_eq!(extent.virt_len / factor, blob.raw_len, "an extent is not a whole blob");
                blob.raw_len
            })
            .sum();
        assert!(work <= member.file_len, "{work} bytes of work for {}", member.file_len);
    }
}

/// Encodes `bytes` under `mode` and proves the result: reads and restores, through the real reader.
fn encode_and_prove(scratch: &Scratch, bytes: &[u8], mode: Mode) -> (PathBuf, EncodeReport) {
    let tag = format!("{mode:?}");
    let input = scratch.file(&format!("{tag}.input"), bytes);
    let output = scratch.path(&format!("{tag}.ziptensors"));
    let report = encode_file(&input, &output, &options(mode)).expect("encodes");
    assert_container_invariants(&output, &report);

    let container = Container::open(&output, &OpenOptions::default()).expect("opens");
    let member = container.single().expect("one member");
    assert_eq!(member.len(), bytes.len() as u64);
    assert_eq!(member.blake3(), *blake3::hash(bytes).as_bytes());
    member.verify().expect("verifies");

    let mut read = vec![0; bytes.len()];
    member.read_at(0, &mut read).expect("reads");
    assert!(read == bytes, "read_at differs from the source under {mode:?}");

    let restored = scratch.path(&format!("{tag}.restored"));
    member.restore_file(&restored).expect("restores");
    let restored_bytes = fs::read(&restored).expect("restored");
    assert!(restored_bytes == bytes, "the restored file differs under {mode:?}");
    fs::remove_file(&restored).expect("scratch file");
    (output, report)
}

/// [`encode_and_prove`] in all modes, and every tensor of the oracle read back by name.
fn prove_fixture(fixture: &Fixture, profile: Profile) {
    for mode in MODES {
        let scratch = Scratch::new();
        let (output, report) = encode_and_prove(&scratch, &fixture.bytes, mode);
        assert_eq!(report.members[0].profile, profile, "under {mode:?}");
        let container = Container::open(&output, &OpenOptions::default()).expect("opens");
        let member = container.single().expect("one member");
        assert_eq!(member.tensors().len(), fixture.tensors.len(), "under {mode:?}");
        assert_tensors_read_back(&member, fixture);
    }
}

fn assert_tensors_read_back(member: &Member, fixture: &Fixture) {
    for tensor in &fixture.tensors {
        let info = member.tensors().iter().find(|row| row.name == tensor.name).expect("listed");
        assert_eq!((info.offset, info.len), (tensor.offset, tensor.len), "{}", tensor.name);
        assert_eq!(info.shape, tensor.shape, "{}", tensor.name);
        assert_eq!(info.dtype, Dtype::from_wire(tensor.dtype_wire), "{}", tensor.name);
        let mut dst = vec![0; usize::try_from(tensor.len).expect("fits")];
        member.decode_tensor_into(&tensor.name, &mut dst).expect("decodes");
        let start = usize::try_from(tensor.offset).expect("fits");
        assert!(dst == fixture.bytes[start..start + dst.len()], "{}", tensor.name);
    }
}

/// Elements of `elem` in about [`TENSOR_BYTES`].
fn spec(name: &str, elem: Elem) -> TensorSpec {
    let (elements, block) = elem.block();
    let blocks = TENSOR_BYTES / block;
    TensorSpec::new(name, &[blocks * elements], elem)
}

fn safetensors_of(elem: Elem) -> Fixture {
    fixtures::safetensors(u64::from(elem.dtype_wire()), &[spec("tensor", elem)])
}

fn gguf_of(elem: Elem) -> Fixture {
    fixtures::gguf(u64::from(elem.dtype_wire()), &[spec("tensor", elem)], &GgufOptions::default())
}

/// The blob of the one big tensor of a file, the largest there is.
fn data_blob(index: &Index) -> &BlobRecord {
    index.blobs().iter().max_by_key(|blob| blob.raw_len).expect("has a blob")
}

/// Proves a one-tensor file in all modes, and checks the layout of its data blob per mode.
fn prove_dtype(fixture: &Fixture, profile: Profile, layout_of: impl Fn(Mode) -> Layout) {
    prove_fixture(fixture, profile);
    for mode in MODES {
        let scratch = Scratch::new();
        let (output, _) = encode_and_prove(&scratch, &fixture.bytes, mode);
        let (_, index) = index_of(&output);
        let blob = data_blob(&index);
        let chain = index.chain(blob.chain_id).expect("valid");
        assert_eq!(chain.layout(), layout_of(mode), "under {mode:?}");
        assert_eq!(blob.blake3 == [0; 32], mode != Mode::Max, "identity under {mode:?}");
        assert!(blob.chunk_count > 1, "the tensor must span chunks");
        assert_eq!(blob.chunk_size % chain.block_bytes(), 0);
    }
}

#[test]
fn f16_roundtrips_in_every_mode() {
    let planes = |_| Layout::Planes(ElementWidth::Two);
    prove_dtype(&safetensors_of(Elem::F16), Profile::Safetensors, planes);
}

#[test]
fn bf16_roundtrips_in_every_mode() {
    let planes = |_| Layout::Planes(ElementWidth::Two);
    prove_dtype(&safetensors_of(Elem::Bf16), Profile::Safetensors, planes);
}

#[test]
fn f32_roundtrips_in_every_mode() {
    let planes = |_| Layout::Planes(ElementWidth::Four);
    prove_dtype(&safetensors_of(Elem::F32), Profile::Safetensors, planes);
}

#[test]
fn f64_roundtrips_in_every_mode() {
    let planes = |_| Layout::Planes(ElementWidth::Eight);
    prove_dtype(&safetensors_of(Elem::F64), Profile::Safetensors, planes);
}

#[test]
fn fp8_roundtrips_in_every_mode() {
    prove_dtype(&safetensors_of(Elem::F8E4M3), Profile::Safetensors, |_| Layout::Whole);
    prove_dtype(&safetensors_of(Elem::F8E5M2), Profile::Safetensors, |_| Layout::Whole);
}

#[test]
fn i8_roundtrips_in_every_mode() {
    prove_dtype(&safetensors_of(Elem::I8), Profile::Safetensors, |_| Layout::Whole);
}

#[test]
fn u8_roundtrips_in_every_mode() {
    prove_dtype(&safetensors_of(Elem::U8), Profile::Safetensors, |_| Layout::Whole);
}

#[test]
fn nibble_data_roundtrips_in_every_mode() {
    // No source format can name packed 4-bit data, so it arrives tagged `U8`, and is stored.
    let mut fixture = safetensors_of(Elem::U8);
    let tensor = &fixture.tensors[0];
    let start = usize::try_from(tensor.offset).expect("fits");
    let end = start + usize::try_from(tensor.len).expect("fits");
    let mut rng = Rng::new(0x4B1D);
    for byte in &mut fixture.bytes[start..end] {
        let word = rng.next_u64();
        let nibble = |bits: u64| u8::try_from((bits & 3) + 6).expect("under 16");
        *byte = (nibble(word) << 4) | nibble(word >> 8);
    }
    prove_dtype(&fixture, Profile::Safetensors, |_| Layout::Whole);
}

/// The layout of a quantized tensor: stored in `fast`, fields otherwise.
fn quantized(table: FieldTable) -> impl Fn(Mode) -> Layout {
    move |mode| if mode == Mode::Fast { Layout::Whole } else { Layout::Fields(table) }
}

#[test]
fn q8_0_roundtrips_in_every_mode() {
    prove_dtype(&gguf_of(Elem::Q80), Profile::Gguf, quantized(FieldTable::Q80));
}

#[test]
fn q4_k_roundtrips_in_every_mode() {
    prove_dtype(&gguf_of(Elem::Q4K), Profile::Gguf, quantized(FieldTable::Q4K));
}

#[test]
fn q5_k_roundtrips_in_every_mode() {
    prove_dtype(&gguf_of(Elem::Q5K), Profile::Gguf, quantized(FieldTable::Q5K));
}

#[test]
fn q6_k_roundtrips_in_every_mode() {
    prove_dtype(&gguf_of(Elem::Q6K), Profile::Gguf, quantized(FieldTable::Q6K));
}

#[test]
fn q8_0_chunks_hold_whole_blocks_under_one_mib() {
    let scratch = Scratch::new();
    let (output, _) = encode_and_prove(&scratch, &gguf_of(Elem::Q80).bytes, Mode::Balanced);
    let (_, index) = index_of(&output);
    assert_eq!(data_blob(&index).chunk_size, 1_048_560);
}

#[test]
fn safetensors_profile_roundtrips() {
    let specs = [
        spec("big.f32", Elem::F32),
        TensorSpec::new("small.bf16", &[16], Elem::Bf16),
        spec("big.f16", Elem::F16),
        TensorSpec::new("mid.u8", &[8192], Elem::U8),
    ];
    prove_fixture(&fixtures::safetensors(1, &specs), Profile::Safetensors);
}

#[test]
fn gguf_profile_roundtrips() {
    let specs = [
        spec("blk.0.weight", Elem::Q80),
        TensorSpec::new("blk.0.norm", &[64], Elem::F32),
        spec("blk.1.weight", Elem::Q6K),
    ];
    prove_fixture(&fixtures::gguf(2, &specs, &GgufOptions::default()), Profile::Gguf);
}

/// The ggml block types without a field table have rows, and their bytes stay in the skeleton:
/// no blob and no report class carries their dtype, so the output is what it was before they were
/// registered, apart from the tensor table.
#[test]
fn gguf_block_types_without_a_field_table_are_rows_over_the_skeleton() {
    let specs: Vec<TensorSpec> = Elem::UNSPLIT_GGML
        .iter()
        .enumerate()
        .map(|(index, &elem)| {
            let (elements, _) = elem.block();
            TensorSpec::new(&format!("blk.{index}.weight"), &[elements * 64, 3], elem)
        })
        .chain([spec("f32", Elem::F32)])
        .collect();
    let fixture = fixtures::gguf(31, &specs, &GgufOptions::default());
    prove_fixture(&fixture, Profile::Gguf);
    for mode in MODES {
        let scratch = Scratch::new();
        let (output, report) = encode_and_prove(&scratch, &fixture.bytes, mode);
        let (_, index) = index_of(&output);
        for blob in index.blobs() {
            assert!(
                matches!(blob.dtype_hint, Dtype::Unknown(0) | Dtype::F32 | Dtype::F16 | Dtype::Bf16),
                "a blob of {:?} under {mode:?}",
                blob.dtype_hint
            );
        }
        for class in &report.classes {
            assert!(
                matches!(class.dtype, None | Some(Dtype::F32 | Dtype::F16 | Dtype::Bf16)),
                "a class of {:?} under {mode:?}",
                class.dtype
            );
        }
    }
}

#[test]
fn onnx_profile_roundtrips() {
    let mut model = OnnxModel::default();
    model.initializers.push(OnnxTensor::raw("weights", &[600_000], Elem::F32));
    model.initializers.push(OnnxTensor::stored(
        "packed",
        &[4096],
        Elem::F32,
        OnnxStorage::PackedFloat,
    ));
    model.constants.push(OnnxTensor::raw("constant", &[2048], Elem::F16));
    model.loop_body.push(OnnxTensor::raw("body", &[8], Elem::U8));
    prove_fixture(&fixtures::onnx(3, &model), Profile::Onnx);
}

#[test]
fn pt_zip_profile_roundtrips() {
    let specs = [
        spec("0", Elem::F32),
        spec("1", Elem::F16),
        TensorSpec::new("2", &[32], Elem::U8),
        TensorSpec::new("3", &[8192], Elem::U8),
    ];
    let fixture = fixtures::pt_zip(4, &specs, &PtZipOptions::default());
    prove_fixture(&fixture, Profile::PtZip);
}

#[test]
fn opaque_files_roundtrip() {
    let mut rng = Rng::new(5);
    let mut bytes = rng.bytes(1_500_000);
    bytes.extend(std::iter::repeat_n(b"not a tensor format. ".as_slice(), 100_000).flatten());
    let fixture = Fixture { bytes, tensors: Vec::new() };
    prove_fixture(&fixture, Profile::Opaque);
    for mode in MODES {
        let scratch = Scratch::new();
        let (output, report) = encode_and_prove(&scratch, &fixture.bytes, mode);
        let (_, index) = index_of(&output);
        assert_eq!(index.blobs().len(), 1, "an opaque file is one blob");
        assert_eq!(report.classes.len(), 1);
        assert_eq!(report.classes[0].dtype, None);
        assert!(report.total.entropy_saved > 0, "zstd must find the repeated text");
    }
}

#[test]
fn an_empty_file_roundtrips() {
    for mode in MODES {
        let scratch = Scratch::new();
        let (output, report) = encode_and_prove(&scratch, &[], mode);
        let (_, index) = index_of(&output);
        assert!(index.blobs().is_empty());
        assert!(index.extents().is_empty());
        assert_eq!(report.members[0].len, 0);
        assert_eq!(report.members[0].profile, Profile::Opaque);
    }
}

#[test]
fn a_zero_length_tensor_roundtrips() {
    let specs = [
        TensorSpec::new("empty", &[0, 4], Elem::F32),
        spec("big", Elem::F16),
        TensorSpec::new("scalar_empty", &[0], Elem::U8),
    ];
    let fixture = fixtures::safetensors(6, &specs);
    assert_eq!(fixture.tensors[0].len, 0);
    prove_fixture(&fixture, Profile::Safetensors);
}

/// Sets `bytes` of the first tensor of `fixture`.
fn with_tensor_bytes(mut fixture: Fixture, bytes: &[u8]) -> Fixture {
    let tensor = &fixture.tensors[0];
    assert_eq!(tensor.len, bytes.len() as u64);
    let start = usize::try_from(tensor.offset).expect("fits");
    fixture.bytes[start..start + bytes.len()].copy_from_slice(bytes);
    fixture
}

/// Canonical bytes `narrow` seen through `view`.
fn widened(view: View, narrow: &[u8]) -> Vec<u8> {
    let mut wide = vec![0; narrow.len() * 2];
    widen_into(view, narrow, &mut wide).expect("sizes match");
    wide
}

/// The first tensor of `fixture`.
fn tensor_bytes(fixture: &Fixture) -> &[u8] {
    let tensor = &fixture.tensors[0];
    let start = usize::try_from(tensor.offset).expect("fits");
    &fixture.bytes[start..start + usize::try_from(tensor.len).expect("fits")]
}

/// An `F16` file and its exact `F32` sibling.
fn f16_with_f32_sibling() -> (Fixture, Fixture) {
    let narrow = safetensors_of(Elem::F16);
    let wide = widened(View::F16ToF32, tensor_bytes(&narrow));
    let elements = wide.len() as u64 / 4;
    let sibling = fixtures::safetensors(9, &[TensorSpec::new("tensor", &[elements], Elem::F32)]);
    (narrow, with_tensor_bytes(sibling, &wide))
}

/// Packs `files` under `mode`, in order, and proves each member against its bytes.
fn pack(scratch: &Scratch, mode: Mode, files: &[(&str, &[u8])]) -> (PathBuf, EncodeReport) {
    let output = scratch.path(&format!("{mode:?}.pack"));
    let mut builder = PackBuilder::new(&output, &options(mode)).expect("starts");
    for (name, bytes) in files {
        let input = scratch.file(&format!("{name}.source"), bytes);
        builder.add_file(&input, name).expect("adds");
    }
    let report = builder.finish().expect("commits");
    assert_container_invariants(&output, &report);
    let container = Container::open(&output, &OpenOptions::default()).expect("opens");
    for (name, bytes) in files {
        let member = container.member(name).expect("present");
        let mut read = vec![0; bytes.len()];
        member.read_at(0, &mut read).expect("reads");
        assert!(read == *bytes, "member {name} differs");
    }
    (output, report)
}

/// The data extent of member `name`, the one over the largest blob.
fn data_extent(index: &Index, member: usize) -> (u32, u16) {
    let extents = index.member_extents(&index.members()[member]).expect("in range");
    let extent = extents.iter().max_by_key(|extent| extent.virt_len).expect("has data");
    (extent.blob_id, extent.view_id)
}

#[test]
fn an_f32_member_shares_a_blob_with_its_f16_sibling_in_either_order() {
    let (narrow, wide) = f16_with_f32_sibling();
    let orders = [
        [("a.f16", narrow.bytes.as_slice()), ("b.f32", wide.bytes.as_slice())],
        [("a.f32", wide.bytes.as_slice()), ("b.f16", narrow.bytes.as_slice())],
    ];
    for (order, files) in orders.iter().enumerate() {
        let scratch = Scratch::new();
        let (output, report) = pack(&scratch, Mode::Max, files);
        let (_, index) = index_of(&output);
        let (first, first_view) = data_extent(&index, 0);
        let (second, second_view) = data_extent(&index, 1);
        assert_eq!(first, second, "the data must be one blob");
        let views = if order == 0 { (0, VIEW_F16) } else { (VIEW_F16, 0) };
        assert_eq!((first_view, second_view), views);
        let blob = &index.blobs()[first as usize];
        assert_eq!(blob.dtype_hint, Dtype::F16);
        assert_eq!(blob.blake3, *blake3::hash(tensor_bytes(&narrow)).as_bytes());
        assert_eq!(report.total.dedup_saved, TENSOR_BYTES);
        assert_eq!(report.total.widen_saved, TENSOR_BYTES);
        assert_eq!(report.members[1].blobs_reused, 1);
    }
}

#[test]
fn balanced_widens_and_has_no_identity_so_siblings_stay_apart() {
    let (narrow, wide) = f16_with_f32_sibling();
    let scratch = Scratch::new();
    let files = [("a.f16", narrow.bytes.as_slice()), ("b.f32", wide.bytes.as_slice())];
    let (output, report) = pack(&scratch, Mode::Balanced, &files);
    let (_, index) = index_of(&output);
    assert_ne!(data_extent(&index, 0).0, data_extent(&index, 1).0);
    assert_eq!(data_extent(&index, 1).1, VIEW_F16);
    assert!(index.blobs().iter().all(|blob| blob.blake3 == [0; 32]));
    assert_eq!(report.total.dedup_saved, 0);
}

#[test]
fn fast_keeps_an_f32_tensor_as_it_is() {
    let (_, wide) = f16_with_f32_sibling();
    let scratch = Scratch::new();
    let (output, _) = encode_and_prove(&scratch, &wide.bytes, Mode::Fast);
    let (_, index) = index_of(&output);
    assert!(index.extents().iter().all(|extent| extent.view_id == 0));
}

#[test]
fn an_f32_tensor_of_bf16_values_uses_the_bf16_view() {
    // Exponents far above binary16's range, so only the other view can reproduce them.
    let mut rng = Rng::new(10);
    let narrow: Vec<u8> = (0..TENSOR_BYTES / 2)
        .flat_map(|_| (0x6000 | (rng.next_u64() & 0x0FFF) as u16).to_le_bytes())
        .collect();
    let wide = widened(View::Bf16ToF32, &narrow);
    let elements = wide.len() as u64 / 4;
    let fixture = fixtures::safetensors(11, &[TensorSpec::new("tensor", &[elements], Elem::F32)]);
    let fixture = with_tensor_bytes(fixture, &wide);
    let scratch = Scratch::new();
    let (output, report) = encode_and_prove(&scratch, &fixture.bytes, Mode::Balanced);
    let (_, index) = index_of(&output);
    assert_eq!(data_extent(&index, 0).1, VIEW_BF16);
    assert_eq!(data_blob(&index).dtype_hint, Dtype::Bf16);
    assert_eq!(report.total.widen_saved, TENSOR_BYTES);
}

#[test]
fn safetensors_and_gguf_share_a_blob() {
    // Both builders draw the first tensor from the same sequence of the seed.
    let specs = [spec("weights", Elem::F32)];
    let st = fixtures::safetensors(12, &specs);
    let gg = fixtures::gguf(12, &specs, &GgufOptions::default());
    assert!(tensor_bytes(&st) == tensor_bytes(&gg));
    let scratch = Scratch::new();
    let files = [("model.safetensors", st.bytes.as_slice()), ("model.gguf", gg.bytes.as_slice())];
    let (output, report) = pack(&scratch, Mode::Max, &files);
    let (_, index) = index_of(&output);
    assert_eq!(data_extent(&index, 0).0, data_extent(&index, 1).0);
    assert_eq!(report.members[1].layers.dedup_saved, TENSOR_BYTES);
    assert_eq!(report.members[0].profile, Profile::Safetensors);
    assert_eq!(report.members[1].profile, Profile::Gguf);
}

#[test]
fn opaque_files_share_a_blob_in_max_only() {
    let bytes = Rng::new(13).bytes(3 << 19);
    let files = [("a.bin", bytes.as_slice()), ("b/copy.bin", bytes.as_slice())];
    let scratch = Scratch::new();
    fs::create_dir(scratch.path("b")).expect("scratch directory");
    let (output, report) = pack(&scratch, Mode::Max, &files);
    let (_, index) = index_of(&output);
    assert_eq!(index.blobs().len(), 1);
    assert_eq!(report.total.dedup_saved, bytes.len() as u64);
    let (output, report) = pack(&scratch, Mode::Balanced, &files);
    let (_, index) = index_of(&output);
    assert_eq!(index.blobs().len(), 2);
    assert_eq!(report.total.dedup_saved, 0);
}

#[test]
fn tensors_repeated_inside_one_file_share_a_blob() {
    let mut fixture = fixtures::safetensors(14, &[spec("a", Elem::F16), spec("b", Elem::F16)]);
    let first = tensor_bytes(&fixture).to_vec();
    let start = usize::try_from(fixture.tensors[1].offset).expect("fits");
    fixture.bytes[start..start + first.len()].copy_from_slice(&first);
    let scratch = Scratch::new();
    let (output, report) = encode_and_prove(&scratch, &fixture.bytes, Mode::Max);
    let (_, index) = index_of(&output);
    assert_eq!(index.blobs().len(), 2, "one for the header, one for both tensors");
    assert_eq!(report.members[0].blobs_reused, 1);
}

#[test]
fn max_keeps_a_rotation_where_it_saves_and_restores_exactly() {
    // Exponents 127 and 128 differ in all bits, so one bit of information sits in two planes unless
    // the rotation aligns the exponent with a byte.
    let mut rng = Rng::new(19);
    let wide: Vec<u8> = (0..TENSOR_BYTES / 4)
        .flat_map(|_| {
            let word = rng.next_u64();
            let exponent = 127 + (word >> 40 & 1);
            u32::try_from((exponent << 23) | (word & 0x7F_FFFF)).expect("31 bits").to_le_bytes()
        })
        .collect();
    let fixture = with_tensor_bytes(safetensors_of(Elem::F32), &wide);
    let scratch = Scratch::new();
    let (output, report) = encode_and_prove(&scratch, &fixture.bytes, Mode::Max);
    let (_, index) = index_of(&output);
    let chain = index.chain(data_blob(&index).chain_id).expect("valid");
    assert_eq!(chain.rotl1(), Some(ElementWidth::Four));
    let plain = Scratch::new();
    let (_, balanced) = encode_and_prove(&plain, &fixture.bytes, Mode::Balanced);
    assert!(report.total.stored < balanced.total.stored);
}

#[test]
fn a_tensor_with_no_gain_from_rotation_keeps_the_plain_chain() {
    let scratch = Scratch::new();
    let (output, _) = encode_and_prove(&scratch, &safetensors_of(Elem::F16).bytes, Mode::Max);
    let (_, index) = index_of(&output);
    assert_eq!(index.chain(data_blob(&index).chain_id).expect("valid").rotl1(), None);
}

/// An `F32` tensor of exact values, with the element at index `at` not exact under any view.
fn nearly_exact(at: usize) -> (Fixture, Vec<u8>) {
    let narrow = safetensors_of(Elem::F16);
    let mut wide = widened(View::F16ToF32, tensor_bytes(&narrow));
    let inexact = 0x3F80_0001_u32.to_le_bytes();
    wide[at * 4..at * 4 + 4].copy_from_slice(&inexact);
    let elements = wide.len() as u64 / 4;
    let specs = [
        TensorSpec::new("tensor", &[elements], Elem::F32),
        TensorSpec::new("after", &[60_000], Elem::F16),
    ];
    let fixture = with_tensor_bytes(fixtures::safetensors(15, &specs), &wide);
    (fixture, wide)
}

/// Proves that a late mismatch left a plain identity blob and no trace of the narrow stream.
fn prove_rollback(fixture: &Fixture, mode: Mode, wide: &[u8]) {
    let scratch = Scratch::new();
    let (output, report) = encode_and_prove(&scratch, &fixture.bytes, mode);
    assert_eq!(report.members[0].widen_restreams, 1, "the narrow stream must have started");
    assert_eq!(report.total.widen_saved, 0);
    let (_, index) = index_of(&output);
    assert!(index.extents().iter().all(|extent| extent.view_id == 0));
    let blob = data_blob(&index);
    assert_eq!(blob.raw_len, wide.len() as u64);
    assert_eq!(blob.dtype_hint, Dtype::F32);
}

#[test]
fn a_tensor_whose_last_element_is_not_exact_rolls_back_and_restreams() {
    let elements = (TENSOR_BYTES / 2) as usize;
    let (fixture, wide) = nearly_exact(elements - 1);
    prove_rollback(&fixture, Mode::Balanced, &wide);
    prove_rollback(&fixture, Mode::Max, &wide);
}

#[test]
fn a_mismatch_just_past_the_sample_rolls_back_too() {
    let (fixture, wide) = nearly_exact(SAMPLE_ELEMENTS);
    prove_rollback(&fixture, Mode::Max, &wide);
}

#[test]
fn a_mismatch_inside_the_sample_needs_no_rollback() {
    for at in [0, SAMPLE_ELEMENTS - 1] {
        let (fixture, _) = nearly_exact(at);
        let scratch = Scratch::new();
        let (output, report) = encode_and_prove(&scratch, &fixture.bytes, Mode::Max);
        assert_eq!(report.members[0].widen_restreams, 0, "the sample must have caught it");
        let (_, index) = index_of(&output);
        assert!(index.extents().iter().all(|extent| extent.view_id == 0));
    }
}

#[test]
fn a_rollback_leaves_the_other_blobs_of_the_pack_alone() {
    let elements = (TENSOR_BYTES / 2) as usize;
    let (nearly, _) = nearly_exact(elements - 1);
    let (narrow, wide) = f16_with_f32_sibling();
    let scratch = Scratch::new();
    let files = [
        ("first.f16", narrow.bytes.as_slice()),
        ("nearly.f32", nearly.bytes.as_slice()),
        ("sibling.f32", wide.bytes.as_slice()),
    ];
    let (output, report) = pack(&scratch, Mode::Max, &files);
    assert_eq!(report.members[1].widen_restreams, 1);
    assert_eq!(report.members[2].blobs_reused, 1);
    let (_, index) = index_of(&output);
    assert_eq!(data_extent(&index, 0).0, data_extent(&index, 2).0);
}

/// A file that takes every path of the pipeline: all kinds of blob, a duplicate, a late mismatch.
fn busy_fixture() -> Fixture {
    let elements = (TENSOR_BYTES / 2) as usize;
    let (mut fixture, _) = nearly_exact(elements / 2);
    let more = fixtures::gguf(
        16,
        &[spec("q", Elem::Q5K), spec("f", Elem::F32), spec("h", Elem::Bf16)],
        &GgufOptions::default(),
    );
    fixture.bytes.extend_from_slice(&more.bytes);
    fixture.tensors.clear();
    fixture
}

#[test]
fn output_bytes_do_not_depend_on_the_thread_count_or_the_run() {
    let inputs = [
        safetensors_of(Elem::F32).bytes,
        gguf_of(Elem::Q80).bytes,
        nearly_exact((TENSOR_BYTES / 2 - 1) as usize).0.bytes,
        f16_with_f32_sibling().0.bytes,
        f16_with_f32_sibling().1.bytes,
        busy_fixture().bytes,
    ];
    let scratch = Scratch::new();
    let paths: Vec<PathBuf> = inputs
        .iter()
        .enumerate()
        .map(|(at, bytes)| scratch.file(&format!("input-{at}"), bytes))
        .collect();
    for mode in MODES {
        let mut outputs = Vec::new();
        for threads in [1, 2, 3, 7, 1] {
            let output = scratch.path("out");
            let settings = EncodeOptions { mode, threads };
            let mut builder = PackBuilder::new(&output, &settings).expect("starts");
            for (at, path) in paths.iter().enumerate() {
                builder.add_file(path, &format!("input-{at}")).expect("adds");
            }
            builder.finish().expect("commits");
            outputs.push(fs::read(&output).expect("output"));
        }
        assert!(outputs.windows(2).all(|pair| pair[0] == pair[1]), "under {mode:?}");
    }
}

#[test]
fn many_small_fragments_open_with_default_limits() {
    // Each storage is one range, and the ZIP headers between them are one skeleton fragment each.
    let specs: Vec<TensorSpec> =
        (0..3000).map(|at| TensorSpec::new(&format!("{at}"), &[1024], Elem::F32)).collect();
    let fixture = fixtures::pt_zip(17, &specs, &PtZipOptions::default());
    for mode in MODES {
        let scratch = Scratch::new();
        let (output, report) = encode_and_prove(&scratch, &fixture.bytes, mode);
        let (_, index) = index_of(&output);
        assert!(index.extents().len() > 6000, "{} extents", index.extents().len());
        assert_eq!(report.verify.bytes, fixture.bytes.len() as u64);

        // Fragments are separate blobs, never ranges of one, so no bound of the reader comes near.
        // `assert_container_invariants` compared the work with the member; this asks the format layer
        // for the same with no allowance beyond the floor.
        let strict = Limits { max_decode_amplification: 1, ..Limits::default() };
        check_decode_budget(&index, 0..1, &strict).expect("within bound");
        let skeleton = index.blobs().iter().filter(|blob| blob.dtype_hint == Dtype::Unknown(0));
        assert!(skeleton.into_iter().all(|blob| blob.chunk_count == 1));
    }
}

/// Chunk bytes of `blob`, in order.
fn stored_chunks(file: &[u8], index: &Index, blob: &BlobRecord) -> Vec<Vec<u8>> {
    let chunks = index.blob_chunks(blob).expect("in range");
    chunks
        .iter()
        .map(|chunk| {
            let start = usize::try_from(chunk.file_offset).expect("fits");
            file[start..start + chunk.stored_len as usize].to_vec()
        })
        .collect()
}

/// Asserts that each blob of `copy` is a blob of `source` with the same stored bytes and records.
fn assert_blobs_copied(copy: &Path, source: &Path) {
    let (_, copied) = index_of(copy);
    let (_, original) = index_of(source);
    let (copy_file, source_file) = (fs::read(copy).expect("copy"), fs::read(source).expect("source"));
    let mut known: HashMap<Vec<Vec<u8>>, &BlobRecord> = HashMap::new();
    for blob in original.blobs() {
        known.insert(stored_chunks(&source_file, &original, blob), blob);
    }
    for blob in copied.blobs() {
        let bytes = stored_chunks(&copy_file, &copied, blob);
        let from = known.get(&bytes).expect("every stored chunk comes from the source");
        assert_eq!(blob.blake3, from.blake3);
        assert_eq!(blob.raw_len, from.raw_len);
        assert_eq!(blob.chunk_size, from.chunk_size);
        assert_eq!(blob.dtype_hint, from.dtype_hint);
        assert_eq!(copied.chain(blob.chain_id).ok(), original.chain(from.chain_id).ok());
        let hashes = |index: &Index, blob: &BlobRecord| -> Vec<(u64, u32)> {
            let chunks = index.blob_chunks(blob).expect("in range");
            chunks.iter().map(|chunk| (chunk.xxh3_decoded, chunk.stored_len)).collect()
        };
        assert_eq!(hashes(&copied, blob), hashes(&original, from));
    }
}

#[test]
fn add_member_from_writes_no_new_compressed_bytes() {
    let (narrow, wide) = f16_with_f32_sibling();
    for mode in MODES {
        let scratch = Scratch::new();
        let tensors = safetensors_of(Elem::Bf16);
        let files = [
            ("a.f16", narrow.bytes.as_slice()),
            ("b.f32", wide.bytes.as_slice()),
            ("c.bf16", tensors.bytes.as_slice()),
        ];
        let (source, _) = pack(&scratch, mode, &files);
        let container = Container::open(&source, &OpenOptions::default()).expect("opens");

        let merged = scratch.path("merged");
        // The target's mode is irrelevant to a copy.
        let mut builder = PackBuilder::new(&merged, &options(Mode::Max)).expect("starts");
        for name in ["c.bf16", "b.f32", "a.f16"] {
            let report = builder.add_member_from(&container, name).expect("copies");
            assert_eq!(report.compressed_bytes, 0);
            assert_eq!(report.name, name);
            assert_eq!(report.widen_restreams, 0);
        }
        let report = builder.finish().expect("commits");
        assert_container_invariants_of_a_copy(&merged, &report);
        assert_blobs_copied(&merged, &source);

        let (_, before) = index_of(&source);
        let (_, after) = index_of(&merged);
        assert_eq!(after.blobs().len(), before.blobs().len());
        assert_eq!(after.chunks().len(), before.chunks().len());
        let copied: u64 = report.members.iter().map(|member| member.copied_bytes).sum();
        let stored: u64 = before.chunks().iter().map(|chunk| u64::from(chunk.stored_len)).sum();
        assert_eq!(copied, stored);

        let container = Container::open(&merged, &OpenOptions::default()).expect("opens");
        let names: Vec<&str> = container.members().iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["c.bf16", "b.f32", "a.f16"]);
        for (name, bytes) in files {
            let member = container.member(name).expect("present");
            let mut read = vec![0; bytes.len()];
            member.read_at(0, &mut read).expect("reads");
            assert!(read == bytes, "member {name} differs after the copy under {mode:?}");
            assert_eq!(member.tensors().len(), 1);
            assert_eq!(member.tensors()[0].name, "tensor");
        }
    }
}

/// [`assert_container_invariants`] for a merge: layers count what was copied, in the same way.
fn assert_container_invariants_of_a_copy(path: &Path, report: &EncodeReport) {
    assert_container_invariants(path, report);
    assert_eq!(report.members.iter().map(|m| m.compressed_bytes).sum::<u64>(), 0);
}

#[test]
fn a_copy_reuses_a_blob_this_pack_compressed_itself() {
    let scratch = Scratch::new();
    let bytes = safetensors_of(Elem::F16).bytes;
    let (source, _) = pack(&scratch, Mode::Max, &[("source", bytes.as_slice())]);
    let container = Container::open(&source, &OpenOptions::default()).expect("opens");
    let input = scratch.file("again", &bytes);
    let merged = scratch.path("merged");
    let mut builder = PackBuilder::new(&merged, &options(Mode::Max)).expect("starts");
    builder.add_file(&input, "again").expect("adds");
    let copied = builder.add_member_from(&container, "source").expect("copies");
    assert_eq!(copied.copied_bytes, 0, "all its blobs were here already");
    assert_eq!(copied.blobs_reused, 2);
    let report = builder.finish().expect("commits");
    assert_container_invariants(&merged, &report);
}

#[test]
fn add_member_from_refuses_a_missing_member_and_keeps_going() {
    let scratch = Scratch::new();
    let (source, _) = pack(&scratch, Mode::Max, &[("source", b"bytes".as_slice())]);
    let container = Container::open(&source, &OpenOptions::default()).expect("opens");
    let merged = scratch.path("merged");
    let mut builder = PackBuilder::new(&merged, &options(Mode::Max)).expect("starts");
    let error = builder.add_member_from(&container, "nothing").expect_err("no such member");
    assert!(matches!(error, Error::MemberNotFound { .. }), "{error:?}");
    builder.add_member_from(&container, "source").expect("copies");
    let error = builder.add_member_from(&container, "source").expect_err("used twice");
    assert!(matches!(error, Error::MalformedIndex { section: "MEMB", .. }), "{error:?}");
    builder.finish().expect("commits");
}

#[test]
fn a_changed_input_stops_the_commit_and_leaves_nothing() {
    let scratch = Scratch::new();
    let input = scratch.file("input", &[7; 5000]);
    let output = scratch.path("output");
    let mut builder = PackBuilder::new(&output, &options(Mode::Balanced)).expect("starts");
    builder.add_file(&input, "input").expect("adds");
    fs::write(&input, [7; 5001]).expect("rewrites");
    let error = builder.finish().expect_err("the input changed");
    assert!(matches!(error, Error::InputChanged { ref path } if *path == input), "{error:?}");
    assert_eq!(scratch.entries(), ["input"]);
}

#[test]
fn a_deleted_input_stops_the_commit_too() {
    let scratch = Scratch::new();
    let input = scratch.file("input", &[7; 5000]);
    let output = scratch.path("output");
    let mut builder = PackBuilder::new(&output, &options(Mode::Fast)).expect("starts");
    builder.add_file(&input, "input").expect("adds");
    fs::remove_file(&input).expect("deletes");
    let error = builder.finish().expect_err("the input is gone");
    assert!(matches!(error, Error::InputChanged { .. }), "{error:?}");
    assert!(scratch.entries().is_empty());
}

#[test]
fn a_failed_commit_keeps_what_was_at_the_output() {
    let scratch = Scratch::new();
    let input = scratch.file("input", &[7; 5000]);
    let output = scratch.file("output", b"the previous file");
    let mut builder = PackBuilder::new(&output, &options(Mode::Fast)).expect("starts");
    builder.add_file(&input, "input").expect("adds");
    fs::write(&input, [8; 7]).expect("rewrites");
    builder.finish().expect_err("the input changed");
    assert_eq!(fs::read(&output).expect("still there"), b"the previous file");
}

#[test]
fn a_dropped_builder_leaves_nothing() {
    let scratch = Scratch::new();
    let input = scratch.file("input", &[7; 5000]);
    let mut builder = PackBuilder::new(&scratch.path("out"), &options(Mode::Fast)).expect("starts");
    builder.add_file(&input, "input").expect("adds");
    drop(builder);
    assert_eq!(scratch.entries(), ["input"]);
}

#[cfg(unix)]
#[test]
fn a_symbolic_link_is_refused_and_the_builder_goes_on() {
    let scratch = Scratch::new();
    let input = scratch.file("input", &[7; 5000]);
    let link = scratch.path("link");
    std::os::unix::fs::symlink(&input, &link).expect("creates a link");
    let output = scratch.path("output");
    let mut builder = PackBuilder::new(&output, &options(Mode::Fast)).expect("starts");
    let error = builder.add_file(&link, "link").expect_err("a link is not stored");
    assert!(matches!(error, Error::SymlinkInput { ref path } if *path == link), "{error:?}");
    builder.add_file(&input, "input").expect("adds");
    let report = builder.finish().expect("commits");
    assert_eq!(report.members.len(), 1);
    let error = encode_file(&link, &output, &options(Mode::Fast)).expect_err("a link is not stored");
    assert!(matches!(error, Error::SymlinkInput { .. }), "{error:?}");
}

#[test]
fn encoding_a_file_onto_itself_is_refused_and_leaves_it_alone() {
    let scratch = Scratch::new();
    let input = scratch.file("input", &[7; 5000]);
    let error = encode_file(&input, &input, &options(Mode::Fast)).expect_err("same path");
    assert!(matches!(error, Error::OutputIsInput { .. }), "{error:?}");
    assert_eq!(fs::read(&input).expect("still there"), [7; 5000]);
    assert_eq!(scratch.entries(), ["input"]);
}

#[test]
fn restoring_onto_the_container_is_refused_and_leaves_it_alone() {
    let scratch = Scratch::new();
    let (output, _) = pack(&scratch, Mode::Fast, &[("a", b"one".as_slice()), ("b", b"two".as_slice())]);
    let before = fs::read(&output).expect("the pack");
    let container = Container::open(&output, &OpenOptions::default()).expect("opens");
    let member = container.member("a").expect("present");
    let same = |error: Error| matches!(error, Error::OutputIsInput { .. });
    assert!(same(member.restore_file(&output).expect_err("same path")));
    #[cfg(unix)]
    {
        let (soft, hard) = (scratch.path("soft"), scratch.path("hard"));
        std::os::unix::fs::symlink(&output, &soft).expect("links");
        fs::hard_link(&output, &hard).expect("links");
        assert!(same(member.restore_file(&soft).expect_err("a link to it")));
        assert!(same(member.restore_file(&hard).expect_err("another name of it")));
    }
    assert!(fs::read(&output).expect("still there") == before);
    let other = scratch.path("restored");
    member.restore_file(&other).expect("another path is fine");
    assert_eq!(fs::read(&other).expect("restored"), b"one");
}

#[test]
fn layer_report_of_a_file_says_what_its_encode_said() {
    let (narrow, wide) = f16_with_f32_sibling();
    let quantized = gguf_of(Elem::Q4K);
    let files = [
        ("a.f16", narrow.bytes.as_slice()),
        ("b.f32", wide.bytes.as_slice()),
        ("c.gguf", quantized.bytes.as_slice()),
        ("d.again", narrow.bytes.as_slice()),
    ];
    for mode in MODES {
        let scratch = Scratch::new();
        let (output, report) = pack(&scratch, mode, &files);
        let container = Container::open(&output, &OpenOptions::default()).expect("opens");
        let (total, classes) = layer_report(&container).expect("reports");
        assert_eq!(total, report.total, "under {mode:?}");
        assert_eq!(classes, report.classes, "under {mode:?}");
    }
}

#[test]
fn a_file_that_cannot_be_read_leaves_the_builder_usable() {
    let scratch = Scratch::new();
    let input = scratch.file("input", &[7; 5000]);
    let output = scratch.path("output");
    let mut builder = PackBuilder::new(&output, &options(Mode::Max)).expect("starts");
    let missing = scratch.path("missing");
    let error = builder.add_file(&missing, "missing").expect_err("no such file");
    assert!(matches!(error, Error::Io { .. }), "{error:?}");
    builder.add_file(&input, "input").expect("adds");
    let report = builder.finish().expect("commits");
    assert_container_invariants(&output, &report);
}

#[test]
fn member_names_a_restore_would_refuse_are_refused_here() {
    let scratch = Scratch::new();
    let input = scratch.file("input", &[7; 16]);
    let mut builder = PackBuilder::new(&scratch.path("out"), &options(Mode::Fast)).expect("starts");
    for name in ["", "../x", "/etc/x", "a:b", "a\\..\\b"] {
        let error = builder.add_file(&input, name).expect_err("unsafe");
        assert!(matches!(error, Error::UnsafeMemberName { .. }), "{name:?}: {error:?}");
    }
    builder.add_file(&input, "nested/ok").expect("adds");
    let error = builder.add_file(&input, "nested/ok").expect_err("used twice");
    assert!(matches!(error, Error::MalformedIndex { section: "MEMB", .. }), "{error:?}");
}

#[test]
fn an_empty_pack_is_a_valid_file() {
    let scratch = Scratch::new();
    let output = scratch.path("output");
    let report = PackBuilder::new(&output, &options(Mode::Max))
        .expect("starts")
        .finish()
        .expect("commits");
    assert!(report.members.is_empty());
    let container = Container::open(&output, &OpenOptions::default()).expect("opens");
    assert!(container.members().is_empty());
}

#[test]
fn an_empty_member_sits_between_others() {
    let bytes = safetensors_of(Elem::F16).bytes;
    let scratch = Scratch::new();
    let files = [("a", bytes.as_slice()), ("empty", [].as_slice()), ("c", bytes.as_slice())];
    let (output, _) = pack(&scratch, Mode::Max, &files);
    let container = Container::open(&output, &OpenOptions::default()).expect("opens");
    let member = container.member("empty").expect("present");
    assert!(member.is_empty());
    member.verify().expect("verifies");
}

#[test]
fn the_report_splits_bytes_by_class() {
    let specs = [spec("a", Elem::F16), spec("b", Elem::U8), spec("c", Elem::F8E4M3)];
    let fixture = fixtures::safetensors(18, &specs);
    let scratch = Scratch::new();
    let (_, report) = encode_and_prove(&scratch, &fixture.bytes, Mode::Balanced);
    let classes: HashMap<Option<Dtype>, u64> =
        report.classes.iter().map(|class| (class.dtype, class.layers.raw)).collect();
    assert_eq!(classes.get(&Some(Dtype::F16)), Some(&TENSOR_BYTES));
    assert_eq!(classes.get(&Some(Dtype::U8)), Some(&TENSOR_BYTES));
    assert_eq!(classes.get(&Some(Dtype::F8E4M3)), Some(&TENSOR_BYTES));
    assert_eq!(
        classes.get(&None).copied(),
        Some(fixture.bytes.len() as u64 - 3 * TENSOR_BYTES),
        "the rest is the header"
    );
    let stored = report.classes.iter().find(|class| class.dtype == Some(Dtype::U8));
    let stored = stored.expect("listed").layers;
    assert_eq!(stored.entropy_saved, 0, "U8 is stored");
    assert!(stored.overhead > 0, "each stored chunk has a prologue");
    let floats = report.classes.iter().find(|class| class.dtype == Some(Dtype::F16));
    assert!(floats.expect("listed").layers.entropy_saved > 0);
}

#[test]
fn a_stored_chunk_compresses_only_where_the_policy_says() {
    // The stream words of a `fast` quantized blob and of a `balanced` one, on bytes zstd can shrink.
    let fixture = gguf_of(Elem::Q4K);
    let low: Vec<u8> = tensor_bytes(&fixture).iter().map(|byte| byte & 1).collect();
    let fixture = with_tensor_bytes(fixture, &low);
    let compressed = |mode: Mode| -> bool {
        let scratch = Scratch::new();
        let (output, _) = encode_and_prove(&scratch, &fixture.bytes, mode);
        let (_, index) = index_of(&output);
        let file = fs::read(&output).expect("output");
        let chunks = stored_chunks(&file, &index, data_blob(&index));
        let streams = index.chain(data_blob(&index).chain_id).expect("valid").stream_count();
        chunks.iter().any(|chunk| {
            let (words, _) = chunk[..4 * streams].as_chunks::<4>();
            words.iter().any(|word| word[3] == STREAM_ZSTD)
        })
    };
    assert!(!compressed(Mode::Fast));
    assert!(compressed(Mode::Balanced));
}

#[test]
fn names_in_a_pack_stay_unique_and_hold_nested_paths() {
    let scratch = Scratch::new();
    let names = ["dir/a", "dir/b", "c"];
    let files: Vec<(&str, &[u8])> = names.iter().map(|name| (*name, b"x".as_slice())).collect();
    fs::create_dir(scratch.path("dir")).expect("scratch directory");
    let (output, _) = pack(&scratch, Mode::Fast, &files);
    let container = Container::open(&output, &OpenOptions::default()).expect("opens");
    let found: HashSet<&str> = container.members().iter().map(|m| m.name.as_str()).collect();
    assert_eq!(found, names.into_iter().collect());
}

/// Names of the base tensors of [`finetune_pair`], as the base publishes them, with their dtype and
/// what the finetune does to each.
const DELTA_TENSORS: [(&str, Elem, Edit); 4] = [
    ("double_blocks.0.img_attn.proj.weight", Elem::Bf16, Edit::Sparse),
    ("double_blocks.0.img_attn.norm.query_norm.scale", Elem::Bf16, Edit::None),
    ("single_blocks.0.linear1.weight", Elem::Bf16, Edit::Zero),
    ("img_in.weight", Elem::F32, Edit::Sparse),
];

/// What a finetune of [`finetune_pair`] did to a base tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Edit {
    /// Left it as it was: a dedup hit, no delta blob.
    None,
    /// Flipped the lowest bit of every 14th byte: a sparse residual, so a delta blob.
    Sparse,
    /// Replaced it with zeros: alone it stores in almost nothing, against the base it would not.
    Zero,
}

/// The finetune's name of a base tensor: another prefix, and RMS norm scales called `weight`.
fn finetune_name(base: &str) -> String {
    let name = base
        .strip_suffix("norm.scale")
        .map_or_else(|| base.to_owned(), |head| format!("{head}norm.weight"));
    format!("model.diffusion_model.{name}")
}

/// A base file and its finetune: the same seed draws the same tensors, under the finetune's names,
/// then [`DELTA_TENSORS`] says what changes.
fn finetune_pair() -> (Fixture, Fixture) {
    let base_specs: Vec<TensorSpec> =
        DELTA_TENSORS.iter().map(|(name, elem, _)| spec(name, *elem)).collect();
    let tuned_specs: Vec<TensorSpec> = DELTA_TENSORS
        .iter()
        .map(|(name, elem, _)| spec(&finetune_name(name), *elem))
        .collect();
    let base = fixtures::safetensors(31, &base_specs);
    let mut tuned = fixtures::safetensors(31, &tuned_specs);
    for (tensor, (_, _, edit)) in tuned.tensors.iter().zip(DELTA_TENSORS) {
        let start = usize::try_from(tensor.offset).expect("fits");
        let bytes = &mut tuned.bytes[start..start + usize::try_from(tensor.len).expect("fits")];
        match edit {
            Edit::None => {}
            Edit::Sparse => bytes.iter_mut().step_by(14).for_each(|byte| *byte ^= 1),
            Edit::Zero => bytes.fill(0),
        }
    }
    (base, tuned)
}

/// Packs `base` as "base", then `tuned` as "tuned" with "base" as its delta base, under `mode`, and
/// proves both members against their bytes.
fn pack_delta(
    scratch: &Scratch,
    mode: Mode,
    threads: usize,
    base: &[u8],
    tuned: &[u8],
) -> (PathBuf, EncodeReport) {
    let output = scratch.path(&format!("{mode:?}-{threads}.delta"));
    let base_path = scratch.file("base.safetensors", base);
    let tuned_path = scratch.file("tuned.safetensors", tuned);
    let settings = EncodeOptions { mode, threads };
    let mut builder = PackBuilder::new(&output, &settings).expect("starts");
    builder.add_file(&base_path, "base").expect("adds");
    builder.add_file_delta(&tuned_path, "tuned", "base").expect("adds against the base");
    let report = builder.finish().expect("commits");
    assert_container_invariants(&output, &report);
    let container = Container::open(&output, &OpenOptions::default()).expect("opens");
    for (name, bytes) in [("base", base), ("tuned", tuned)] {
        let member = container.member(name).expect("present");
        let mut read = vec![0; bytes.len()];
        member.read_at(0, &mut read).expect("reads");
        assert!(read == bytes, "member {name} differs under {mode:?}");
    }
    (output, report)
}

/// Required feature bits of the container at `path`.
fn required_features(path: &Path) -> u64 {
    let bytes = fs::read(path).expect("container");
    let header: [u8; 64] = bytes[..64].try_into().expect("64 bytes");
    parse_header(&header).expect("valid header").required_features
}

/// The blob of the extent of member `member` that starts at `offset`.
fn blob_at(index: &Index, member: usize, offset: u64) -> &BlobRecord {
    let extents = index.member_extents(&index.members()[member]).expect("in range");
    let extent = extents.iter().find(|extent| extent.virt_offset == offset).expect("an extent there");
    &index.blobs()[extent.blob_id as usize]
}

/// The id of `blob` in `index`.
fn id_of(index: &Index, blob: &BlobRecord) -> u32 {
    let at = index.blobs().iter().position(|other| std::ptr::eq(other, blob)).expect("in the index");
    u32::try_from(at).expect("fits")
}

#[test]
fn a_finetune_stores_changed_tensors_as_delta_blobs_against_its_base() {
    let (base, tuned) = finetune_pair();
    let scratch = Scratch::new();
    let (output, report) = pack_delta(&scratch, Mode::Max, 3, &base.bytes, &tuned.bytes);
    assert_eq!(report.members[0].delta_blobs, 0);
    assert_eq!(report.members[1].delta_blobs, 2, "the two sparse edits");
    assert_eq!(required_features(&output), FEATURE_DELTA);

    let (_, index) = index_of(&output);
    let deltas = index.blobs().iter().filter(|blob| blob.kind == BLOB_KIND_DELTA).count();
    assert_eq!(deltas, 2);
    for ((base_tensor, tuned_tensor), (_, _, edit)) in
        base.tensors.iter().zip(&tuned.tensors).zip(DELTA_TENSORS)
    {
        let base_blob = blob_at(&index, 0, base_tensor.offset);
        let tuned_blob = blob_at(&index, 1, tuned_tensor.offset);
        assert_eq!(base_blob.kind, BLOB_KIND_CHUNKED);
        match edit {
            Edit::None => {
                assert!(std::ptr::eq(base_blob, tuned_blob), "{}: dedup", base_tensor.name);
            }
            Edit::Sparse => {
                assert_eq!(tuned_blob.kind, BLOB_KIND_DELTA, "{}", base_tensor.name);
                let params = index.delta_params(tuned_blob).expect("valid");
                assert_eq!(params.base_blob, id_of(&index, base_blob), "{}", base_tensor.name);
                assert_eq!(params.base_offset, 0);
                assert_eq!(tuned_blob.chunk_size, base_blob.chunk_size);
                assert!(index.chain(tuned_blob.chain_id).expect("valid").delta().is_some());
            }
            Edit::Zero => {
                assert_eq!(tuned_blob.kind, BLOB_KIND_CHUNKED, "{}: delta loses", base_tensor.name);
                assert!(!std::ptr::eq(base_blob, tuned_blob));
            }
        }
    }

    // The tensors read back by their own names, and the delta saves against the finetune alone.
    let container = Container::open(&output, &OpenOptions::default()).expect("opens");
    assert_tensors_read_back(&container.member("tuned").expect("present"), &tuned);
    let alone = Scratch::new();
    let (_, single) = encode_and_prove(&alone, &tuned.bytes, Mode::Max);
    let tuned_stored = report.members[1].layers.stored;
    assert!(tuned_stored * 2 < single.total.stored, "{tuned_stored} against {}", single.total.stored);
    let (total, classes) = layer_report(&container).expect("reports");
    assert_eq!(total, report.total);
    assert_eq!(classes, report.classes);
}

#[test]
fn the_delta_member_restores_and_verifies_on_its_own() {
    let (base, tuned) = finetune_pair();
    let scratch = Scratch::new();
    let (output, _) = pack_delta(&scratch, Mode::Max, 2, &base.bytes, &tuned.bytes);
    let container = Container::open(&output, &OpenOptions::default()).expect("opens");
    let member = container.member("tuned").expect("present");
    member.verify().expect("verifies");
    let restored = scratch.path("tuned.restored");
    member.restore_file(&restored).expect("restores");
    assert!(fs::read(&restored).expect("restored") == tuned.bytes);
    for tensor in &tuned.tensors {
        // A read that starts and ends inside chunks of a delta blob.
        let mut window = vec![0; 3000];
        let offset = tensor.offset + (1 << 20) - 1000;
        member.read_at(offset, &mut window).expect("reads");
        let start = usize::try_from(offset).expect("fits");
        assert!(window == tuned.bytes[start..start + 3000], "{}", tensor.name);
    }
}

#[test]
fn delta_output_does_not_depend_on_the_thread_count() {
    let (base, tuned) = finetune_pair();
    let scratch = Scratch::new();
    let outputs: Vec<Vec<u8>> = [1, 2, 5]
        .into_iter()
        .map(|threads| {
            let (output, _) = pack_delta(&scratch, Mode::Max, threads, &base.bytes, &tuned.bytes);
            fs::read(output).expect("output")
        })
        .collect();
    assert!(outputs.windows(2).all(|pair| pair[0] == pair[1]));
}

#[test]
fn modes_other_than_max_ignore_the_base_and_set_no_feature() {
    let (base, tuned) = finetune_pair();
    for mode in [Mode::Fast, Mode::Balanced] {
        let scratch = Scratch::new();
        let (output, report) = pack_delta(&scratch, mode, 2, &base.bytes, &tuned.bytes);
        assert_eq!(report.members[1].delta_blobs, 0, "under {mode:?}");
        assert_eq!(required_features(&output), 0, "under {mode:?}");
        let (_, index) = index_of(&output);
        assert!(index.blobs().iter().all(|blob| blob.kind == BLOB_KIND_CHUNKED));
        assert!(index.params().is_empty());
    }
    // A max pack without a base sets no feature either.
    let scratch = Scratch::new();
    let files = [("a", base.bytes.as_slice()), ("b", tuned.bytes.as_slice())];
    let (output, _) = pack(&scratch, Mode::Max, &files);
    assert_eq!(required_features(&output), 0);
}

#[test]
fn a_base_must_be_a_member_added_from_a_file_in_every_mode() {
    let (base, tuned) = finetune_pair();
    let scratch = Scratch::new();
    let base_path = scratch.file("base", &base.bytes);
    let tuned_path = scratch.file("tuned", &tuned.bytes);
    let (source, _) = pack(&scratch, Mode::Max, &[("copied", base.bytes.as_slice())]);
    let container = Container::open(&source, &OpenOptions::default()).expect("opens");
    for mode in MODES {
        let output = scratch.path("out");
        let mut builder = PackBuilder::new(&output, &options(mode)).expect("starts");
        let error = builder.add_file_delta(&tuned_path, "tuned", "base").expect_err("no base yet");
        assert!(matches!(error, Error::MemberNotFound { ref name } if name == "base"), "{error:?}");
        builder.add_member_from(&container, "copied").expect("copies");
        let error = builder.add_file_delta(&tuned_path, "tuned", "copied").expect_err("not a file");
        assert!(matches!(error, Error::MemberNotFound { .. }), "{error:?}");
        builder.add_file(&base_path, "base").expect("adds");
        builder.add_file_delta(&tuned_path, "tuned", "base").expect("the builder went on");
        builder.finish().expect("commits");
    }
}

#[test]
fn a_base_file_changed_after_it_was_added_is_input_changed() {
    let (base, tuned) = finetune_pair();
    let scratch = Scratch::new();
    let base_path = scratch.file("base", &base.bytes);
    let tuned_path = scratch.file("tuned", &tuned.bytes);
    let output = scratch.path("out");
    let mut builder = PackBuilder::new(&output, &options(Mode::Max)).expect("starts");
    builder.add_file(&base_path, "base").expect("adds");
    fs::write(&base_path, &base.bytes[..base.bytes.len() - 2]).expect("rewrites");
    let error = builder.add_file_delta(&tuned_path, "tuned", "base").expect_err("base changed");
    assert!(matches!(error, Error::InputChanged { ref path } if *path == base_path), "{error:?}");
    let error = builder.finish().expect_err("the base input changed");
    assert!(matches!(error, Error::InputChanged { .. }), "{error:?}");
    assert_eq!(scratch.entries(), ["base", "tuned"]);
}

#[test]
fn a_copy_of_the_delta_member_brings_its_base_blobs() {
    let (base, tuned) = finetune_pair();
    let scratch = Scratch::new();
    let (source, _) = pack_delta(&scratch, Mode::Max, 3, &base.bytes, &tuned.bytes);
    let container = Container::open(&source, &OpenOptions::default()).expect("opens");
    let (_, original) = index_of(&source);

    let alone = scratch.path("alone");
    let mut builder = PackBuilder::new(&alone, &options(Mode::Max)).expect("starts");
    let copied = builder.add_member_from(&container, "tuned").expect("copies");
    assert_eq!(copied.delta_blobs, 2);
    let report = builder.finish().expect("commits");
    assert_container_invariants_of_a_copy(&alone, &report);
    assert_blobs_copied(&alone, &source);
    assert_eq!(required_features(&alone), FEATURE_DELTA);
    let (_, index) = index_of(&alone);
    // The tuned member's own blobs, and the two bases that no extent of it reaches.
    let member_blobs: HashSet<u32> = index
        .member_extents(&index.members()[0])
        .expect("in range")
        .iter()
        .map(|extent| extent.blob_id)
        .collect();
    assert_eq!(index.blobs().len(), member_blobs.len() + 2);
    for blob in index.blobs().iter().filter(|blob| blob.kind == BLOB_KIND_DELTA) {
        let params = index.delta_params(blob).expect("valid");
        assert!(params.base_blob < id_of(&index, blob));
        assert!(!member_blobs.contains(&params.base_blob), "a base no extent reaches");
    }
    let copy = Container::open(&alone, &OpenOptions::default()).expect("opens");
    let member = copy.member("tuned").expect("present");
    member.verify().expect("verifies");
    assert_tensors_read_back(&member, &tuned);
    let (total, classes) = layer_report(&copy).expect("reports");
    assert_eq!(total, report.total);
    assert_eq!(classes, report.classes);

    // Both members, in either order, need no blob twice: the bases are reused by the source's id.
    for order in [["base", "tuned"], ["tuned", "base"]] {
        let both = scratch.path("both");
        let mut builder = PackBuilder::new(&both, &options(Mode::Max)).expect("starts");
        for name in order {
            builder.add_member_from(&container, name).expect("copies");
        }
        let report = builder.finish().expect("commits");
        assert_container_invariants_of_a_copy(&both, &report);
        let (_, index) = index_of(&both);
        assert_eq!(index.blobs().len(), original.blobs().len(), "{order:?}");
        let stored: u64 = original.chunks().iter().map(|chunk| u64::from(chunk.stored_len)).sum();
        let copied: u64 = report.members.iter().map(|member| member.copied_bytes).sum();
        assert_eq!(copied, stored, "{order:?}");
        let copy = Container::open(&both, &OpenOptions::default()).expect("opens");
        copy.member("tuned").expect("present").verify().expect("verifies");
        let (total, _) = layer_report(&copy).expect("reports");
        assert_eq!(total, report.total, "{order:?}");
    }
}

/// Bytes of a BF16 tensor that are also whole `Q8_0` blocks, and its `Q8_0` sibling in a GGUF file
/// holding those very bytes: one identity, two chunk sizes (1 MiB and 1 048 560).
fn q8_sibling(bytes: &[u8]) -> Fixture {
    let blocks = bytes.len() as u64 / 34;
    let specs = [TensorSpec::new("q8", &[blocks * 32], Elem::Q80)];
    let gguf = fixtures::gguf(32, &specs, &GgufOptions::default());
    with_tensor_bytes(gguf, bytes)
}

/// A base of one BF16 tensor of whole `Q8_0` blocks and its sparse finetune.
fn q8_sized_pair() -> (Fixture, Fixture) {
    let elements = 34 * 40_000;
    let base = fixtures::safetensors(33, &[TensorSpec::new("w", &[elements], Elem::Bf16)]);
    let mut tuned = base.clone();
    let start = usize::try_from(tuned.tensors[0].offset).expect("fits");
    let end = start + usize::try_from(tuned.tensors[0].len).expect("fits");
    tuned.bytes[start..end].iter_mut().step_by(14).for_each(|byte| *byte ^= 1);
    (base, tuned)
}

#[test]
fn a_base_blob_of_another_chunk_size_gets_no_delta_blob() {
    let (base, tuned) = q8_sized_pair();
    let sibling = q8_sibling(tensor_bytes(&base));
    let scratch = Scratch::new();
    let paths = [
        scratch.file("q8.gguf", &sibling.bytes),
        scratch.file("base", &base.bytes),
        scratch.file("tuned", &tuned.bytes),
    ];
    let output = scratch.path("out");
    let mut builder = PackBuilder::new(&output, &options(Mode::Max)).expect("starts");
    builder.add_file(&paths[0], "q8.gguf").expect("adds");
    let added = builder.add_file(&paths[1], "base").expect("adds");
    assert_eq!(added.layers.dedup_saved, tensor_bytes(&base).len() as u64, "the Q8_0 blob serves");
    let report = builder.add_file_delta(&paths[2], "tuned", "base").expect("adds");
    assert_eq!(report.delta_blobs, 0, "the base's blob has chunks of 1 048 560 bytes");
    let report = builder.finish().expect("commits");
    assert_container_invariants(&output, &report);
    assert_eq!(required_features(&output), 0);
    let container = Container::open(&output, &OpenOptions::default()).expect("opens");
    container.member("tuned").expect("present").verify().expect("verifies");
}

#[test]
fn a_copy_rests_a_delta_blob_only_on_a_base_of_the_same_chunk_size() {
    let (base, tuned) = q8_sized_pair();
    let scratch = Scratch::new();
    let (source, _) = pack_delta(&scratch, Mode::Max, 2, &base.bytes, &tuned.bytes);
    let container = Container::open(&source, &OpenOptions::default()).expect("opens");
    let identity = *blake3::hash(tensor_bytes(&base)).as_bytes();

    let sibling = q8_sibling(tensor_bytes(&base));
    let merged = scratch.path("merged");
    let mut builder = PackBuilder::new(&merged, &options(Mode::Max)).expect("starts");
    builder.add_file(&scratch.file("q8.gguf", &sibling.bytes), "q8.gguf").expect("adds");
    let copied = builder.add_member_from(&container, "tuned").expect("copies");
    assert_eq!(copied.delta_blobs, 1);
    let report = builder.finish().expect("commits");
    assert_container_invariants(&merged, &report);
    let (_, index) = index_of(&merged);
    let same = index.blobs().iter().filter(|blob| blob.blake3 == identity).count();
    assert_eq!(same, 2, "the Q8_0 blob does not serve, so the base was copied");
    let delta = index.blobs().iter().find(|blob| blob.kind == BLOB_KIND_DELTA).expect("copied");
    let base_blob = &index.blobs()[index.delta_params(delta).expect("valid").base_blob as usize];
    assert_eq!(base_blob.blake3, identity);
    assert_eq!(base_blob.chunk_size, delta.chunk_size);
    let copy = Container::open(&merged, &OpenOptions::default()).expect("opens");
    copy.member("tuned").expect("present").verify().expect("verifies");
}

#[test]
fn a_copy_rests_a_delta_blob_on_a_base_this_pack_made_itself() {
    let (base, tuned) = finetune_pair();
    let scratch = Scratch::new();
    let (source, _) = pack_delta(&scratch, Mode::Max, 2, &base.bytes, &tuned.bytes);
    let (_, original) = index_of(&source);
    let container = Container::open(&source, &OpenOptions::default()).expect("opens");

    let merged = scratch.path("merged");
    let mut builder = PackBuilder::new(&merged, &options(Mode::Max)).expect("starts");
    builder.add_file(&scratch.file("again", &base.bytes), "again").expect("adds");
    let copied = builder.add_member_from(&container, "tuned").expect("copies");
    assert_eq!(copied.delta_blobs, 2);
    // Its header, two delta blobs and the zero tensor: the bases and the unchanged tensor are here.
    assert_eq!(copied.blobs_added, 4, "{copied:?}");
    let report = builder.finish().expect("commits");
    assert_container_invariants(&merged, &report);
    let (_, index) = index_of(&merged);
    assert_eq!(index.blobs().len(), original.blobs().len(), "no base was copied");
    let again = index.members()[0];
    let made: HashSet<u32> = index
        .member_extents(&again)
        .expect("in range")
        .iter()
        .map(|extent| extent.blob_id)
        .collect();
    for blob in index.blobs().iter().filter(|blob| blob.kind == BLOB_KIND_DELTA) {
        let base_blob = index.delta_params(blob).expect("valid").base_blob;
        assert!(made.contains(&base_blob), "the base is a blob of the member added from a file");
    }
    let copy = Container::open(&merged, &OpenOptions::default()).expect("opens");
    let member = copy.member("tuned").expect("present");
    member.verify().expect("verifies");
    assert_tensors_read_back(&member, &tuned);
    let (total, classes) = layer_report(&copy).expect("reports");
    assert_eq!(total, report.total);
    assert_eq!(classes, report.classes);
}

#[test]
fn a_finetune_of_a_finetune_rests_only_on_plain_blobs() {
    let specs = [spec("a", Elem::Bf16), spec("b", Elem::Bf16)];
    let base = fixtures::safetensors(34, &specs);
    let edit = |fixture: &mut Fixture, tensor: usize, step: usize| {
        let at = &fixture.tensors[tensor];
        let start = usize::try_from(at.offset).expect("fits");
        let end = start + usize::try_from(at.len).expect("fits");
        fixture.bytes[start..end].iter_mut().step_by(step).for_each(|byte| *byte ^= 1);
    };
    // `a` is a delta blob of the first finetune, so the second cannot rest on it; `b` the first left
    // as it was, an extent over the base's own blob, which the second can rest on.
    let mut first = base.clone();
    edit(&mut first, 0, 14);
    let mut second = first.clone();
    edit(&mut second, 0, 22);
    edit(&mut second, 1, 14);
    let scratch = Scratch::new();
    let paths = [
        scratch.file("base", &base.bytes),
        scratch.file("first", &first.bytes),
        scratch.file("second", &second.bytes),
    ];
    let output = scratch.path("out");
    let mut builder = PackBuilder::new(&output, &options(Mode::Max)).expect("starts");
    builder.add_file(&paths[0], "base").expect("adds");
    assert_eq!(builder.add_file_delta(&paths[1], "first", "base").expect("adds").delta_blobs, 1);
    assert_eq!(builder.add_file_delta(&paths[2], "second", "first").expect("adds").delta_blobs, 1);
    let report = builder.finish().expect("commits");
    assert_container_invariants(&output, &report);
    assert_eq!(required_features(&output), FEATURE_DELTA);

    let (_, index) = index_of(&output);
    let a_of = |member: usize, fixture: &Fixture| blob_at(&index, member, fixture.tensors[0].offset);
    let b_of = |member: usize, fixture: &Fixture| blob_at(&index, member, fixture.tensors[1].offset);
    assert_eq!(a_of(1, &first).kind, BLOB_KIND_DELTA);
    assert_eq!(a_of(2, &second).kind, BLOB_KIND_CHUNKED, "its paired blob is a delta blob");
    let b = b_of(2, &second);
    assert_eq!(b.kind, BLOB_KIND_DELTA);
    let rests_on = index.delta_params(b).expect("valid").base_blob;
    assert_eq!(rests_on, id_of(&index, b_of(0, &base)), "the base's blob, through the first");
    let container = Container::open(&output, &OpenOptions::default()).expect("opens");
    for (name, fixture) in [("base", &base), ("first", &first), ("second", &second)] {
        let member = container.member(name).expect("present");
        member.verify().expect("verifies");
        assert_tensors_read_back(&member, fixture);
    }
}

#[test]
fn blobs_a_failed_copy_left_behind_count_in_the_report() {
    let (base, _) = finetune_pair();
    let scratch = Scratch::new();
    let (source, _) = pack(&scratch, Mode::Max, &[("base", base.bytes.as_slice())]);
    let container = Container::open(&source, &OpenOptions::default()).expect("opens");
    // Cut the source inside the last tensor's blob: the copy takes the blobs before it and fails.
    let (_, original) = index_of(&source);
    let last = blob_at(&original, 0, base.tensors[3].offset);
    let cut = original.blob_chunks(last).expect("in range")[0].file_offset + 10;
    fs::OpenOptions::new().write(true).open(&source).expect("opens").set_len(cut).expect("cuts");

    let merged = scratch.path("merged");
    let mut builder = PackBuilder::new(&merged, &options(Mode::Max)).expect("starts");
    builder.add_member_from(&container, "base").expect_err("the source is cut");
    let input = scratch.file("other", &[7; 5000]);
    builder.add_file(&input, "other").expect("the builder went on");
    let report = builder.finish().expect("commits");
    // The data region holds the blobs of the failed copy, and the report says so.
    assert_container_invariants(&merged, &report);
    assert!(report.total.overhead > report.members[0].layers.overhead);
    let copy = Container::open(&merged, &OpenOptions::default()).expect("opens");
    let (total, classes) = layer_report(&copy).expect("reports");
    assert_eq!(total, report.total);
    assert_eq!(classes, report.classes);
}
