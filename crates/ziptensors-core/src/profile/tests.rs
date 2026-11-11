//! File: profile/tests.rs
//! Purpose: tests of `scan`, `normalize` and `classify_weight` over generated files of every
//! format.
//!
//! Main responsibilities: each parser finds what its builder put in; malformed, truncated, flipped
//! and hostile input gives an opaque or in-bounds plan and never an `Err` or a panic; only a failed
//! read is an `Err`.
//!
//! Implementation notes: the builders are in `tests/common/fixtures.rs`. No test reads a real
//! model, and the single temporary file goes under the system temp directory. Legacy torch
//! prefixes are written out as bytes here; nothing depends on torch.

#[path = "../../tests/common/fixtures.rs"]
mod fixtures;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use fixtures::{
    Elem, Fixture, GgufOptions, Msg, OnnxModel, OnnxStorage, OnnxTensor, PtZipFlaw,
    PtZipOptions, Rng, TensorSpec,
};

use super::{MIN_RANGE_BYTES, WeightKind, classify_weight, legacy_torch, normalize, scan};
use crate::error::Error;
use crate::io::{FileSource, RandomAccess};
use crate::types::{Dtype, Profile, RangePlan};

/// A source held in memory.
struct Memory(Vec<u8>);

impl RandomAccess for Memory {
    fn len(&self) -> u64 {
        u64::try_from(self.0.len()).expect("test buffer fits u64")
    }

    fn read_exact_at(&self, off: u64, buf: &mut [u8]) -> Result<(), Error> {
        let start = usize::try_from(off).expect("test offset fits usize");
        buf.copy_from_slice(&self.0[start..start + buf.len()]);
        Ok(())
    }
}

/// A vast source: `head` at the start, `tail` at the end, zeros between. Counts bytes served.
struct Vast {
    len: u64,
    head: Vec<u8>,
    tail: Vec<u8>,
    served: AtomicU64,
    largest: AtomicU64,
}

impl Vast {
    fn new(len: u64, head: Vec<u8>, tail: Vec<u8>) -> Self {
        Self { len, head, tail, served: AtomicU64::new(0), largest: AtomicU64::new(0) }
    }

    fn byte_at(&self, pos: u64) -> u8 {
        let tail_start = self.len - u64::try_from(self.tail.len()).expect("fits");
        if pos >= tail_start {
            let index = usize::try_from(pos - tail_start).expect("inside the tail");
            return self.tail[index];
        }
        usize::try_from(pos).ok().and_then(|index| self.head.get(index)).copied().unwrap_or(0)
    }
}

impl RandomAccess for Vast {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_exact_at(&self, off: u64, buf: &mut [u8]) -> Result<(), Error> {
        let len = u64::try_from(buf.len()).expect("fits");
        self.served.fetch_add(len, Ordering::Relaxed);
        self.largest.fetch_max(len, Ordering::Relaxed);
        for (byte, pos) in buf.iter_mut().zip(off..) {
            *byte = self.byte_at(pos);
        }
        Ok(())
    }
}

/// A source whose reads past `good` bytes fail.
struct Failing {
    bytes: Vec<u8>,
    good: u64,
}

impl RandomAccess for Failing {
    fn len(&self) -> u64 {
        u64::try_from(self.bytes.len()).expect("fits")
    }

    fn read_exact_at(&self, off: u64, buf: &mut [u8]) -> Result<(), Error> {
        let end = off + u64::try_from(buf.len()).expect("fits");
        if end > self.good {
            return Err(Error::Io {
                op: "read_exact_at",
                path: PathBuf::from("mock"),
                source: std::io::ErrorKind::Other.into(),
            });
        }
        let start = usize::try_from(off).expect("fits");
        buf.copy_from_slice(&self.bytes[start..start + buf.len()]);
        Ok(())
    }
}

fn scan_bytes(bytes: &[u8]) -> RangePlan {
    scan(&Memory(bytes.to_vec())).expect("malformed content is not an error")
}

fn assert_opaque(bytes: &[u8]) {
    let plan = scan_bytes(bytes);
    assert_eq!(plan.profile, Profile::Opaque);
    assert_eq!(plan.ranges, []);
    assert_eq!(plan.tensors, []);
}

/// Asserts that `plan` reports exactly the tensors of `fixture`, as rows, and as ranges where the
/// type has one.
fn assert_found(plan: &RangePlan, profile: Profile, fixture: &Fixture) {
    assert_eq!(plan.profile, profile);
    let rows: HashMap<&str, _> = plan.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    assert_eq!(rows.len(), fixture.tensors.len(), "{:?}", plan.tensors);
    let typed = fixture.tensors.iter().filter(|want| has_range(want.dtype_wire)).count();
    assert_eq!(plan.ranges.len(), typed, "{:?}", plan.ranges);
    for want in &fixture.tensors {
        let row = rows.get(want.name.as_str()).expect("the tensor is in the plan");
        let dtype = Dtype::from_wire(want.dtype_wire);
        assert_eq!((row.offset, row.len), (want.offset, want.len), "{}", want.name);
        assert_eq!(row.shape, want.shape, "{}", want.name);
        assert_eq!(row.dtype, dtype, "{}", want.name);
        let found = plan
            .ranges
            .iter()
            .any(|r| (r.offset, r.len, r.hint) == (want.offset, want.len, dtype));
        assert_eq!(found, has_range(want.dtype_wire), "range of {}", want.name);
    }
}

/// Whether a tensor of dtype `wire` gets a range. The ggml block types without a field table,
/// wire ids 0x0034 to 0x0047, get a row only: their bytes stay in the skeleton.
fn has_range(wire: u16) -> bool {
    !(0x0034..=0x0047).contains(&wire)
}

/// Every plan a scan can give, once normalized, must be fit for a writer.
fn assert_usable(bytes: &[u8]) {
    let file_len = u64::try_from(bytes.len()).expect("fits");
    let plan = normalize(scan_bytes(bytes), file_len);
    let mut end = 0;
    for range in &plan.ranges {
        assert!(range.offset >= end, "{:?}", plan.ranges);
        end = range.offset + range.len;
        assert!(end <= file_len);
        assert!(range.len >= MIN_RANGE_BYTES);
    }
    let mut names = std::collections::HashSet::new();
    for tensor in &plan.tensors {
        assert!(tensor.offset + tensor.len <= file_len);
        assert!(tensor.shape.len() <= 16);
        assert!(names.insert(tensor.name.as_str()));
    }
}

fn every_truncation(bytes: &[u8]) {
    for cut in 0..bytes.len() {
        assert_usable(&bytes[..cut]);
    }
}

fn every_byte_flipped(bytes: &[u8]) {
    let mut copy = bytes.to_vec();
    for index in 0..bytes.len() {
        copy[index] ^= 0xFF;
        assert_usable(&copy);
        copy[index] = bytes[index];
    }
}

fn random_mutations(bytes: &[u8], seed: u64, rounds: u32) {
    let mut rng = Rng::new(seed);
    let len = u64::try_from(bytes.len()).expect("fits");
    for _ in 0..rounds {
        let mut copy = bytes.to_vec();
        for _ in 0..8 {
            let index = usize::try_from(rng.next_u64() % len).expect("fits");
            copy[index] = rng.next_u64().to_le_bytes()[0];
        }
        assert_usable(&copy);
    }
}

fn small_safetensors() -> Fixture {
    fixtures::safetensors(
        1,
        &[
            TensorSpec::new("a.weight", &[4, 4], Elem::F32),
            TensorSpec::new("b.bias", &[3], Elem::F16),
            TensorSpec::new("c.quant_state", &[16], Elem::U8),
        ],
    )
}

fn small_gguf() -> Fixture {
    fixtures::gguf(
        2,
        &[TensorSpec::new("v", &[8], Elem::F32), TensorSpec::new("q", &[32, 2], Elem::Q80)],
        &GgufOptions::default(),
    )
}

fn small_onnx() -> Fixture {
    fixtures::onnx(
        3,
        &OnnxModel {
            initializers: vec![
                OnnxTensor::raw("init", &[4], Elem::F32),
                OnnxTensor::stored("floats", &[4], Elem::F32, OnnxStorage::PackedFloat),
            ],
            constants: vec![OnnxTensor::raw("const", &[2], Elem::I64)],
            then_branch: vec![OnnxTensor::raw("then", &[2], Elem::F16)],
            else_branch: vec![OnnxTensor::raw("else", &[2], Elem::U8)],
            loop_body: vec![OnnxTensor::raw("body", &[2], Elem::F64)],
            attributes_first: false,
        },
    )
}

fn small_pt_zip() -> Fixture {
    fixtures::pt_zip(
        4,
        &[TensorSpec::new("", &[16], Elem::F32), TensorSpec::new("", &[32], Elem::U8)],
        &PtZipOptions::default(),
    )
}

fn small_pt_zip_with(options: &PtZipOptions) -> Vec<u8> {
    fixtures::pt_zip(4, &[TensorSpec::new("", &[16], Elem::F32)], options).bytes
}

fn deflated_pt_zip() -> Fixture {
    let options = PtZipOptions { deflated_code: true, ..PtZipOptions::default() };
    fixtures::pt_zip(5, &[TensorSpec::new("", &[16], Elem::F32)], &options)
}

#[test]
fn safetensors_ranges_match_the_builder_for_every_tag() {
    let elems = [
        Elem::Bool,
        Elem::U8,
        Elem::I8,
        Elem::U16,
        Elem::I16,
        Elem::U32,
        Elem::I32,
        Elem::U64,
        Elem::I64,
        Elem::F16,
        Elem::Bf16,
        Elem::F32,
        Elem::F64,
        Elem::F8E4M3,
        Elem::F8E5M2,
    ];
    let specs: Vec<TensorSpec> = elems
        .iter()
        .enumerate()
        .map(|(index, &elem)| TensorSpec::new(&format!("t{index}"), &[3, 5], elem))
        .collect();
    let fixture = fixtures::safetensors(10, &specs);
    assert_found(&scan_bytes(&fixture.bytes), Profile::Safetensors, &fixture);
}

#[test]
fn safetensors_keeps_the_declared_tag_and_ignores_metadata() {
    // Packed NF4 weights and the JSON blob beside them are tagged U8 and stay U8.
    let fixture = fixtures::safetensors(
        11,
        &[
            TensorSpec::new("layer.weight", &[6000, 1], Elem::U8),
            TensorSpec::new("layer.weight.quant_state.bitsandbytes__nf4", &[64], Elem::U8),
            TensorSpec::new("widened", &[1024], Elem::F32),
        ],
    );
    let plan = scan_bytes(&fixture.bytes);
    assert_found(&plan, Profile::Safetensors, &fixture);
    assert!(plan.tensors.iter().all(|tensor| tensor.name != "__metadata__"));
}

#[test]
fn safetensors_names_are_unescaped_and_empty_tensors_are_kept() {
    let fixture = fixtures::safetensors(
        12,
        &[
            TensorSpec::new("quote\"back\\slash", &[1], Elem::U8),
            TensorSpec::new("слой", &[1], Elem::U8),
            TensorSpec::new("empty", &[0, 3], Elem::F32),
            TensorSpec::new("scalar", &[], Elem::F32),
        ],
    );
    assert_found(&scan_bytes(&fixture.bytes), Profile::Safetensors, &fixture);
}

#[test]
fn safetensors_with_no_tensors_is_recognized() {
    let plan = scan_bytes(&fixtures::safetensors_raw(b"{}", &[]));
    assert_eq!(plan.profile, Profile::Safetensors);
    assert_eq!(plan.tensors, []);
    let plan = scan_bytes(&fixtures::safetensors_raw(br#"{"__metadata__":{"a":"b"}}  "#, &[]));
    assert_eq!(plan.profile, Profile::Safetensors);
}

#[test]
fn safetensors_ignores_unknown_keys_and_marks_unknown_tags() {
    let header = br#"{"t":{"extra":[1,{"x":2}],"dtype":"C64","shape":[1],"data_offsets":[0,8]}}"#;
    let plan = scan_bytes(&fixtures::safetensors_raw(header, &[0; 8]));
    assert_eq!(plan.tensors[0].dtype, Dtype::Unknown(0));
    assert_eq!(plan.ranges[0].hint, Dtype::Unknown(0));
}

#[test]
fn malformed_safetensors_headers_are_opaque() {
    let headers: [&[u8]; 18] = [
        b"{",
        b"{]",
        b"{\"a\"",
        br#"{"a":1}"#,
        br#"{"a":[]}"#,
        br#"{"a":{}}"#,
        br#"{"a":{"dtype":"F32","shape":[1]}}"#,
        br#"{"a":{"dtype":"F32","data_offsets":[0,4]}}"#,
        br#"{"a":{"shape":[1],"data_offsets":[0,4]}}"#,
        br#"{"a":{"dtype":1,"shape":[1],"data_offsets":[0,4]}}"#,
        br#"{"a":{"dtype":"F32","shape":[1.5],"data_offsets":[0,4]}}"#,
        br#"{"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}"#,
        br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#,
        br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0]}}"#,
        br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,99999999999999999999]}}"#,
        br#"{"a":{"dtype":"F32","shape":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            "data_offsets":[0,4]}}"#,
        br#"{"a":{"dtype":"F32","shape":[1],
            "data_offsets":[18446744073709551615,18446744073709551615]}}"#,
        b"{} trailing",
    ];
    for header in headers {
        assert_opaque(&fixtures::safetensors_raw(header, &[0; 32]));
    }
    assert_opaque(&fixtures::safetensors_raw(b"{\"\xFF\":{}}", &[]));
}

#[test]
fn deep_metadata_is_opaque_and_does_not_overflow_the_stack() {
    let header = format!(r#"{{"__metadata__":{}}}"#, "[".repeat(200_000));
    assert_opaque(&fixtures::safetensors_raw(header.as_bytes(), &[]));
}

#[test]
fn gguf_ranges_match_the_builder_for_every_type_and_version() {
    let elems = [
        Elem::F32,
        Elem::F16,
        Elem::Bf16,
        Elem::F64,
        Elem::I8,
        Elem::I16,
        Elem::I32,
        Elem::I64,
    ];
    let mut specs: Vec<TensorSpec> = elems
        .iter()
        .enumerate()
        .map(|(index, &elem)| TensorSpec::new(&format!("t{index}"), &[7, 3], elem))
        .collect();
    specs.push(TensorSpec::new("q8", &[64, 3], Elem::Q80));
    specs.push(TensorSpec::new("q4", &[256, 2], Elem::Q4K));
    specs.push(TensorSpec::new("q5", &[512], Elem::Q5K));
    specs.push(TensorSpec::new("q6", &[256, 2, 2], Elem::Q6K));
    for version in [2, 3] {
        for alignment in [None, Some(32), Some(64), Some(4096)] {
            let options = GgufOptions { version, alignment, ..GgufOptions::default() };
            let fixture = fixtures::gguf(20, &specs, &options);
            assert_found(&scan_bytes(&fixture.bytes), Profile::Gguf, &fixture);
        }
    }
}

#[test]
fn gguf_padding_stays_out_of_the_ranges() {
    let fixture = fixtures::gguf(
        21,
        &[TensorSpec::new("a", &[3], Elem::F32), TensorSpec::new("b", &[3], Elem::F32)],
        &GgufOptions { alignment: Some(64), ..GgufOptions::default() },
    );
    let plan = scan_bytes(&fixture.bytes);
    assert_found(&plan, Profile::Gguf, &fixture);
    assert_eq!(plan.ranges[1].offset - plan.ranges[0].offset, 64);
    assert_eq!(plan.ranges[0].len, 12);
}

#[test]
fn gguf_tensor_of_an_unmapped_type_stays_in_the_skeleton() {
    let fixture = fixtures::gguf(
        22,
        &[TensorSpec::new("a", &[3], Elem::F32)],
        &GgufOptions { unmapped_tensor: true, ..GgufOptions::default() },
    );
    assert_found(&scan_bytes(&fixture.bytes), Profile::Gguf, &fixture);
}

#[test]
fn gguf_rows_without_whole_blocks_are_opaque() {
    let mut fixture = fixtures::gguf(
        23,
        &[TensorSpec::new("q", &[32], Elem::Q80)],
        &GgufOptions::default(),
    );
    let needle = 32_u64.to_le_bytes();
    let at = fixture
        .bytes
        .windows(8)
        .position(|window| window == needle)
        .expect("the dimension is in the tensor info");
    fixture.bytes[at] = 33;
    assert_opaque(&fixture.bytes);
}

#[test]
fn gguf_block_types_without_a_field_table_get_rows_and_no_range() {
    let specs: Vec<TensorSpec> = Elem::UNSPLIT_GGML
        .iter()
        .enumerate()
        .map(|(index, &elem)| {
            let (elements, _) = elem.block();
            TensorSpec::new(&format!("u{index}"), &[elements * 2, 3], elem)
        })
        .chain([TensorSpec::new("f", &[7, 3], Elem::F32)])
        .collect();
    for alignment in [None, Some(64)] {
        let options = GgufOptions { alignment, ..GgufOptions::default() };
        let fixture = fixtures::gguf(24, &specs, &options);
        let plan = scan_bytes(&fixture.bytes);
        assert_found(&plan, Profile::Gguf, &fixture);
        assert_eq!(plan.ranges.len(), 1, "only the F32 tensor is typed");
        for (want, elem) in fixture.tensors.iter().zip(Elem::UNSPLIT_GGML) {
            let (elements, block) = elem.block();
            assert_eq!(want.len, 2 * 3 * block, "{elem:?}");
            assert_eq!(want.shape, [elements * 2, 3], "{elem:?}");
        }
    }
}

#[test]
fn gguf_block_types_without_whole_blocks_are_opaque() {
    for elem in Elem::UNSPLIT_GGML {
        let (elements, _) = elem.block();
        let mut fixture = fixtures::gguf(
            25,
            &[TensorSpec::new("q", &[elements * 3, 5], elem)],
            &GgufOptions::default(),
        );
        let needle = (elements * 3).to_le_bytes();
        let at = fixture
            .bytes
            .windows(8)
            .position(|window| window == needle)
            .expect("the dimension is in the tensor info");
        fixture.bytes[at..at + 8].copy_from_slice(&(elements * 3 + 1).to_le_bytes());
        assert_opaque(&fixture.bytes);
    }
}

/// A GGUF head with the given version and counts, and `rest` after it.
fn gguf_head(version: u32, tensors: u64, pairs: u64, rest: &[u8]) -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&version.to_le_bytes());
    bytes.extend_from_slice(&tensors.to_le_bytes());
    bytes.extend_from_slice(&pairs.to_le_bytes());
    bytes.extend_from_slice(rest);
    bytes
}

/// The key-value bytes of one `general.alignment` of `value_type`.
fn gguf_alignment(value_type: u32, value: &[u8]) -> Vec<u8> {
    let mut bytes = 17_u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"general.alignment");
    bytes.extend_from_slice(&value_type.to_le_bytes());
    bytes.extend_from_slice(value);
    bytes
}

#[test]
fn malformed_gguf_headers_are_opaque() {
    assert_opaque(b"GGUF");
    for version in [0, 1, 4, 0x0300_0000] {
        assert_opaque(&gguf_head(version, 0, 0, &[]));
    }
    assert_opaque(&gguf_head(3, u64::MAX, 0, &[]));
    assert_opaque(&gguf_head(3, 0, u64::MAX, &[]));
    assert_opaque(&gguf_head(3, 1, 0, &[0; 8]));
    assert_opaque(&gguf_head(3, 0, 1, &gguf_alignment(4, &0_u32.to_le_bytes())));
    assert_opaque(&gguf_head(3, 0, 1, &gguf_alignment(4, &3_u32.to_le_bytes())));
    assert_opaque(&gguf_head(3, 0, 1, &gguf_alignment(10, &32_u64.to_le_bytes())));

    let mut unknown_type = 1_u64.to_le_bytes().to_vec();
    unknown_type.extend_from_slice(b"k");
    unknown_type.extend_from_slice(&99_u32.to_le_bytes());
    assert_opaque(&gguf_head(3, 0, 1, &unknown_type));
}

#[test]
fn gguf_with_no_tensors_is_recognized() {
    assert_eq!(scan_bytes(&gguf_head(2, 0, 0, &[])).profile, Profile::Gguf);
}

#[test]
fn gguf_arrays_nested_too_deeply_are_opaque() {
    let mut value = 1_u64.to_le_bytes().to_vec();
    value.extend_from_slice(b"k");
    value.extend_from_slice(&9_u32.to_le_bytes());
    for _ in 0..64 {
        value.extend_from_slice(&9_u32.to_le_bytes());
        value.extend_from_slice(&1_u64.to_le_bytes());
    }
    assert_opaque(&gguf_head(3, 0, 1, &value));
}

#[test]
fn onnx_ranges_match_the_builder_through_nodes_and_subgraphs() {
    let fixture = small_onnx();
    assert_found(&scan_bytes(&fixture.bytes), Profile::Onnx, &fixture);
}

#[test]
fn onnx_every_element_type_with_raw_data_maps() {
    let elems = [
        Elem::F32,
        Elem::U8,
        Elem::I8,
        Elem::U16,
        Elem::I16,
        Elem::I32,
        Elem::I64,
        Elem::Bool,
        Elem::F16,
        Elem::F64,
        Elem::U32,
        Elem::U64,
        Elem::Bf16,
    ];
    let initializers = elems
        .iter()
        .enumerate()
        .map(|(index, &elem)| OnnxTensor::raw(&format!("t{index}"), &[3, 2], elem))
        .collect();
    let fixture = fixtures::onnx(30, &OnnxModel { initializers, ..OnnxModel::default() });
    assert_found(&scan_bytes(&fixture.bytes), Profile::Onnx, &fixture);
}

#[test]
fn onnx_external_and_varint_tensors_stay_in_the_skeleton() {
    let fixture = fixtures::onnx(
        31,
        &OnnxModel {
            initializers: vec![
                OnnxTensor::raw("kept", &[4], Elem::F32),
                OnnxTensor::stored("outside", &[4], Elem::F32, OnnxStorage::External),
                OnnxTensor::stored("ints", &[4], Elem::I32, OnnxStorage::Varint),
            ],
            ..OnnxModel::default()
        },
    );
    assert_eq!(fixture.tensors.len(), 1);
    assert_found(&scan_bytes(&fixture.bytes), Profile::Onnx, &fixture);
}

#[test]
fn onnx_constant_takes_its_name_from_a_later_output() {
    let fixture = fixtures::onnx(
        32,
        &OnnxModel {
            constants: vec![OnnxTensor::raw("late", &[2], Elem::F32)],
            attributes_first: true,
            ..OnnxModel::default()
        },
    );
    assert_found(&scan_bytes(&fixture.bytes), Profile::Onnx, &fixture);
}

#[test]
fn onnx_unnamed_tensor_gets_a_range_and_no_row() {
    let mut tensor = Msg::new();
    tensor.varint(2, 1).bytes_field(9, &[0; 8]);
    let mut graph = Msg::new();
    graph.message(5, tensor);
    let plan = scan_bytes(&fixtures::onnx_model_bytes(graph).bytes);
    assert_eq!(plan.profile, Profile::Onnx);
    assert_eq!(plan.ranges.len(), 1);
    assert_eq!(plan.tensors, []);
}

#[test]
fn onnx_subgraphs_nest_up_to_the_cap_and_no_further() {
    assert_eq!(scan_bytes(&fixtures::onnx_nested(64)).ranges.len(), 1);
    assert_opaque(&fixtures::onnx_nested(65));
    assert_opaque(&fixtures::onnx_nested(2000));
}

#[test]
fn onnx_lengths_that_leave_their_message_are_opaque() {
    // The inner field claims 6 bytes, and the graph holding it only has 3 bytes.
    let graph = [(5 << 3) | 2, 6, 0];
    let mut model = Msg::new();
    model.varint(1, 8).bytes_field(7, &graph).bytes_field(2, &[0; 16]);
    assert_opaque(&model.bytes);
}

#[test]
fn data_that_only_resembles_protobuf_is_opaque() {
    let mut no_graph = Msg::new();
    no_graph.varint(1, 8).bytes_field(2, b"fixture");
    assert_opaque(&no_graph.bytes);

    let mut no_version = Msg::new();
    no_version.bytes_field(7, &[]);
    assert_opaque(&no_version.bytes);

    let mut unknown_field = Msg::new();
    unknown_field.varint(1, 8).bytes_field(7, &[]).varint(99, 1);
    assert_opaque(&unknown_field.bytes);

    let mut wrong_wire = Msg::new();
    wrong_wire.varint(1, 8).varint(7, 1);
    assert_opaque(&wrong_wire.bytes);

    let mut trailing = fixtures::onnx_model_bytes(Msg::new()).bytes;
    trailing.push(0x08);
    assert_opaque(&trailing);
}

#[test]
fn legacy_pickle_and_other_bytes_are_opaque() {
    let mut pickle = vec![0x80, 0x02, 0x8A, 0x0A];
    pickle.extend_from_slice(&Rng::new(40).bytes(4096));
    assert_opaque(&pickle);
    assert_opaque(&[]);
    assert_opaque(b"x");
    assert_opaque(b"12345678");
    assert_opaque(&Rng::new(41).bytes(65536));
    assert_opaque(&vec![0; 65536]);
    assert_opaque(&vec![0xFF; 65536]);
}

fn large_storages() -> [TensorSpec; 4] {
    [
        TensorSpec::new("", &[8192], Elem::F32),
        TensorSpec::new("", &[16384], Elem::F16),
        TensorSpec::new("", &[4096], Elem::F64),
        TensorSpec::new("", &[32768], Elem::U8),
    ]
}

#[test]
fn pt_zip_storages_match_the_builder_with_width_from_the_trial() {
    let fixture = fixtures::pt_zip(50, &large_storages(), &PtZipOptions::default());
    let plan = scan_bytes(&fixture.bytes);
    assert_found(&plan, Profile::PtZip, &fixture);
    assert!(fixture.tensors.iter().all(|tensor| tensor.offset % 64 == 0));
}

#[test]
fn pt_zip_without_zip64_records_and_with_another_prefix_is_read() {
    let options = PtZipOptions {
        prefix: "best".to_owned(),
        torch_layout: false,
        alignment: 8,
        ..PtZipOptions::default()
    };
    let fixture = fixtures::pt_zip(51, &large_storages(), &options);
    assert_found(&scan_bytes(&fixture.bytes), Profile::PtZip, &fixture);
}

#[test]
fn short_pt_zip_storages_are_flat_bytes_with_no_sampling() {
    let fixture = small_pt_zip();
    assert_found(&scan_bytes(&fixture.bytes), Profile::PtZip, &fixture);
}

#[test]
fn pt_zip_flaws_are_opaque() {
    let flaws = [
        PtZipFlaw::CompressedPickle,
        PtZipFlaw::CompressedStorage,
        PtZipFlaw::Zip64Marker,
        PtZipFlaw::Zip64Disagrees,
        PtZipFlaw::LocalNameDiffers,
        PtZipFlaw::OverlappingEntries,
        PtZipFlaw::NoPickle,
        PtZipFlaw::Encrypted,
    ];
    let storages = [TensorSpec::new("", &[1024], Elem::F32)];
    for flaw in flaws {
        let options = PtZipOptions { flaw, ..PtZipOptions::default() };
        assert_opaque(&fixtures::pt_zip(52, &storages, &options).bytes);
    }
    let options = PtZipOptions {
        flaw: PtZipFlaw::LocalSizeDiffers,
        torch_layout: false,
        ..PtZipOptions::default()
    };
    assert_opaque(&fixtures::pt_zip(53, &storages, &options).bytes);
}

#[test]
fn a_deflated_code_entry_stays_in_the_skeleton_of_a_readable_archive() {
    // Sizes differ on this entry, so it also proves the layout check uses the bytes in the file.
    for torch_layout in [true, false] {
        let options = PtZipOptions { deflated_code: true, torch_layout, ..PtZipOptions::default() };
        let fixture = fixtures::pt_zip(54, &large_storages(), &options);
        let plan = scan_bytes(&fixture.bytes);
        assert_found(&plan, Profile::PtZip, &fixture);
        assert!(plan.tensors.iter().all(|tensor| !tensor.name.contains("code")));
    }
}

#[test]
fn a_stored_entry_stating_two_sizes_is_opaque() {
    let options = PtZipOptions { torch_layout: false, ..PtZipOptions::default() };
    let mut bytes = small_pt_zip_with(&options);
    let central = bytes
        .windows(4)
        .position(|window| window == b"PK\x01\x02")
        .expect("the archive has a central directory");
    bytes[central + 24] ^= 1;
    assert_opaque(&bytes);
}

#[test]
fn pt_zip_with_trailing_bytes_is_opaque() {
    let mut bytes = small_pt_zip().bytes;
    bytes.push(0);
    assert_opaque(&bytes);
}

#[test]
fn truncation_at_every_offset_is_safe() {
    for fixture in
        [small_safetensors(), small_gguf(), small_onnx(), small_pt_zip(), deflated_pt_zip()]
    {
        every_truncation(&fixture.bytes);
    }
}

#[test]
fn truncating_a_recognized_header_is_opaque_at_every_offset_inside_it() {
    let fixture = small_gguf();
    let first_tensor = usize::try_from(fixture.tensors[0].offset).expect("fits");
    for cut in 0..first_tensor - 32 {
        assert_opaque(&fixture.bytes[..cut]);
    }
    let fixture = small_safetensors();
    let header_end = usize::try_from(fixture.tensors[0].offset).expect("fits");
    for cut in 0..header_end {
        assert_opaque(&fixture.bytes[..cut]);
    }
}

#[test]
fn truncated_data_keeps_only_what_is_in_the_file() {
    let fixture = fixtures::safetensors(
        60,
        &[TensorSpec::new("a", &[2048], Elem::F32), TensorSpec::new("b", &[2048], Elem::F32)],
    );
    let keep = usize::try_from(fixture.tensors[1].offset + 16).expect("fits");
    let file_len = u64::try_from(keep).expect("fits");
    let plan = normalize(scan_bytes(&fixture.bytes[..keep]), file_len);
    assert_eq!(plan.profile, Profile::Safetensors);
    assert_eq!(plan.ranges.len(), 1);
    assert_eq!(plan.tensors.len(), 1);
    assert_eq!(plan.tensors[0].name, "a");
}

#[test]
fn every_byte_flip_is_safe() {
    for fixture in
        [small_safetensors(), small_gguf(), small_onnx(), small_pt_zip(), deflated_pt_zip()]
    {
        every_byte_flipped(&fixture.bytes);
    }
}

#[test]
fn random_mutations_are_safe() {
    let fixtures = [small_safetensors(), small_gguf(), small_onnx(), small_pt_zip()];
    for (seed, fixture) in (70..).zip(&fixtures) {
        random_mutations(&fixture.bytes, seed, 2000);
    }
}

#[test]
fn normalized_plans_keep_small_tensors_by_name() {
    let fixture = fixtures::safetensors(
        80,
        &[
            TensorSpec::new("small", &[16], Elem::F32),
            TensorSpec::new("just.under", &[1023], Elem::F32),
            TensorSpec::new("exactly", &[1024], Elem::F32),
        ],
    );
    let file_len = u64::try_from(fixture.bytes.len()).expect("fits");
    let plan = normalize(scan_bytes(&fixture.bytes), file_len);
    assert_eq!(plan.tensors.len(), 3);
    assert_eq!(plan.ranges.len(), 1);
    assert_eq!(plan.ranges[0].len, MIN_RANGE_BYTES);
}

#[test]
fn hostile_overlap_and_duplicate_names_normalize() {
    let header = br#"{"a":{"dtype":"U8","shape":[8192],"data_offsets":[0,8192]},
        "b":{"dtype":"U8","shape":[8192],"data_offsets":[4096,12288]},
        "a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
    let bytes = fixtures::safetensors_raw(header, &[0; 12288]);
    let plan = normalize(scan_bytes(&bytes), u64::try_from(bytes.len()).expect("fits"));
    assert_eq!(plan.ranges.len(), 1);
    assert_eq!(plan.tensors.len(), 2);
    assert_eq!(plan.tensors[0].len, 8192);
}

/// A scratch file under the system temp directory, removed on drop.
struct TempFile(PathBuf);

impl TempFile {
    fn new(extension: &str, bytes: &[u8]) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let name = format!("ziptensors-profile-{}-{id}.{extension}", std::process::id());
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, bytes).expect("scratch file must be writable");
        Self(path)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        // A leftover in the temp directory is harmless and a panic in drop would abort.
        if let Err(error) = std::fs::remove_file(&self.0) {
            eprintln!("cannot remove {}: {error}", self.0.display());
        }
    }
}

#[test]
fn a_name_that_lies_changes_nothing() {
    let cases = [
        ("safetensors", small_gguf(), Profile::Gguf),
        ("gguf", small_safetensors(), Profile::Safetensors),
        ("onnx", small_pt_zip(), Profile::PtZip),
        ("pt", small_onnx(), Profile::Onnx),
    ];
    for (extension, fixture, profile) in cases {
        let file = TempFile::new(extension, &fixture.bytes);
        let source = FileSource::open(&file.0).expect("open the scratch file");
        let plan = scan(&source).expect("scan a readable file");
        assert_found(&plan, profile, &fixture);
    }
}

/// Scans a vast source, asserting no read went over 1 MiB.
fn scan_vast(source: &Vast) -> RangePlan {
    let plan = scan(source).expect("hostile sizes are not an error");
    assert!(source.largest.load(Ordering::Relaxed) <= 1 << 20);
    plan
}

const VAST: u64 = 1 << 60;

#[test]
fn hostile_safetensors_header_length_is_refused_before_reading() {
    let mut head = (1_u64 << 50).to_le_bytes().to_vec();
    head.push(b'{');
    let source = Vast::new(VAST, head, Vec::new());
    assert_eq!(scan_vast(&source).profile, Profile::Opaque);
    assert!(source.served.load(Ordering::Relaxed) < 1 << 20);
}

#[test]
fn hostile_gguf_counts_are_refused_before_reading() {
    let source = Vast::new(VAST, gguf_head(3, u64::MAX, u64::MAX, &[]), Vec::new());
    assert_eq!(scan_vast(&source).profile, Profile::Opaque);
    assert!(source.served.load(Ordering::Relaxed) < 1 << 20);
}

#[test]
fn hostile_gguf_string_array_stops_at_the_header_cap() {
    // Zeros read as empty strings, so the declared count alone would run for centuries.
    let mut value = 1_u64.to_le_bytes().to_vec();
    value.extend_from_slice(b"k");
    value.extend_from_slice(&9_u32.to_le_bytes());
    value.extend_from_slice(&8_u32.to_le_bytes());
    value.extend_from_slice(&u64::MAX.to_le_bytes());
    let source = Vast::new(VAST, gguf_head(3, 0, 1, &value), Vec::new());
    assert_eq!(scan_vast(&source).profile, Profile::Opaque);
    assert!(source.served.load(Ordering::Relaxed) < 512 << 20);
}

#[test]
fn hostile_gguf_tensor_size_reads_no_data() {
    // A tensor that claims 2^42 bytes gets a row and no data is read; `normalize` removes it.
    let mut info = 1_u64.to_le_bytes().to_vec();
    info.extend_from_slice(b"t");
    info.extend_from_slice(&1_u32.to_le_bytes());
    info.extend_from_slice(&(1_u64 << 40).to_le_bytes());
    info.extend_from_slice(&0_u32.to_le_bytes());
    info.extend_from_slice(&0_u64.to_le_bytes());
    let source = Vast::new(VAST, gguf_head(3, 1, 0, &info), Vec::new());
    let plan = scan_vast(&source);
    assert_eq!(plan.profile, Profile::Gguf);
    assert_eq!(plan.ranges[0].len, 1 << 42);
    let plan = normalize(plan, 1 << 20);
    assert_eq!((plan.ranges, plan.tensors), (Vec::new(), Vec::new()));
}

#[test]
fn hostile_onnx_lengths_do_not_read_the_claimed_bytes() {
    let mut model = Msg::new();
    model.varint(1, 8);
    model.raw_varint((7 << 3) | 2);
    model.raw_varint(u64::MAX);
    let source = Vast::new(VAST, model.bytes.clone(), Vec::new());
    assert_eq!(scan_vast(&source).profile, Profile::Opaque);

    let mut model = Msg::new();
    model.varint(1, 8);
    model.raw_varint((2 << 3) | 2);
    model.raw_varint(VAST / 2);
    let source = Vast::new(VAST, model.bytes, Vec::new());
    assert_eq!(scan_vast(&source).profile, Profile::Opaque);
    assert!(source.served.load(Ordering::Relaxed) < 1 << 20);
}

#[test]
fn hostile_zip_directory_is_refused() {
    let mut end = b"PK\x05\x06".to_vec();
    end.extend_from_slice(&[0; 4]);
    end.extend_from_slice(&[0xFE, 0xFF, 0xFE, 0xFF]);
    end.extend_from_slice(&64_u32.to_le_bytes());
    end.extend_from_slice(&128_u32.to_le_bytes());
    end.extend_from_slice(&[0; 2]);
    let source = Vast::new(VAST, b"PK\x03\x04".to_vec(), end);
    assert_eq!(scan_vast(&source).profile, Profile::Opaque);
    assert!(source.served.load(Ordering::Relaxed) < 1 << 20);
}

#[test]
fn only_failed_reads_are_errors() {
    for fixture in [small_safetensors(), small_gguf(), small_onnx(), small_pt_zip()] {
        let source = Failing { bytes: fixture.bytes, good: 4 };
        let error = scan(&source).expect_err("the read failed");
        assert!(matches!(error, Error::Io { op: "read_exact_at", .. }), "{error:?}");
    }
    let empty = Failing { bytes: Vec::new(), good: 0 };
    assert_eq!(scan(&empty).expect("nothing to read").profile, Profile::Opaque);
}

fn classify_bytes(bytes: &[u8]) -> Option<WeightKind> {
    classify_weight(&Memory(bytes.to_vec())).expect("unrecognized content is not an error")
}

/// The first pickle of a legacy `torch.save` in pickle `protocol` 2 to 5: torch's
/// `MAGIC_NUMBER`, as `CPython`'s `pickle.dumps(0x1950A86A20F9469CFC6C, protocol)` writes it.
fn legacy_magic(protocol: u8) -> Vec<u8> {
    let mut bytes = vec![0x80, protocol];
    if protocol >= 4 {
        bytes.push(0x95);
        bytes.extend_from_slice(&13_u64.to_le_bytes());
    }
    bytes.extend_from_slice(&[
        0x8A, 0x0A, 0x6C, 0xFC, 0x9C, 0x46, 0xF9, 0x20, 0x6A, 0xA8, 0x50, 0x19, 0x2E,
    ]);
    bytes
}

/// A legacy save's start: the magic pickle, then `PROTOCOL_VERSION` 1001 and arbitrary bytes.
fn legacy_file(protocol: u8) -> Vec<u8> {
    let mut bytes = legacy_magic(protocol);
    bytes.extend_from_slice(&[0x80, protocol.min(3), 0x4D, 0xE9, 0x03, 0x2E]);
    bytes.extend_from_slice(&Rng::new(90 + u64::from(protocol)).bytes(4096));
    bytes
}

#[test]
fn every_parsed_format_is_a_weight_file() {
    let cases = [
        (small_safetensors().bytes, Profile::Safetensors),
        (small_gguf().bytes, Profile::Gguf),
        (small_onnx().bytes, Profile::Onnx),
        (small_pt_zip().bytes, Profile::PtZip),
        (deflated_pt_zip().bytes, Profile::PtZip),
    ];
    for (bytes, profile) in cases {
        assert_eq!(classify_bytes(&bytes), Some(WeightKind::Parsed(profile)));
    }
}

#[test]
fn legacy_torch_saves_are_weight_files_in_protocols_two_to_five() {
    for protocol in 2..=5 {
        assert_eq!(classify_bytes(&legacy_file(protocol)), Some(WeightKind::LegacyTorch));
        assert_eq!(classify_bytes(&legacy_magic(protocol)), Some(WeightKind::LegacyTorch));
    }
}

#[test]
fn a_prefix_one_byte_off_is_not_a_weight_file() {
    for protocol in 2..=5 {
        let magic = legacy_magic(protocol);
        for index in 0..magic.len() {
            // 0x10 rather than 0x01: flipping the low bit of the protocol byte would only turn
            // one accepted protocol into another.
            let mut near = magic.clone();
            near[index] ^= 0x10;
            assert_eq!(classify_bytes(&near), None, "protocol {protocol}, byte {index}");
        }
        assert_eq!(classify_bytes(&magic[..magic.len() - 1]), None, "protocol {protocol}");
    }
    for protocol in [0, 1, 6] {
        let mut other = legacy_magic(5);
        other[1] = protocol;
        assert_eq!(classify_bytes(&other), None, "protocol {protocol}");
    }
    let mut long_frame = legacy_magic(4);
    long_frame[3] = 14;
    long_frame.push(0x2E);
    assert_eq!(classify_bytes(&long_frame), None);
}

#[test]
fn the_legacy_check_reads_only_the_prefix() {
    let source = Vast::new(VAST, legacy_magic(5), Vec::new());
    assert!(legacy_torch::matches(&source).expect("the prefix is readable"));
    assert!(source.largest.load(Ordering::Relaxed) <= 32);
    assert!(source.served.load(Ordering::Relaxed) <= 32);
}

#[test]
fn other_content_is_not_a_weight_file() {
    let mut png = b"\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR".to_vec();
    png.extend_from_slice(&Rng::new(95).bytes(4096));
    assert_eq!(classify_bytes(&png), None);
    assert_eq!(classify_bytes(&[]), None);
    assert_eq!(classify_bytes(b"{\"weight_map\": {}}"), None);
    let no_pickle = PtZipOptions { flaw: PtZipFlaw::NoPickle, ..PtZipOptions::default() };
    assert_eq!(classify_bytes(&small_pt_zip_with(&no_pickle)), None);
    let fixture = small_safetensors();
    let header_end = usize::try_from(fixture.tensors[0].offset).expect("fits");
    for cut in [0, 8, 9, header_end / 2, header_end - 1] {
        assert_eq!(classify_bytes(&fixture.bytes[..cut]), None, "cut at {cut}");
    }
}

#[test]
fn a_failed_read_while_classifying_is_an_error() {
    for good in [0, 9] {
        let source = Failing { bytes: legacy_file(4), good };
        let error = classify_weight(&source).expect_err("the read failed");
        assert!(matches!(error, Error::Io { op: "read_exact_at", .. }), "{error:?}");
    }
}
