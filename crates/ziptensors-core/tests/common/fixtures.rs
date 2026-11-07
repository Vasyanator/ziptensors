//! File: tests/common/fixtures.rs
//! Purpose: seeded builders of small source files in each supported format, for tests.
//!
//! Main responsibilities:
//! - one builder per format, each returning the bytes and an oracle of what they hold;
//! - payloads with the byte statistics of real tensors, reproducible from a seed;
//! - knobs that make a file hostile or malformed in a named way.
//!
//! Key structures: [`Rng`], [`Elem`], [`TensorSpec`], [`Fixture`], [`Expected`], [`GgufOptions`],
//! [`OnnxModel`], [`PtZipOptions`], [`Msg`].
//!
//! Key functions: [`safetensors`], [`gguf`], [`onnx`], [`pt_zip`], [`payload`].
//!
//! Implementation notes: include from a test target with
//! `#[path = "common/fixtures.rs"] mod fixtures;`. This file depends on nothing of the crate, so
//! unit tests and integration tests can share it and it stays an independent statement of each
//! format, which is the point of an oracle. Dtypes are therefore wire ids: compare them through
//! `Dtype::from_wire`. Builders panic on a nonsensical spec, since that is a bug in a test.

#![allow(dead_code, reason = "each test target uses a different subset of these builders")]

/// Splitmix64. Small, seedable, and the same on every platform.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// Starts the sequence of `seed`.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `count` bytes with no structure.
    pub fn bytes(&mut self, count: usize) -> Vec<u8> {
        (0..count.div_ceil(8))
            .flat_map(|_| self.next_u64().to_le_bytes())
            .take(count)
            .collect()
    }
}

/// An element type, stated independently of the crate under test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Elem {
    Bool,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16,
    Bf16,
    F32,
    F64,
    F8E4M3,
    F8E5M2,
    Q80,
    Q4K,
    Q5K,
    Q6K,
    Q40,
    Q41,
    Q50,
    Q51,
    Q81,
    Q2K,
    Q3K,
    Q8K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq1S,
    Iq4Nl,
    Iq3S,
    Iq2S,
    Iq4Xs,
    Iq1M,
    Tq10,
    Tq20,
    Mxfp4,
}

impl Elem {
    /// The ggml block types with no field table, in the order of their wire ids.
    pub const UNSPLIT_GGML: [Self; 20] = [
        Self::Q40,
        Self::Q41,
        Self::Q50,
        Self::Q51,
        Self::Q81,
        Self::Q2K,
        Self::Q3K,
        Self::Q8K,
        Self::Iq2Xxs,
        Self::Iq2Xs,
        Self::Iq3Xxs,
        Self::Iq1S,
        Self::Iq4Nl,
        Self::Iq3S,
        Self::Iq2S,
        Self::Iq4Xs,
        Self::Iq1M,
        Self::Tq10,
        Self::Tq20,
        Self::Mxfp4,
    ];
}

impl Elem {
    /// Wire id of the matching dtype, from the spec registry.
    pub fn dtype_wire(self) -> u16 {
        match self {
            Self::Bool => 0x0001,
            Self::U8 => 0x0002,
            Self::I8 => 0x0003,
            Self::U16 => 0x0004,
            Self::I16 => 0x0005,
            Self::U32 => 0x0006,
            Self::I32 => 0x0007,
            Self::U64 => 0x0008,
            Self::I64 => 0x0009,
            Self::F16 => 0x0010,
            Self::Bf16 => 0x0011,
            Self::F32 => 0x0012,
            Self::F64 => 0x0013,
            Self::F8E4M3 => 0x0020,
            Self::F8E5M2 => 0x0021,
            Self::Q80 => 0x0030,
            Self::Q4K => 0x0031,
            Self::Q5K => 0x0032,
            Self::Q6K => 0x0033,
            Self::Q40 => 0x0034,
            Self::Q41 => 0x0035,
            Self::Q50 => 0x0036,
            Self::Q51 => 0x0037,
            Self::Q81 => 0x0038,
            Self::Q2K => 0x0039,
            Self::Q3K => 0x003A,
            Self::Q8K => 0x003B,
            Self::Iq2Xxs => 0x003C,
            Self::Iq2Xs => 0x003D,
            Self::Iq3Xxs => 0x003E,
            Self::Iq1S => 0x003F,
            Self::Iq4Nl => 0x0040,
            Self::Iq3S => 0x0041,
            Self::Iq2S => 0x0042,
            Self::Iq4Xs => 0x0043,
            Self::Iq1M => 0x0044,
            Self::Tq10 => 0x0045,
            Self::Tq20 => 0x0046,
            Self::Mxfp4 => 0x0047,
        }
    }

    /// `(elements per block, bytes per block)`, from `ggml-common.h` for the quantized types, as
    /// `GGML_QUANT_SIZES` of the reference `gguf` Python package lists them.
    pub fn block(self) -> (u64, u64) {
        match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 => (1, 1),
            Self::U16 | Self::I16 | Self::F16 | Self::Bf16 => (1, 2),
            Self::U32 | Self::I32 | Self::F32 => (1, 4),
            Self::U64 | Self::I64 | Self::F64 => (1, 8),
            Self::Q80 => (32, 34),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Q40 | Self::Iq4Nl => (32, 18),
            Self::Q41 => (32, 20),
            Self::Q50 => (32, 22),
            Self::Q51 => (32, 24),
            Self::Q81 => (32, 40),
            Self::Q2K => (256, 84),
            Self::Q3K | Self::Iq3S => (256, 110),
            Self::Q8K => (256, 292),
            Self::Iq2Xxs | Self::Tq20 => (256, 66),
            Self::Iq2Xs => (256, 74),
            Self::Iq3Xxs => (256, 98),
            Self::Iq1S => (256, 50),
            Self::Iq2S => (256, 82),
            Self::Iq4Xs => (256, 136),
            Self::Iq1M => (256, 56),
            Self::Tq10 => (256, 54),
            Self::Mxfp4 => (32, 17),
        }
    }

    /// Tag in a safetensors header. `None` for types safetensors cannot name.
    pub fn safetensors_tag(self) -> Option<&'static str> {
        match self {
            Self::Bool => Some("BOOL"),
            Self::U8 => Some("U8"),
            Self::I8 => Some("I8"),
            Self::U16 => Some("U16"),
            Self::I16 => Some("I16"),
            Self::U32 => Some("U32"),
            Self::I32 => Some("I32"),
            Self::U64 => Some("U64"),
            Self::I64 => Some("I64"),
            Self::F16 => Some("F16"),
            Self::Bf16 => Some("BF16"),
            Self::F32 => Some("F32"),
            Self::F64 => Some("F64"),
            Self::F8E4M3 => Some("F8_E4M3"),
            Self::F8E5M2 => Some("F8_E5M2"),
            Self::Q80
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Q40
            | Self::Q41
            | Self::Q50
            | Self::Q51
            | Self::Q81
            | Self::Q2K
            | Self::Q3K
            | Self::Q8K
            | Self::Iq2Xxs
            | Self::Iq2Xs
            | Self::Iq3Xxs
            | Self::Iq1S
            | Self::Iq4Nl
            | Self::Iq3S
            | Self::Iq2S
            | Self::Iq4Xs
            | Self::Iq1M
            | Self::Tq10
            | Self::Tq20
            | Self::Mxfp4 => None,
        }
    }

    /// ggml type id. `None` for types GGUF has no id for.
    pub fn ggml_type(self) -> Option<u32> {
        match self {
            Self::F32 => Some(0),
            Self::F16 => Some(1),
            Self::Q80 => Some(8),
            Self::Q4K => Some(12),
            Self::Q5K => Some(13),
            Self::Q6K => Some(14),
            Self::I8 => Some(24),
            Self::I16 => Some(25),
            Self::I32 => Some(26),
            Self::I64 => Some(27),
            Self::F64 => Some(28),
            Self::Bf16 => Some(30),
            Self::Q40 => Some(2),
            Self::Q41 => Some(3),
            Self::Q50 => Some(6),
            Self::Q51 => Some(7),
            Self::Q81 => Some(9),
            Self::Q2K => Some(10),
            Self::Q3K => Some(11),
            Self::Q8K => Some(15),
            Self::Iq2Xxs => Some(16),
            Self::Iq2Xs => Some(17),
            Self::Iq3Xxs => Some(18),
            Self::Iq1S => Some(19),
            Self::Iq4Nl => Some(20),
            Self::Iq3S => Some(21),
            Self::Iq2S => Some(22),
            Self::Iq4Xs => Some(23),
            Self::Iq1M => Some(29),
            Self::Tq10 => Some(34),
            Self::Tq20 => Some(35),
            Self::Mxfp4 => Some(39),
            Self::Bool
            | Self::U8
            | Self::U16
            | Self::U32
            | Self::U64
            | Self::F8E4M3
            | Self::F8E5M2 => None,
        }
    }

    /// `TensorProto.DataType` value. `None` for types ONNX has no id for.
    pub fn onnx_type(self) -> Option<u64> {
        match self {
            Self::F32 => Some(1),
            Self::U8 => Some(2),
            Self::I8 => Some(3),
            Self::U16 => Some(4),
            Self::I16 => Some(5),
            Self::I32 => Some(6),
            Self::I64 => Some(7),
            Self::Bool => Some(9),
            Self::F16 => Some(10),
            Self::F64 => Some(11),
            Self::U32 => Some(12),
            Self::U64 => Some(13),
            Self::Bf16 => Some(16),
            Self::F8E4M3
            | Self::F8E5M2
            | Self::Q80
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Q40
            | Self::Q41
            | Self::Q50
            | Self::Q51
            | Self::Q81
            | Self::Q2K
            | Self::Q3K
            | Self::Q8K
            | Self::Iq2Xxs
            | Self::Iq2Xs
            | Self::Iq3Xxs
            | Self::Iq1S
            | Self::Iq4Nl
            | Self::Iq3S
            | Self::Iq2S
            | Self::Iq4Xs
            | Self::Iq1M
            | Self::Tq10
            | Self::Tq20
            | Self::Mxfp4 => None,
        }
    }
}

/// `bytes` of tensor data whose byte planes look like `elem`: floats get a skewed sign and exponent
/// over a noisy mantissa, integers small values, one-byte and quantized types plain noise. A
/// partial element at the end is cut.
pub fn payload(rng: &mut Rng, elem: Elem, bytes: usize) -> Vec<u8> {
    let (_, block) = elem.block();
    let mut out = Vec::with_capacity(bytes + 8);
    while out.len() < bytes {
        let word = rng.next_u64();
        let sign = word >> 63;
        let binade = (word >> 60) & 3;
        match (elem, block) {
            (Elem::F16 | Elem::Bf16, _) => {
                let high = u16::try_from(0x3800 | (sign << 15) | (binade << 8)).expect("16 bits");
                let low = u16::try_from(word & 0xFF).expect("8 bits");
                out.extend_from_slice(&(high | low).to_le_bytes());
            }
            (Elem::F32, _) => {
                let exponent = 120 + binade;
                let bits = (sign << 31) | (exponent << 23) | (word & 0x7F_FFFF);
                out.extend_from_slice(&u32::try_from(bits).expect("32 bits").to_le_bytes());
            }
            (Elem::F64, _) => {
                let exponent = 1016 + binade;
                let bits = (sign << 63) | (exponent << 52) | (word & 0xF_FFFF_FFFF_FFFF);
                out.extend_from_slice(&bits.to_le_bytes());
            }
            (_, 2) => out.extend_from_slice(&(word & 0xFF).to_le_bytes()[..2]),
            (_, 4) => out.extend_from_slice(&(word & 0xFFF).to_le_bytes()[..4]),
            (_, 8) => out.extend_from_slice(&(word & 0xF_FFFF).to_le_bytes()),
            _ => out.extend_from_slice(&word.to_le_bytes()),
        }
    }
    out.truncate(bytes);
    out
}

/// One tensor to put in a file. `shape` is in the order of the target format.
#[derive(Debug, Clone)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<u64>,
    pub elem: Elem,
}

impl TensorSpec {
    /// A tensor named `name`.
    pub fn new(name: &str, shape: &[u64], elem: Elem) -> Self {
        Self { name: name.to_owned(), shape: shape.to_vec(), elem }
    }

    /// Its size in bytes. The first dimension of a quantized type must be whole blocks.
    pub fn byte_len(&self) -> u64 {
        let (elements, block) = self.elem.block();
        let first = self.shape.first().copied().unwrap_or(1);
        assert_eq!(first % elements, 0, "{} is not whole blocks", self.name);
        self.shape.iter().skip(1).product::<u64>() * (first / elements) * block
    }

    fn data(&self, rng: &mut Rng) -> Vec<u8> {
        payload(rng, self.elem, usize::try_from(self.byte_len()).expect("test tensor fits usize"))
    }
}

/// What a builder put in a file: one tensor, with an absolute offset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expected {
    pub name: String,
    pub offset: u64,
    pub len: u64,
    pub shape: Vec<u64>,
    pub dtype_wire: u16,
}

/// A generated file and the oracle of its contents.
#[derive(Debug, Clone)]
pub struct Fixture {
    pub bytes: Vec<u8>,
    /// Tensors a parser must find, in file order of their description.
    pub tensors: Vec<Expected>,
}

fn to_u64(value: usize) -> u64 {
    u64::try_from(value).expect("test size fits u64")
}

/// `text` as a JSON string, quoted.
fn json_string(text: &str) -> String {
    let mut out = String::from("\"");
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            _ => out.push(ch),
        }
    }
    out.push('"');
    out
}

/// A safetensors file from a raw header text and data section, for malformed headers.
pub fn safetensors_raw(header: &[u8], data: &[u8]) -> Vec<u8> {
    let mut bytes = to_u64(header.len()).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(data);
    bytes
}

/// A safetensors file holding `specs` in order. Its `__metadata__` block states things that must be
/// ignored, including a false `data_offsets`, and the header is padded with spaces to 8 bytes.
pub fn safetensors(seed: u64, specs: &[TensorSpec]) -> Fixture {
    let mut rng = Rng::new(seed);
    let mut data = Vec::new();
    let metadata = r#""__metadata__":{"format":"pt","data_offsets":"[0,9999999]","dtype":"F32"}"#;
    let mut entries = vec![metadata.to_owned()];
    let mut placed = Vec::new();
    for spec in specs {
        let tag = spec.elem.safetensors_tag().expect("type has a safetensors tag");
        let begin = to_u64(data.len());
        data.extend_from_slice(&spec.data(&mut rng));
        let end = to_u64(data.len());
        let shape: Vec<String> = spec.shape.iter().map(u64::to_string).collect();
        entries.push(format!(
            r#"{}:{{"dtype":"{tag}","shape":[{}],"data_offsets":[{begin},{end}]}}"#,
            json_string(&spec.name),
            shape.join(",")
        ));
        placed.push((spec, begin, end - begin));
    }
    let mut header = format!("{{{}}}", entries.join(",")).into_bytes();
    header.resize(header.len().next_multiple_of(8), b' ');
    let base = 8 + to_u64(header.len());
    let tensors = placed
        .into_iter()
        .map(|(spec, begin, len)| Expected {
            name: spec.name.clone(),
            offset: base + begin,
            len,
            shape: spec.shape.clone(),
            dtype_wire: spec.elem.dtype_wire(),
        })
        .collect();
    Fixture { bytes: safetensors_raw(&header, &data), tensors }
}

/// Options of [`gguf`].
#[derive(Debug, Clone)]
pub struct GgufOptions {
    /// Value of the version field.
    pub version: u32,
    /// Value of `general.alignment`. `None` leaves the key out, which means 32.
    pub alignment: Option<u32>,
    /// Strings in a tokenizer-like array value.
    pub tokens: usize,
    /// Adds a tensor of ggml type 4, the retired `Q4_2`, which has no dtype. It is left out of the
    /// oracle.
    pub unmapped_tensor: bool,
}

impl Default for GgufOptions {
    fn default() -> Self {
        Self { version: 3, alignment: None, tokens: 5, unmapped_tensor: false }
    }
}

fn gguf_string(out: &mut Vec<u8>, text: &str) {
    out.extend_from_slice(&to_u64(text.len()).to_le_bytes());
    out.extend_from_slice(text.as_bytes());
}

fn gguf_key(out: &mut Vec<u8>, key: &str, value_type: u32) {
    gguf_string(out, key);
    out.extend_from_slice(&value_type.to_le_bytes());
}

/// A GGUF file holding `specs` in order, with data padded by `0xAA` to the alignment, so that a
/// range that swallows padding is caught. `shape` is written as it stands, fastest dimension first.
pub fn gguf(seed: u64, specs: &[TensorSpec], options: &GgufOptions) -> Fixture {
    let mut rng = Rng::new(seed);
    let alignment = u64::from(options.alignment.unwrap_or(32));

    let mut kv = Vec::new();
    let mut kv_count = 4_u64;
    gguf_key(&mut kv, "general.architecture", 8);
    gguf_string(&mut kv, "fixture");
    gguf_key(&mut kv, "tokenizer.ggml.tokens", 9);
    kv.extend_from_slice(&8_u32.to_le_bytes());
    kv.extend_from_slice(&to_u64(options.tokens).to_le_bytes());
    for token in 0..options.tokens {
        gguf_string(&mut kv, &format!("token-{token}"));
    }
    gguf_key(&mut kv, "tokenizer.ggml.scores", 9);
    kv.extend_from_slice(&6_u32.to_le_bytes());
    kv.extend_from_slice(&3_u64.to_le_bytes());
    kv.extend_from_slice(&[0; 12]);
    gguf_key(&mut kv, "general.flag", 7);
    kv.push(1);
    if let Some(value) = options.alignment {
        kv_count += 1;
        gguf_key(&mut kv, "general.alignment", 4);
        kv.extend_from_slice(&value.to_le_bytes());
    }

    let mut data = Vec::new();
    let mut infos = Vec::new();
    let mut placed = Vec::new();
    for spec in specs {
        let ggml_type = spec.elem.ggml_type().expect("type has a ggml id");
        gguf_string(&mut infos, &spec.name);
        infos.extend_from_slice(&u32::try_from(spec.shape.len()).expect("few dims").to_le_bytes());
        for dim in &spec.shape {
            infos.extend_from_slice(&dim.to_le_bytes());
        }
        infos.extend_from_slice(&ggml_type.to_le_bytes());
        infos.extend_from_slice(&to_u64(data.len()).to_le_bytes());
        placed.push((spec, to_u64(data.len())));
        data.extend_from_slice(&spec.data(&mut rng));
        let padded = to_u64(data.len()).next_multiple_of(alignment);
        data.resize(usize::try_from(padded).expect("test size fits usize"), 0xAA);
    }
    let mut tensor_count = to_u64(specs.len());
    if options.unmapped_tensor {
        tensor_count += 1;
        gguf_string(&mut infos, "legacy.q4_2");
        infos.extend_from_slice(&1_u32.to_le_bytes());
        infos.extend_from_slice(&32_u64.to_le_bytes());
        infos.extend_from_slice(&4_u32.to_le_bytes());
        infos.extend_from_slice(&to_u64(data.len()).to_le_bytes());
        data.extend_from_slice(&rng.bytes(18));
    }

    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&options.version.to_le_bytes());
    bytes.extend_from_slice(&tensor_count.to_le_bytes());
    bytes.extend_from_slice(&kv_count.to_le_bytes());
    bytes.extend_from_slice(&kv);
    bytes.extend_from_slice(&infos);
    let base = to_u64(bytes.len()).next_multiple_of(alignment);
    bytes.resize(usize::try_from(base).expect("small"), 0xAA);
    bytes.extend_from_slice(&data);
    let tensors = placed
        .into_iter()
        .map(|(spec, offset)| Expected {
            name: spec.name.clone(),
            offset: base + offset,
            len: spec.byte_len(),
            shape: spec.shape.clone(),
            dtype_wire: spec.elem.dtype_wire(),
        })
        .collect();
    Fixture { bytes, tensors }
}

/// A protobuf message under construction, tracking where the tensor bytes in it end up.
#[derive(Debug, Clone, Default)]
pub struct Msg {
    pub bytes: Vec<u8>,
    marks: Vec<Expected>,
}

impl Msg {
    /// An empty message.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a varint with no tag.
    pub fn raw_varint(&mut self, mut value: u64) {
        while value >= 0x80 {
            self.bytes.push(u8::try_from(value & 0x7F).expect("7 bits") | 0x80);
            value >>= 7;
        }
        self.bytes.push(u8::try_from(value).expect("under 0x80"));
    }

    /// Appends a varint field.
    pub fn varint(&mut self, field: u64, value: u64) -> &mut Self {
        self.raw_varint(field << 3);
        self.raw_varint(value);
        self
    }

    /// Appends a length-delimited field.
    pub fn bytes_field(&mut self, field: u64, value: &[u8]) -> &mut Self {
        self.raw_varint((field << 3) | 2);
        self.raw_varint(to_u64(value.len()));
        self.bytes.extend_from_slice(value);
        self
    }

    /// Appends a nested message, carrying its tensors along.
    pub fn message(&mut self, field: u64, value: Self) -> &mut Self {
        self.raw_varint((field << 3) | 2);
        self.raw_varint(to_u64(value.bytes.len()));
        let shift = to_u64(self.bytes.len());
        self.bytes.extend_from_slice(&value.bytes);
        self.marks.extend(value.marks.into_iter().map(|mut mark| {
            mark.offset += shift;
            mark
        }));
        self
    }

    /// Appends tensor bytes as a length-delimited field and records them for the oracle.
    fn tensor_bytes(&mut self, field: u64, value: &[u8], mut mark: Expected) {
        self.raw_varint((field << 3) | 2);
        self.raw_varint(to_u64(value.len()));
        mark.offset = to_u64(self.bytes.len());
        mark.len = to_u64(value.len());
        self.bytes.extend_from_slice(value);
        self.marks.push(mark);
    }
}

/// How an ONNX tensor stores its values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxStorage {
    /// `raw_data`.
    Raw,
    /// Packed `float_data`.
    PackedFloat,
    /// `int32_data` as varints, which is no byte run. Not in the oracle.
    Varint,
    /// `external_data` naming a file that does not exist. Not in the oracle.
    External,
}

/// One ONNX tensor.
#[derive(Debug, Clone)]
pub struct OnnxTensor {
    pub spec: TensorSpec,
    pub storage: OnnxStorage,
}

impl OnnxTensor {
    /// A tensor stored as `raw_data`.
    pub fn raw(name: &str, shape: &[u64], elem: Elem) -> Self {
        Self { spec: TensorSpec::new(name, shape, elem), storage: OnnxStorage::Raw }
    }

    /// A tensor stored some other way.
    pub fn stored(name: &str, shape: &[u64], elem: Elem, storage: OnnxStorage) -> Self {
        Self { spec: TensorSpec::new(name, shape, elem), storage }
    }
}

/// What an [`onnx`] model holds.
#[derive(Debug, Clone, Default)]
pub struct OnnxModel {
    /// Tensors of `graph.initializer`.
    pub initializers: Vec<OnnxTensor>,
    /// `Constant` nodes. Their tensors have no name of their own; the node output carries it.
    pub constants: Vec<OnnxTensor>,
    /// Initializers of the `then_branch` subgraph of an `If` node.
    pub then_branch: Vec<OnnxTensor>,
    /// `Constant` nodes in the `else_branch` subgraph of that node.
    pub else_branch: Vec<OnnxTensor>,
    /// Initializers of the `body` subgraph of a `Loop` node.
    pub loop_body: Vec<OnnxTensor>,
    /// Serializes `Constant` node attributes before the outputs, as protobuf permits.
    pub attributes_first: bool,
}

fn onnx_tensor(rng: &mut Rng, tensor: &OnnxTensor, own_name: bool) -> Msg {
    let spec = &tensor.spec;
    let mut msg = Msg::new();
    let mut dims = Msg::new();
    for &dim in &spec.shape {
        dims.raw_varint(dim);
    }
    msg.bytes_field(1, &dims.bytes);
    msg.varint(2, spec.elem.onnx_type().expect("type has an ONNX id"));
    if own_name {
        msg.bytes_field(8, spec.name.as_bytes());
    }
    let mark = Expected {
        name: spec.name.clone(),
        offset: 0,
        len: 0,
        shape: spec.shape.clone(),
        dtype_wire: spec.elem.dtype_wire(),
    };
    match tensor.storage {
        OnnxStorage::Raw => msg.tensor_bytes(9, &spec.data(rng), mark),
        OnnxStorage::PackedFloat => msg.tensor_bytes(4, &spec.data(rng), mark),
        OnnxStorage::Varint => {
            let mut packed = Msg::new();
            for byte in spec.data(rng) {
                packed.raw_varint(u64::from(byte));
            }
            msg.bytes_field(5, &packed.bytes);
        }
        OnnxStorage::External => {
            let mut entry = Msg::new();
            entry.bytes_field(1, b"location").bytes_field(2, b"../outside.bin");
            msg.message(13, entry);
            msg.varint(14, 1);
        }
    }
    msg
}

fn onnx_constant(rng: &mut Rng, tensor: &OnnxTensor, attributes_first: bool) -> Msg {
    let mut attribute = Msg::new();
    attribute.bytes_field(1, b"value").message(5, onnx_tensor(rng, tensor, false)).varint(20, 4);
    let mut node = Msg::new();
    if attributes_first {
        node.message(5, attribute);
        node.bytes_field(2, tensor.spec.name.as_bytes());
        node.bytes_field(4, b"Constant");
    } else {
        node.bytes_field(2, tensor.spec.name.as_bytes());
        node.bytes_field(4, b"Constant");
        node.message(5, attribute);
    }
    node
}

fn onnx_subgraph(
    rng: &mut Rng,
    name: &str,
    initializers: &[OnnxTensor],
    constants: &[OnnxTensor],
) -> Msg {
    let mut graph = Msg::new();
    for tensor in constants {
        graph.message(1, onnx_constant(rng, tensor, false));
    }
    graph.bytes_field(2, name.as_bytes());
    for tensor in initializers {
        graph.message(5, onnx_tensor(rng, tensor, true));
    }
    graph
}

fn onnx_control_node(op_type: &str, attributes: Vec<(&str, Msg)>) -> Msg {
    let mut node = Msg::new();
    node.bytes_field(1, b"cond").bytes_field(2, b"result").bytes_field(4, op_type.as_bytes());
    for (name, graph) in attributes {
        let mut attribute = Msg::new();
        attribute.bytes_field(1, name.as_bytes()).message(6, graph).varint(20, 5);
        node.message(5, attribute);
    }
    node
}

/// Wraps `graph` in the fields of a `ModelProto`.
pub fn onnx_model_bytes(graph: Msg) -> Msg {
    let mut opset = Msg::new();
    opset.varint(2, 17);
    let mut model = Msg::new();
    model.varint(1, 8).bytes_field(2, b"fixture").message(7, graph).message(8, opset);
    model
}

/// An ONNX model holding `model`. Its oracle lists only tensors with a byte run in the file.
pub fn onnx(seed: u64, model: &OnnxModel) -> Fixture {
    let mut rng = Rng::new(seed);
    let mut graph = Msg::new();
    for tensor in &model.constants {
        graph.message(1, onnx_constant(&mut rng, tensor, model.attributes_first));
    }
    if !model.then_branch.is_empty() || !model.else_branch.is_empty() {
        let then_graph = onnx_subgraph(&mut rng, "then", &model.then_branch, &[]);
        let else_graph = onnx_subgraph(&mut rng, "else", &[], &model.else_branch);
        let attributes = vec![("then_branch", then_graph), ("else_branch", else_graph)];
        graph.message(1, onnx_control_node("If", attributes));
    }
    if !model.loop_body.is_empty() {
        let body = onnx_subgraph(&mut rng, "body", &model.loop_body, &[]);
        graph.message(1, onnx_control_node("Loop", vec![("body", body)]));
    }
    graph.bytes_field(2, b"main");
    for tensor in &model.initializers {
        graph.message(5, onnx_tensor(&mut rng, tensor, true));
    }
    let Msg { bytes, marks } = onnx_model_bytes(graph);
    Fixture { bytes, tensors: marks }
}

/// An ONNX model with `depth` nested `If` subgraphs and one 8-byte tensor at the bottom.
pub fn onnx_nested(depth: usize) -> Vec<u8> {
    let mut graph = Msg::new();
    let tensor = OnnxTensor::raw("deep", &[2], Elem::F32);
    graph.message(5, onnx_tensor(&mut Rng::new(1), &tensor, true));
    for _ in 0..depth {
        let mut outer = Msg::new();
        outer.message(1, onnx_control_node("If", vec![("then_branch", graph)]));
        graph = outer;
    }
    onnx_model_bytes(graph).bytes
}

/// A way of making a [`pt_zip`] archive fail its checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PtZipFlaw {
    /// A valid archive.
    #[default]
    None,
    /// `data.pkl` is marked deflated in both headers.
    CompressedPickle,
    /// A storage is marked deflated in both headers.
    CompressedStorage,
    /// The classic end record says the central directory offset is a ZIP64 marker.
    Zip64Marker,
    /// The ZIP64 end record states a different entry count.
    Zip64Disagrees,
    /// A central name differs from its local header.
    LocalNameDiffers,
    /// A central record points at the same local header as another entry.
    OverlappingEntries,
    /// There is no `data.pkl` entry.
    NoPickle,
    /// An entry is marked encrypted.
    Encrypted,
    /// Sizes in a local header disagree, with the data descriptor flag clear.
    LocalSizeDiffers,
}

/// Options of [`pt_zip`].
#[derive(Debug, Clone)]
pub struct PtZipOptions {
    /// Top directory of the archive.
    pub prefix: String,
    /// Emits ZIP64 end records and data descriptor local headers, as `torch.save` does.
    pub torch_layout: bool,
    /// Data alignment, from padding in the local extra field. 64 in `torch.save`.
    pub alignment: u64,
    /// Adds a deflated `code/` entry whose stored bytes are shorter than its content, as a
    /// `TorchScript` archive has. The bytes are filler and nothing may inflate them.
    pub deflated_code: bool,
    /// The defect to build in.
    pub flaw: PtZipFlaw,
}

impl Default for PtZipOptions {
    fn default() -> Self {
        Self {
            prefix: "archive".to_owned(),
            torch_layout: true,
            alignment: 64,
            deflated_code: false,
            flaw: PtZipFlaw::None,
        }
    }
}

/// One ZIP entry being written.
struct ZipEntry {
    name: String,
    flags: u16,
    method: u16,
    /// Bytes in the file.
    stored: u32,
    /// Bytes of the content.
    size: u32,
    local_offset: u32,
}

fn le16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn le32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn le64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn to_u32(value: usize) -> u32 {
    u32::try_from(value).expect("test size fits u32")
}

/// Appends a local header, padding and data. Returns the data's offset.
fn zip_add(
    out: &mut Vec<u8>,
    entries: &mut Vec<ZipEntry>,
    mut entry: ZipEntry,
    data: &[u8],
    options: &PtZipOptions,
) -> u64 {
    entry.local_offset = to_u32(out.len());
    let descriptor = entry.flags & (1 << 3) != 0;
    let name_len = entry.name.len();
    let unpadded = to_u64(out.len() + 30 + name_len + 4);
    let pad = unpadded.next_multiple_of(options.alignment) - unpadded;
    let pad = usize::try_from(pad).expect("small pad");
    le32(out, 0x0403_4B50);
    le16(out, 45);
    le16(out, entry.flags);
    le16(out, entry.method);
    le16(out, 0);
    le16(out, 0);
    le32(out, 0);
    let local = |size: u32| if descriptor { 0 } else { size };
    let wrong = u32::from(options.flaw == PtZipFlaw::LocalSizeDiffers);
    le32(out, local(entry.stored) + wrong);
    le32(out, local(entry.size) + wrong);
    le16(out, u16::try_from(name_len).expect("short name"));
    le16(out, u16::try_from(pad + 4).expect("small pad"));
    out.extend_from_slice(entry.name.as_bytes());
    out.extend_from_slice(b"FB");
    le16(out, u16::try_from(pad).expect("small pad"));
    out.resize(out.len() + pad, b'Z');
    let offset = to_u64(out.len());
    out.extend_from_slice(data);
    if descriptor {
        le32(out, 0x0807_4B50);
        le32(out, 0);
        le32(out, entry.stored);
        le32(out, entry.size);
    }
    entries.push(entry);
    offset
}

/// A `PyTorch`-style archive: filler `data.pkl` (no pickle, nothing may read it), `byteorder`, one
/// `data/N` entry per spec, and `version`. Each spec's `shape` gives the element count by its
/// product. The oracle states what a width trial must report: the elem for storages of 4 KiB or
/// more, so give those a float elem or `U8`; `U8` and flat for shorter ones.
pub fn pt_zip(seed: u64, specs: &[TensorSpec], options: &PtZipOptions) -> Fixture {
    let mut rng = Rng::new(seed);
    let prefix = &options.prefix;
    let flags = if options.torch_layout { 0x0808 } else { 0x0800 };
    let mut bytes = Vec::new();
    let mut entries = Vec::new();
    let mut tensors = Vec::new();
    let entry = |name: String, flags, method, size| ZipEntry {
        name,
        flags,
        method,
        stored: size,
        size,
        local_offset: 0,
    };

    if options.flaw != PtZipFlaw::NoPickle {
        let filler = rng.bytes(64);
        let method = if options.flaw == PtZipFlaw::CompressedPickle { 8 } else { 0 };
        let pickle = entry(format!("{prefix}/data.pkl"), flags, method, to_u32(filler.len()));
        zip_add(&mut bytes, &mut entries, pickle, &filler, options);
    }
    let order = b"little";
    let flags_order = if options.flaw == PtZipFlaw::Encrypted { flags | 1 } else { flags };
    let header = entry(format!("{prefix}/byteorder"), flags_order, 0, to_u32(order.len()));
    zip_add(&mut bytes, &mut entries, header, order, options);
    if options.deflated_code {
        let filler = rng.bytes(40);
        let mut code = entry(format!("{prefix}/code/__torch__.py"), flags, 8, 300);
        code.stored = to_u32(filler.len());
        zip_add(&mut bytes, &mut entries, code, &filler, options);
    }
    for (index, spec) in specs.iter().enumerate() {
        let data = spec.data(&mut rng);
        let name = format!("{prefix}/data/{index}");
        let method = if options.flaw == PtZipFlaw::CompressedStorage { 8 } else { 0 };
        let storage = entry(name.clone(), flags, method, to_u32(data.len()));
        let offset = zip_add(&mut bytes, &mut entries, storage, &data, options);
        let len = to_u64(data.len());
        let (shape, dtype_wire) = if len < 4096 {
            (vec![len], Elem::U8.dtype_wire())
        } else {
            (vec![len / spec.elem.block().1], spec.elem.dtype_wire())
        };
        tensors.push(Expected { name, offset, len, shape, dtype_wire });
    }
    let version = b"3\n";
    let trailer = entry(format!("{prefix}/version"), flags, 0, to_u32(version.len()));
    zip_add(&mut bytes, &mut entries, trailer, version, options);

    match options.flaw {
        PtZipFlaw::LocalNameDiffers => entries[0].name.push('x'),
        PtZipFlaw::OverlappingEntries => entries[1].local_offset = entries[0].local_offset,
        PtZipFlaw::None
        | PtZipFlaw::CompressedPickle
        | PtZipFlaw::CompressedStorage
        | PtZipFlaw::Zip64Marker
        | PtZipFlaw::Zip64Disagrees
        | PtZipFlaw::NoPickle
        | PtZipFlaw::Encrypted
        | PtZipFlaw::LocalSizeDiffers => {}
    }
    zip_central_and_end(&mut bytes, &entries, options);
    Fixture { bytes, tensors }
}

/// Appends the central directory and the end records.
fn zip_central_and_end(bytes: &mut Vec<u8>, entries: &[ZipEntry], options: &PtZipOptions) {
    let directory = to_u32(bytes.len());
    for entry in entries {
        le32(bytes, 0x0201_4B50);
        le16(bytes, 45);
        le16(bytes, 45);
        le16(bytes, entry.flags);
        le16(bytes, entry.method);
        le16(bytes, 0);
        le16(bytes, 0);
        le32(bytes, 0);
        le32(bytes, entry.stored);
        le32(bytes, entry.size);
        le16(bytes, u16::try_from(entry.name.len()).expect("short name"));
        le16(bytes, 0);
        le16(bytes, 0);
        le16(bytes, 0);
        le16(bytes, 0);
        le32(bytes, 0);
        le32(bytes, entry.local_offset);
        bytes.extend_from_slice(entry.name.as_bytes());
    }
    let size = to_u32(bytes.len()) - directory;
    let count = u16::try_from(entries.len()).expect("few entries");
    if options.torch_layout {
        let record = to_u64(bytes.len());
        let stated = u64::from(count) + u64::from(options.flaw == PtZipFlaw::Zip64Disagrees);
        le32(bytes, 0x0606_4B50);
        le64(bytes, 44);
        le16(bytes, 0x031E);
        le16(bytes, 45);
        le32(bytes, 0);
        le32(bytes, 0);
        le64(bytes, stated);
        le64(bytes, stated);
        le64(bytes, u64::from(size));
        le64(bytes, u64::from(directory));
        le32(bytes, 0x0706_4B50);
        le32(bytes, 0);
        le64(bytes, record);
        le32(bytes, 1);
    }
    le32(bytes, 0x0605_4B50);
    le16(bytes, 0);
    le16(bytes, 0);
    le16(bytes, count);
    le16(bytes, count);
    le32(bytes, size);
    le32(bytes, if options.flaw == PtZipFlaw::Zip64Marker { u32::MAX } else { directory });
    le16(bytes, 0);
}
