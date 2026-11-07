//! File: profile/onnx.rs
//! Purpose: a protobuf wire walker that finds tensor bytes in an ONNX model.
//!
//! Main responsibilities:
//! - [`matches`]: ONNX has no magic, so a bounded check of the top-level `ModelProto` fields;
//! - [`parse`]: descends through the graph, its nodes and subgraphs to every `TensorProto`, and
//!   reports `raw_data` and packed `float_data`.
//!
//! Key functions: [`matches`], [`parse`].
//!
//! Implementation notes: field numbers come from `onnx.proto`, and no protobuf dependency is
//! involved. Only those two fields are byte runs in file order. Varint fields and unpacked floats
//! are not, so such a tensor gets no range and no row. A tensor with `external_data` is ignored and
//! its location is never read or opened. A `Constant` tensor often has no name of its own, so it
//! takes the first output of its node; a tensor left unnamed gets a range but no row. Each message
//! is walked once, so work is linear in the file, and a budget on visited fields bounds it further.

use super::plan::MAX_TENSOR_DIMS;
use super::reader::Reader;
use super::{MAX_NAME_BYTES, MAX_TENSORS, ParseResult, Stop};
use crate::types::{Dtype, Profile, RangePlan, TensorInfo, TypedRange};

/// Most top-level fields the structural check walks before giving up.
const MAX_TOP_LEVEL_FIELDS: u32 = 64 * 1024;

/// Most fields one walk visits. Real models stay far below it: tensor payloads count as one field.
const MAX_FIELDS: u64 = 1 << 26;

/// Deepest nesting of subgraphs followed.
const MAX_GRAPH_DEPTH: u32 = 64;

/// `ModelProto.ir_version`.
const MODEL_IR_VERSION: u64 = 1;
/// `ModelProto.graph`.
const MODEL_GRAPH: u64 = 7;

/// `GraphProto.node`.
const GRAPH_NODE: u64 = 1;
/// `GraphProto.initializer`.
const GRAPH_INITIALIZER: u64 = 5;

/// `NodeProto.output`.
const NODE_OUTPUT: u64 = 2;
/// `NodeProto.attribute`.
const NODE_ATTRIBUTE: u64 = 5;

/// `AttributeProto.t`.
const ATTRIBUTE_TENSOR: u64 = 5;
/// `AttributeProto.g`.
const ATTRIBUTE_GRAPH: u64 = 6;
/// `AttributeProto.tensors`.
const ATTRIBUTE_TENSORS: u64 = 10;
/// `AttributeProto.graphs`.
const ATTRIBUTE_GRAPHS: u64 = 11;

/// `TensorProto.dims`.
const TENSOR_DIMS: u64 = 1;
/// `TensorProto.data_type`.
const TENSOR_DATA_TYPE: u64 = 2;
/// `TensorProto.float_data`.
const TENSOR_FLOAT_DATA: u64 = 4;
/// `TensorProto.name`.
const TENSOR_NAME: u64 = 8;
/// `TensorProto.raw_data`.
const TENSOR_RAW_DATA: u64 = 9;
/// `TensorProto.external_data`.
const TENSOR_EXTERNAL_DATA: u64 = 13;
/// `TensorProto.data_location`.
const TENSOR_DATA_LOCATION: u64 = 14;
/// `TensorProto.DataLocation.EXTERNAL`.
const LOCATION_EXTERNAL: u64 = 1;

/// A span of the source: `len` bytes from `start`.
#[derive(Debug, Clone, Copy)]
struct Span {
    start: u64,
    len: u64,
}

impl Span {
    /// End of the span. [`next_field`] proved it lies inside the source.
    fn end(self) -> u64 {
        self.start + self.len
    }
}

/// The value of one protobuf field.
#[derive(Debug, Clone, Copy)]
enum Value {
    /// Wire type 0.
    Varint(u64),
    /// Wire type 2, already skipped by the reader.
    Bytes(Span),
    /// Wire types 1 and 5, skipped.
    Fixed,
}

/// Reads the tag and value at the cursor and leaves the cursor after the value, at or before
/// `end`. A length-delimited value that leaves `end`, which is what a false length looks like, and
/// a group or reserved wire type all stop as malformed.
fn next_field(reader: &mut Reader<'_>, end: u64) -> ParseResult<(u64, Value)> {
    let tag = reader.varint()?;
    let number = tag >> 3;
    if number == 0 {
        return Err(Stop::Malformed("protobuf field number 0"));
    }
    let value = match tag & 7 {
        0 => Value::Varint(reader.varint()?),
        1 => {
            reader.skip(8)?;
            Value::Fixed
        }
        2 => {
            let len = reader.varint()?;
            let start = reader.pos();
            reader.skip(len)?;
            Value::Bytes(Span { start, len })
        }
        5 => {
            reader.skip(4)?;
            Value::Fixed
        }
        _ => return Err(Stop::Malformed("unsupported protobuf wire type")),
    };
    if reader.pos() > end {
        return Err(Stop::Malformed("protobuf field leaves its message"));
    }
    Ok((number, value))
}

/// Whether the whole source is a sequence of `ModelProto` fields holding a version and a graph.
///
/// The field walk touches a few small reads, since payloads are skipped. Any mismatch is `false`.
pub(super) fn matches(reader: &mut Reader<'_>) -> ParseResult<bool> {
    match model_fields_found(reader) {
        Ok(found) => Ok(found),
        Err(Stop::Malformed(_)) => Ok(false),
        Err(Stop::Io(error)) => Err(Stop::Io(error)),
    }
}

fn model_fields_found(reader: &mut Reader<'_>) -> ParseResult<bool> {
    reader.seek(0)?;
    let end = reader.len();
    let (mut has_version, mut has_graph) = (false, false);
    let mut visited = 0;
    while reader.pos() < end {
        visited += 1;
        if visited > MAX_TOP_LEVEL_FIELDS {
            return Ok(false);
        }
        let (number, value) = next_field(reader, end)?;
        if !is_model_field(number, value) {
            return Ok(false);
        }
        has_version |= number == MODEL_IR_VERSION;
        has_graph |= number == MODEL_GRAPH;
    }
    Ok(has_version && has_graph)
}

/// Whether `number` is a field of `ModelProto` carried with its proper wire type.
fn is_model_field(number: u64, value: Value) -> bool {
    match value {
        Value::Varint(_) => matches!(number, 1 | 5),
        Value::Bytes(_) => matches!(number, 2 | 3 | 4 | 6 | 7 | 8 | 14 | 20 | 25 | 26),
        Value::Fixed => false,
    }
}

/// Walks the model at the start of the source. [`matches`] has accepted the top level.
pub(super) fn parse(reader: &mut Reader<'_>) -> ParseResult<RangePlan> {
    reader.seek(0)?;
    let end = reader.len();
    let mut walker = Walker {
        reader,
        plan: RangePlan { profile: Profile::Onnx, ranges: Vec::new(), tensors: Vec::new() },
        fields_left: MAX_FIELDS,
    };
    while walker.reader.pos() < end {
        let (number, value) = walker.field(end)?;
        if let (MODEL_GRAPH, Value::Bytes(span)) = (number, value) {
            walker.graph(span, 0)?;
        }
    }
    Ok(walker.plan)
}

/// The tensor fields that matter, gathered in any order.
struct RawTensor {
    name: String,
    dims: Vec<u64>,
    data_type: u64,
    raw: Option<Span>,
    floats: Option<Span>,
    external: bool,
}

/// One walk of a model.
struct Walker<'r, 'a> {
    reader: &'r mut Reader<'a>,
    plan: RangePlan,
    fields_left: u64,
}

impl Walker<'_, '_> {
    /// Reads one field, charging the budget.
    fn field(&mut self, end: u64) -> ParseResult<(u64, Value)> {
        self.fields_left =
            self.fields_left.checked_sub(1).ok_or(Stop::Malformed("ONNX has too many fields"))?;
        next_field(self.reader, end)
    }

    /// Walks a `GraphProto`. On return the cursor is at the end of `span`, as for all messages.
    fn graph(&mut self, span: Span, depth: u32) -> ParseResult<()> {
        if depth > MAX_GRAPH_DEPTH {
            return Err(Stop::Malformed("ONNX subgraphs nest too deeply"));
        }
        self.reader.seek(span.start)?;
        while self.reader.pos() < span.end() {
            match self.field(span.end())? {
                (GRAPH_NODE, Value::Bytes(node)) => self.node(node, depth)?,
                (GRAPH_INITIALIZER, Value::Bytes(tensor)) => {
                    let tensor = self.tensor(tensor)?;
                    self.add_tensor(tensor, None)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Walks a `NodeProto`. Attribute tensors wait for the end, since `output` may follow them.
    fn node(&mut self, span: Span, depth: u32) -> ParseResult<()> {
        self.reader.seek(span.start)?;
        let mut output = None;
        let mut tensors = Vec::new();
        while self.reader.pos() < span.end() {
            match self.field(span.end())? {
                (NODE_OUTPUT, Value::Bytes(name)) if output.is_none() => {
                    output = Some(self.string(name)?);
                }
                (NODE_ATTRIBUTE, Value::Bytes(attribute)) => {
                    self.attribute(attribute, depth, &mut tensors)?;
                }
                _ => {}
            }
        }
        for tensor in tensors {
            self.add_tensor(tensor, output.as_deref())?;
        }
        Ok(())
    }

    /// Walks an `AttributeProto`, descending into its subgraphs at once.
    fn attribute(
        &mut self,
        span: Span,
        depth: u32,
        tensors: &mut Vec<RawTensor>,
    ) -> ParseResult<()> {
        self.reader.seek(span.start)?;
        while self.reader.pos() < span.end() {
            match self.field(span.end())? {
                (ATTRIBUTE_TENSOR | ATTRIBUTE_TENSORS, Value::Bytes(tensor)) => {
                    if tensors.len() >= MAX_TENSORS {
                        return Err(Stop::Malformed("ONNX node has too many tensors"));
                    }
                    tensors.push(self.tensor(tensor)?);
                }
                (ATTRIBUTE_GRAPH | ATTRIBUTE_GRAPHS, Value::Bytes(graph)) => {
                    self.graph(graph, depth + 1)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Reads a `TensorProto`. As in protobuf, the last occurrence of a singular field wins.
    fn tensor(&mut self, span: Span) -> ParseResult<RawTensor> {
        self.reader.seek(span.start)?;
        let mut tensor = RawTensor {
            name: String::new(),
            dims: Vec::new(),
            data_type: 0,
            raw: None,
            floats: None,
            external: false,
        };
        while self.reader.pos() < span.end() {
            match self.field(span.end())? {
                (TENSOR_DIMS, Value::Varint(dim)) => push_dim(&mut tensor.dims, dim)?,
                (TENSOR_DIMS, Value::Bytes(packed)) => self.packed_dims(packed, &mut tensor.dims)?,
                (TENSOR_DATA_TYPE, Value::Varint(data_type)) => tensor.data_type = data_type,
                (TENSOR_FLOAT_DATA, Value::Bytes(run)) => tensor.floats = Some(run),
                (TENSOR_NAME, Value::Bytes(name)) => tensor.name = self.string(name)?,
                (TENSOR_RAW_DATA, Value::Bytes(run)) => tensor.raw = Some(run),
                (TENSOR_EXTERNAL_DATA, _)
                | (TENSOR_DATA_LOCATION, Value::Varint(LOCATION_EXTERNAL)) => {
                    tensor.external = true;
                }
                _ => {}
            }
        }
        Ok(tensor)
    }

    /// Reads a packed run of dimensions.
    fn packed_dims(&mut self, span: Span, dims: &mut Vec<u64>) -> ParseResult<()> {
        self.reader.seek(span.start)?;
        while self.reader.pos() < span.end() {
            push_dim(dims, self.reader.varint()?)?;
        }
        if self.reader.pos() > span.end() {
            return Err(Stop::Malformed("ONNX dims leave their field"));
        }
        Ok(())
    }

    /// Reads a string field and returns to the end of it.
    fn string(&mut self, span: Span) -> ParseResult<String> {
        self.reader.seek(span.start)?;
        let cap = u64::try_from(MAX_NAME_BYTES).unwrap_or(u64::MAX);
        let bytes = self.reader.take(span.len, cap)?;
        String::from_utf8(bytes).map_err(|_| Stop::Malformed("ONNX name is not UTF-8"))
    }

    /// Adds the bytes of a finished tensor to the plan, named after `fallback` if it has no name.
    fn add_tensor(&mut self, tensor: RawTensor, fallback: Option<&str>) -> ParseResult<()> {
        if tensor.external {
            return Ok(());
        }
        let payload = match (tensor.raw, tensor.floats) {
            (Some(raw), _) => Some((raw, dtype_of(tensor.data_type))),
            (None, Some(floats)) => Some((floats, Dtype::F32)),
            (None, None) => None,
        };
        let Some((span, dtype)) = payload else {
            return Ok(());
        };
        if self.plan.ranges.len() >= MAX_TENSORS {
            return Err(Stop::Malformed("ONNX has too many tensors"));
        }
        let range = TypedRange { offset: span.start, len: span.len, hint: dtype };
        self.plan.ranges.push(range);
        let name = if tensor.name.is_empty() {
            fallback.unwrap_or_default().to_owned()
        } else {
            tensor.name
        };
        if !name.is_empty() {
            let (offset, len) = (span.start, span.len);
            self.plan.tensors.push(TensorInfo { name, offset, len, shape: tensor.dims, dtype });
        }
        Ok(())
    }
}

/// Appends one dimension, enforcing the table's limit before the vector grows.
fn push_dim(dims: &mut Vec<u64>, dim: u64) -> ParseResult<()> {
    if dims.len() >= MAX_TENSOR_DIMS {
        return Err(Stop::Malformed("ONNX tensor has too many dimensions"));
    }
    dims.push(dim);
    Ok(())
}

/// Maps `TensorProto.DataType`. An element type with no [`Dtype`] means no dtype was determined.
fn dtype_of(data_type: u64) -> Dtype {
    match data_type {
        1 => Dtype::F32,
        2 => Dtype::U8,
        3 => Dtype::I8,
        4 => Dtype::U16,
        5 => Dtype::I16,
        6 => Dtype::I32,
        7 => Dtype::I64,
        9 => Dtype::Bool,
        10 => Dtype::F16,
        11 => Dtype::F64,
        12 => Dtype::U32,
        13 => Dtype::U64,
        16 => Dtype::Bf16,
        _ => Dtype::from_wire(0),
    }
}

#[cfg(test)]
mod tests {
    use super::dtype_of;
    use crate::types::Dtype;

    #[test]
    fn element_types_map_to_dtypes() {
        assert_eq!(dtype_of(1), Dtype::F32);
        assert_eq!(dtype_of(16), Dtype::Bf16);
        for string_or_complex_or_unknown in [0, 8, 14, 15, 17, u64::MAX] {
            assert_eq!(dtype_of(string_or_complex_or_unknown), Dtype::Unknown(0));
        }
    }
}
