//! File: profile/plan.rs
//! Purpose: turns a parser's raw plan into one that is safe to hand to a writer.
//!
//! Main responsibilities:
//! - drops ranges that leave the file, are empty or too short, or overlap one already kept;
//! - trims each range to a whole number of blocks of its dtype;
//! - keeps a tensor row reachable by name unless it breaks a rule of the tensor table.
//!
//! Key functions: [`normalize`]. Key constants: [`MIN_RANGE_BYTES`], [`MAX_TENSOR_DIMS`].
//!
//! Implementation notes: this enforces the spec invariants a source file could otherwise violate
//! through the tensor table (unique names, range inside the file, at most 16 dimensions), and
//! bounds the names, so a hostile input costs ratio and never makes a writer's index fail
//! validation.

use std::cmp::Reverse;
use std::collections::HashSet;

use crate::types::{RangePlan, TensorInfo, TypedRange};

/// A range shorter than this stays in the skeleton, spec section 11. Its tensor stays reachable.
pub const MIN_RANGE_BYTES: u64 = 4096;

/// Most dimensions of a tensor table row, spec section 4.
pub(super) const MAX_TENSOR_DIMS: usize = 16;

/// Most bytes of tensor names kept per file. Real tables hold a few MiB at most, and the names of a
/// hostile table must not fill the spec's default 256 MiB index cap.
const MAX_NAME_TABLE_BYTES: usize = 64 << 20;

/// Makes `plan` valid for a source of `file_len` bytes.
///
/// `ranges` come back sorted by offset and pairwise disjoint, each inside the file, a whole number
/// of blocks of its hint, and at least [`MIN_RANGE_BYTES`] long. A dtype with no known block is not
/// trimmed. At equal offsets the longest range wins, otherwise the earlier one does, so the result
/// does not depend on parser order. Trimmed tail bytes fall to the skeleton.
///
/// `tensors` keep their order. A row is removed only if it leaves the file, has over 16 dimensions,
/// repeats a name, where the first row wins, or would take the names past
/// [`MAX_NAME_TABLE_BYTES`]. Rows whose range was dropped or trimmed stay exactly as they were. The
/// profile is kept. Normalizing twice changes nothing.
#[must_use]
pub fn normalize(plan: RangePlan, file_len: u64) -> RangePlan {
    let RangePlan { profile, ranges, tensors } = plan;
    let total = ranges.len();
    let ranges = disjoint_ranges(ranges, file_len);
    let row_count = tensors.len();
    let tensors = valid_rows(tensors, file_len);
    tracing::debug!(
        file_len,
        ranges = ranges.len(),
        dropped_ranges = total - ranges.len(),
        dropped_rows = row_count - tensors.len(),
        "plan normalized"
    );
    RangePlan { profile, ranges, tensors }
}

/// Trims every range, then keeps a disjoint subset in offset order.
fn disjoint_ranges(ranges: Vec<TypedRange>, file_len: u64) -> Vec<TypedRange> {
    let mut trimmed: Vec<TypedRange> =
        ranges.into_iter().filter_map(|range| trim(range, file_len)).collect();
    trimmed.sort_by_key(|range| (range.offset, Reverse(range.len)));
    let mut kept = Vec::with_capacity(trimmed.len());
    let mut end = 0;
    for range in trimmed {
        if range.offset >= end {
            // `trim` proved offset + len <= file_len.
            end = range.offset + range.len;
            kept.push(range);
        }
    }
    kept
}

/// Cuts `range` to whole blocks. `None` if it leaves the file or ends up under the minimum.
fn trim(range: TypedRange, file_len: u64) -> Option<TypedRange> {
    let end = range.offset.checked_add(range.len)?;
    if end > file_len {
        return None;
    }
    let block = u64::from(range.hint.block_bytes().unwrap_or(1)).max(1);
    let len = range.len - range.len % block;
    (len >= MIN_RANGE_BYTES).then_some(TypedRange { len, ..range })
}

/// Removes the rows the tensor table cannot hold.
fn valid_rows(mut tensors: Vec<TensorInfo>, file_len: u64) -> Vec<TensorInfo> {
    let mut seen = HashSet::with_capacity(tensors.len());
    let mut name_budget = MAX_NAME_TABLE_BYTES;
    tensors.retain(|tensor| {
        let inside = tensor.offset.checked_add(tensor.len).is_some_and(|end| end <= file_len);
        let keep = inside
            && tensor.shape.len() <= MAX_TENSOR_DIMS
            && tensor.name.len() <= name_budget
            && seen.insert(tensor.name.clone());
        if keep {
            name_budget -= tensor.name.len();
        }
        keep
    });
    tensors
}

#[cfg(test)]
mod tests {
    use super::{MAX_NAME_TABLE_BYTES, MIN_RANGE_BYTES, normalize};
    use crate::types::{Dtype, Profile, RangePlan, TensorInfo, TypedRange};

    const MIN: u64 = MIN_RANGE_BYTES;

    fn range(offset: u64, len: u64, hint: Dtype) -> TypedRange {
        TypedRange { offset, len, hint }
    }

    fn row(name: &str, offset: u64, len: u64) -> TensorInfo {
        TensorInfo { name: name.to_owned(), offset, len, shape: vec![len], dtype: Dtype::U8 }
    }

    fn plan(ranges: Vec<TypedRange>, tensors: Vec<TensorInfo>) -> RangePlan {
        RangePlan { profile: Profile::Safetensors, ranges, tensors }
    }

    fn ranges_of(input: Vec<TypedRange>, file_len: u64) -> Vec<TypedRange> {
        normalize(plan(input, Vec::new()), file_len).ranges
    }

    #[test]
    fn ranges_leaving_the_file_are_dropped() {
        let cases = [
            range(0, MIN + 1, Dtype::U8),
            range(1, MIN, Dtype::U8),
            range(u64::MAX, MIN, Dtype::U8),
            range(u64::MAX - 1, u64::MAX, Dtype::U8),
        ];
        assert_eq!(ranges_of(cases.to_vec(), MIN), []);
        assert_eq!(ranges_of(vec![range(0, MIN, Dtype::U8)], MIN).len(), 1);
    }

    #[test]
    fn short_and_empty_ranges_are_dropped_but_their_rows_stay() {
        let out = normalize(
            plan(
                vec![range(0, 0, Dtype::F32), range(4, MIN - 1, Dtype::U8)],
                vec![row("empty", 0, 0), row("small", 4, MIN - 1)],
            ),
            MIN * 2,
        );
        assert_eq!(out.ranges, []);
        assert_eq!(out.tensors.len(), 2);
    }

    #[test]
    fn ranges_are_trimmed_to_their_block_and_rows_are_not() {
        let q4 = Dtype::Q4K.block_bytes().map(u64::from).expect("known block");
        let out = normalize(
            plan(
                vec![
                    range(100, MIN * 2 + 3, Dtype::F32),
                    range(MIN * 3, q4 * 100 + 7, Dtype::Q4K),
                    range(MIN * 300, MIN + 5, Dtype::Unknown(0x7777)),
                ],
                vec![row("a", 100, MIN * 2 + 3)],
            ),
            MIN * 400,
        );
        let lens: Vec<u64> = out.ranges.iter().map(|r| r.len).collect();
        assert_eq!(lens, [MIN * 2, q4 * 100, MIN + 5]);
        assert_eq!(out.tensors[0].len, MIN * 2 + 3);
    }

    #[test]
    fn a_trim_that_falls_under_the_minimum_drops_the_range() {
        let block = Dtype::Q6K.block_bytes().map(u64::from).expect("known block");
        let len = MIN.div_ceil(block) * block - 1;
        assert_eq!(ranges_of(vec![range(0, len, Dtype::Q6K)], MIN * 2), []);
    }

    #[test]
    fn overlaps_keep_the_earlier_range_whatever_the_input_order() {
        let a = range(0, MIN * 2, Dtype::U8);
        let b = range(MIN, MIN * 2, Dtype::U8);
        let c = range(MIN * 2, MIN, Dtype::U8);
        for input in [[a, b, c], [c, b, a], [b, a, c], [b, c, a]] {
            assert_eq!(ranges_of(input.to_vec(), MIN * 4), [a, c]);
        }
    }

    #[test]
    fn at_one_offset_the_longest_range_wins() {
        let short = range(0, MIN, Dtype::U8);
        let long = range(0, MIN * 2, Dtype::F32);
        assert_eq!(ranges_of(vec![short, long], MIN * 2), [long]);
        assert_eq!(ranges_of(vec![long, short], MIN * 2), [long]);
    }

    #[test]
    fn a_trimmed_tail_does_not_block_its_neighbour() {
        let first = range(0, MIN + 3, Dtype::F32);
        let second = range(MIN, MIN, Dtype::U8);
        let out = ranges_of(vec![first, second], MIN * 3);
        assert_eq!(out, [range(0, MIN, Dtype::F32), second]);
    }

    #[test]
    fn rows_the_table_cannot_hold_are_removed_and_order_is_kept() {
        let mut deep = row("deep", 0, 1);
        deep.shape = vec![1; 17];
        let mut deepest_valid = row("sixteen", 0, 1);
        deepest_valid.shape = vec![1; 16];
        let input = vec![
            row("b", 8, 2),
            row("outside", 9, 2),
            row("overflow", u64::MAX, 2),
            deep,
            deepest_valid,
            row("b", 0, 1),
            row("a", 10, 0),
        ];
        let out = normalize(plan(Vec::new(), input), 10);
        let names: Vec<&str> = out.tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["b", "sixteen", "a"]);
        assert_eq!(out.tensors[0].offset, 8);
    }

    #[test]
    fn names_past_the_table_budget_are_removed() {
        let name = |ch: &str, size: usize| ch.repeat(size);
        let input = vec![
            row(&name("a", MAX_NAME_TABLE_BYTES / 2), 0, 0),
            row(&name("b", MAX_NAME_TABLE_BYTES / 2 - 1), 0, 0),
            row(&name("c", 2), 0, 0),
            row("d", 0, 0),
        ];
        let out = normalize(plan(Vec::new(), input), 0);
        assert_eq!(out.tensors.len(), 3);
        assert_eq!(out.tensors[2].name, "d");
        assert_eq!(normalize(out.clone(), 0), out);
    }

    #[test]
    fn normalizing_twice_changes_nothing() {
        let input = plan(
            vec![
                range(0, MIN + 3, Dtype::F32),
                range(MIN - 1, MIN, Dtype::U8),
                range(MIN * 2, 10, Dtype::U8),
            ],
            vec![row("a", 0, MIN), row("a", 1, 1)],
        );
        let once = normalize(input, MIN * 3);
        assert_eq!(normalize(once.clone(), MIN * 3), once);
    }

    #[test]
    fn the_profile_survives_even_with_nothing_left() {
        let out = normalize(plan(vec![range(0, 8, Dtype::U8)], Vec::new()), 0);
        assert_eq!(out.profile, Profile::Safetensors);
        assert_eq!(out.ranges, []);
    }
}
