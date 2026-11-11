//! File: `format/validate_tests.rs`
//! Purpose: tests of `validate.rs`: at least one reject test per invariant of spec section 9, the
//! accepted boundary of each rule, hostile values at `u64::MAX`, and cost under shared ranges.
//!
//! Implementation notes: each test corrupts one field of the valid `Fixture`, or of its variant
//! with a delta blob for invariant 13. Invariant 12 is the
//! writer's commit protocol and the second half of invariant 10 is decode-time; neither touches
//! this module. The file is separate only because of its length.

use std::cmp::Reverse;

use super::{RangeMax, check_decode_budget, check_restore_name, validate};
use crate::chain::{RawStep, STEP_PLANES, STEP_PRIVATE_FIRST, STEP_RESERVED_DELTA_FIRST, STEP_WHOLE};
use crate::chain::{VIEW_BF16_TO_F32, VIEW_F16_TO_F32};
use crate::error::Error;
use crate::format::builder::IndexBuilder;
use crate::format::footer::Footer;
use crate::format::header::Header;
use crate::format::index::Index;
use crate::format::limits::Limits;
use crate::format::records::{
    ChainRecord, ChunkRecord, DeltaParams, ExtentRecord, StrRef, TensorRecord,
};
use crate::format::testkit::{DATA_END, FILE_LEN, Fixture};
use crate::chain::MAX_CHUNK_SIZE;
use crate::format::MemberRecord;
use crate::types::Dtype;

/// Applies `corrupt` to a valid fixture and expects `MalformedIndex` for `section`.
fn rejected(section: &str, corrupt: impl FnOnce(&mut Fixture)) {
    let mut fixture = Fixture::valid();
    corrupt(&mut fixture);
    let error = fixture.check().expect_err("corrupted");
    assert!(
        matches!(&error, Error::MalformedIndex { section: got, .. } if *got == section),
        "{error:?}"
    );
}

/// [`rejected`] over [`Fixture::with_delta`].
fn delta_rejected(section: &str, corrupt: impl FnOnce(&mut Fixture)) {
    let mut fixture = Fixture::with_delta();
    corrupt(&mut fixture);
    let error = fixture.check().expect_err("corrupted");
    assert!(
        matches!(&error, Error::MalformedIndex { section: got, .. } if *got == section),
        "{error:?}"
    );
}

/// [`accepted`] over [`Fixture::with_delta`].
fn delta_accepted(change: impl FnOnce(&mut Fixture)) {
    let mut fixture = Fixture::with_delta();
    change(&mut fixture);
    fixture.check().expect("still valid");
}

/// Applies `change` to a valid fixture and expects it to stay valid.
fn accepted(change: impl FnOnce(&mut Fixture)) {
    let mut fixture = Fixture::valid();
    change(&mut fixture);
    fixture.check().expect("still valid");
}

/// Start of the bytes of the fixture's data region that no chunk uses, for a chunk that must grow.
const FREE_DATA: u64 = 150;

/// Small deterministic generator, xorshift64*. Tests use it for repeatable data.
struct Prng(u64);

impl Prng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

/// The header of a file without delta blobs.
fn header() -> Header {
    Header::current()
}

#[test]
fn the_fixture_is_valid() {
    Fixture::valid().check().expect("valid");
    Fixture::with_delta().check().expect("valid");
}

#[test]
fn inv1_validate_applies_the_footer_layout() {
    let fixture = Fixture::valid();
    let limits = Limits::default();
    let index = Index::parse(&fixture.serialize(), &limits).expect("parses");
    let error =
        validate(&index, &header(), &Fixture::footer(), FILE_LEN + 1, &limits).expect_err("wrong length");
    assert!(matches!(error, Error::MalformedIndex { section: "footer", .. }));
    let footer = Footer { index_offset: 63, ..Fixture::footer() };
    let error = validate(&index, &header(), &footer, FILE_LEN, &limits).expect_err("index in header");
    assert!(matches!(error, Error::MalformedIndex { section: "footer", .. }));
}

#[test]
fn inv2_validate_applies_the_index_cap() {
    let raw = Fixture::valid().serialize();
    let index = Index::parse(&raw, &Limits::default()).expect("parses");
    let limits = Limits { max_index_bytes: 1023, ..Limits::default() };
    let error =
        validate(&index, &header(), &Fixture::footer(), FILE_LEN, &limits).expect_err("over cap");
    assert!(matches!(error, Error::BudgetExceeded { what: "index", .. }));
}

#[test]
fn inv3_member_name_must_be_in_the_heap() {
    rejected("STRS", |f| f.members[0].name = StrRef { off: 19, len: 1 });
    // `off + len` overflows u32; `len` stays under the name budget, which is checked first.
    rejected("STRS", |f| f.members[0].name = StrRef { off: u32::MAX, len: 2 });
    rejected("STRS", |f| f.tensors[0].name = StrRef { off: 20, len: 0 });
}

#[test]
fn inv3_names_must_be_utf8() {
    rejected("STRS", |f| f.strings[0] = 0xFF);
    rejected("STRS", |f| f.strings[15] = 0xFF);
}

#[test]
fn inv3_utf8_is_required_only_of_referenced_bytes() {
    accepted(|f| f.strings.extend_from_slice(&[0xFF, 0xFE]));
}

#[test]
fn inv3_member_names_must_be_unique() {
    rejected("MEMB", |f| f.members[1].name = f.members[0].name);
    rejected("MEMB", |f| {
        f.strings.extend_from_slice(b"a.bin");
        f.members[1].name = StrRef { off: 19, len: 5 };
    });
}

#[test]
fn inv3_tensor_names_must_be_unique_within_a_member() {
    rejected("MEMB", |f| f.tensors[1].name = f.tensors[0].name);
    rejected("MEMB", |f| {
        f.strings.extend_from_slice(b"t0");
        f.tensors[1].name = StrRef { off: 19, len: 2 };
    });
}

#[test]
fn inv3_the_same_tensor_name_in_two_members_is_legal() {
    accepted(|f| {
        let duplicate = TensorRecord { virt_len: 4, ..f.tensors[0] };
        f.tensors.push(duplicate);
        f.members[1].tensor_count = 1;
    });
}

#[test]
fn inv3_ndim_limit() {
    rejected("TENS", |f| {
        f.dims = vec![1; 17];
        f.tensors[0].ndim = 17;
    });
    rejected("TENS", |f| f.tensors[0].ndim = u16::MAX);
    accepted(|f| {
        f.dims.extend_from_slice(&[1; 16]);
        f.tensors[0].shape_off = 3;
        f.tensors[0].ndim = 16;
    });
    accepted(|f| f.tensors[0].ndim = 0);
}

#[test]
fn inv3_tensor_range_of_a_member_must_be_in_tens() {
    rejected("MEMB", |f| f.members[0].tensor_count = 3);
    rejected("MEMB", |f| f.members[1].first_tensor = 3);
    rejected("MEMB", |f| {
        f.members[0].first_tensor = u32::MAX;
        f.members[0].tensor_count = u32::MAX;
    });
    accepted(|f| f.members[2].first_tensor = 2);
}

#[test]
fn inv4_chunks_must_lie_in_the_data_region() {
    rejected("CHNK", |f| f.chunks[0].file_offset = 63);
    rejected("CHNK", |f| f.chunks[0].file_offset = 0);
    rejected("CHNK", |f| f.chunks[4].file_offset = DATA_END - 23);
    accepted(|f| f.chunks[4].file_offset = DATA_END - 24);
}

#[test]
fn inv4_chunk_end_overflow_is_a_rejection() {
    rejected("CHNK", |f| f.chunks[0].file_offset = u64::MAX);
    rejected("CHNK", |f| f.chunks[0].file_offset = u64::MAX - 4);
}

#[test]
fn inv4_partial_overlap_is_rejected() {
    rejected("CHNK", |f| f.chunks[1].file_offset = 68);
    rejected("CHNK", |f| f.chunks[3].file_offset = 90);
}

#[test]
fn inv4_same_offset_with_another_length_is_rejected() {
    rejected("CHNK", |f| {
        f.chunks[1].file_offset = 64;
    });
}

#[test]
fn inv4_identical_ranges_are_legal() {
    accepted(|f| {
        f.chunks[1].stored_len = 5;
        f.chunks[1].file_offset = 64;
    });
    accepted(|f| {
        let first = f.chunks[0];
        f.chunks.push(first);
        f.chunks.push(first);
    });
}

#[test]
fn inv4_a_range_nested_in_identical_pairs_is_rejected() {
    rejected("CHNK", |f| {
        let first = f.chunks[3];
        f.chunks.push(first);
        f.chunks.push(ChunkRecord { file_offset: first.file_offset + 1, stored_len: 2, ..first });
    });
}

#[test]
fn inv4_unreferenced_data_bytes_are_legal() {
    accepted(|f| {
        f.chunks[3].file_offset = 90;
        f.chunks[4].file_offset = 150;
    });
}

#[test]
fn inv5_chunk_range_must_be_in_chnk() {
    rejected("BLOB", |f| f.blobs[1].first_chunk = 4);
    rejected("BLOB", |f| f.blobs[1].first_chunk = u64::MAX);
    rejected("BLOB", |f| f.blobs[1].first_chunk = u64::MAX - 1);
}

#[test]
fn inv5_chunk_count_must_be_the_ceiling() {
    rejected("BLOB", |f| f.blobs[0].chunk_count = 2);
    rejected("BLOB", |f| f.blobs[0].chunk_count = 4);
    rejected("BLOB", |f| f.blobs[0].raw_len = 13);
    rejected("BLOB", |f| f.blobs[0].raw_len = u64::MAX);
    accepted(|f| {
        f.blobs[0].raw_len = 9;
        f.chunks[2].stored_len = 5;
        f.extents[0].virt_len = 9;
        f.extents[1].virt_offset = 9;
        f.members[0].file_len = 25;
        f.tensors[0].virt_len = 9;
        f.tensors[1].virt_offset = 9;
    });
    accepted(|f| {
        f.blobs[0].raw_len = 0;
        f.blobs[0].chunk_count = 0;
        f.extents[0].view_id = 0;
        f.extents[0].blob_id = 1;
        f.extents[0].virt_len = 10;
        f.extents[0].blob_offset = 0;
    });
}

#[test]
fn inv5_chunk_size_limits() {
    rejected("BLOB", |f| f.blobs[0].chunk_size = 0);
    rejected("BLOB", |f| {
        f.blobs[0].chunk_size = MAX_CHUNK_SIZE + 1;
        f.blobs[0].chunk_count = 1;
    });
    rejected("BLOB", |f| {
        f.blobs[0].chunk_size = u32::MAX;
        f.blobs[0].chunk_count = 1;
    });
    accepted(|f| {
        f.blobs[0].chunk_size = MAX_CHUNK_SIZE;
        f.blobs[0].chunk_count = 1;
        f.blobs[0].first_chunk = 0;
        f.chunks[0].stored_len = 5;
    });
}

#[test]
fn inv5_chunk_size_and_raw_len_must_be_multiples_of_the_chain_block() {
    rejected("BLOB", |f| {
        f.blobs[1].chunk_size = 6;
        f.blobs[1].chunk_count = 3;
    });
    rejected("BLOB", |f| {
        f.blobs[1].raw_len = 14;
    });
}

#[test]
fn inv5_chain_id_must_be_in_range() {
    rejected("BLOB", |f| f.blobs[0].chain_id = 2);
    rejected("BLOB", |f| f.blobs[0].chain_id = u32::MAX);
}

#[test]
fn inv5_chains_must_satisfy_section_5_2() {
    rejected("CHAN", |f| f.chains[0].step_count = 0);
    rejected("CHAN", |f| f.chains[1].step_count = 3);
    rejected("CHAN", |f| f.chains[1].step_count = u16::MAX);
    rejected("CHAN", |f| f.chains[1].first_step = 3);
    rejected("CHAN", |f| f.chains[0].first_step = u32::MAX);
    rejected("CHAN", |f| {
        f.steps.swap(1, 2);
    });
    rejected("CHAN", |f| {
        let whole = RawStep { codec_id: STEP_WHOLE, param0: 0, param1: 0 };
        f.steps = vec![whole; 8];
        f.chains = vec![
            ChainRecord { first_step: 0, step_count: 1 },
            ChainRecord { first_step: 0, step_count: 8 },
        ];
    });
}

#[test]
fn inv5_stored_len_bounds_of_inner_chunks() {
    rejected("BLOB", |f| f.chunks[0].stored_len = 3);
    rejected("BLOB", |f| {
        f.chunks[0].file_offset = FREE_DATA;
        f.chunks[0].stored_len = 9;
    });
    rejected("BLOB", |f| f.chunks[3].stored_len = 15);
    accepted(|f| f.chunks[0].stored_len = 4);
    accepted(|f| f.chunks[3].stored_len = 16);
}

#[test]
fn inv5_stored_len_bound_of_the_last_chunk_uses_its_raw_length() {
    rejected("BLOB", |f| {
        f.chunks[2].file_offset = FREE_DATA;
        f.chunks[2].stored_len = 7;
    });
    accepted(|f| f.chunks[4].stored_len = 16);
    rejected("BLOB", |f| f.chunks[2].stored_len = 3);
}

#[test]
fn inv6_extents_must_partition_the_member() {
    rejected("MEMB", |f| f.extents[1].virt_offset = 11);
    rejected("MEMB", |f| f.extents[1].virt_offset = 9);
    rejected("MEMB", |f| f.extents[0].virt_offset = 1);
    rejected("MEMB", |f| f.members[0].file_len = 25);
    rejected("MEMB", |f| f.members[0].file_len = 27);
    rejected("MEMB", |f| f.members[0].file_len = u64::MAX);
}

#[test]
fn inv6_extents_must_be_sorted() {
    rejected("MEMB", |f| {
        f.extents.swap(0, 1);
    });
}

#[test]
fn inv6_extent_range_must_be_in_extn() {
    rejected("MEMB", |f| f.members[0].extent_count = 4);
    rejected("MEMB", |f| f.members[2].first_extent = 4);
    rejected("MEMB", |f| f.members[0].first_extent = u64::MAX);
}

#[test]
fn inv6_every_extent_has_bytes() {
    rejected("EXTN", |f| f.extents[1].virt_len = 0);
}

#[test]
fn inv6_a_member_of_length_0_has_no_extents() {
    rejected("MEMB", |f| {
        f.members[2].first_extent = 2;
        f.members[2].extent_count = 1;
    });
    rejected("MEMB", |f| f.members[1].extent_count = 0);
}

#[test]
fn inv6_extent_overflow_at_u64_max_is_a_rejection() {
    rejected("EXTN", |f| f.extents[1].virt_offset = u64::MAX);
    rejected("EXTN", |f| {
        f.extents[0].virt_len = u64::MAX;
        f.extents[1].virt_offset = u64::MAX;
    });
}

#[test]
fn inv6_extents_may_be_shared_between_members() {
    accepted(|f| {
        f.members[1].first_extent = 0;
        f.members[1].extent_count = 2;
        f.members[1].file_len = 26;
    });
}

#[test]
fn inv6_the_tail_of_an_extent_run_cannot_be_a_member() {
    rejected("MEMB", |f| {
        f.members[1].first_extent = 1;
        f.members[1].file_len = 16;
    });
}

#[test]
fn inv7_identity_range_must_lie_in_the_blob() {
    rejected("EXTN", |f| f.extents[0].virt_len = 11);
    rejected("EXTN", |f| f.extents[0].blob_offset = 1);
    rejected("EXTN", |f| f.extents[0].blob_offset = u64::MAX);
    rejected("EXTN", |f| f.extents[0].blob_id = 2);
    rejected("EXTN", |f| f.extents[0].blob_id = u32::MAX);
}

#[test]
fn inv7_widened_range_counts_half_the_virtual_length() {
    rejected("EXTN", |f| f.extents[1].blob_offset = 10);
    rejected("EXTN", |f| f.extents[1].blob_offset = u64::MAX - 1);
    accepted(|f| {
        f.extents[1].blob_offset = 0;
        f.extents[1].view_id = VIEW_BF16_TO_F32;
    });
}

#[test]
fn inv7_widened_range_must_be_element_aligned() {
    rejected("EXTN", |f| f.extents[1].blob_offset = 7);
    rejected("EXTN", |f| {
        f.extents[1].virt_len = 14;
        f.members[0].file_len = 24;
        f.tensors[1].virt_len = 14;
    });
    rejected("EXTN", |f| {
        f.extents[1].virt_len = 15;
        f.members[0].file_len = 25;
        f.tensors[1].virt_len = 15;
    });
}

#[test]
fn inv7_views_have_no_alignment_on_virt_offset() {
    accepted(|f| {
        f.blobs[0].raw_len = 9;
        f.chunks[2].stored_len = 5;
        f.extents[0].virt_len = 9;
        f.extents[1].virt_offset = 9;
        f.members[0].file_len = 25;
        f.tensors[0].virt_len = 9;
        f.tensors[1].virt_offset = 9;
        assert_eq!(f.extents[1].view_id, VIEW_F16_TO_F32);
    });
}

#[test]
fn inv7_blob_kind_and_params() {
    rejected("BLOB", |f| f.blobs[0].params_off = 1);
    rejected("BLOB", |f| f.blobs[0].params_len = 1);
}

#[test]
fn inv8_tensor_range_must_lie_in_its_member() {
    rejected("MEMB", |f| f.tensors[1].virt_len = 17);
    rejected("MEMB", |f| f.tensors[1].virt_offset = 11);
    accepted(|f| {
        f.tensors[1].virt_offset = 26;
        f.tensors[1].virt_len = 0;
    });
    rejected("MEMB", |f| {
        f.tensors[1].virt_offset = 27;
        f.tensors[1].virt_len = 0;
    });
}

#[test]
fn inv8_tensor_range_overflow_is_a_rejection() {
    rejected("TENS", |f| {
        f.tensors[1].virt_offset = u64::MAX;
        f.tensors[1].virt_len = 1;
    });
    rejected("TENS", |f| {
        f.tensors[1].virt_offset = 1;
        f.tensors[1].virt_len = u64::MAX;
    });
}

#[test]
fn inv8_shape_range_must_be_in_shap() {
    rejected("TENS", |f| f.tensors[1].shape_off = 2);
    rejected("TENS", |f| f.tensors[1].shape_off = u32::MAX);
}

#[test]
fn inv8_dtype_size_is_not_checked_against_the_range() {
    accepted(|f| {
        f.tensors[0].dtype = Dtype::F64;
        f.tensors[0].virt_len = 3;
    });
    accepted(|f| f.tensors[0].dtype = Dtype::Unknown(0x7777));
}

#[test]
fn inv9_unknown_steps_are_rejected_as_codecs() {
    for id in [0, 0x0103, VIEW_F16_TO_F32, STEP_RESERVED_DELTA_FIRST, 0x0500, STEP_PRIVATE_FIRST] {
        let mut fixture = Fixture::valid();
        fixture.steps[0].codec_id = id;
        let error = fixture.check().expect_err("unknown step");
        assert!(matches!(error, Error::UnknownCodec { id: got } if got == id), "{error:?}");
    }
}

#[test]
fn inv9_a_step_no_chain_references_is_still_checked() {
    let mut fixture = Fixture::valid();
    fixture.steps.push(RawStep { codec_id: 0x7777, param0: 0, param1: 0 });
    let error = fixture.check().expect_err("unknown step");
    assert!(matches!(error, Error::UnknownCodec { id: 0x7777 }));
    rejected("CHST", |f| f.steps.push(RawStep { codec_id: STEP_PLANES, param0: 3, param1: 0 }));
}

#[test]
fn inv9_a_chain_no_blob_references_is_still_checked() {
    rejected("CHAN", |f| f.chains.push(ChainRecord { first_step: 1, step_count: 1 }));
}

#[test]
fn inv9_bad_step_parameters_are_rejected() {
    rejected("CHST", |f| f.steps[0].param0 = 1);
    rejected("CHST", |f| f.steps[1].param1 = 1);
    rejected("CHST", |f| f.steps[2].param0 = 3);
}

#[test]
fn inv9_unknown_views_are_rejected_as_codecs() {
    for id in [0x0100, 0x0300, 0x0303, 0x0400, 0x8000, u16::MAX] {
        let mut fixture = Fixture::valid();
        fixture.extents[0].view_id = id;
        let error = fixture.check().expect_err("unknown view");
        assert!(matches!(error, Error::UnknownCodec { id: got } if got == id), "{error:?}");
    }
}

#[test]
fn inv9_unknown_blob_kinds_are_rejected() {
    for kind in [2, 3, u8::MAX] {
        rejected("BLOB", |f| f.blobs[0].kind = kind);
        delta_rejected("BLOB", |f| f.blobs[0].kind = kind);
    }
}

#[test]
fn inv13_the_delta_fixture_has_the_pairing_of_section_7a() {
    let fixture = Fixture::with_delta();
    let index = fixture.check().expect("valid");
    let params = index.delta_params(&index.blobs()[2]).expect("an entry");
    assert_eq!(params, DeltaParams { base_blob: 0, base_offset: 4 });
    assert_eq!((params.base_chunk(0, 4), params.base_chunk(1, 4)), (Some(1), Some(2)));
    assert!(index.params_required());
}

#[test]
fn inv13_the_feature_bit_and_delta_blobs_go_together() {
    delta_rejected("BLOB", |f| f.required_features = 0);
    rejected("header", |f| f.required_features = crate::chain::FEATURE_DELTA);
    // A kind 1 blob in a file without the bit is refused before its entry is looked at.
    rejected("BLOB", |f| f.blobs[0].kind = 1);
    // With the bit, the missing entry is what fails.
    rejected("PARM", |f| {
        f.blobs[0].kind = 1;
        f.required_features = crate::chain::FEATURE_DELTA;
        f.params = vec![0; 16];
    });
}

#[test]
fn inv13_parm_is_required_exactly_in_a_file_with_delta_blobs() {
    // Parameters nothing uses are legal, but not with the required bit in a file without the feature.
    rejected("PARM", |f| f.params = vec![0; 16]);
    let mut fixture = Fixture::with_delta();
    let mut raw = fixture.serialize();
    clear_parm_required_bit(&mut raw);
    let limits = Limits::default();
    let index = Index::parse(&raw, &limits).expect("parses");
    let error = validate(&index, &fixture.header(), &Fixture::footer(), FILE_LEN, &limits)
        .expect_err("PARM without its required bit");
    assert!(matches!(error, Error::MalformedIndex { section: "PARM", .. }), "{error:?}");
    // The same file without delta blobs and without the feature bit is valid: PARM is skipped.
    fixture.blobs.truncate(2);
    fixture.extents.truncate(3);
    fixture.members.truncate(3);
    let mut raw = fixture.serialize();
    clear_parm_required_bit(&mut raw);
    let index = Index::parse(&raw, &limits).expect("parses");
    validate(&index, &header(), &Fixture::footer(), FILE_LEN, &limits).expect("optional PARM");
}

/// Clears the required bit of `PARM`, the last of the ten table entries [`Fixture::serialize`]
/// writes when there are parameters.
fn clear_parm_required_bit(raw: &mut [u8]) {
    let at = 8 + 9 * 32;
    assert_eq!(&raw[at..at + 4], b"PARM");
    raw[at + 4] = 0;
}

#[test]
fn inv13_the_parm_entry_has_its_size_alignment_and_zero_field() {
    delta_rejected("PARM", |f| f.blobs[2].params_len = 0);
    delta_rejected("PARM", |f| f.blobs[2].params_len = 8);
    delta_rejected("PARM", |f| f.blobs[2].params_len = 24);
    delta_rejected("PARM", |f| f.blobs[2].params_off = 4);
    delta_rejected("PARM", |f| f.blobs[2].params_off = 8);
    delta_rejected("PARM", |f| f.blobs[2].params_off = u64::MAX - 7);
    delta_rejected("PARM", |f| f.params[4] = 1);
    delta_rejected("PARM", |f| f.params.truncate(15));
    // Entries may sit anywhere aligned in the heap, with bytes nothing uses around them.
    delta_accepted(|f| {
        let entry = f.params.clone();
        f.params = [vec![0xEE; 8], entry, vec![0xEE; 3]].concat();
        f.blobs[2].params_off = 8;
    });
}

#[test]
fn inv13_the_base_is_an_earlier_blob_of_kind_0() {
    let base = |id| move |f: &mut Fixture| f.params[..4].copy_from_slice(&u32::to_le_bytes(id));
    delta_rejected("BLOB", base(2));
    delta_rejected("BLOB", base(3));
    delta_rejected("BLOB", base(u32::MAX));
    // Blob 1 is earlier and of kind 0, but has chunks of 8, not 4.
    delta_rejected("BLOB", base(1));
    // A delta blob is never a base: blob 3 over blob 2 is refused, so is a chain of deltas.
    delta_rejected("BLOB", |f| {
        let delta = f.blobs[2];
        let short = crate::format::records::BlobRecord { raw_len: 2, chunk_count: 1, ..delta };
        f.blobs.push(crate::format::records::BlobRecord { first_chunk: 6, ..short });
        f.params.extend(DeltaParams { base_blob: 2, base_offset: 0 }.encode());
        f.blobs[3].params_off = 16;
    });
}

#[test]
fn inv13_the_delta_blob_lies_in_its_base_on_chunk_boundaries() {
    let offset = |at: u64| move |f: &mut Fixture| f.params[8..].copy_from_slice(&at.to_le_bytes());
    delta_accepted(offset(0));
    delta_accepted(offset(4));
    delta_rejected("BLOB", offset(2));
    delta_rejected("BLOB", offset(8));
    delta_rejected("BLOB", offset(u64::MAX - 3));
    // Two bytes over the short last chunk of the base reach its end exactly.
    delta_accepted(|f| {
        f.blobs[2].raw_len = 2;
        f.blobs[2].chunk_count = 1;
        f.params[8..].copy_from_slice(&8_u64.to_le_bytes());
        f.extents[3].virt_len = 2;
        f.members[3].file_len = 2;
        f.chunks[5].stored_len = 6;
    });
}

#[test]
fn inv13_the_chain_has_a_delta_step_exactly_on_a_delta_blob() {
    delta_rejected("BLOB", |f| f.blobs[2].chain_id = 0);
    rejected("BLOB", |f| {
        f.steps.push(RawStep { codec_id: crate::chain::STEP_DELTA_XOR, param0: 2, param1: 0 });
        f.steps.push(RawStep { codec_id: STEP_WHOLE, param0: 0, param1: 0 });
        f.chains.push(ChainRecord { first_step: 3, step_count: 2 });
        f.blobs[0].chain_id = 2;
    });
    // The delta width joins the chain block: 8 does not divide a chunk_size of 4.
    delta_rejected("BLOB", |f| f.steps[3].param0 = 8);
}

#[test]
fn inv9_unknown_profile_and_dtype_are_legal() {
    accepted(|f| {
        f.members[0].profile = crate::Profile::Unknown(0xEE);
        f.blobs[0].dtype_hint = Dtype::Unknown(0xEEEE);
    });
}

#[test]
fn inv10_budget_is_checked_over_the_members_to_decode() {
    let index = Fixture::valid().check().expect("valid");
    let limits = Limits { max_total_decoded: 30, ..Limits::default() };
    check_decode_budget(&index, 0..3, &limits).expect("26 + 4 + 0 fits");
    let limits = Limits { max_total_decoded: 29, ..Limits::default() };
    let error = check_decode_budget(&index, 0..3, &limits).expect_err("over budget");
    assert!(matches!(
        error,
        Error::BudgetExceeded { what: "total decoded size", declared: 30, limit: 29 }
    ));
    check_decode_budget(&index, [1, 2], &limits).expect("only the small members");
    check_decode_budget(&index, [], &limits).expect("nothing to decode");
}

#[test]
fn inv10_budget_error_reports_the_whole_sum_not_where_it_crossed() {
    let index = Fixture::valid().check().expect("valid");
    // The first member alone (26) crosses this limit; the error still names all three (30).
    let limits = Limits { max_total_decoded: 10, ..Limits::default() };
    let error = check_decode_budget(&index, 0..3, &limits).expect_err("over budget");
    assert!(
        matches!(error, Error::BudgetExceeded { what: "total decoded size", declared: 30, limit: 10 }),
        "{error:?}"
    );
}

#[test]
fn inv10_budget_sum_overflow_is_a_rejection_under_any_limit() {
    let mut fixture = Fixture::valid();
    fixture.members[0].file_len = u64::MAX;
    fixture.members[1].file_len = u64::MAX;
    let index = Index::parse(&fixture.serialize(), &Limits::default()).expect("parses");
    let limits = Limits { max_total_decoded: u64::MAX, ..Limits::default() };
    check_decode_budget(&index, [0], &limits).expect("one member fits");
    let error = check_decode_budget(&index, [0, 1], &limits).expect_err("overflow");
    assert!(matches!(error, Error::BudgetExceeded { declared: u64::MAX, .. }));
}

#[test]
fn inv10_budget_reports_a_bad_member_index() {
    let index = Fixture::valid().check().expect("valid");
    let error = check_decode_budget(&index, [3], &Limits::default()).expect_err("no member 3");
    assert!(matches!(error, Error::MemberNotFound { .. }));
}

/// An index with one blob of `raw_len` bytes in chunks of 1 MiB, and a member of `count` extents of
/// `virt_len` bytes, one chunk apart. Parsed only: the chunks themselves are not needed.
fn work_index(raw_len: u64, count: u64, virt_len: u64) -> Index {
    let mut fixture = Fixture::valid();
    fixture.blobs.truncate(1);
    fixture.blobs[0].raw_len = raw_len;
    fixture.blobs[0].chunk_size = 1 << 20;
    fixture.blobs[0].chunk_count = u32::try_from(raw_len.div_ceil(1 << 20)).expect("few chunks");
    let extent =
        ExtentRecord { virt_offset: 0, virt_len, blob_id: 0, view_id: 0, blob_offset: 0 };
    fixture.extents = (0..count)
        .map(|at| ExtentRecord { virt_offset: at * virt_len, blob_offset: at << 20, ..extent })
        .collect();
    fixture.members.truncate(1);
    fixture.members[0].file_len = count * virt_len;
    fixture.members[0].extent_count = u32::try_from(count).expect("few extents");
    fixture.members[0].tensor_count = 0;
    fixture.tensors.clear();
    Index::parse(&fixture.serialize(), &Limits::default()).expect("parses")
}

#[test]
fn decode_work_has_a_boundary_at_amplification_times_length_plus_floor() {
    let limits = Limits::default();
    let index = work_index(1 << 30, 256, 1);
    check_decode_budget(&index, [0], &limits).expect("256 MiB is within the floor");
    let index = work_index(1 << 30, 257, 1);
    let error = check_decode_budget(&index, [0], &limits).expect_err("one chunk over");
    let (declared, limit) = (257 << 20, 16 * 257 + (256 << 20));
    assert!(
        matches!(
            error,
            Error::BudgetExceeded { what: "decode work", declared: d, limit: l }
                if d == declared && l == limit
        ),
        "{error:?}"
    );
    let limits = Limits { max_decode_amplification: 1 << 13, ..Limits::default() };
    check_decode_budget(&index, [0], &limits).expect("a caller may allow more");
    check_decode_budget(&index, [], &Limits::default()).expect("nothing to decode");
}

#[test]
fn decode_work_has_one_floor_however_many_members_there_are() {
    let limits = Limits::default();
    let index = work_index(1 << 30, 200, 1);
    check_decode_budget(&index, [0], &limits).expect("one member is within the floor");
    let error = check_decode_budget(&index, [0, 0], &limits).expect_err("two are not");
    let (declared, limit) = (400 << 20, 16 * 400 + (256 << 20));
    assert!(
        matches!(
            error,
            Error::BudgetExceeded { what: "decode work", declared: d, limit: l }
                if d == declared && l == limit
        ),
        "{error:?}"
    );
}

#[test]
fn decode_work_counts_every_chunk_an_extent_touches() {
    let chunk = 1_u64 << 20;
    let work = |blob_offset, virt_len, view_id, raw_len| {
        let mut fixture = Fixture::valid();
        (fixture.blobs[0].raw_len, fixture.blobs[0].chunk_size) = (raw_len, 1 << 20);
        let index = Index::parse(&fixture.serialize(), &Limits::default()).expect("parses");
        let extent = ExtentRecord { virt_offset: 0, virt_len, blob_id: 0, view_id, blob_offset };
        super::extent_work(&extent, &index).expect("valid")
    };
    assert_eq!(work(0, 1, 0, 8 * chunk), chunk);
    assert_eq!(work(chunk - 1, 1, 0, 8 * chunk), chunk);
    assert_eq!(work(chunk - 1, 2, 0, 8 * chunk), 2 * chunk);
    assert_eq!(work(chunk, chunk, 0, 8 * chunk), chunk);
    // The last chunk of a blob is short, and counts as what it holds.
    assert_eq!(work(0, chunk + 5, 0, chunk + 5), chunk + 5);
    assert_eq!(work(chunk, 1, 0, chunk + 5), 5);
    // A view doubles the length in the member, not the bytes decoded.
    assert_eq!(work(0, 2 * chunk, VIEW_F16_TO_F32, 8 * chunk), chunk);
    assert_eq!(work(chunk - 2, 8, VIEW_F16_TO_F32, 8 * chunk), 2 * chunk);
}

#[test]
fn decode_work_of_a_delta_extent_adds_the_base_chunks_it_pairs_with() {
    let chunk = 1_u64 << 20;
    // Base blob 0 of 8 chunks and a delta blob 2 of `raw_len` bytes at base chunk `base_chunk`.
    let work = |base_chunk: u64, raw_len: u64, blob_offset: u64, virt_len: u64| {
        let mut fixture = Fixture::with_delta();
        (fixture.blobs[0].raw_len, fixture.blobs[0].chunk_size) = (8 * chunk, 1 << 20);
        (fixture.blobs[2].raw_len, fixture.blobs[2].chunk_size) = (raw_len, 1 << 20);
        fixture.params[8..].copy_from_slice(&(base_chunk * chunk).to_le_bytes());
        let index = Index::parse(&fixture.serialize(), &Limits::default()).expect("parses");
        let extent = ExtentRecord { virt_offset: 0, virt_len, blob_id: 2, view_id: 0, blob_offset };
        super::extent_work(&extent, &index).expect("valid")
    };
    assert_eq!(work(0, 4 * chunk, 0, 4 * chunk), 8 * chunk);
    assert_eq!(work(3, 2 * chunk, chunk, 1), 2 * chunk);
    // A short delta blob still decodes a whole base chunk per chunk, and counts it.
    assert_eq!(work(5, 2, 0, 2), chunk + 2);
    assert_eq!(work(6, chunk + 2, 0, chunk + 2), 3 * chunk + 2);
}

#[test]
fn decode_work_survives_values_at_the_top_of_u64() {
    let mut fixture = Fixture::valid();
    fixture.blobs[0].raw_len = u64::MAX - 1;
    fixture.blobs[0].chunk_size = 2;
    fixture.extents[0].virt_len = u64::MAX - 1;
    fixture.extents[1].virt_offset = u64::MAX - 1;
    fixture.extents[1].blob_id = 0;
    fixture.extents[1].view_id = 0;
    fixture.extents[1].virt_len = 1;
    fixture.extents[1].blob_offset = 0;
    fixture.members[0].file_len = u64::MAX;
    fixture.members[1].extent_count = 0;
    fixture.members[1].file_len = 0;
    let index = Index::parse(&fixture.serialize(), &Limits::default()).expect("parses");
    let limits = Limits { max_decode_amplification: u64::MAX, ..Limits::default() };
    check_decode_budget(&index, [0], &limits).expect("the limit is over the work");
    let limits = Limits { max_decode_amplification: 0, ..Limits::default() };
    let error = check_decode_budget(&index, [0], &limits).expect_err("over");
    assert!(
        matches!(error, Error::BudgetExceeded { what: "decode work", declared: u64::MAX, .. }),
        "{error:?}"
    );
}

#[test]
fn inv11_unsafe_member_names_are_rejected() {
    let names = [
        "",
        "a\0b",
        "/etc/passwd",
        "\\windows",
        "..",
        "../x",
        "a/../b",
        "a/..",
        "a\\..\\b",
        "a/..\\b",
        "..\\x",
        "C:\\x",
        "C:x",
        "file.txt:stream",
        ":",
    ];
    for name in names {
        let error = check_restore_name(name).expect_err(name);
        assert!(matches!(&error, Error::UnsafeMemberName { name: got } if got == name));
    }
}

#[test]
fn inv11_ordinary_names_are_accepted() {
    for name in ["a", "a/b.bin", "dir\\file", "..a", "a..", "a/..b/c", "...", "модель.bin"] {
        check_restore_name(name).expect(name);
    }
}

#[test]
fn name_work_budget_stops_before_any_name_is_used() {
    let limits = Limits { max_name_work_bytes: 18, ..Limits::default() };
    let raw = Fixture::valid().serialize();
    let index = Index::parse(&raw, &limits).expect("parses");
    let error = validate(&index, &header(), &Fixture::footer(), FILE_LEN, &limits)
        .expect_err("over budget");
    assert!(matches!(
        error,
        Error::BudgetExceeded { what: "name bytes", declared: 19, limit: 18 }
    ));
    let limits = Limits { max_name_work_bytes: 19, ..Limits::default() };
    validate(&index, &header(), &Fixture::footer(), FILE_LEN, &limits).expect("exactly on budget");
}

#[test]
fn range_max_agrees_with_a_scan() {
    let mut prng = Prng(0x5EED);
    for len in 0..40 {
        let values: Vec<u64> = (0..len).map(|_| prng.below(1000)).collect();
        let tree = RangeMax::new(&values);
        let reversed: Vec<Reverse<u64>> = values.iter().copied().map(Reverse).collect();
        let reversed_tree = RangeMax::new(&reversed);
        for start in 0..=values.len() {
            for end in start..=values.len() {
                assert_eq!(tree.max(start..end), values[start..end].iter().copied().max());
                assert_eq!(
                    reversed_tree.max(start..end).map(|Reverse(min)| min),
                    values[start..end].iter().copied().min()
                );
            }
        }
    }
}

/// A hostile sharing pattern must not multiply two counts. With 200 000 records on each side a
/// walk per owner would take hours; these three tests finish at once or hang.
const SHARED: usize = 200_000;
/// [`SHARED`] as a `u32` field value.
const SHARED_U32: u32 = 200_000;

fn shared_index(fixture: &Fixture) -> Index {
    let limits = Limits { max_index_bytes: 1 << 30, ..Limits::default() };
    Index::parse(&fixture.serialize(), &limits).expect("parses")
}

fn validate_shared(index: &Index) -> Result<(), Error> {
    let limits = Limits { max_index_bytes: 1 << 30, ..Limits::default() };
    let footer = Footer { index_offset: 1 << 40, ..Fixture::footer() };
    validate(index, &header(), &footer, (1 << 40) + 100 + 64, &limits).map(drop)
}

#[test]
fn many_blobs_over_all_chunks_cost_no_product() {
    let mut fixture = Fixture::valid();
    fixture.chunks = vec![ChunkRecord { file_offset: 64, xxh3_decoded: 0, stored_len: 5 }; SHARED];
    let blob = fixture.blobs[0];
    fixture.blobs = vec![blob; SHARED];
    fixture.blobs.iter_mut().for_each(|blob| {
        blob.raw_len = SHARED as u64;
        blob.chunk_size = 1;
        blob.chunk_count = SHARED_U32;
    });
    fixture.extents.iter_mut().for_each(|extent| extent.blob_id = 0);
    validate_shared(&shared_index(&fixture)).expect("valid");
}

#[test]
fn many_members_over_all_extents_cost_no_product() {
    let mut fixture = Fixture::valid();
    let extent =
        ExtentRecord { virt_offset: 0, virt_len: 1, blob_id: 0, view_id: 0, blob_offset: 0 };
    fixture.extents = (0..SHARED as u64)
        .map(|at| ExtentRecord { virt_offset: at, blob_offset: at % 10, ..extent })
        .collect();
    let member = MemberRecord {
        file_len: SHARED as u64,
        first_extent: 0,
        extent_count: SHARED_U32,
        first_tensor: 0,
        tensor_count: 0,
        ..fixture.members[0]
    };
    fixture.strings = vec![b'x'; SHARED];
    fixture.tensors.clear();
    fixture.dims.clear();
    // Each name is a distinct length of the one run of `x`.
    fixture.members = (1..=500_u32)
        .map(|len| MemberRecord { name: StrRef { off: 0, len }, ..member })
        .collect();
    let index = shared_index(&fixture);
    validate_shared(&index).expect("valid");
    let limits = Limits::default();
    check_decode_budget(&index, 0..500, &limits).expect("the work of each is within its bound");
}

#[test]
fn many_members_over_all_tensors_cost_no_product() {
    let mut fixture = Fixture::valid();
    let digits = (0..SHARED).flat_map(|id| format!("{id:06}").into_bytes());
    fixture.strings = digits.collect();
    fixture.tensors = (0..SHARED_U32)
        .map(|id| TensorRecord {
            name: StrRef { off: id * 6, len: 6 },
            virt_offset: 0,
            virt_len: 1,
            shape_off: 0,
            ndim: 0,
            dtype: Dtype::U8,
        })
        .collect();
    let member = fixture.members[0];
    fixture.members = (0..SHARED_U32)
        .map(|id| MemberRecord {
            name: StrRef { off: id * 6, len: 6 },
            first_tensor: 0,
            tensor_count: SHARED_U32,
            ..member
        })
        .collect();
    fixture.members.iter_mut().skip(1).for_each(|member| {
        member.first_extent = 3;
        member.extent_count = 0;
        member.file_len = 0;
    });
    let error = validate_shared(&shared_index(&fixture)).expect_err("second member too short");
    assert!(matches!(error, Error::MalformedIndex { section: "MEMB", .. }));
    fixture.members.iter_mut().skip(1).for_each(|member| {
        member.first_extent = 0;
        member.extent_count = 2;
        member.file_len = 26;
    });
    validate_shared(&shared_index(&fixture)).expect("valid");
}

/// A delta over a random earlier kind 0 blob of `blobs`, with its chunk size, when the chain made
/// of `steps` after a delta step and the bytes left in the base after a chunk boundary allow it.
/// Returns the base id, `base_offset`, the chain and the room left in the base, in whole blocks.
fn pick_delta_base(
    prng: &mut Prng,
    blobs: &[crate::format::records::BlobRecord],
    steps: &[crate::chain::Step],
) -> Option<(usize, u64, crate::chain::Chain, u64)> {
    use crate::chain::{Chain, DeltaOp, ElementWidth, Step};
    let id = usize::try_from(prng.below(blobs.len() as u64 + 1)).expect("small");
    let pick = prng.below(1 << 20);
    let base = blobs.get(id)?;
    let steps = [&[Step::Delta(DeltaOp::Xor(ElementWidth::Two))], steps].concat();
    let chain = Chain::new(&steps).expect("valid");
    let block = u64::from(chain.block_bytes());
    let chunk = u64::from(base.chunk_size);
    let base_offset = chunk * (pick % u64::from(base.chunk_count));
    let room = (base.raw_len - base_offset) / block * block;
    let usable = base.kind == 0 && chunk.is_multiple_of(block) && room > 0;
    usable.then_some((id, base_offset, chain, room))
}

/// Builds a random valid index through the public builder calls: blobs of each chain kind, some of
/// them delta blobs over an earlier one, members over them. Returns the raw index, the header, and
/// its footer offsets.
fn random_valid_index(prng: &mut Prng) -> (Vec<u8>, Header, Footer, u64) {
    use crate::chain::{Chain, ElementWidth, FEATURE_DELTA, FieldTable, Layout, Step};
    use crate::format::records::{BlobRecord, DELTA_PARAMS_LEN};

    let layouts = [
        Layout::Whole,
        Layout::Planes(ElementWidth::Two),
        Layout::Planes(ElementWidth::Eight),
        Layout::Fields(FieldTable::Q80),
        Layout::Fields(FieldTable::Q6K),
    ];
    let mut builder = IndexBuilder::new();
    let mut offset = 64;
    let blob_count = 1 + prng.below(8);
    let mut blobs: Vec<BlobRecord> = Vec::new();
    let mut header = Header::current();
    for _ in 0..blob_count {
        let layout = layouts[usize::try_from(prng.below(5)).expect("small")];
        let mut steps = vec![Step::Layout(layout)];
        if prng.below(2) == 0 {
            steps.insert(0, Step::Rotl1(ElementWidth::Four));
        }
        let delta = pick_delta_base(prng, &blobs, &steps);
        let chain = match delta {
            Some((_, _, chain, _)) => chain,
            None => Chain::new(&steps).expect("valid"),
        };
        let chain_id = builder.add_chain(chain).expect("fits");
        let block = u64::from(chain.block_bytes());
        let mut chunk_size = block * (1 + prng.below(4));
        let mut raw_len = block * (1 + prng.below(40));
        let mut params = (0, 0);
        if let Some((id, base_offset, _, room)) = delta {
            chunk_size = u64::from(blobs[id].chunk_size);
            raw_len = block * (1 + prng.below(room / block));
            let base_blob = u32::try_from(id).expect("small");
            let entry = DeltaParams { base_blob, base_offset };
            params = builder.add_params(&entry.encode()).expect("fits");
            assert_eq!(params.1, DELTA_PARAMS_LEN);
            header.required_features = FEATURE_DELTA;
        }
        let chunk_count = raw_len.div_ceil(chunk_size);
        let mut first_chunk = None;
        let prologue = 4 * chain.stream_count() as u64;
        for at in 0..chunk_count {
            let raw = if at + 1 == chunk_count { raw_len - at * chunk_size } else { chunk_size };
            let stored = prologue + prng.below(raw + 1);
            let chunk = ChunkRecord {
                file_offset: offset,
                xxh3_decoded: prng.next(),
                stored_len: u32::try_from(stored).expect("small"),
            };
            offset += stored + prng.below(3);
            let id = builder.add_chunk(chunk);
            first_chunk.get_or_insert(id);
        }
        let blob = BlobRecord {
            blake3: [0; 32],
            raw_len,
            first_chunk: first_chunk.expect("at least one chunk"),
            chunk_count: u32::try_from(chunk_count).expect("small"),
            chunk_size: u32::try_from(chunk_size).expect("small"),
            chain_id,
            kind: u8::from(delta.is_some()),
            dtype_hint: Dtype::from_wire(u16::try_from(prng.below(0x40)).expect("small")),
            params_off: params.0,
            params_len: params.1,
        };
        builder.add_blob(blob).expect("fits");
        blobs.push(blob);
    }
    for (id, blob) in blobs.iter().enumerate() {
        let name = builder.add_string(&format!("m/{id}")).expect("fits");
        let first_extent = builder.add_extent(ExtentRecord {
            virt_offset: 0,
            virt_len: blob.raw_len,
            blob_id: u32::try_from(id).expect("small"),
            view_id: 0,
            blob_offset: 0,
        });
        builder
            .add_member(MemberRecord {
                name,
                file_len: blob.raw_len,
                blake3: [0; 32],
                first_extent,
                extent_count: 1,
                first_tensor: 0,
                tensor_count: 0,
                profile: crate::Profile::Opaque,
            })
            .expect("fits");
    }
    let raw = builder.serialize();
    let footer = Footer::for_index(offset, 50, &raw, crate::format::GENERATION_INITIAL);
    (raw, header, footer, offset + 50 + 64)
}

#[test]
fn random_valid_indexes_pass_and_survive_a_rewrite() {
    let mut prng = Prng(0x1234_5678);
    let mut deltas = 0;
    for _ in 0..200 {
        let (raw, header, footer, file_len) = random_valid_index(&mut prng);
        let limits = Limits::default();
        footer.check_index(&raw).expect("hash");
        let index = Index::parse(&raw, &limits).expect("parses");
        validate(&index, &header, &footer, file_len, &limits).expect("valid");
        let blob_count = u32::try_from(index.members().len()).expect("few");
        check_decode_budget(&index, 0..blob_count, &limits).expect("whole blobs are within budget");
        deltas += index.blobs().iter().filter(|blob| blob.kind == 1).count();
        // `Fixture` writes the whole heap as one entry at offset 0, which gives the same bytes.
        assert_eq!(Fixture::from_index(&index).serialize(), raw);
    }
    assert!(deltas > 50, "only {deltas} delta blobs were generated");
}

/// Runs the whole pipeline on arbitrary bytes, under both headers. The outcome does not matter, only
/// that it returns.
fn run_on(raw: &[u8]) {
    let limits = Limits::default();
    if let Ok(index) = Index::parse(raw, &limits) {
        for header in [header(), Fixture::with_delta().header()] {
            if validate(&index, &header, &Fixture::footer(), FILE_LEN, &limits).is_ok() {
                let count = u32::try_from(index.members().len()).unwrap_or(0);
                check_decode_budget(&index, 0..count, &limits).ok();
            }
        }
        for member in index.members() {
            index.string(member.name).ok();
            index.member_extents(member).ok();
            index.member_tensors(member).ok();
        }
    }
}

#[test]
fn no_truncation_panics() {
    for raw in [Fixture::valid().serialize(), Fixture::with_delta().serialize()] {
        for len in 0..raw.len() {
            run_on(&raw[..len]);
        }
    }
}

#[test]
fn no_single_byte_change_panics() {
    for raw in [Fixture::valid().serialize(), Fixture::with_delta().serialize()] {
        for at in 0..raw.len() {
            for value in [0x00, 0x01, 0x7F, 0x80, 0xFF] {
                let mut bytes = raw.clone();
                bytes[at] = value;
                run_on(&bytes);
            }
        }
    }
}

#[test]
fn no_random_mutation_panics() {
    let mut prng = Prng(0x0BAD_5EED);
    for round in 0..20_000 {
        let raw = if round % 2 == 0 { Fixture::valid() } else { Fixture::with_delta() }.serialize();
        let mut bytes = raw.clone();
        for _ in 0..=prng.below(16) {
            let at = usize::try_from(prng.below(raw.len() as u64)).expect("small");
            bytes[at] = prng.next().to_le_bytes()[0];
        }
        run_on(&bytes);
    }
}

#[test]
fn random_bytes_never_panic() {
    let mut prng = Prng(0xFEED_FACE);
    for _ in 0..5_000 {
        let len = usize::try_from(prng.below(512)).expect("small");
        let bytes: Vec<u8> = (0..len).map(|_| prng.next().to_le_bytes()[3]).collect();
        run_on(&bytes);
    }
}
