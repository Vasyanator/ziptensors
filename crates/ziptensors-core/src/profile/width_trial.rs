//! File: `profile/width_trial.rs`
//! Purpose: guesses the element width of bytes whose type no parser could name.
//!
//! Main responsibilities: [`width_trial`], which compares the order-0 entropy of the byte planes a
//! candidate width would produce.
//!
//! Implementation notes: uniform elements give byte planes with different statistics, so splitting
//! at the right width lowers the mean entropy and a wrong width mixes the planes back together.
//! Candidates are only comparable if they measure the same bytes with the same estimator. The
//! plug-in estimate reads low on short planes, and neighbouring parts of a tensor differ, so every
//! candidate covers the whole sample with [`MAX_PLANES`] histograms of one length. A narrow width
//! cuts its planes into slices to get there.

/// Candidate widths above one byte, smallest first.
const WIDTHS: [u8; 3] = [2, 4, 8];

/// The widest candidate.
const MAX_WIDTH: u8 = 8;

/// Number of byte planes of the widest candidate.
const MAX_PLANES: usize = 8;

/// Below this there is too little data per plane for the entropies to mean anything.
const MIN_SAMPLE_BYTES: usize = 4096;

/// Bytes of the sample that are looked at. Keeps every count well inside `u32`.
const MAX_SAMPLE_BYTES: usize = 1 << 20;

/// Bits per byte a width must save over no split at all to count as evidence.
const MIN_GAIN_BITS: f64 = 0.1;

/// Costs this close to the best are ties. Data of width 4 costs the same at width 8, since each
/// plane appears twice, so a tie must go to the smaller width.
const TIE_BITS: f64 = 0.03;

/// Returns the element width of `sample`, one of 1, 2, 4 or 8. 1 means no evidence of a wider
/// element, which is also the answer for a sample under 4 KiB.
///
/// `sample` must start on an element boundary of the data. Only its first 1 MiB is used. The answer
/// is a hint that picks a byte plane split: it says nothing about what the elements are.
#[must_use]
pub fn width_trial(sample: &[u8]) -> u8 {
    let sample = sample.get(..MAX_SAMPLE_BYTES).unwrap_or(sample);
    if sample.len() < MIN_SAMPLE_BYTES {
        return 1;
    }
    let plane_len = sample.len() / usize::from(MAX_WIDTH);
    let costs = WIDTHS.map(|width| plane_cost(sample, width, plane_len));
    let best = costs.iter().copied().fold(f64::INFINITY, f64::min);
    if plane_cost(sample, 1, plane_len) - best < MIN_GAIN_BITS {
        return 1;
    }
    match costs.map(|cost| cost <= best + TIE_BITS) {
        [true, _, _] => 2,
        [false, true, _] => 4,
        [false, false, _] => 8,
    }
}

/// Mean entropy in bits per byte of `sample` split into `width` byte planes, each measured over
/// slices of `plane_len` elements. `width` divides [`MAX_PLANES`], and `sample` holds that many
/// histograms' worth of bytes, so the count of histograms is always [`MAX_PLANES`].
fn plane_cost(sample: &[u8], width: u8, plane_len: usize) -> f64 {
    let width_bytes = usize::from(width);
    let mut bits = 0.0;
    for slice in sample.chunks_exact(plane_len * width_bytes).take(MAX_PLANES / width_bytes) {
        let mut histograms = [[0_u32; 256]; MAX_PLANES];
        for element in slice.chunks_exact(width_bytes) {
            for (histogram, &byte) in histograms.iter_mut().zip(element) {
                histogram[usize::from(byte)] += 1;
            }
        }
        bits += histograms.iter().take(width_bytes).map(plane_entropy).sum::<f64>();
    }
    bits / f64::from(MAX_WIDTH)
}

/// Order-0 entropy of one plane, in bits per byte.
fn plane_entropy(histogram: &[u32; 256]) -> f64 {
    let total = f64::from(histogram.iter().sum::<u32>());
    let counts = histogram.iter().filter(|&&count| count > 0);
    counts.map(|&count| f64::from(count) / total).map(|p| -p * p.log2()).sum()
}

#[cfg(test)]
mod tests {
    use super::{MIN_SAMPLE_BYTES, width_trial};

    /// Splitmix64 output bytes, for data with no structure.
    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut state = seed;
        move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// Binary32 values over a few binades with a random sign, as trained weights are.
    fn floats32(seed: u64, count: usize) -> Vec<u8> {
        let mut next = rng(seed);
        (0..count)
            .flat_map(|_| {
                let word = next();
                let fraction = word >> 8 & 0x7F_FFFF;
                let bits = (word >> 63 << 31) | ((120 + (word & 3)) << 23) | fraction;
                u32::try_from(bits).expect("32 bits").to_le_bytes()
            })
            .collect()
    }

    /// Binary64 values of the same kind.
    fn floats64(seed: u64, count: usize) -> Vec<u8> {
        let mut next = rng(seed);
        (0..count)
            .flat_map(|_| {
                let word = next();
                let fraction = word >> 8 & 0xF_FFFF_FFFF_FFFF;
                ((word >> 63 << 63) | ((1016 + (word & 3)) << 52) | fraction).to_le_bytes()
            })
            .collect()
    }

    #[test]
    fn four_byte_floats_are_width_four() {
        for seed in 0..8 {
            assert_eq!(width_trial(&floats32(seed, 8192)), 4, "seed {seed}");
        }
    }

    #[test]
    fn eight_byte_floats_are_width_eight() {
        for seed in 0..8 {
            assert_eq!(width_trial(&floats64(seed, 4096)), 8, "seed {seed}");
        }
    }

    #[test]
    fn two_byte_data_is_width_two() {
        for seed in 0..8 {
            let mut next = rng(seed);
            // The shape of binary16 weights: a skewed high byte over a noisy low byte.
            let bytes: Vec<u8> = (0..16384)
                .flat_map(|_| {
                    let word = next();
                    let high = 0x38 + u8::try_from(word >> 61).expect("3 bits");
                    [word.to_le_bytes()[0], high]
                })
                .collect();
            assert_eq!(width_trial(&bytes), 2, "seed {seed}");
        }
    }

    #[test]
    fn small_integers_in_eight_bytes_are_width_eight() {
        let mut next = rng(5);
        let bytes: Vec<u8> = (0..4096).flat_map(|_| (next() >> 44).to_le_bytes()).collect();
        assert_eq!(width_trial(&bytes), 8);
    }

    #[test]
    fn noise_and_text_and_constants_are_width_one() {
        let mut next = rng(9);
        let noise: Vec<u8> = (0..4096).flat_map(|_| next().to_le_bytes()).collect();
        assert_eq!(width_trial(&noise), 1);
        assert_eq!(width_trial(&vec![0; 65536]), 1);
        let words = ["the ", "quick ", "brown ", "fox ", "jumps ", "over ", "a ", "lazy ", "dog "];
        let text: String =
            (0..4096).map(|_| words[usize::try_from(next() % 9).expect("under 9")]).collect();
        assert_eq!(width_trial(text.as_bytes()), 1);
    }

    #[test]
    fn a_short_sample_is_no_evidence() {
        let bytes = floats32(1, 1024);
        assert_eq!(width_trial(&bytes[..MIN_SAMPLE_BYTES - 4]), 1);
        assert_eq!(width_trial(&bytes[..MIN_SAMPLE_BYTES]), 4);
        assert_eq!(width_trial(&[]), 1);
    }

    #[test]
    fn a_change_of_statistics_inside_the_sample_does_not_favour_a_wide_width() {
        // Two parts of one tensor with different scales. A candidate that saw only the first part
        // would cost less or more than one that saw both, for reasons unrelated to the width.
        for seed in 0..8 {
            let mut bytes = floats32(seed, 2048);
            let wide_range = floats32(seed + 100, 2048);
            let (elements, _) = wide_range.as_chunks::<4>();
            let shifted =
                elements.iter().flat_map(|&[a, b, c, d]| [a, b, c ^ 0x55, d & 0x07 | 0x38]);
            bytes.extend(shifted);
            assert_eq!(width_trial(&bytes), 4, "seed {seed}");
        }
    }

    #[test]
    fn a_long_sample_is_cut_and_still_answers() {
        assert_eq!(width_trial(&floats32(2, 1 << 19)), 4);
    }
}
