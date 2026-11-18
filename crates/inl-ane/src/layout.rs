//! The host side of a package's function contract: which functions exist,
//! how texts are packed into one call, and the masks and pooling matrices that
//! go with a packing. Pure Rust, no Core ML, so every rule here is unit tested.
//!
//! A packed call places texts back to back from row 0 of an `S`-row function.
//! Attention never crosses texts (the bias inputs block it) and RoPE is
//! relative, so a text gets the same embedding packed as alone.

use half::f16;

/// The additive bias of a blocked (query, key) pair.
pub(crate) const BLOCKED: f16 = f16::from_f32_const(-10_000.0);
const OPEN: f16 = f16::ZERO;

/// EmbeddingGemma 2's sliding window, also the block and halo of the
/// windowed functions.
pub(crate) const WINDOW: usize = 512;
/// Key columns of one `band_bias` row: a 512-row block and a 512-row halo on
/// each side.
pub(crate) const BAND_COLUMNS: usize = 3 * WINDOW;
/// Pool weight of a long (winl) call: 2^-10 on every token row instead of 1/n,
/// which underflows fp16 inside the graph; the scale cancels in the L2 norm.
const LONG_POOL: f16 = f16::from_f32_const(0.000_976_562_5);
/// KURE-Sparse-nano's sliding window reaches this many rows on each side
/// (`local_attention / 2`).
const SPLADE_HALF_WINDOW: usize = 64;
/// Rows per block of a SPLADE function's `block_max` output; texts start on a
/// block boundary so a block never mixes two texts.
pub(crate) const SPLADE_BLOCK: usize = 8;

/// The architectures a package can hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// `google/embeddinggemma-2`: bidirectional, mean pooling.
    Gemma2,
    /// `Qwen/Qwen3-Embedding-0.6B`: causal, last-token pooling.
    Qwen3,
    /// `nlpai-lab/KURE-Sparse-nano`: ModernBERT masked LM, SPLADE max pooling
    /// (learned-sparse, not dense).
    ModernBertSplade,
}

/// How a function attends and pools; decides its inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// `embed_S`: full attention, `global_bias` + `local_bias`, mean pool.
    Bucket,
    /// `winp{S}_b512`: windowed packing, `band_bias` + `global_bias`, mean pool.
    Winp,
    /// `winl{S}_b512`: one long text, `band_bias` + `key_bias`, 2^-10 pool.
    Winl,
    /// `embed_S`: causal attention, `causal_bias`, last-token pool.
    Causal,
    /// `splade_S`: `global_bias` + `local_bias`, a `keep` row mask, block max
    /// of the shifted vocabulary logits.
    Splade,
}

impl Family {
    /// True for a learned-sparse (SPLADE) family, false for a dense one.
    pub fn is_sparse(self) -> bool {
        self == Self::ModernBertSplade
    }

    /// The Core ML package (`<name>.mlpackage`) holding the text functions.
    pub(crate) fn package_name(self) -> &'static str {
        match self {
            Self::Gemma2 => "EmbeddingGemma2Text",
            Self::Qwen3 => "Qwen3EmbeddingText",
            Self::ModernBertSplade => "ModernBertSpladeText",
        }
    }

    pub(crate) fn from_package_name(name: &str) -> Option<Self> {
        [Self::Gemma2, Self::Qwen3, Self::ModernBertSplade]
            .into_iter()
            .find(|family| family.package_name() == name)
    }

    /// Rows a text's offset in a call is a multiple of.
    pub(crate) fn alignment(self) -> usize {
        match self {
            Self::ModernBertSplade => SPLADE_BLOCK,
            Self::Gemma2 | Self::Qwen3 => 1,
        }
    }

    /// Texts one `length`-row call holds: the pool rows of a dense function;
    /// a SPLADE function has no pool rows, only its blocks bound it.
    pub(crate) fn max_texts(self, length: usize) -> usize {
        match self {
            Self::ModernBertSplade => length / SPLADE_BLOCK,
            Self::Gemma2 | Self::Qwen3 => slots(length),
        }
    }

    /// Packing lengths, ascending: every call of up to this many rows.
    pub(crate) fn packing_lengths(self) -> &'static [usize] {
        match self {
            Self::Gemma2 => &[64, 128, 256, 512, 768, 1024, 1280, 1536, 2048],
            Self::Qwen3 | Self::ModernBertSplade => &[64, 128, 256, 512, 768, 1024, 1536, 2048],
        }
    }

    /// Single-text lengths beyond the packing ones, ascending.
    pub(crate) fn long_lengths(self) -> &'static [usize] {
        match self {
            Self::Gemma2 => &[3072, 4096, 6144, 8192],
            Self::Qwen3 | Self::ModernBertSplade => &[],
        }
    }

    /// Measured latency of one call per packing length (only the ratios
    /// matter to the planner).
    fn cost(self, length: usize) -> f64 {
        let table: &[(usize, f64)] = match self {
            // ANEEmbed `bench --buckets`, Apple M5 Ultra.
            Self::Gemma2 => &[
                (64, 3.16),
                (128, 5.19),
                (256, 10.71),
                (512, 26.47),
                (768, 44.97),
                (1024, 68.93),
                (1280, 93.47),
                (1536, 114.53),
                (2048, 170.34),
            ],
            // `scripts/ane` packages, `MLModel.predict` per function, M4 Max.
            Self::Qwen3 => &[
                (64, 8.7),
                (128, 12.1),
                (256, 30.1),
                (512, 83.8),
                (768, 149.0),
                (1024, 237.3),
                (1536, 464.6),
                (2048, 1659.4),
            ],
            // `scripts/ane` package, `MLModel.predict` per function, M4 Max.
            Self::ModernBertSplade => &[
                (64, 3.9),
                (128, 5.7),
                (256, 13.0),
                (512, 35.5),
                (768, 75.1),
                (1024, 129.2),
                (1536, 297.3),
                (2048, 723.3),
            ],
        };
        table
            .iter()
            .find(|(l, _)| *l == length)
            .map_or(length as f64, |(_, cost)| *cost)
    }

    /// Scale applied to a token-table row before the fp16 cast.
    pub(crate) fn row_scale(self) -> f32 {
        match self {
            // Gemma multiplies the embedding by sqrt(hidden) = sqrt(512).
            Self::Gemma2 => (512f32).sqrt(),
            Self::Qwen3 | Self::ModernBertSplade => 1.0,
        }
    }

    pub(crate) fn kind(self, length: usize) -> Kind {
        match self {
            Self::Gemma2 if self.long_lengths().contains(&length) => Kind::Winl,
            Self::Gemma2 if length > 1280 => Kind::Winp,
            Self::Gemma2 => Kind::Bucket,
            Self::Qwen3 => Kind::Causal,
            Self::ModernBertSplade => Kind::Splade,
        }
    }

    pub(crate) fn function_name(self, length: usize) -> String {
        match self.kind(length) {
            Kind::Bucket | Kind::Causal => format!("embed_{length}"),
            Kind::Winp => format!("winp{length}_b{WINDOW}"),
            Kind::Winl => format!("winl{length}_b{WINDOW}"),
            Kind::Splade => format!("splade_{length}"),
        }
    }

    /// The input names of a function, in the package's order.
    pub(crate) fn input_names(self, length: usize) -> &'static [&'static str] {
        match self.kind(length) {
            Kind::Bucket => &["inputs_embeds", "global_bias", "local_bias", "pool"],
            Kind::Winp => &["inputs_embeds", "band_bias", "global_bias", "pool"],
            Kind::Winl => &["inputs_embeds", "band_bias", "key_bias", "pool"],
            Kind::Causal => &["inputs_embeds", "causal_bias", "pool"],
            Kind::Splade => &["inputs_embeds", "global_bias", "local_bias", "keep"],
        }
    }

    /// The function lengths a context of `max_length` tokens needs: every
    /// packing length, then the long lengths up to the one that holds it.
    pub(crate) fn lengths_for(self, max_length: usize) -> Vec<usize> {
        let mut lengths = self.packing_lengths().to_vec();
        let packed_max = *lengths.last().expect("a family has packing lengths");
        if max_length > packed_max {
            for &length in self.long_lengths() {
                lengths.push(length);
                if length >= max_length {
                    break;
                }
            }
        }
        lengths
    }
}

/// The distinct function lengths `calls` use, ascending. A function that no call
/// of a request uses never has to be loaded for it.
pub(crate) fn used_lengths(calls: &[Call]) -> Vec<usize> {
    let mut lengths: Vec<usize> = calls.iter().map(|call| call.length).collect();
    lengths.sort_unstable();
    lengths.dedup();
    lengths
}

/// Texts one call of an `S`-row packing function holds.
pub(crate) fn slots(length: usize) -> usize {
    (length / 32).max(8)
}

/// One text in a call: rows `offset..offset + len`, pool row = its index in
/// the call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Segment {
    /// Index of the text in the caller's list.
    pub item: usize,
    pub offset: usize,
    pub len: usize,
}

/// One function call: a length and the texts packed into it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Call {
    pub length: usize,
    pub segments: Vec<Segment>,
}

/// Calls covering every text exactly once, packing texts of up to the largest
/// packing length; a longer text gets a call of the smallest long length that
/// holds it.
///
/// For every packing length C the packing texts go first-fit-decreasing into
/// bins of C rows and C's slot count, each bin shrinks to the smallest length
/// that holds it, and the cheapest C (by [`Family::cost`]) wins: small C means
/// many cheap calls, large C fewer expensive ones.
///
/// A text takes its count rounded up to [`Family::alignment`] rows, so every
/// offset is a multiple of it; the segment's `len` stays the count.
///
/// Every count must be at least 1 and at most the largest available length.
pub(crate) fn plan(family: Family, lengths: &[usize], counts: &[usize]) -> Vec<Call> {
    let rows: Vec<usize> = counts
        .iter()
        .map(|&count| count.next_multiple_of(family.alignment()))
        .collect();
    let packing: Vec<usize> = family
        .packing_lengths()
        .iter()
        .copied()
        .filter(|length| lengths.contains(length))
        .collect();
    let packed_max = packing.last().copied().unwrap_or(0);
    let mut calls = Vec::new();
    let mut order: Vec<usize> = Vec::new();
    for (item, &count) in counts.iter().enumerate() {
        debug_assert!(count >= 1);
        if rows[item] > packed_max {
            let length = lengths
                .iter()
                .copied()
                .find(|&length| length >= count && !packing.contains(&length))
                .expect("count within the loaded lengths");
            calls.push(Call {
                length,
                segments: vec![Segment {
                    item,
                    offset: 0,
                    len: count,
                }],
            });
        } else {
            order.push(item);
        }
    }
    if order.is_empty() {
        return calls;
    }
    order.sort_by(|&a, &b| rows[b].cmp(&rows[a]).then(a.cmp(&b)));
    let mut best: Option<(f64, Vec<Call>)> = None;
    for &capacity in &packing {
        let packed = pack(family, &order, &rows, counts, capacity, &packing);
        let cost: f64 = packed.iter().map(|call| family.cost(call.length)).sum();
        if best.as_ref().is_none_or(|(best_cost, _)| cost < *best_cost) {
            best = Some((cost, packed));
        }
    }
    calls.extend(best.expect("at least one packing length").1);
    calls
}

/// First-fit-decreasing bins of `capacity` rows; a text takes `rows[item]`
/// rows and keeps `counts[item]` as its length.
fn pack(
    family: Family,
    order: &[usize],
    rows: &[usize],
    counts: &[usize],
    capacity: usize,
    packing: &[usize],
) -> Vec<Call> {
    let slots_per_bin = family.max_texts(capacity);
    let mut bins: Vec<(Vec<usize>, usize)> = Vec::new();
    let mut open: Vec<usize> = Vec::new();
    for &item in order {
        let count = rows[item];
        if count > capacity {
            bins.push((vec![item], count));
            continue;
        }
        let fit = open
            .iter()
            .position(|&bin| bins[bin].1 + count <= capacity && bins[bin].0.len() < slots_per_bin);
        match fit {
            Some(position) => {
                let bin = open[position];
                bins[bin].0.push(item);
                bins[bin].1 += count;
                if bins[bin].0.len() == slots_per_bin || bins[bin].1 == capacity {
                    open.remove(position);
                }
            }
            None => {
                bins.push((vec![item], count));
                open.push(bins.len() - 1);
            }
        }
    }
    bins.into_iter()
        .map(|(items, used)| {
            let length = packing
                .iter()
                .copied()
                .find(|&length| length >= used && family.max_texts(length) >= items.len())
                .expect("a bin always fits its capacity's length");
            let mut offset = 0;
            let segments = items
                .into_iter()
                .map(|item| {
                    let segment = Segment {
                        item,
                        offset,
                        len: counts[item],
                    };
                    offset += rows[item];
                    segment
                })
                .collect();
            Call { length, segments }
        })
        .collect()
}

/// Which keys a query row of a text may attend to.
#[derive(Clone, Copy)]
enum Mask {
    /// Every row of the same text.
    Full,
    /// Rows of the same text at most this far away.
    Window(usize),
    /// Rows of the same text up to the query row itself.
    Causal,
}

/// The text a row belongs to, given segments ordered by offset.
fn segment_of(segments: &[Segment], row: usize) -> Option<&Segment> {
    segments
        .iter()
        .find(|segment| row >= segment.offset && row < segment.offset + segment.len)
}

/// `[S, S]` bias: 0 where query and key are in the same text (and within the
/// mask), else [`BLOCKED`]; a padding row attends to itself only, so no row is
/// fully masked.
fn square_bias(segments: &[Segment], length: usize, mask: Mask, out: &mut [f16]) {
    debug_assert_eq!(out.len(), length * length);
    for (row, target) in out.chunks_exact_mut(length).enumerate() {
        target.fill(BLOCKED);
        match segment_of(segments, row) {
            Some(segment) => {
                let end = segment.offset + segment.len;
                let (lower, upper) = match mask {
                    Mask::Full => (segment.offset, end),
                    Mask::Window(window) => (
                        segment.offset.max(row.saturating_sub(window)),
                        end.min(row + window + 1),
                    ),
                    Mask::Causal => (segment.offset, row + 1),
                };
                target[lower..upper].fill(OPEN);
            }
            None => target[row] = OPEN,
        }
    }
}

/// `[S, 1536]` band bias of the windowed functions: row `r`, column `c` is
/// query `r` against key `(r / 512) * 512 - 512 + c`, open when both are in the
/// same text within the sliding window, or the key is the row itself.
fn band_bias(segments: &[Segment], length: usize, out: &mut [f16]) {
    debug_assert_eq!(out.len(), length * BAND_COLUMNS);
    for (row, target) in out.chunks_exact_mut(BAND_COLUMNS).enumerate() {
        target.fill(BLOCKED);
        // Column of key k for this row: k - first_key, first_key = block start - halo.
        let first_key = (row / WINDOW) * WINDOW;
        let column = |key: usize| key + WINDOW - first_key;
        match segment_of(segments, row) {
            Some(segment) => {
                let lower = segment.offset.max(row.saturating_sub(WINDOW));
                let upper = (segment.offset + segment.len).min(row + WINDOW + 1);
                target[column(lower)..column(upper)].fill(OPEN);
            }
            None => target[column(row)] = OPEN,
        }
    }
}

/// `[K, S]` pooling matrix: row k = 1/n_k over text k's rows (mean pooling),
/// unused rows 0.
fn mean_pool(segments: &[Segment], length: usize, out: &mut [f16]) {
    out.fill(f16::ZERO);
    for (slot, segment) in segments.iter().enumerate() {
        let weight = f16::from_f32(1.0 / segment.len as f32);
        let row = &mut out[slot * length..(slot + 1) * length];
        row[segment.offset..segment.offset + segment.len].fill(weight);
    }
}

/// `[K, S]` pooling matrix: row k = 1 at text k's last row (last-token
/// pooling), unused rows 0.
fn last_pool(segments: &[Segment], length: usize, out: &mut [f16]) {
    out.fill(f16::ZERO);
    for (slot, segment) in segments.iter().enumerate() {
        out[slot * length + segment.offset + segment.len - 1] = f16::ONE;
    }
}

/// `[S, 1]` row mask of a SPLADE function: 1 on a text's rows whose token is
/// not a prefix marker (`[Q]` / `[D]` are attended but never pooled), 0 on
/// prefix, alignment and padding rows. `ids[segment.item]` are the text's
/// tokens.
pub(crate) fn fill_keep(segments: &[Segment], ids: &[Vec<u32>], prefix: &[u32], out: &mut [f16]) {
    out.fill(f16::ZERO);
    for segment in segments {
        let tokens = &ids[segment.item];
        debug_assert_eq!(tokens.len(), segment.len);
        for (row, id) in out[segment.offset..segment.offset + segment.len]
            .iter_mut()
            .zip(tokens)
        {
            if !prefix.contains(id) {
                *row = f16::ONE;
            }
        }
    }
}

/// Fill input `name` of a `length`-row function of `family` for `segments`
/// (all but `inputs_embeds`, which comes from the token table, and `keep`,
/// which comes from the token ids through [`fill_keep`]).
pub(crate) fn fill_input(
    family: Family,
    length: usize,
    name: &str,
    segments: &[Segment],
    out: &mut [f16],
) {
    let kind = family.kind(length);
    match (kind, name) {
        (Kind::Bucket | Kind::Winp | Kind::Splade, "global_bias") => {
            square_bias(segments, length, Mask::Full, out)
        }
        (Kind::Bucket, "local_bias") => square_bias(segments, length, Mask::Window(WINDOW), out),
        (Kind::Splade, "local_bias") => {
            square_bias(segments, length, Mask::Window(SPLADE_HALF_WINDOW), out)
        }
        (Kind::Winp | Kind::Winl, "band_bias") => band_bias(segments, length, out),
        (Kind::Bucket | Kind::Winp, "pool") => mean_pool(segments, length, out),
        (Kind::Winl, "key_bias") => {
            let used = segments.iter().map(|s| s.offset + s.len).max().unwrap_or(0);
            out[..used].fill(OPEN);
            out[used..].fill(BLOCKED);
        }
        (Kind::Winl, "pool") => {
            let used = segments.iter().map(|s| s.offset + s.len).max().unwrap_or(0);
            out[..used].fill(LONG_POOL);
            out[used..].fill(f16::ZERO);
        }
        (Kind::Causal, "causal_bias") => square_bias(segments, length, Mask::Causal, out),
        (Kind::Causal, "pool") => last_pool(segments, length, out),
        (kind, name) => unreachable!("{kind:?} function has no input {name}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segs(lens: &[usize]) -> Vec<Segment> {
        let mut offset = 0;
        lens.iter()
            .enumerate()
            .map(|(item, &len)| {
                let segment = Segment { item, offset, len };
                offset += len;
                segment
            })
            .collect()
    }

    fn open(value: f16) -> bool {
        value == OPEN
    }

    #[test]
    fn plan_covers_every_text_once_within_each_calls_rows_and_slots() {
        let family = Family::Gemma2;
        let lengths = family.lengths_for(8192);
        let counts: Vec<usize> = (0..300).map(|i| 1 + (i * 37 + i * i * 11) % 2600).collect();
        let calls = plan(family, &lengths, &counts);
        let mut seen = vec![0usize; counts.len()];
        for call in &calls {
            assert!(lengths.contains(&call.length));
            let used: usize = call.segments.iter().map(|s| s.len).sum();
            assert!(used <= call.length, "{call:?}");
            if family.kind(call.length) == Kind::Winl {
                assert_eq!(call.segments.len(), 1);
            } else {
                assert!(call.segments.len() <= slots(call.length));
            }
            let mut offset = 0;
            for segment in &call.segments {
                assert_eq!(segment.offset, offset, "texts are back to back");
                assert_eq!(segment.len, counts[segment.item]);
                offset += segment.len;
                seen[segment.item] += 1;
            }
            // The call is the smallest length that holds it.
            let smaller = lengths.iter().copied().filter(|&l| l < call.length);
            for length in smaller {
                let fits = used <= length
                    && (family.kind(length) != Kind::Winl || call.segments.len() == 1)
                    && (family.kind(length) == Kind::Winl || call.segments.len() <= slots(length));
                let long_only =
                    family.kind(length) == Kind::Winl && family.kind(call.length) != Kind::Winl;
                assert!(!fits || long_only, "{call:?} fits {length}");
            }
        }
        assert!(seen.iter().all(|&n| n == 1), "{seen:?}");
    }

    #[test]
    fn short_texts_share_one_call() {
        let calls = plan(
            Family::Gemma2,
            &Family::Gemma2.lengths_for(2048),
            &[5, 7, 9],
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].length, 64);
    }

    #[test]
    fn long_lengths_load_only_when_the_context_needs_them() {
        let family = Family::Gemma2;
        assert_eq!(family.lengths_for(2048), family.packing_lengths());
        assert_eq!(family.lengths_for(3000).last(), Some(&3072));
        assert_eq!(family.lengths_for(5000).last(), Some(&6144));
        assert_eq!(family.function_name(64), "embed_64");
        assert_eq!(family.function_name(2048), "winp2048_b512");
        assert_eq!(family.function_name(8192), "winl8192_b512");
    }

    #[test]
    fn square_bias_keeps_texts_apart_and_padding_on_itself() {
        let length = 16;
        let segments = segs(&[5, 4]);
        let mut out = vec![f16::ZERO; length * length];
        square_bias(&segments, length, Mask::Full, &mut out);
        let at = |i: usize, j: usize| open(out[i * length + j]);
        let text = |row: usize| segment_of(&segments, row).map(|s| s.item);
        for i in 0..length {
            for j in 0..length {
                let expected = match (text(i), text(j)) {
                    (Some(a), Some(b)) => a == b,
                    _ => i == j,
                };
                assert_eq!(at(i, j), expected, "({i}, {j})");
            }
        }
    }

    #[test]
    fn local_bias_is_the_global_bias_within_the_sliding_window() {
        let length = 1280;
        let segments = segs(&[1100, 100]);
        let mut global = vec![f16::ZERO; length * length];
        let mut local = vec![f16::ZERO; length * length];
        square_bias(&segments, length, Mask::Full, &mut global);
        square_bias(&segments, length, Mask::Window(WINDOW), &mut local);
        for i in 0..length {
            for j in 0..length {
                let expected = open(global[i * length + j]) && i.abs_diff(j) <= WINDOW;
                assert_eq!(open(local[i * length + j]), expected, "({i}, {j})");
            }
        }
    }

    #[test]
    fn band_bias_is_the_local_bias_seen_through_the_band() {
        let length = 2048;
        let segments = segs(&[1300, 37, 600]);
        let mut local = vec![f16::ZERO; length * length];
        let mut band = vec![f16::ZERO; length * BAND_COLUMNS];
        square_bias(&segments, length, Mask::Window(WINDOW), &mut local);
        band_bias(&segments, length, &mut band);
        for row in 0..length {
            for column in 0..BAND_COLUMNS {
                let key = (row / WINDOW * WINDOW + column).checked_sub(WINDOW);
                let Some(key) = key.filter(|&key| key < length) else {
                    continue; // never read by the graph
                };
                assert_eq!(
                    open(band[row * BAND_COLUMNS + column]),
                    open(local[row * length + key]),
                    "row {row} key {key}"
                );
            }
        }
    }

    #[test]
    fn mean_pool_averages_each_text_and_leaves_spare_slots_empty() {
        let length = 64;
        let segments = segs(&[4, 10]);
        let mut out = vec![f16::ONE; slots(length) * length];
        mean_pool(&segments, length, &mut out);
        let row = |k: usize| &out[k * length..(k + 1) * length];
        assert!(row(0)[..4].iter().all(|&v| v == f16::from_f32(0.25)));
        assert!(row(0)[4..].iter().all(|&v| v == f16::ZERO));
        assert!(row(1)[4..14].iter().all(|&v| v == f16::from_f32(0.1)));
        assert!((2..slots(length)).all(|k| row(k).iter().all(|&v| v == f16::ZERO)));
    }

    #[test]
    fn a_long_call_pools_and_keys_only_the_texts_rows() {
        let length = 3072;
        let segments = segs(&[2600]);
        let mut key = vec![f16::ZERO; length];
        let mut pool = vec![f16::ONE; length];
        fill_input(Family::Gemma2, length, "key_bias", &segments, &mut key);
        fill_input(Family::Gemma2, length, "pool", &segments, &mut pool);
        assert!(key[..2600].iter().all(|&v| v == OPEN));
        assert!(key[2600..].iter().all(|&v| v == BLOCKED));
        assert!(pool[..2600].iter().all(|&v| v == LONG_POOL));
        assert!(pool[2600..].iter().all(|&v| v == f16::ZERO));
    }

    #[test]
    fn qwen3_causal_bias_sees_only_earlier_rows_of_the_same_text() {
        let length = 16;
        let segments = segs(&[5, 4]);
        let mut out = vec![f16::ZERO; length * length];
        fill_input(Family::Qwen3, length, "causal_bias", &segments, &mut out);
        let text = |row: usize| segment_of(&segments, row).map(|s| s.item);
        for i in 0..length {
            for j in 0..length {
                let expected = match (text(i), text(j)) {
                    (Some(a), Some(b)) => a == b && j <= i,
                    _ => i == j,
                };
                let value = out[i * length + j];
                assert_eq!(open(value), expected, "({i}, {j})");
                assert!(open(value) || value == BLOCKED);
            }
        }
    }

    #[test]
    fn qwen3_pools_the_last_row_of_each_text() {
        let length = 64;
        let segments = segs(&[4, 10]);
        let mut out = vec![f16::ONE; slots(length) * length];
        fill_input(Family::Qwen3, length, "pool", &segments, &mut out);
        let row = |k: usize| &out[k * length..(k + 1) * length];
        for (k, last) in [(0, 3), (1, 13)] {
            for (column, &value) in row(k).iter().enumerate() {
                let expected = if column == last { f16::ONE } else { f16::ZERO };
                assert_eq!(value, expected, "slot {k} column {column}");
            }
        }
        assert!((2..slots(length)).all(|k| row(k).iter().all(|&v| v == f16::ZERO)));
    }

    #[test]
    fn qwen3_packs_up_to_2048_rows_in_causal_buckets() {
        let family = Family::Qwen3;
        assert_eq!(
            Family::from_package_name("Qwen3EmbeddingText"),
            Some(family)
        );
        assert_eq!(
            family.lengths_for(8192),
            [64, 128, 256, 512, 768, 1024, 1536, 2048]
        );
        assert_eq!(family.function_name(1536), "embed_1536");
        assert_eq!(
            family.input_names(2048),
            ["inputs_embeds", "causal_bias", "pool"]
        );
        let counts: Vec<usize> = (0..200).map(|i| 1 + (i * 53 + i * i * 7) % 2048).collect();
        let calls = plan(family, &family.lengths_for(2048), &counts);
        let mut seen = vec![0usize; counts.len()];
        for call in &calls {
            let used: usize = call.segments.iter().map(|s| s.len).sum();
            assert!(used <= call.length && call.segments.len() <= slots(call.length));
            for segment in &call.segments {
                seen[segment.item] += 1;
            }
        }
        assert!(seen.iter().all(|&n| n == 1));
        assert_eq!(
            plan(family, &family.lengths_for(2048), &[3, 9, 11]).len(),
            1
        );
    }

    #[test]
    fn qwen3_keeps_short_texts_out_of_its_slowest_function() {
        // Per token, 2048 rows run 8.5x slower than 128 on the Neural
        // Engine, so packing 300-token passages into big calls wastes time.
        let family = Family::Qwen3;
        let counts = vec![300; 64];
        let calls = plan(family, &family.lengths_for(2048), &counts);
        assert!(
            calls.iter().all(|call| call.length < 1536),
            "{:?}",
            calls.iter().map(|call| call.length).collect::<Vec<_>>()
        );
    }

    #[test]
    fn splade_texts_start_on_block_boundaries_and_never_share_a_block() {
        // The graph max-reduces 8-row blocks; a block that held the end of one
        // text and the start of the next would leak terms between them.
        let family = Family::ModernBertSplade;
        let lengths = family.lengths_for(2048);
        let counts: Vec<usize> = (0..200).map(|i| 1 + (i * 37 + i * i * 11) % 2048).collect();
        let calls = plan(family, &lengths, &counts);
        let mut seen = vec![0usize; counts.len()];
        for call in &calls {
            let mut end = 0;
            for segment in &call.segments {
                assert_eq!(segment.offset % SPLADE_BLOCK, 0, "{call:?}");
                assert!(segment.offset >= end, "{call:?}");
                assert_eq!(segment.len, counts[segment.item]);
                end = (segment.offset + segment.len).next_multiple_of(SPLADE_BLOCK);
                seen[segment.item] += 1;
            }
            assert!(end <= call.length, "{call:?}");
        }
        assert!(seen.iter().all(|&n| n == 1));
        // 32 queries of 5 tokens take 8 rows each: two 128-row calls of 16
        // texts, more than a dense function's 8 pool rows.
        let calls = plan(family, &lengths, &[5; 32]);
        assert_eq!(calls.len(), 2, "{calls:?}");
        assert!(calls
            .iter()
            .all(|call| call.length == 128 && call.segments.len() == 16));
    }

    #[test]
    fn splade_keep_masks_prefix_markers_alignment_and_padding() {
        let segments = [
            Segment {
                item: 0,
                offset: 0,
                len: 4,
            },
            Segment {
                item: 1,
                offset: 8,
                len: 3,
            },
        ];
        let ids = vec![vec![5, 50001, 7, 3], vec![5, 50000, 3]];
        let mut out = vec![f16::ONE; 16];
        fill_keep(&segments, &ids, &[50000, 50001], &mut out);
        let keep: Vec<f32> = out.iter().map(|v| v.to_f32()).collect();
        assert_eq!(
            keep,
            [1., 0., 1., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.]
        );
    }

    #[test]
    fn splade_local_bias_opens_64_rows_each_side_within_the_text() {
        let segments = segs(&[200]);
        let length = 256;
        let mut out = vec![f16::ZERO; length * length];
        fill_input(
            Family::ModernBertSplade,
            length,
            "local_bias",
            &segments,
            &mut out,
        );
        let row = &out[100 * length..101 * length];
        let opened: Vec<usize> = (0..length).filter(|&k| open(row[k])).collect();
        assert_eq!(opened, (36..=164).collect::<Vec<_>>());
    }
}

#[cfg(test)]
mod lazy_tests {
    use super::*;

    /// The functions a request touches are exactly the call lengths of its plan,
    /// so a loader that creates a function on first use needs no others.
    #[test]
    fn lengths_used_are_the_distinct_call_lengths_of_the_plan() {
        let family = Family::ModernBertSplade;
        let lengths = family.lengths_for(2048);
        // One short query is one call of the smallest function: the other seven
        // of SPLADE's eight are never needed for it.
        let calls = plan(family, &lengths, &[9]);
        assert_eq!(used_lengths(&calls), vec![lengths[0]]);
        // Every used length is a loaded length, ascending and distinct, and
        // every call's length is among them.
        let counts = [9, 40, 200, 700, 1500, 9, 40];
        let calls = plan(family, &lengths, &counts);
        let used = used_lengths(&calls);
        assert!(used.windows(2).all(|pair| pair[0] < pair[1]), "{used:?}");
        assert!(
            used.iter().all(|length| lengths.contains(length)),
            "{used:?}"
        );
        assert!(calls.iter().all(|call| used.contains(&call.length)));
        // The planner may pack a short text into a long call, so what matters is
        // that the set is the plan's, not the whole table.
        let tiny = used_lengths(&plan(family, &lengths, &[3, 5, 7]));
        assert_eq!(tiny.len(), 1, "{tiny:?}");
        assert!(tiny.len() < lengths.len());
    }
}
