use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

use serde::Serialize;

use crate::epoch::EpochRow;
use crate::features::{COLUMN_KINDS, COLUMN_NAMES, ColumnKind, FieldAgg, merge_field};
use crate::template::WalletTemplate;
use crate::vector::{CORE_AXES, EXTENDED_AXES, HEURISTIC_AXES, parse_key};

#[derive(Debug, Serialize)]
pub struct Window {
    pub start_height: u32,
    pub end_height: u32,
    pub epochs: u64,
}

#[derive(Debug, Serialize)]
pub struct Totals {
    pub txs: u64,
    pub defects: u64,
}

#[derive(Debug, Serialize)]
pub struct TemplatePoint {
    pub start_height: u32,
    pub matches: u64,
    /// Fraction of the epoch's transactions consistent with this template — the
    /// measured anonymity set, as a share of the population.
    pub share: f64,
}

/// One observed value on an axis, wallet-agnostic: the share of *all* classified
/// transactions carrying this value, irrespective of what any other axis of the same
/// transaction reads. This is the number a wallet developer needs when choosing a
/// value for a single axis — "if I emit this, I blend with `share` of the chain."
///
/// `share` is the aggregate over the whole window — a single number that can hide
/// enormous swings from epoch to epoch. `min_epoch_share` / `max_epoch_share` are the
/// smallest and largest per-epoch share this value ever took across the window, so a
/// reader can see whether `share` is a stable point estimate or the midpoint of a wide
/// range. Computed only over epochs with `txs > 0`; an epoch this value was never
/// observed in (but which did classify at least one transaction) contributes a 0.0 to
/// that epoch's share — see `build_report`'s bounds pass for why absence is treated as
/// a real 0% observation rather than skipped.
#[derive(Debug, Serialize)]
pub struct AxisValueShare {
    pub value: String,
    pub count: u64,
    pub share: f64,
    pub min_epoch_share: f64,
    pub max_epoch_share: f64,
}

/// The wallet-agnostic summary of one fingerprint axis: every value it was observed
/// to take, plus the derived numbers ("best blend" / "worst blend" / how concentrated
/// the axis is) that make the value table actionable without a wallet in the picture.
#[derive(Debug, Serialize)]
pub struct AxisSummary {
    /// Every observed value, sorted by share descending (ties broken by value name for
    /// reproducibility). A developer needs to see every option on the axis, not only
    /// the extremes, since the choice is among all of them.
    pub values: Vec<AxisValueShare>,
    pub distinct_values: u64,
    /// The value with the largest anonymity set on this axis — the "best blend" choice.
    pub largest_value: String,
    pub largest_share: f64,
    /// The value with the smallest anonymity set on this axis — the choice that stands
    /// out the most.
    pub smallest_value: String,
    pub smallest_share: f64,
    /// Stability indicator: the amplitude (max − min, in share units) of the *dominant*
    /// value's (`largest_value`'s) per-epoch share across the window. A large amplitude
    /// means the aggregate `largest_share` above is not a stable point — the axis's
    /// composition swings substantially between epochs, typically because it tracks the
    /// share of some traffic population (e.g. an OP_RETURN-tagging service) that itself
    /// varies day to day. A small amplitude means the aggregate is close to what every
    /// individual epoch actually looked like — a property of the population being
    /// measured, not an artifact of which epochs happened to be sampled.
    pub dominant_amplitude: f64,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub window: Window,
    pub totals: Totals,
    pub axis_marginals: BTreeMap<String, BTreeMap<String, u64>>,
    /// Wallet-agnostic per-axis anonymity sets: for every axis, every observed value's
    /// share of the population plus the derived largest/smallest numbers. This
    /// is the primary output of the survey — it answers "what anon set do I get on
    /// axis Y if I choose value X," with no notion of a wallet's full vector at all.
    pub axis_summaries: BTreeMap<String, AxisSummary>,
    /// Per CORE axis, per value: the conditional anonymity-set distribution (see
    /// `ConditionalAnon`). Answers "if my wallet uses value X on axis Y, how small is
    /// the anonymity set I land in?" — the dashboard's headline identifiability number.
    pub conditional_anonymity: BTreeMap<String, BTreeMap<String, ConditionalAnon>>,
    /// The change-identification heuristic's own abstention rate: the `Indeterminate`
    /// share of `change_position` (see `HEURISTIC_AXES`). Surfaced at the top level,
    /// not buried inside `axis_summaries`, because abstention is a property of the
    /// heuristic's method, not of the chain it measures.
    pub change_heuristic_abstention_rate: f64,
    /// Illustrative appendix only: given a wallet's *full* fingerprint vector, what
    /// share of the population is consistent with it. A narrower, wallet-shaped
    /// question than `axis_summaries` above, which is why it is no longer the
    /// headline.
    pub template_series: BTreeMap<String, Vec<TemplatePoint>>,
    /// Per-transaction boolean signals (`EpochRow::aux_counts`), aggregated the same
    /// way as `axis_marginals` — flag name -> share of all classified transactions
    /// where it holds. These are NOT axes: a transaction carries all, some, or none of
    /// them simultaneously, so shares here do not sum to 1.0 and should not be read
    /// alongside the per-axis-value tables above as if they were another axis.
    pub aux_flags: BTreeMap<String, AuxShare>,
    /// Derived from the SAME `output_types` value counts as the `output_types` row in
    /// `axis_summaries` — never recomputed from anything else, so this can never
    /// disagree with that table. Not mutually exclusive (a transaction paying to both a
    /// base58 and a bech32m output counts in both), so the three family shares can sum
    /// to more than 1.0.
    pub encoding_families: EncodingFamilies,
    /// One entry per feature column ever observed in `EpochRow::field_aggs` across the
    /// window (see `features.rs`'s ~173-column catalog), each column's `FieldAgg`
    /// merged across every epoch via `merge_field`. Empty when no epoch in the window
    /// carried `field_aggs` at all (older `epochs.jsonl` written before that field
    /// existed) — never an error, since a field-less report is still a valid report of
    /// everything else above.
    pub fields: Vec<FieldReport>,
}

/// One categorical value's share within its field's `FieldReport::values` — the same
/// shape as `AxisValueShare` but scoped to one column's own tallies (raw accumulator
/// keys, e.g. `"0"`/`"1"` for a `Bool`/`OneHot` column, or the signed integer string a
/// `Ordinal` column's index was folded under — see `fold_field`'s doc comment in
/// `features.rs`). Labels are applied later, at display time; this stays unlabelled.
#[derive(Debug, Serialize)]
pub struct FieldValueShare {
    pub value: String,
    pub count: u64,
    pub share: f64,
}

/// Distributional summary of one `Numeric`-kind field, derived from its merged
/// `count`/`sum`/`min`/`max`/`hist` (see `numeric_stats_from_hist`). `median`/`p90` are
/// estimated by linear interpolation within the histogram bucket containing the
/// respective rank; `min`/`max` are the exact running reductions, not estimates.
#[derive(Debug, Serialize, Clone, PartialEq)]
pub struct Stats {
    pub min: f64,
    pub median: f64,
    pub mean: f64,
    pub p90: f64,
    pub max: f64,
    pub count: u64,
}

/// One histogram bucket of a `Numeric`-kind field, in the same bucket order as
/// `FieldAgg::Numeric::hist`. `lo`/`hi` are the bucket's edges (see the module-level
/// bucket semantics: bucket 0 is `(-inf, edges[0])`, the last bucket is
/// `[edges[last], +inf)` — both open ends serialize as JSON `null` via `f64::INFINITY`/
/// `NEG_INFINITY`, since serde_json renders non-finite floats that way).
#[derive(Debug, Serialize)]
pub struct HistBucket {
    pub lo: f64,
    pub hi: f64,
    pub count: u64,
    pub share: f64,
}

/// One feature column's window-level rollup: every epoch's `FieldAgg` for this column
/// merged via `merge_field`, then rendered per its `ColumnKind`. Exactly one of
/// `values`/(`stats`+`hist`) is populated, matching whether the column is categorical
/// (`Bool`/`OneHot`/`Ordinal`) or `Numeric`.
#[derive(Debug, Serialize)]
pub struct FieldReport {
    pub name: String,
    /// Catalog section id (`"00"`..`"11"`) the column belongs to — see `section_for`.
    pub section: &'static str,
    /// `"bool"` | `"onehot"` | `"ordinal"` | `"numeric"`, derived from `ColumnKind`.
    pub kind: &'static str,
    pub total: u64,
    pub values: Vec<FieldValueShare>,
    pub stats: Option<Stats>,
    pub hist: Option<Vec<HistBucket>>,
}

/// One aggregated auxiliary flag: how many (and what share of) all classified
/// transactions had it set.
#[derive(Debug, Serialize)]
pub struct AuxShare {
    pub count: u64,
    pub share: f64,
}

/// One address-encoding family's share of all classified transactions having at least
/// one output belonging to it.
#[derive(Debug, Serialize)]
pub struct EncodingFamilyShare {
    pub family: String,
    pub count: u64,
    pub share: f64,
}

/// The three address-encoding families derived from `output_types`, plus the remainder
/// that belongs to none of them (`OpReturn`, `NonStandard`, and any other output type
/// with no standard address encoding). `families` are not mutually exclusive; `families`
/// plus `none` together account for every classified transaction exactly once in the
/// sense that every transaction contributes to `none` XOR to at least one family.
#[derive(Debug, Serialize)]
pub struct EncodingFamilies {
    /// base58, bech32, bech32m — in that order.
    pub families: Vec<EncodingFamilyShare>,
    pub none_count: u64,
    pub none_share: f64,
    pub total: u64,
}

/// True for any axis defined under the CURRENT `CORE_AXES` / `EXTENDED_AXES` /
/// `HEURISTIC_AXES` sets. Used to filter `EpochRow::axis_counts` at aggregation time,
/// so a row written under a since-retired axis set (e.g. `change_type_matches_inputs`,
/// deleted entirely — see
/// `docs/superpowers/specs/2026-07-29-remove-heuristic-axes-from-sparsity-design.md`)
/// does not resurrect that axis's marginal/summary in a report regenerated from
/// already-scanned data. This is the per-axis-marginal counterpart to
/// `vector::strip_retired_axes`, which does the equivalent filtering for joint-vector
/// keys.
fn is_current_axis(axis: &str) -> bool {
    CORE_AXES.contains(&axis) || EXTENDED_AXES.contains(&axis) || HEURISTIC_AXES.contains(&axis)
}

#[derive(Debug)]
pub enum ReportError {
    Io(std::io::Error),
    Parse(serde_json::Error),
}

impl std::fmt::Display for ReportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::Parse(e) => write!(f, "parsing epoch row: {e}"),
        }
    }
}

impl std::error::Error for ReportError {}

/// Turns one axis's raw value -> count map into its wallet-agnostic summary: every
/// value's share of the axis total, sorted descending, plus the largest/smallest
/// derived numbers. Returns `None` only for an axis with no observed values at
/// all (which cannot occur for any axis present in `axis_marginals`, since every key
/// there was inserted alongside at least one value/count pair).
///
/// `bounds` supplies each value's (min, max) per-epoch share across the window (see
/// `init_value_share_bounds`/`fold_value_share_bounds_row`), keyed the same as `values`. A value present in
/// `values` but absent from `bounds` (should not happen — the bounds pass
/// is seeded from the same marginal — but defended against here rather than assumed)
/// falls back to `(0.0, 0.0)` rather than propagating an `Option`/panic into the report.
fn axis_summary(
    values: &BTreeMap<String, u64>,
    bounds: &BTreeMap<String, (f64, f64)>,
) -> Option<AxisSummary> {
    if values.is_empty() {
        return None;
    }
    let total: u64 = values.values().sum();
    let mut shares: Vec<AxisValueShare> = values
        .iter()
        .map(|(value, count)| {
            let (min_epoch_share, max_epoch_share) =
                bounds.get(value).copied().unwrap_or((0.0, 0.0));
            AxisValueShare {
                value: value.clone(),
                count: *count,
                share: if total == 0 {
                    0.0
                } else {
                    *count as f64 / total as f64
                },
                min_epoch_share,
                max_epoch_share,
            }
        })
        .collect();
    // Largest share first; ties broken by value name so the report is byte-reproducible.
    shares.sort_by(|a, b| {
        b.share
            .partial_cmp(&a.share)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.value.cmp(&b.value))
    });

    let largest = shares.first().expect("checked values non-empty above");
    let smallest = shares.last().expect("checked values non-empty above");
    let (largest_value, largest_share) = (largest.value.clone(), largest.share);
    let dominant_amplitude = largest.max_epoch_share - largest.min_epoch_share;
    let (smallest_value, smallest_share) = (smallest.value.clone(), smallest.share);

    Some(AxisSummary {
        distinct_values: shares.len() as u64,
        largest_value,
        largest_share,
        smallest_value,
        smallest_share,
        dominant_amplitude,
        values: shares,
    })
}

/// For every axis, for every value ever observed on it, the minimum and maximum share
/// (count / epoch txs) that value took in any single epoch across the whole window.
///
/// Computed in a pass separate from the marginal-aggregation loop in `build_report`,
/// because a value's full set of *possible* epochs to bound against (every axis/value
/// pair that appears anywhere in `axis_marginals`) is only known once every row has
/// been aggregated — a single forward pass would leave a value that first appears in a
/// later epoch with no bound contribution from earlier epochs it was legitimately
/// absent from. See `fold_value_share_bounds_row` for the two deliberate policy choices
/// governing what "minimum" and "maximum" mean here.
///
/// Seeds the bounds accumulator: every axis/value pair that appears anywhere in
/// `axis_marginals`, each starting at `(+inf, -inf)` so the first real observation
/// always wins the initial min/max comparison. Split out from the fold below (see
/// `fold_value_share_bounds_row`) so both `build_report` (one in-memory slice) and
/// `build_report_from_path` (a second streamed pass over disk) can seed once, fold row
/// by row, then finalize — without either needing the full row set materialized at the
/// point this bounds pass runs, only `axis_marginals`, which the FIRST pass already
/// produced in full.
fn init_value_share_bounds(
    axis_marginals: &BTreeMap<String, BTreeMap<String, u64>>,
) -> BTreeMap<String, BTreeMap<String, (f64, f64)>> {
    axis_marginals
        .iter()
        .map(|(axis, values)| {
            let per_value = values
                .keys()
                .map(|value| (value.clone(), (f64::INFINITY, f64::NEG_INFINITY)))
                .collect();
            (axis.clone(), per_value)
        })
        .collect()
}

/// Folds one `EpochRow` into an in-progress bounds accumulator (see
/// `init_value_share_bounds` for how it's seeded). This is the per-row body of what
/// used to be a single function, `per_epoch_value_share_bounds`, taking `&[EpochRow]`
/// directly; factored out so `build_report_from_path` can drive it one row at a time
/// off a second streamed pass over disk instead of an in-memory slice.
///
/// Two deliberate policy choices, both load-bearing for what the resulting min/max
/// actually mean:
///
/// - Epochs with `txs == 0` are excluded entirely (no division by zero, and — just as
///   important — they must NOT be folded in as a 0% observation, since a day with no
///   classified transactions carries no information about this axis at all; treating
///   it as 0% would artificially drag every value's minimum down to 0 for a day that
///   simply produced no data).
/// - Within an epoch that DID classify transactions (`txs > 0`), a value with no entry
///   in that epoch's `axis_counts` for this axis is treated as a genuine 0% observation
///   for that epoch, not as a missing measurement — the axis was measured that epoch
///   (`txs > 0`), and this value simply did not occur in it. This is what lets the
///   minimum surface "a value that vanishes on some days" as the volatility finding it
///   is, rather than hiding it by skipping those epochs.
fn fold_value_share_bounds_row(
    bounds: &mut BTreeMap<String, BTreeMap<String, (f64, f64)>>,
    row: &EpochRow,
) {
    if row.txs == 0 {
        return;
    }
    for (axis, per_value) in bounds.iter_mut() {
        let counts_this_epoch = row.axis_counts.get(axis);
        for (value, (min, max)) in per_value.iter_mut() {
            let count = counts_this_epoch
                .and_then(|c| c.get(value))
                .copied()
                .unwrap_or(0);
            let share = count as f64 / row.txs as f64;
            if share < *min {
                *min = share;
            }
            if share > *max {
                *max = share;
            }
        }
    }
}

/// A value whose count in `axis_marginals` is non-zero must have been contributed by
/// at least one txs > 0 epoch (counts only ever come from classified transactions, and
/// a classified transaction implies its epoch's txs > 0), so in practice every pair
/// gets a finite bound from `fold_value_share_bounds_row`. This guards the theoretical
/// remainder (e.g. a value with a zero count surviving into the marginal from a
/// hand-built test fixture) so `f64::INFINITY` / `NEG_INFINITY` never reach
/// `report.json`, where they would serialize as `null` and violate the "always a
/// number" contract every other share in this report upholds.
fn finalize_value_share_bounds(
    mut bounds: BTreeMap<String, BTreeMap<String, (f64, f64)>>,
) -> BTreeMap<String, BTreeMap<String, (f64, f64)>> {
    for per_value in bounds.values_mut() {
        for (min, max) in per_value.values_mut() {
            if !min.is_finite() {
                *min = 0.0;
            }
            if !max.is_finite() {
                *max = 0.0;
            }
        }
    }
    bounds
}

/// Tokenizes an `output_types` axis value string — the `Debug` formatting of a sorted,
/// deduplicated `Vec<OutputType>` produced by `FingerprintVector::axis_value`, e.g.
/// `"[P2wpkh, OpReturn]"` or `"[P2pkh]"` — into its individual type names. Splits on the
/// literal separator `Debug` uses for a `Vec` rather than matching substrings, so a
/// variant name that happens to be a textual prefix of another (`P2pk` / `P2pkh`) cannot
/// cause a false match.
fn output_type_tokens(debug_repr: &str) -> Vec<&str> {
    debug_repr
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(", ")
        .filter(|s| !s.is_empty())
        .collect()
}

const BASE58_TYPES: &[&str] = &["P2pkh", "P2sh"];
const BECH32_TYPES: &[&str] = &["P2wpkh", "P2wsh"];
const BECH32M_TYPES: &[&str] = &["P2tr"];

/// Derives the three address-encoding families from the SAME `output_types` value ->
/// count map that feeds the `output_types` row of `axis_summaries`, so this can never
/// disagree with the table printed above it in the Markdown report. Each family counts a
/// transaction if ANY of its (possibly several) outputs belongs to that family — so a
/// transaction paying to both a P2PKH and a P2TR output is counted in both `base58` and
/// `bech32m`, and the three shares can sum to more than 1.0. `OpReturn`, `NonStandard`,
/// and any other output type with no standard address encoding belong to none of the
/// three families; such transactions are counted in `none` instead.
fn encoding_families(output_types_marginal: &BTreeMap<String, u64>) -> EncodingFamilies {
    let total: u64 = output_types_marginal.values().sum();
    let mut base58 = 0u64;
    let mut bech32 = 0u64;
    let mut bech32m = 0u64;
    let mut none = 0u64;

    for (key, count) in output_types_marginal {
        let tokens = output_type_tokens(key);
        let has_base58 = tokens.iter().any(|t| BASE58_TYPES.contains(t));
        let has_bech32 = tokens.iter().any(|t| BECH32_TYPES.contains(t));
        let has_bech32m = tokens.iter().any(|t| BECH32M_TYPES.contains(t));
        if has_base58 {
            base58 += count;
        }
        if has_bech32 {
            bech32 += count;
        }
        if has_bech32m {
            bech32m += count;
        }
        if !has_base58 && !has_bech32 && !has_bech32m {
            none += count;
        }
    }

    let share = |c: u64| {
        if total == 0 {
            0.0
        } else {
            c as f64 / total as f64
        }
    };

    EncodingFamilies {
        families: vec![
            EncodingFamilyShare {
                family: "base58".to_string(),
                count: base58,
                share: share(base58),
            },
            EncodingFamilyShare {
                family: "bech32".to_string(),
                count: bech32,
                share: share(bech32),
            },
            EncodingFamilyShare {
                family: "bech32m".to_string(),
                count: bech32m,
                share: share(bech32m),
            },
        ],
        none_count: none,
        none_share: share(none),
        total,
    }
}

/// Everything one streaming pass over the epoch rows folds into. Filled in one row at
/// a time by `fold_row`, so at most one `EpochRow` needs to be resident regardless of
/// how large the file behind it is — see `build_report_from_path`, which drives this
/// straight off disk. `build_report` drives the exact same fold off an in-memory
/// slice, so the two never disagree.
struct Aggregates {
    totals: Totals,
    axis_marginals: BTreeMap<String, BTreeMap<String, u64>>,
    /// Chain-wide CORE joint-vector key -> transaction count, accumulated across epochs.
    /// Each key's count IS the anonymity set its transactions share; this feeds
    /// `conditional_anonymity` (the per-value "how identifying" number the dashboard leads
    /// with). No re-scan: it is rebuilt from the per-epoch `vectors` already on disk.
    vectors: BTreeMap<String, u64>,
    aux_counts: BTreeMap<String, u64>,
    /// Every supplied template gets exactly one `TemplatePoint` per row, defaulting to
    /// zero matches — a template that matches nothing anywhere must show up as an
    /// explicit 0 / 0.0% in every epoch, not be silently absent (that absence is
    /// indistinguishable from a typo in the template's axis name, and is itself the
    /// most alarming finding the tool can produce: a wallet with no anonymity set).
    template_series: BTreeMap<String, Vec<TemplatePoint>>,
    /// Column name -> its `FieldAgg` merged (via `merge_field`) across every epoch seen
    /// so far. Stays empty if no epoch's `field_aggs` has ever been non-empty, which is
    /// exactly what makes `Report::fields` come out `[]` for a legacy scan — see
    /// `build_field_reports`.
    field_aggs: BTreeMap<String, FieldAgg>,
    start_height: Option<u32>,
    end_height: u32,
    epochs: u64,
}

impl Aggregates {
    fn new(templates: &[WalletTemplate]) -> Self {
        Self {
            totals: Totals { txs: 0, defects: 0 },
            axis_marginals: BTreeMap::new(),
            vectors: BTreeMap::new(),
            aux_counts: BTreeMap::new(),
            template_series: templates
                .iter()
                .map(|t| (t.name.clone(), Vec::new()))
                .collect(),
            field_aggs: BTreeMap::new(),
            start_height: None,
            end_height: 0,
            epochs: 0,
        }
    }
}

/// Folds one `EpochRow` into `acc`. This is the whole per-row body `build_report` used
/// to run inline over a `Vec<EpochRow>` it held in full — factored out here so the
/// identical logic can be driven either by an in-memory slice (`build_report`) or by
/// rows read and dropped one at a time as they stream off disk
/// (`build_report_from_path`), with no behavioural difference between the two.
///
/// Template matching is recomputed HERE, at report time, from each row's
/// `vectors_extended` histogram — not read from `EpochRow::template_matches` (see that
/// field's doc comment in epoch.rs for why it is legacy). `vectors_extended`'s keys
/// carry every one of the 17 axes (core + extended; see `FingerprintVector::key_for`
/// and `all_axes` in epoch.rs), so any template is matchable purely from data already
/// on disk. This is what makes a `wallets.toml` correction take effect in seconds
/// instead of requiring a multi-hour rescan.
fn fold_row(acc: &mut Aggregates, row: &EpochRow, templates: &[WalletTemplate]) {
    acc.totals.txs += row.txs;
    acc.totals.defects += row.defects;
    acc.epochs += 1;
    if acc.start_height.is_none() {
        acc.start_height = Some(row.start_height);
    }
    acc.end_height = row.end_height;

    for (axis, values) in &row.axis_counts {
        // Compatibility filter: a row scanned under a since-retired axis set (e.g.
        // `change_type_matches_inputs`, deleted entirely) may still carry that axis's
        // marginal on disk. Skipping it here — rather than only in `epoch.rs`'s
        // `ingest`, which only governs FUTURE scans — keeps a report regenerated
        // against already-scanned data free of axes that no longer exist in code. See
        // `is_current_axis`.
        if !is_current_axis(axis) {
            continue;
        }
        let entry = acc.axis_marginals.entry(axis.clone()).or_default();
        for (value, count) in values {
            *entry.entry(value.clone()).or_insert(0) += count;
        }
    }
    for (name, count) in &row.aux_counts {
        *acc.aux_counts.entry(name.clone()).or_insert(0) += count;
    }
    // Chain-wide CORE joint-vector counts, for `conditional_anonymity`.
    for (key, count) in &row.vectors {
        *acc.vectors.entry(key.clone()).or_insert(0) += count;
    }

    // Per-field rollup (Task 4): merge this row's `field_aggs` into the running
    // window-level aggregate, one column at a time. A column seen for the first time
    // is cloned in wholesale rather than merged against a freshly-`new_field_agg`
    // zero value — cheaper, and avoids needing this function to know each column's
    // `ColumnKind` just to seed an identity element `merge_field` would fold away anyway.
    for (name, agg) in &row.field_aggs {
        match acc.field_aggs.get_mut(name) {
            Some(existing) => merge_field(existing, agg),
            None => {
                acc.field_aggs.insert(name.clone(), agg.clone());
            }
        }
    }

    // Parse each distinct joint-vector key once per row and test it against every
    // template, rather than re-parsing per template.
    let mut matches_this_row: BTreeMap<&str, u64> =
        templates.iter().map(|t| (t.name.as_str(), 0u64)).collect();
    for (key, count) in &row.vectors_extended {
        let axes = parse_key(key);
        for template in templates {
            if template.matches_axes(&axes) {
                *matches_this_row
                    .get_mut(template.name.as_str())
                    .expect("matches_this_row pre-populated with every template name") += count;
            }
        }
    }
    for template in templates {
        let matches = matches_this_row[template.name.as_str()];
        acc.template_series
            .get_mut(&template.name)
            .expect("template_series pre-populated with every template name")
            .push(TemplatePoint {
                start_height: row.start_height,
                matches,
                share: if row.txs == 0 {
                    0.0
                } else {
                    matches as f64 / row.txs as f64
                },
            });
    }
}

/// Turns a completed `Aggregates` fold plus its bounds pass into the final `Report`.
/// Shared tail of `build_report` and `build_report_from_path` — the only difference
/// between the two is how `acc` and `axis_value_bounds` were produced (in-memory slice
/// vs. two streamed passes over disk).
/// Anonymity-set bucket boundaries: `(lo, hi, label)`, contiguous and covering every
/// non-negative count. Used by `conditional_anonymity` to bucket each joint vector's
/// anonymity-set size.
const TAIL_MASS_BOUNDS: [(u64, u64, &str); 7] = [
    (1, 1, "exactly 1"),
    (2, 9, "2-9"),
    (10, 99, "10-99"),
    (100, 999, "100-999"),
    (1_000, 9_999, "1,000-9,999"),
    (10_000, 99_999, "10,000-99,999"),
    (100_000, u64::MAX, ">=100,000"),
];

/// For one CORE axis value, the conditional distribution of CORE-joint anonymity-set
/// sizes among the transactions that use it: `buckets` maps the `TAIL_MASS_BOUNDS`
/// labels to the transaction count landing in each, and `share_lt10`/`share_lt100` are
/// the fractions of those transactions in an anonymity set below 10 / 100 — the "how
/// identifying is this value" signal. Computed from the aggregated joint-vector map, so
/// no per-transaction data or re-scan is needed.
#[derive(Debug, Serialize)]
pub struct ConditionalAnon {
    pub buckets: BTreeMap<String, u64>,
    pub share_lt10: f64,
    pub share_lt100: f64,
}

/// The bucket label a given anonymity-set size falls in. Every non-negative count
/// matches exactly one `TAIL_MASS_BOUNDS` bound.
fn bucket_label_for(count: u64) -> &'static str {
    TAIL_MASS_BOUNDS
        .iter()
        .find(|&&(lo, hi, _)| count >= lo && count <= hi)
        .map(|&(_, _, label)| label)
        .unwrap_or("")
}

/// For every CORE axis value, the conditional anonymity-set distribution among the
/// transactions using it. Each joint vector's own count IS the anonymity set its
/// transactions share; that count is added, weighted by itself (the tx count), into the
/// bucket of every CORE axis value the vector carries.
fn conditional_anonymity(
    vectors: &BTreeMap<String, u64>,
) -> BTreeMap<String, BTreeMap<String, ConditionalAnon>> {
    // axis -> value -> (buckets, total, lt10, lt100)
    struct Acc {
        buckets: BTreeMap<String, u64>,
        total: u64,
        lt10: u64,
        lt100: u64,
    }
    let mut acc: BTreeMap<&str, BTreeMap<String, Acc>> =
        CORE_AXES.iter().map(|&a| (a, BTreeMap::new())).collect();

    for (key, &count) in vectors {
        let parsed = parse_key(key);
        let label = bucket_label_for(count).to_string();
        for &axis in CORE_AXES {
            let Some(value) = parsed.get(axis) else {
                continue;
            };
            let e = acc
                .get_mut(axis)
                .unwrap()
                .entry(value.clone())
                .or_insert_with(|| Acc {
                    buckets: BTreeMap::new(),
                    total: 0,
                    lt10: 0,
                    lt100: 0,
                });
            *e.buckets.entry(label.clone()).or_insert(0) += count;
            e.total += count;
            if count < 10 {
                e.lt10 += count;
            }
            if count < 100 {
                e.lt100 += count;
            }
        }
    }

    acc.into_iter()
        .map(|(axis, values)| {
            let m = values
                .into_iter()
                .map(|(v, e)| {
                    let t = e.total as f64;
                    (
                        v,
                        ConditionalAnon {
                            buckets: e.buckets,
                            share_lt10: if t > 0.0 { e.lt10 as f64 / t } else { 0.0 },
                            share_lt100: if t > 0.0 { e.lt100 as f64 / t } else { 0.0 },
                        },
                    )
                })
                .collect();
            (axis.to_string(), m)
        })
        .collect()
}

/// Looks up a feature column's `ColumnKind` by name. `COLUMN_NAMES[2..]` and
/// `COLUMN_KINDS` are index-aligned (see both const docs in `features.rs`), so this is
/// a linear search over the ~173-entry catalog rather than a per-transaction hot path —
/// called once per distinct column name when building `Report::fields`, never inside
/// the epoch fold.
fn column_kind_for(name: &str) -> Option<&'static ColumnKind> {
    COLUMN_NAMES[2..]
        .iter()
        .zip(COLUMN_KINDS.iter())
        .find(|(n, _)| **n == name)
        .map(|(_, k)| k)
}

/// The catalog section id (`"00"`..`"11"`) a feature column belongs to, following the
/// grouping comments in `COLUMN_NAMES` (features.rs) and `FEATURES.md`. Every one of the
/// 173 `COLUMN_NAMES[2..]` columns has an explicit arm; the two identity columns
/// (`txid`/`block_height`, section `"00"`) never carry a `FieldAgg` and so never reach
/// this function in practice, but are included for documentation of the full 00..11
/// space. A name reaching the fallback arm indicates a column was added to
/// `COLUMN_NAMES` without a section assignment here — panics loudly rather than
/// silently mis-filing it, since `Report::fields` is the dashboard's own catalog view.
fn section_for(name: &str) -> &'static str {
    match name {
        "txid" | "block_height" => "00",

        // 01 — booleans + tri-state pairs
        "op_return"
        | "round_feerate"
        | "low_s_yes"
        | "round_fee"
        | "round_payment"
        | "uih1"
        | "uih2"
        | "address_reuse"
        | "same_block_parent"
        | "changeless"
        | "low_r_yes"
        | "low_r_indeterminate"
        | "uncompressed_pubkey_yes"
        | "uncompressed_pubkey_indeterminate" => "01",

        // 02 — nominal one-hot families (version/nsequence/nlocktime/order/subtype/sighash/input_types)
        "version_1"
        | "version_2"
        | "version_3"
        | "version_other"
        | "nsequence_cake_group_c"
        | "nsequence_lone_0x01"
        | "nsequence_rbf"
        | "nsequence_final"
        | "nsequence_max"
        | "nsequence_mixed_other"
        | "nlocktime_zero"
        | "nlocktime_anti_fee_snipe"
        | "nlocktime_backdated"
        | "nlocktime_future"
        | "nlocktime_timestamp"
        | "input_order_bip69"
        | "input_order_value_ascending"
        | "input_order_value_descending"
        | "input_order_age_ascending"
        | "input_order_other"
        | "input_order_indeterminate"
        | "output_order_bip69"
        | "output_order_value_ascending"
        | "output_order_value_descending"
        | "output_order_age_ascending"
        | "output_order_other"
        | "output_order_indeterminate"
        | "input_subtype_p2sh_p2wpkh"
        | "input_subtype_p2sh_multisig"
        | "input_subtype_p2sh_other"
        | "input_subtype_p2wsh_multisig"
        | "input_subtype_p2wsh_other"
        | "input_subtype_taproot_key_path"
        | "input_subtype_taproot_script_path"
        | "input_subtype_bare"
        | "input_subtype_mixed"
        | "input_subtype_indeterminate"
        | "sighash_taproot_default"
        | "sighash_taproot_explicit"
        | "sighash_all"
        | "sighash_none"
        | "sighash_single"
        | "sighash_acp_all"
        | "sighash_acp_none"
        | "sighash_acp_single"
        | "sighash_other"
        | "sighash_mixed"
        | "sighash_na"
        | "input_types_uniform_p2pkh"
        | "input_types_uniform_p2sh"
        | "input_types_uniform_p2wpkh"
        | "input_types_uniform_p2wsh"
        | "input_types_uniform_p2tr"
        | "input_types_uniform_op_return"
        | "input_types_uniform_nonstandard"
        | "input_types_uniform_p2pk"
        | "input_types_mixed"
        | "input_types_unknown" => "02",

        // 03 — ordinal index columns
        "ecdsa_sigs_index"
        | "input_age_index"
        | "output_structure_index"
        | "feerate_bucket_index"
        | "locktime_offset_index"
        | "locktime_offset_not_applicable" => "03",

        // 04 — change heuristic (position/type, plus the round-number agreement pair)
        "change_position_first"
        | "change_position_last"
        | "change_position_middle"
        | "change_position_indeterminate"
        | "change_type_present"
        | "change_detected_by_round_number"
        | "change_heuristics_agree" => "04",

        // 05 — nSequence/nLockTime bit-level aggregates (Groups A+B, plus the nlocktime opt-in fingerprint)
        "bip125_rbf"
        | "all_inputs_final"
        | "bip68_active"
        | "bip68_type_time"
        | "nsequence_reserved_bits_set"
        | "bip68_relative_value_max"
        | "nlocktime_dead"
        | "nlocktime_optin_without_use" => "05",

        // 06 — counts and positional signal (Group C, positional type encoding, multisig config, sig-op cost)
        "input_count"
        | "output_count"
        | "input_count_p2pkh"
        | "input_count_p2sh"
        | "input_count_p2wpkh"
        | "input_count_p2wsh"
        | "input_count_p2tr"
        | "input_count_op_return"
        | "input_count_nonstandard"
        | "input_count_p2pk"
        | "output_count_p2pkh"
        | "output_count_p2sh"
        | "output_count_p2wpkh"
        | "output_count_p2wsh"
        | "output_count_p2tr"
        | "output_count_op_return"
        | "output_count_nonstandard"
        | "output_count_p2pk"
        | "first_output_type_index"
        | "last_output_type_index"
        | "first_output_matches_input_type"
        | "last_output_matches_input_type"
        | "outputs_type_grouped"
        | "first_input_type_index"
        | "last_input_type_index"
        | "inputs_type_grouped"
        | "input_type_at_pos_0"
        | "input_type_at_pos_1"
        | "input_type_at_pos_2"
        | "input_type_at_pos_3"
        | "input_type_at_pos_4"
        | "input_type_at_pos_5"
        | "input_type_at_pos_6"
        | "input_type_at_pos_7"
        | "input_type_at_pos_8"
        | "input_type_at_pos_9"
        | "output_type_at_pos_0"
        | "output_type_at_pos_1"
        | "output_type_at_pos_2"
        | "output_type_at_pos_3"
        | "output_type_at_pos_4"
        | "output_type_at_pos_5"
        | "output_type_at_pos_6"
        | "output_type_at_pos_7"
        | "output_type_at_pos_8"
        | "output_type_at_pos_9"
        | "multisig_m"
        | "multisig_n"
        | "multisig_mixed"
        | "sigop_count" => "06",

        // 07 — feerate (raw + derived + the two appended weight-unit analogues)
        "fee_sat"
        | "vsize"
        | "weight"
        | "feerate_sat_per_vb"
        | "feerate_sat_per_kwu"
        | "fee_is_multiple_of_vsize"
        | "feerate_over_block_min"
        | "feerate_over_prev_block_min"
        | "fee_is_multiple_of_kwu"
        | "fee_is_multiple_of_weight" => "07",

        // 08 — amounts (digit-structure measures + output-value-structure columns)
        "max_output_value_sat"
        | "hamming_decimal"
        | "hamming_base2"
        | "hamming_base3"
        | "decimal_sig_fig_span"
        | "hamming_decimal_min_over_outputs"
        | "distinct_output_value_count"
        | "max_equal_value_output_count" => "08",

        // 09 — taproot
        "taproot_max_merkle_depth"
        | "taproot_script_path_input_count"
        | "taproot_keyspend_non_default_sighash" => "09",

        // 10 — standardness
        "is_standard"
        | "nonstd_version"
        | "nonstd_weight"
        | "nonstd_output_type"
        | "nonstd_dust"
        | "nonstd_multi_op_return"
        | "nonstd_bare_multisig" => "10",

        // 11 — protocol markers
        "has_inscription_envelope" | "has_runestone" => "11",

        other => unreachable!(
            "column {other:?} has no section_for mapping — add one when a new feature \
             column is introduced"
        ),
    }
}

/// The value at rank `p` (0.0..=1.0) of a `Numeric` field's merged histogram, estimated
/// by the standard grouped-data formula: find the bucket containing the
/// `k = ceil(p * count)`-th smallest item (1-indexed), then linearly interpolate within
/// that bucket assuming its items are evenly spread across its width. The two open-ended
/// buckets (index 0 = underflow `(-inf, edges[0])`, index `edges.len()` = overflow
/// `[edges[last], +inf)`) have no finite width to interpolate across, so a rank landing
/// in either clamps to the field's exact `min`/`max` instead (per the task's resolved
/// ambiguity). `count == 0` returns `0.0` — callers must not invoke this when there is no
/// data (see `numeric_stats_from_hist`, which guards this case before calling in).
fn percentile_from_hist(
    edges: &[f64],
    count: u64,
    min: f64,
    max: f64,
    hist: &[u64],
    p: f64,
) -> f64 {
    if count == 0 {
        return 0.0;
    }
    let k = ((p * count as f64).ceil() as u64).clamp(1, count);
    let mut cumulative: u64 = 0;
    for (i, &h) in hist.iter().enumerate() {
        if h == 0 {
            continue;
        }
        if k <= cumulative + h {
            if i == 0 {
                return min;
            }
            if i == edges.len() {
                return max;
            }
            let lo = edges[i - 1];
            let hi = edges[i];
            let position_in_bucket = (k - cumulative) as f64; // 1..=h
            let fraction = (position_in_bucket - 0.5) / h as f64;
            return lo + fraction * (hi - lo);
        }
        cumulative += h;
    }
    // Every bucket accounted for and `k <= count`, so this is unreachable as long as
    // `hist`'s buckets sum to `count` (an invariant `merge_field`/`fold_field` maintain).
    max
}

/// Turns a `Numeric` field's merged `count`/`sum`/`min`/`max`/`hist` into its dashboard
/// `Stats`: `mean` is exact (`sum/count`); `median`/`p90` are estimated via
/// `percentile_from_hist`; `min`/`max` pass through unchanged, since those two are exact
/// running reductions, not estimates. `count == 0` (a `Numeric` column that was created
/// but never folded — e.g. a hand-built fixture) returns all-zero `Stats` rather than
/// propagating `sum/0` `NaN` or the `f64::INFINITY`/`NEG_INFINITY` `new_field_agg` seeds
/// `min`/`max` start at.
pub fn numeric_stats_from_hist(
    edges: &[f64],
    count: u64,
    sum: f64,
    min: f64,
    max: f64,
    hist: &[u64],
) -> Stats {
    if count == 0 {
        return Stats {
            min: 0.0,
            median: 0.0,
            mean: 0.0,
            p90: 0.0,
            max: 0.0,
            count: 0,
        };
    }
    let mean = sum / count as f64;
    let median = percentile_from_hist(edges, count, min, max, hist, 0.5);
    let p90 = percentile_from_hist(edges, count, min, max, hist, 0.9);
    Stats {
        min,
        median,
        mean,
        p90,
        max,
        count,
    }
}

/// Renders one column's merged `FieldAgg` into its `FieldReport`, per `kind`'s variant.
/// Categorical variants (`Bool`/`OneHot`/`Ordinal`) populate `values` (sorted by count
/// descending, ties broken by value string for reproducibility) and leave
/// `stats`/`hist` `None`; `Numeric` populates `stats`/`hist` and leaves `values` empty.
fn field_report(name: &str, kind: &ColumnKind, agg: &FieldAgg) -> FieldReport {
    let section = section_for(name);
    match (kind, agg) {
        (ColumnKind::Bool, FieldAgg::Categorical(m))
        | (ColumnKind::OneHot { .. }, FieldAgg::Categorical(m))
        | (ColumnKind::Ordinal { .. }, FieldAgg::Categorical(m)) => {
            let kind_label = match kind {
                ColumnKind::Bool => "bool",
                ColumnKind::OneHot { .. } => "onehot",
                ColumnKind::Ordinal { .. } => "ordinal",
                ColumnKind::Numeric { .. } => unreachable!("matched above"),
            };
            let total: u64 = m.values().sum();
            let mut values: Vec<FieldValueShare> = m
                .iter()
                .map(|(value, count)| FieldValueShare {
                    value: value.clone(),
                    count: *count,
                    share: if total == 0 {
                        0.0
                    } else {
                        *count as f64 / total as f64
                    },
                })
                .collect();
            values.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.value.cmp(&b.value)));
            FieldReport {
                name: name.to_string(),
                section,
                kind: kind_label,
                total,
                values,
                stats: None,
                hist: None,
            }
        }
        (
            ColumnKind::Numeric { edges },
            FieldAgg::Numeric {
                count,
                sum,
                min,
                max,
                hist,
            },
        ) => {
            let stats = numeric_stats_from_hist(edges, *count, *sum, *min, *max, hist);
            let total_hist: u64 = hist.iter().sum();
            let hist_out: Vec<HistBucket> = hist
                .iter()
                .enumerate()
                .map(|(i, &h)| {
                    let lo = if i == 0 {
                        f64::NEG_INFINITY
                    } else {
                        edges[i - 1]
                    };
                    let hi = if i == edges.len() {
                        f64::INFINITY
                    } else {
                        edges[i]
                    };
                    HistBucket {
                        lo,
                        hi,
                        count: h,
                        share: if total_hist == 0 {
                            0.0
                        } else {
                            h as f64 / total_hist as f64
                        },
                    }
                })
                .collect();
            FieldReport {
                name: name.to_string(),
                section,
                kind: "numeric",
                total: *count,
                values: Vec::new(),
                stats: Some(stats),
                hist: Some(hist_out),
            }
        }
        _ => unreachable!("a FieldAgg's variant must match the ColumnKind it was created from"),
    }
}

/// Builds `Report::fields` from the window's merged `field_aggs` (see
/// `Aggregates::field_aggs`, folded across every epoch by `merge_field` in `fold_row`).
/// Empty when `field_aggs` is empty — i.e. no epoch in the window carried it at all,
/// which is exactly the "old `epochs.jsonl`, no `field_aggs`" compatibility case the task
/// requires: such a report must still succeed, just with `fields: []`. Sorted by column
/// name (the map's natural `BTreeMap` order) for a reproducible `report.json`.
fn build_field_reports(field_aggs: &BTreeMap<String, FieldAgg>) -> Vec<FieldReport> {
    field_aggs
        .iter()
        .map(|(name, agg)| {
            let kind = column_kind_for(name).unwrap_or_else(|| {
                panic!("field_aggs carries column {name:?}, which is not in COLUMN_NAMES")
            });
            field_report(name, kind, agg)
        })
        .collect()
}

fn assemble_report(
    acc: Aggregates,
    axis_value_bounds: BTreeMap<String, BTreeMap<String, (f64, f64)>>,
) -> Report {
    let Aggregates {
        totals,
        axis_marginals,
        vectors,
        aux_counts,
        template_series,
        field_aggs,
        start_height,
        end_height,
        epochs,
    } = acc;

    let axis_summaries: BTreeMap<String, AxisSummary> = axis_marginals
        .iter()
        .filter_map(|(axis, values)| {
            axis_summary(values, &axis_value_bounds[axis]).map(|summary| (axis.clone(), summary))
        })
        .collect();

    let aux_flags: BTreeMap<String, AuxShare> = aux_counts
        .iter()
        .map(|(name, count)| {
            let share = if totals.txs == 0 {
                0.0
            } else {
                *count as f64 / totals.txs as f64
            };
            (
                name.clone(),
                AuxShare {
                    count: *count,
                    share,
                },
            )
        })
        .collect();

    let encoding_families = encoding_families(
        axis_marginals
            .get("output_types")
            .unwrap_or(&BTreeMap::new()),
    );

    // The change-identification heuristic's own abstention rate: the `Indeterminate`
    // share of `change_position` (equivalently `change_type`, which goes
    // `Indeterminate` together with it whenever the heuristic itself is inconclusive —
    // `change_position` additionally counts sub-2-output transactions as
    // `Indeterminate`, which is why the doc comment and the design both name
    // `change_position` specifically as the source of this number).
    let change_heuristic_abstention_rate = axis_marginals
        .get("change_position")
        .map(|values| {
            let total: u64 = values.values().sum();
            let indeterminate = values.get("Indeterminate").copied().unwrap_or(0);
            if total == 0 {
                0.0
            } else {
                indeterminate as f64 / total as f64
            }
        })
        .unwrap_or(0.0);

    Report {
        window: Window {
            start_height: start_height.unwrap_or(0),
            end_height,
            epochs,
        },
        totals,
        axis_marginals,
        axis_summaries,
        conditional_anonymity: conditional_anonymity(&vectors),
        change_heuristic_abstention_rate,
        template_series,
        aux_flags,
        encoding_families,
        fields: build_field_reports(&field_aggs),
    }
}

pub fn build_report(rows: &[EpochRow], templates: &[WalletTemplate]) -> Report {
    let mut acc = Aggregates::new(templates);
    for row in rows {
        fold_row(&mut acc, row, templates);
    }

    let mut bounds = init_value_share_bounds(&acc.axis_marginals);
    for row in rows {
        fold_value_share_bounds_row(&mut bounds, row);
    }
    let axis_value_bounds = finalize_value_share_bounds(bounds);

    assemble_report(acc, axis_value_bounds)
}

/// The streaming counterpart to `build_report`: drives the identical `fold_row` /
/// bounds-pass logic straight off disk via `epoch_rows`, so at most one `EpochRow` (a
/// few hundred bytes to low tens of KB of raw JSON, versus the multi-GB `Vec<EpochRow>`
/// `build_report` requires its caller to hold) is ever resident. Reads the file twice —
/// once to fold totals/marginals/vectors, once more for the per-epoch min/max bounds
/// pass, which genuinely needs `axis_marginals` (the first pass's output) before it can
/// even initialize its accumulator (see `fold_value_share_bounds_row`'s doc comment on
/// why a single forward pass cannot do this). Two sequential linear scans of the file
/// are cheap next to holding the whole dataset in memory at once, which is the actual
/// scaling problem this function exists to avoid.
fn build_report_from_path(
    epochs_path: &Path,
    templates: &[WalletTemplate],
) -> Result<Report, ReportError> {
    let mut acc = Aggregates::new(templates);
    for row in epoch_rows(epochs_path)? {
        fold_row(&mut acc, &row?, templates);
    }

    let mut bounds = init_value_share_bounds(&acc.axis_marginals);
    for row in epoch_rows(epochs_path)? {
        fold_value_share_bounds_row(&mut bounds, &row?);
    }
    let axis_value_bounds = finalize_value_share_bounds(bounds);

    Ok(assemble_report(acc, axis_value_bounds))
}

/// Streams every `EpochRow` out of a scan's JSONL output, in file order, one row at a
/// time — the file is read with a `BufReader` and each line is parsed and yielded as it
/// is read, so a caller that discards each row after using it (see
/// `build_report_from_path` and `write_series_csv` in `bin/lumen.rs`) never holds more
/// than one row's worth of the dataset in memory, however large the file on disk. The
/// eager `read_epoch_rows` below is a thin `Vec`-collecting wrapper over this, kept for
/// call sites (mainly tests) that genuinely want the whole dataset in memory at once.
pub fn epoch_rows(
    epochs_path: &Path,
) -> Result<impl Iterator<Item = Result<EpochRow, ReportError>>, ReportError> {
    let file = std::fs::File::open(epochs_path).map_err(ReportError::Io)?;
    // A `.zst` suffix means the finished epochs file was zstd-compressed for storage
    // (a scan's JSONL is large and compresses heavily). Decompress transparently; the
    // stream stays line-at-a-time either way, so memory is still one row at a time.
    let reader: Box<dyn BufRead> = if epochs_path.extension().is_some_and(|e| e == "zst") {
        Box::new(BufReader::new(
            zstd::stream::read::Decoder::new(file).map_err(ReportError::Io)?,
        ))
    } else {
        Box::new(BufReader::new(file))
    };
    Ok(reader.lines().filter_map(|line| match line {
        Ok(l) if l.trim().is_empty() => None,
        Ok(l) => Some(serde_json::from_str::<EpochRow>(&l).map_err(ReportError::Parse)),
        Err(e) => Some(Err(ReportError::Io(e))),
    }))
}

/// Read every `EpochRow` out of a scan's JSONL output, in file order, into memory at
/// once. See `epoch_rows` for the streaming version this delegates to.
pub fn read_epoch_rows(epochs_path: &Path) -> Result<Vec<EpochRow>, ReportError> {
    epoch_rows(epochs_path)?.collect()
}

/// Build the report from a scan's JSONL output and serialise it to `report.json`.
pub fn write_report(
    epochs_path: &Path,
    out_dir: &Path,
    templates: &[WalletTemplate],
) -> Result<(), ReportError> {
    let report = build_report_from_path(epochs_path, templates)?;
    std::fs::create_dir_all(out_dir).map_err(ReportError::Io)?;

    let json = serde_json::to_string_pretty(&report).map_err(ReportError::Parse)?;
    std::fs::write(out_dir.join("report.json"), json).map_err(ReportError::Io)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::template::Confidence;
    use std::collections::BTreeMap;

    /// Builds a row whose `vectors`/`vectors_extended` keys are real `axis=value`
    /// pairs on the `nsequence` axis (e.g. `nsequence=Rbf`), so a `WalletTemplate`
    /// constraining `nsequence` can actually be matched against them through
    /// `parse_key` + `matches_axes`, exactly as `build_report` does against
    /// production data. `template_matches` is populated only as the legacy
    /// scan-time snapshot, for tests that specifically check report-time matching
    /// against it — `build_report` itself never reads it.
    fn row(start: u32, vectors: &[(&str, u64)], template: (&str, u64)) -> EpochRow {
        let mut axis_counts = BTreeMap::new();
        let mut nseq = BTreeMap::new();
        for (name, count) in vectors {
            nseq.insert((*name).to_string(), *count);
        }
        axis_counts.insert("nsequence".to_string(), nseq);

        EpochRow {
            start_height: start,
            end_height: start + 143,
            txs: vectors.iter().map(|(_, c)| c).sum(),
            defects: 0,
            axis_counts,
            vectors: vectors
                .iter()
                .map(|(n, c)| ((*n).to_string(), *c))
                .collect(),
            vectors_extended: vectors
                .iter()
                .map(|(n, c)| (format!("nsequence={n}"), *c))
                .collect(),
            aux_counts: BTreeMap::new(),
            field_aggs: BTreeMap::new(),
            template_matches: [(template.0.to_string(), template.1)].into_iter().collect(),
        }
    }

    /// A template constraining `nsequence` to `expected` — matches whichever `row()`
    /// buckets carry that value on the `nsequence` axis, and nothing else.
    fn nsequence_template(name: &str, expected: &str) -> WalletTemplate {
        WalletTemplate {
            name: name.to_string(),
            confidence: Confidence::CodePredicted,
            axes: [("nsequence".to_string(), expected.to_string())]
                .into_iter()
                .collect(),
        }
    }

    #[test]
    fn sums_marginals_across_epochs() {
        let rows = vec![
            row(1000, &[("Rbf", 90), ("CakeGroupC", 10)], ("cake", 10)),
            row(1144, &[("Rbf", 80), ("CakeGroupC", 20)], ("cake", 20)),
        ];
        let report = build_report(&rows, &[]);

        assert_eq!(report.window.start_height, 1000);
        assert_eq!(report.window.end_height, 1287);
        assert_eq!(report.totals.txs, 200);
        assert_eq!(report.axis_marginals["nsequence"]["Rbf"], 170);
        assert_eq!(report.axis_marginals["nsequence"]["CakeGroupC"], 30);
    }

    #[test]
    fn template_series_carries_per_epoch_anonymity_sets() {
        let rows = vec![
            row(1000, &[("Rbf", 90), ("CakeGroupC", 10)], ("cake", 10)),
            row(1144, &[("Rbf", 80), ("CakeGroupC", 20)], ("cake", 20)),
        ];
        let cake = nsequence_template("cake", "CakeGroupC");
        let report = build_report(&rows, std::slice::from_ref(&cake));

        let series = &report.template_series["cake"];
        assert_eq!(series.len(), 2);
        assert_eq!(series[0].start_height, 1000);
        assert_eq!(series[0].matches, 10);
        assert!((series[0].share - 0.1).abs() < 1e-9);
        assert!((series[1].share - 0.2).abs() < 1e-9);
    }

    #[test]
    fn a_template_matching_nothing_anywhere_appears_as_zero_every_epoch_not_absent() {
        // This is the finding that was previously unreportable entirely: a template
        // that matches zero transactions in every epoch used to get no key at all in
        // `EpochRow::template_matches` (the old scan-time-only mechanism), and
        // therefore no entry anywhere in the report — indistinguishable from a typo in
        // the template's axis name, and silently hiding the single most alarming
        // result the tool can produce (a wallet with no measured anonymity set at
        // all). Report-time matching against `vectors_extended` must instead emit an
        // explicit zero for every epoch.
        let rows = vec![
            row(1000, &[("Rbf", 90), ("CakeGroupC", 10)], ("cake", 10)),
            row(1144, &[("Rbf", 100)], ("cake", 0)),
        ];
        let ghost = nsequence_template("ghost-wallet", "NeverObservedOnChain");
        let report = build_report(&rows, std::slice::from_ref(&ghost));

        let series = &report.template_series["ghost-wallet"];
        assert_eq!(
            series.len(),
            2,
            "a wallet matching nothing must still get one point per epoch, not be absent"
        );
        for point in series {
            assert_eq!(
                point.matches, 0,
                "matches must be an explicit 0, not absent"
            );
            assert_eq!(
                point.share, 0.0,
                "share must be an explicit 0.0, not absent"
            );
        }
    }

    #[test]
    fn every_template_gets_exactly_one_point_per_epoch() {
        let rows = vec![
            row(1000, &[("Rbf", 90), ("CakeGroupC", 10)], ("cake", 10)),
            row(1144, &[("Rbf", 80), ("CakeGroupC", 20)], ("cake", 20)),
            row(1288, &[("Rbf", 100)], ("cake", 0)),
        ];
        let templates = vec![
            nsequence_template("cake", "CakeGroupC"),
            nsequence_template("core-like", "Rbf"),
            nsequence_template("ghost", "NeverObservedOnChain"),
        ];
        let report = build_report(&rows, &templates);

        assert_eq!(report.template_series.len(), templates.len());
        for template in &templates {
            assert_eq!(
                report.template_series[&template.name].len(),
                rows.len(),
                "template {} must have exactly one point per epoch",
                template.name
            );
        }
    }

    #[test]
    fn report_time_matching_produces_expected_counts_for_a_known_composition() {
        // Proves report-time matching (`parse_key` + `matches_axes`, driven purely by
        // `EpochRow::vectors_extended`) is correct, not merely plausible: build a
        // fixture whose exact composition is known by construction, accumulate it into
        // a real `EpochRow` via `EpochAccumulator::ingest` (which no longer computes
        // scan-time template matches at all — see `epoch.rs`), run it through
        // `build_report`, and assert the recomputed counts against values derived
        // directly from the fixture's construction, rather than against a second copy
        // of the matching logic (there no longer is one to compare against).
        //
        // A block with only one non-coinbase transaction cannot exercise this: the
        // epoch row would end up with a single distinct `vectors_extended` key, so
        // every template trivially matches "all of 1" or "none of 1" — a bug that
        // summed key COUNTS as if it were counting distinct KEYS would sail through
        // unnoticed. This fixture instead carries four transactions across three
        // distinct fingerprint shapes, with one shape duplicated (so one key carries
        // count 2), forcing `build_report` to sum multiple distinct vectors within the
        // same row.
        use crate::epoch::EpochAccumulator;
        use crate::template::Confidence;
        use crate::vector::tests_support::{
            EPOCH_TEST_HEIGHT, block_with_many, cake_like_tx, legacy_p2pkh_tx,
        };

        // Shape A, x2: the chain-proven Cake-shaped segwit tx (nsequence=CakeGroupC,
        // nlocktime=Zero, low_r=Yes). Calling the fixture twice yields two independent
        // `Transaction`s with identical fingerprint shape, so they collapse into ONE
        // `vectors_extended` key with count 2.
        let cake_a = cake_like_tx();
        let cake_b = cake_like_tx();

        // Shape B: a legacy P2PKH tx at the fixture's default sequence/locktime
        // (nsequence=Max, nlocktime=Zero) — a single legacy input, unlike A's two
        // segwit inputs, and a different nsequence class entirely.
        let mut pubkey = vec![0xab; 33];
        pubkey[0] = 0x02;
        let mut legacy_max = legacy_p2pkh_tx(pubkey.clone());
        legacy_max.input[0].previous_output.vout = 100; // distinct prevout from A's

        // Shape C: the same legacy shape, but RBF-signaling and anti-fee-snipe
        // locktimed (nsequence=Rbf, nlocktime=AntiFeeSnipe since EPOCH_TEST_HEIGHT-1
        // falls in (height-100, height]) — different from both A and B on two axes at
        // once.
        let mut legacy_rbf = legacy_p2pkh_tx(pubkey);
        legacy_rbf.input[0].previous_output.vout = 200; // distinct prevout from A's and B's
        legacy_rbf.input[0].sequence = bitcoin::Sequence(0xffff_fffd);
        legacy_rbf.lock_time =
            bitcoin::absolute::LockTime::from_height(EPOCH_TEST_HEIGHT - 1).unwrap();

        let block = block_with_many(
            vec![cake_a, cake_b, legacy_max, legacy_rbf],
            EPOCH_TEST_HEIGHT,
        );

        let mut acc = EpochAccumulator::new(EPOCH_TEST_HEIGHT, 144);
        acc.ingest(&block);
        let row = acc.flush(EPOCH_TEST_HEIGHT + 144);

        // Sanity-check the fixture itself before trusting it to catch anything: it
        // must actually produce multiple distinct keys with differing counts, or the
        // rest of this test would be exercising the same all-or-nothing case flagged
        // above.
        assert_eq!(
            row.txs, 4,
            "all four transactions must classify, not defect"
        );
        assert_eq!(
            row.vectors_extended.len(),
            3,
            "three genuinely different shapes must produce three distinct \
             vectors_extended keys, got {:?}",
            row.vectors_extended
        );
        assert!(
            row.vectors_extended.values().any(|&count| count > 1),
            "the duplicated Cake-shaped tx must land in one key with count 2, not two \
             separate count-1 keys: {:?}",
            row.vectors_extended
        );

        // Three single-axis, hand-built templates that exhaustively and disjointly
        // partition the fixture by `nsequence` class alone: every transaction above is
        // exactly one of CakeGroupC (A), Max (B), or Rbf (C) by construction, so their
        // expected match counts are known directly from the fixture, not from
        // re-running the matcher a second way.
        let by_nsequence = |name: &str, expected: &str| WalletTemplate {
            name: name.to_string(),
            confidence: Confidence::CodePredicted,
            axes: [("nsequence".to_string(), expected.to_string())]
                .into_iter()
                .collect(),
        };
        let templates = vec![
            by_nsequence("cake-class", "CakeGroupC"),
            by_nsequence("max-class", "Max"),
            by_nsequence("rbf-class", "Rbf"),
        ];

        let report = build_report(std::slice::from_ref(&row), &templates);

        assert_eq!(
            report.template_series["cake-class"][0].matches, 2,
            "shape A (Cake) is duplicated: exactly 2 of the 4 transactions"
        );
        assert_eq!(
            report.template_series["max-class"][0].matches, 1,
            "shape B (legacy, Max nsequence) is exactly 1 of the 4 transactions"
        );
        assert_eq!(
            report.template_series["rbf-class"][0].matches, 1,
            "shape C (legacy, Rbf nsequence) is exactly 1 of the 4 transactions"
        );
        // These three classes are exhaustive and mutually exclusive over this fixture
        // (every transaction has exactly one nsequence class), so they must account for
        // every transaction exactly once.
        let total: u64 = ["cake-class", "max-class", "rbf-class"]
            .iter()
            .map(|name| report.template_series[*name][0].matches)
            .sum();
        assert_eq!(
            total, row.txs,
            "every transaction must land in exactly one nsequence class"
        );

        // Also pin the real shipped "Cake Wallet" template (which constrains more than
        // just `nsequence`) to the same, independently-known count: only the two
        // Cake-shaped transactions satisfy nsequence=CakeGroupC, nlocktime=Zero,
        // low_r=Yes, and uncompressed_pubkey=No all at once — the Max/Rbf-sequenced
        // legacy transactions already contradict it on `nsequence` alone.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("templates/wallets.toml");
        let shipped = crate::template::load_templates(&path).unwrap();
        let cake_wallet = shipped
            .iter()
            .find(|t| t.name == "Cake Wallet")
            .expect("cake template ships");
        let report_with_shipped = build_report(
            std::slice::from_ref(&row),
            std::slice::from_ref(cake_wallet),
        );
        assert_eq!(
            report_with_shipped.template_series["Cake Wallet"][0].matches, 2,
            "the shipped Cake Wallet template must match exactly the two Cake-shaped \
             txs, not all 4 or none"
        );
    }

    #[test]
    fn zero_tx_epoch_yields_zero_share_not_a_division_by_zero() {
        let rows = vec![row(1000, &[], ("cake", 0))];
        let cake = nsequence_template("cake", "CakeGroupC");
        let report = build_report(&rows, std::slice::from_ref(&cake));

        assert_eq!(report.totals.txs, 0);
        let series = &report.template_series["cake"];
        assert_eq!(series.len(), 1);
        assert_eq!(series[0].matches, 0);
        assert_eq!(series[0].share, 0.0);
        assert!(!series[0].share.is_nan(), "0/0 must not produce NaN");
    }

    #[test]
    fn empty_input_produces_a_report_without_panicking_or_nan() {
        let report = build_report(&[], &[]);

        assert_eq!(report.window.epochs, 0);
        assert_eq!(report.totals.txs, 0);
        assert_eq!(report.totals.defects, 0);
        assert!(report.template_series.is_empty());

        assert!(report.axis_summaries.is_empty());

        // Gap 1 / Gap 2 additions must not panic or produce NaN on an empty scan either.
        assert!(report.aux_flags.is_empty());
        assert_eq!(report.encoding_families.total, 0);
        assert!(!report.encoding_families.none_share.is_nan());
        assert_eq!(report.encoding_families.none_share, 0.0);
        for family in &report.encoding_families.families {
            assert!(
                !family.share.is_nan(),
                "family {} share must not be NaN",
                family.family
            );
            assert_eq!(family.share, 0.0);
        }

        // Heuristic-abstention addition must not panic or produce NaN either.
        assert!(!report.change_heuristic_abstention_rate.is_nan());
        assert_eq!(report.change_heuristic_abstention_rate, 0.0);
    }

    #[test]
    fn axis_summary_identifies_largest_and_smallest_share_by_known_distribution() {
        // 90/10 split, wallet-agnostic: build_report never sees a "wallet" at all here,
        // only the nsequence axis's raw value counts, exactly as production data would
        // supply them via EpochRow::axis_counts.
        let rows = vec![row(1000, &[("Rbf", 90), ("CakeGroupC", 10)], ("cake", 10))];
        let report = build_report(&rows, &[]);

        let summary = &report.axis_summaries["nsequence"];
        assert_eq!(summary.largest_value, "Rbf");
        assert!(
            (summary.largest_share - 0.9).abs() < 1e-9,
            "largest share should be 0.9, got {}",
            summary.largest_share
        );
        assert_eq!(summary.smallest_value, "CakeGroupC");
        assert!(
            (summary.smallest_share - 0.1).abs() < 1e-9,
            "smallest share should be 0.1, got {}",
            summary.smallest_share
        );
        assert_eq!(summary.distinct_values, 2);

        // The value table itself must carry every observed value (not only the
        // extremes), sorted by share descending, since a developer needs to see every
        // option on the axis to make a choice among them.
        assert_eq!(summary.values.len(), 2);
        assert_eq!(summary.values[0].value, "Rbf");
        assert_eq!(summary.values[0].count, 90);
        assert_eq!(summary.values[1].value, "CakeGroupC");
        assert_eq!(summary.values[1].count, 10);
    }

    #[test]
    fn axis_summary_of_a_single_valued_axis_has_that_value_as_both_largest_and_smallest() {
        let rows = vec![row(1000, &[("Rbf", 100)], ("core", 100))];
        let report = build_report(&rows, &[]);

        let summary = &report.axis_summaries["nsequence"];
        assert_eq!(summary.largest_value, "Rbf");
        assert_eq!(summary.smallest_value, "Rbf");
        assert_eq!(summary.largest_share, 1.0);
        assert_eq!(summary.smallest_share, 1.0);
        assert_eq!(summary.distinct_values, 1);
        assert_eq!(summary.values.len(), 1);
    }

    #[test]
    fn axis_summary_shares_sum_to_approximately_one() {
        let rows = vec![
            row(1000, &[("Rbf", 37), ("CakeGroupC", 41)], ("cake", 41)),
            row(1144, &[("Rbf", 5), ("CakeGroupC", 17)], ("cake", 17)),
        ];
        let report = build_report(&rows, &[]);

        let summary = &report.axis_summaries["nsequence"];
        let total_share: f64 = summary.values.iter().map(|v| v.share).sum();
        assert!(
            (total_share - 1.0).abs() < 1e-9,
            "shares across all values on an axis must sum to ~1.0, got {total_share}"
        );
    }

    // ------------------------------------------------------------------------------
    // Per-value min/max share across epochs (the stability measure this report adds).
    // ------------------------------------------------------------------------------

    #[test]
    fn per_value_min_max_reflects_the_swing_between_epochs_without_changing_the_aggregate() {
        // Epoch 1: Rbf at 90% (90/100). Epoch 2: Rbf at 50% (50/100). The aggregate
        // over both epochs is (90+50)/(100+100) = 70%, a single number that would
        // otherwise hide the fact that the real per-epoch share ranged from 50% to 90%.
        let rows = vec![
            row(1000, &[("Rbf", 90), ("CakeGroupC", 10)], ("cake", 10)),
            row(1144, &[("Rbf", 50), ("CakeGroupC", 50)], ("cake", 50)),
        ];
        let report = build_report(&rows, &[]);

        let summary = &report.axis_summaries["nsequence"];
        let rbf = summary.values.iter().find(|v| v.value == "Rbf").unwrap();

        assert!(
            (rbf.share - 0.7).abs() < 1e-9,
            "aggregate share must be unchanged by adding min/max: expected 0.7, got {}",
            rbf.share
        );
        assert!(
            (rbf.min_epoch_share - 0.5).abs() < 1e-9,
            "minimum per-epoch share should be 0.5 (epoch 2), got {}",
            rbf.min_epoch_share
        );
        assert!(
            (rbf.max_epoch_share - 0.9).abs() < 1e-9,
            "maximum per-epoch share should be 0.9 (epoch 1), got {}",
            rbf.max_epoch_share
        );

        // The stability indicator on the axis (dominant value = Rbf, since 70% > 30%
        // aggregate for CakeGroupC) must be exactly this same amplitude.
        assert!(
            (summary.dominant_amplitude - 0.4).abs() < 1e-9,
            "dominant_amplitude should be max - min = 0.9 - 0.5 = 0.4, got {}",
            summary.dominant_amplitude
        );
    }

    #[test]
    fn a_zero_tx_epoch_does_not_panic_and_does_not_drag_the_minimum_down() {
        // Two real epochs where Rbf never drops below 50%, plus a third epoch with no
        // classified transactions at all. If the zero-tx epoch were folded in as a
        // (wrongly computed, or worse, divide-by-zero) 0% observation, Rbf's minimum
        // would incorrectly collapse to 0.0 even though Rbf was never actually observed
        // at 0% in any epoch that classified a transaction.
        let rows = vec![
            row(1000, &[("Rbf", 90), ("CakeGroupC", 10)], ("cake", 10)),
            row(1144, &[("Rbf", 50), ("CakeGroupC", 50)], ("cake", 50)),
            row(1288, &[], ("cake", 0)),
        ];
        let report = build_report(&rows, &[]);

        let summary = &report.axis_summaries["nsequence"];
        let rbf = summary.values.iter().find(|v| v.value == "Rbf").unwrap();

        assert!(
            !rbf.min_epoch_share.is_nan() && !rbf.max_epoch_share.is_nan(),
            "a zero-tx epoch must not introduce a NaN via division by zero"
        );
        assert!(
            (rbf.min_epoch_share - 0.5).abs() < 1e-9,
            "the zero-tx epoch must be excluded entirely, not counted as a 0% \
             observation: minimum should stay at 0.5 (from epoch 2), got {}",
            rbf.min_epoch_share
        );
    }

    #[test]
    fn a_value_absent_from_one_epoch_but_present_in_another_has_minimum_zero() {
        // DECISION (see `fold_value_share_bounds_row`'s doc comment): within an epoch
        // that DID classify transactions, a value's absence from that epoch's counts is
        // treated as a genuine 0% observation for that epoch, not as a missing
        // measurement to be skipped. This is deliberate: the whole point of this
        // feature is to surface a value that disappears entirely on some days as
        // instability, not to hide that disappearance by only ever averaging over
        // epochs where the value happened to occur.
        //
        // Epoch 1: CakeGroupC present at 10/100 = 10%. Epoch 2: CakeGroupC entirely
        // absent (only Rbf occurs), but the epoch still classified 100 transactions.
        let rows = vec![
            row(1000, &[("Rbf", 90), ("CakeGroupC", 10)], ("cake", 10)),
            row(1144, &[("Rbf", 100)], ("cake", 0)),
        ];
        let report = build_report(&rows, &[]);

        let summary = &report.axis_summaries["nsequence"];
        let cake = summary
            .values
            .iter()
            .find(|v| v.value == "CakeGroupC")
            .unwrap();

        assert_eq!(
            cake.min_epoch_share, 0.0,
            "absence from a measured (txs > 0) epoch is a real 0% observation, so the \
             minimum must be exactly 0.0, got {}",
            cake.min_epoch_share
        );
        assert!(
            (cake.max_epoch_share - 0.1).abs() < 1e-9,
            "the one epoch CakeGroupC did occur in was 10%, so the maximum should be \
             0.1, got {}",
            cake.max_epoch_share
        );
    }

    // ------------------------------------------------------------------------------
    // Gap 1: auxiliary flags aggregated across epochs into `Report::aux_flags`.
    // ------------------------------------------------------------------------------

    #[test]
    fn aux_flags_aggregate_across_epochs_into_shares() {
        let mut row1 = row(1000, &[("Rbf", 60), ("CakeGroupC", 40)], ("cake", 0));
        row1.aux_counts = [
            ("address_reuse".to_string(), 70u64),
            ("changeless".to_string(), 10),
        ]
        .into_iter()
        .collect();
        let mut row2 = row(1144, &[("Rbf", 100)], ("cake", 0));
        row2.aux_counts = [("address_reuse".to_string(), 2u64), ("uih1".to_string(), 5)]
            .into_iter()
            .collect();

        let report = build_report(&[row1, row2], &[]);

        // 100 txs per row (see `row`'s txs = sum of vector counts), 200 total.
        assert_eq!(report.totals.txs, 200);

        assert_eq!(
            report.aux_flags["address_reuse"].count, 72,
            "70 + 2 across the two epochs"
        );
        assert!(
            (report.aux_flags["address_reuse"].share - 0.36).abs() < 1e-9,
            "72 / 200 = 0.36, got {}",
            report.aux_flags["address_reuse"].share
        );

        assert_eq!(report.aux_flags["changeless"].count, 10);
        assert!((report.aux_flags["changeless"].share - 0.05).abs() < 1e-9);

        assert_eq!(report.aux_flags["uih1"].count, 5);
        assert!((report.aux_flags["uih1"].share - 0.025).abs() < 1e-9);

        // A flag present in only one epoch must not silently vanish or double count.
        assert!(!report.aux_flags.contains_key("uih2"));
    }

    #[test]
    fn aux_flags_of_an_empty_scan_is_empty_not_a_panic() {
        let report = build_report(&[], &[]);
        assert!(report.aux_flags.is_empty());
    }

    #[test]
    fn aux_flags_zero_tx_epoch_yields_zero_share_not_nan() {
        let mut zero_row = row(1000, &[], ("cake", 0));
        zero_row.aux_counts = [("address_reuse".to_string(), 0u64)].into_iter().collect();
        let report = build_report(&[zero_row], &[]);

        // Nothing was ever set true, and there are no transactions at all, so the flag
        // is present at count 0, and its share must be an explicit 0.0, not NaN from a
        // 0/0 division.
        let flag = &report.aux_flags["address_reuse"];
        assert_eq!(flag.count, 0);
        assert_eq!(flag.share, 0.0);
        assert!(!flag.share.is_nan());
    }

    // ------------------------------------------------------------------------------
    // Gap 2: address-encoding families derived from the `output_types` marginal.
    // ------------------------------------------------------------------------------

    #[test]
    fn encoding_families_from_a_hand_built_output_types_distribution() {
        let mut marginal: BTreeMap<String, u64> = BTreeMap::new();
        marginal.insert("[P2pkh]".to_string(), 10);
        marginal.insert("[P2wpkh]".to_string(), 20);
        marginal.insert("[P2tr]".to_string(), 5);
        // Overlap case: one set with both a base58 and a bech32m member must count
        // toward BOTH families, proving the families are not mutually exclusive.
        marginal.insert("[P2pkh, P2tr]".to_string(), 3);
        // OpReturn-only case: must contribute to no family at all.
        marginal.insert("[OpReturn]".to_string(), 7);

        let families = encoding_families(&marginal);

        assert_eq!(families.total, 45);

        let base58 = families
            .families
            .iter()
            .find(|f| f.family == "base58")
            .unwrap();
        assert_eq!(
            base58.count, 13,
            "10 pure P2pkh + 3 from the overlapping [P2pkh, P2tr] set"
        );
        assert!((base58.share - 13.0 / 45.0).abs() < 1e-9);

        let bech32 = families
            .families
            .iter()
            .find(|f| f.family == "bech32")
            .unwrap();
        assert_eq!(bech32.count, 20);

        let bech32m = families
            .families
            .iter()
            .find(|f| f.family == "bech32m")
            .unwrap();
        assert_eq!(
            bech32m.count, 8,
            "5 pure P2tr + 3 from the overlapping [P2pkh, P2tr] set"
        );
        assert!((bech32m.share - 8.0 / 45.0).abs() < 1e-9);

        assert_eq!(
            families.none_count, 7,
            "the OpReturn-only set belongs to no family"
        );
        assert!((families.none_share - 7.0 / 45.0).abs() < 1e-9);

        // Families overlap, so their shares are allowed to sum past 1.0 (13+20+8=41 of
        // 45 counted transaction-slots, i.e. 0.911, still under 1.0 here, but the point
        // is they are NOT required to sum to 1.0 the way axis_summary values are).
        let family_share_sum: f64 = families.families.iter().map(|f| f.share).sum();
        assert!(family_share_sum > 0.0 && !family_share_sum.is_nan());
    }

    #[test]
    fn encoding_families_of_an_empty_marginal_has_no_nan() {
        let families = encoding_families(&BTreeMap::new());
        assert_eq!(families.total, 0);
        assert_eq!(families.none_count, 0);
        assert_eq!(families.none_share, 0.0);
        for family in &families.families {
            assert_eq!(family.count, 0);
            assert_eq!(family.share, 0.0);
            assert!(!family.share.is_nan());
        }
    }

    #[test]
    fn build_report_derives_encoding_families_from_the_same_output_types_marginal_the_axis_table_uses()
     {
        let mut output_types: BTreeMap<String, u64> = BTreeMap::new();
        output_types.insert("[P2wpkh]".to_string(), 30);
        // A set that overlaps bech32m and carries an OpReturn member too — must still
        // count toward bech32m (P2tr present) and NOT toward `none`, since it is not
        // OpReturn-only.
        output_types.insert("[P2tr, OpReturn]".to_string(), 20);

        let mut axis_counts = BTreeMap::new();
        axis_counts.insert("output_types".to_string(), output_types);

        let epoch_row = EpochRow {
            start_height: 1000,
            end_height: 1143,
            txs: 50,
            defects: 0,
            axis_counts,
            vectors: BTreeMap::new(),
            vectors_extended: BTreeMap::new(),
            aux_counts: BTreeMap::new(),
            field_aggs: BTreeMap::new(),
            template_matches: BTreeMap::new(),
        };

        let report = build_report(std::slice::from_ref(&epoch_row), &[]);

        // Must come from the exact same axis_marginals["output_types"] map that feeds
        // the `output_types` row of axis_summaries — same total, by construction.
        assert_eq!(
            report.encoding_families.total,
            report.axis_marginals["output_types"].values().sum::<u64>()
        );

        let bech32 = report
            .encoding_families
            .families
            .iter()
            .find(|f| f.family == "bech32")
            .unwrap();
        assert_eq!(bech32.count, 30);
        let bech32m = report
            .encoding_families
            .families
            .iter()
            .find(|f| f.family == "bech32m")
            .unwrap();
        assert_eq!(bech32m.count, 20);
        assert_eq!(
            report.encoding_families.none_count, 0,
            "the [P2tr, OpReturn] set has a bech32m member, so it must not land in 'none'"
        );
    }

    // ------------------------------------------------------------------------------
    // Compatibility shim: aggregating stored keys written under a retired axis set.
    // ------------------------------------------------------------------------------

    #[test]
    fn is_current_axis_excludes_deleted_circular_axes_but_keeps_heuristic_axes() {
        assert!(!is_current_axis("change_type_matches_inputs"));
        assert!(!is_current_axis("change_type_matches_outputs"));
        assert!(is_current_axis("change_position"));
        assert!(is_current_axis("change_type"));
        assert!(is_current_axis("nsequence"));
    }

    #[test]
    fn build_report_drops_a_retired_axis_marginal_found_in_stale_data() {
        // A row whose axis_counts still carries a retired axis (as `epochs-v2.jsonl`'s
        // would) must not resurrect that axis anywhere in the regenerated report.
        let mut axis_counts = BTreeMap::new();
        axis_counts.insert(
            "change_type_matches_inputs".to_string(),
            [("Yes".to_string(), 10u64)].into_iter().collect(),
        );
        axis_counts.insert(
            "nsequence".to_string(),
            [("Rbf".to_string(), 10u64)].into_iter().collect(),
        );

        let epoch_row = EpochRow {
            start_height: 1000,
            end_height: 1143,
            txs: 10,
            defects: 0,
            axis_counts,
            vectors: BTreeMap::new(),
            vectors_extended: BTreeMap::new(),
            aux_counts: BTreeMap::new(),
            field_aggs: BTreeMap::new(),
            template_matches: BTreeMap::new(),
        };

        let report = build_report(std::slice::from_ref(&epoch_row), &[]);

        assert!(
            !report
                .axis_marginals
                .contains_key("change_type_matches_inputs"),
            "a retired axis's marginal must not survive into a regenerated report"
        );
        assert!(
            !report
                .axis_summaries
                .contains_key("change_type_matches_inputs"),
            "a retired axis must not get an axis_summaries entry either"
        );
        assert!(
            report.axis_marginals.contains_key("nsequence"),
            "current axes are unaffected"
        );
    }

    #[test]
    fn epoch_rows_reads_zstd_identically_to_plain() {
        use std::io::Write;
        let rows = vec![
            row(1000, &[("Rbf", 90), ("Final", 10)], ("core", 100)),
            row(1144, &[("Rbf", 80), ("Final", 20)], ("core", 100)),
        ];
        let jsonl: String = rows
            .iter()
            .map(|r| serde_json::to_string(r).unwrap() + "\n")
            .collect();

        let mut plain = std::env::temp_dir();
        plain.push(format!("wfs-epochs-{}.jsonl", std::process::id()));
        std::fs::write(&plain, &jsonl).unwrap();

        let mut zst = std::env::temp_dir();
        zst.push(format!("wfs-epochs-{}.jsonl.zst", std::process::id()));
        {
            let f = std::fs::File::create(&zst).unwrap();
            let mut enc = zstd::stream::write::Encoder::new(f, 0).unwrap();
            enc.write_all(jsonl.as_bytes()).unwrap();
            enc.finish().unwrap();
        }

        let from_plain = read_epoch_rows(&plain).unwrap();
        let from_zst = read_epoch_rows(&zst).unwrap();
        assert_eq!(from_plain, rows, "plain reader round-trips the rows");
        assert_eq!(
            from_zst, from_plain,
            "the .zst reader yields identical rows"
        );

        let _ = std::fs::remove_file(&plain);
        let _ = std::fs::remove_file(&zst);
    }

    #[test]
    fn conditional_anonymity_buckets_and_shares() {
        use std::collections::BTreeMap;
        // Two CORE-shaped keys sharing sighash=All but differing on nsequence.
        // vecA count 5 (anonymity set < 10), vecB count 500 (set 100-999).
        let mut vectors: BTreeMap<String, u64> = BTreeMap::new();
        vectors.insert("nsequence=Rbf|sighash=All".to_string(), 5);
        vectors.insert("nsequence=Final|sighash=All".to_string(), 500);
        let cond = super::conditional_anonymity(&vectors);
        // sighash=All: 505 txs; 5 in a set <10, 5 in <100 (500 is in 100-999, not <100).
        let all = &cond["sighash"]["All"];
        assert!((all.share_lt10 - 5.0 / 505.0).abs() < 1e-9);
        assert!((all.share_lt100 - 5.0 / 505.0).abs() < 1e-9);
        assert_eq!(all.buckets["2-9"], 5);
        assert_eq!(all.buckets["100-999"], 500);
        // nsequence=Rbf is only vecA: all 5 txs in a set <10.
        let rbf = &cond["nsequence"]["Rbf"];
        assert!((rbf.share_lt10 - 1.0).abs() < 1e-9);
        assert!((rbf.share_lt100 - 1.0).abs() < 1e-9);
    }

    // ------------------------------------------------------------------------------
    // Task 4: `Report::fields` — the per-field rollup of `EpochRow::field_aggs`.
    // ------------------------------------------------------------------------------

    /// Builds a minimal, otherwise-empty `EpochRow` carrying only the given
    /// `field_aggs` — the rest of the row's maps are empty, since these tests exercise
    /// only the `fields` rollup, not axis/template aggregation (already covered above).
    fn field_row(start: u32, txs: u64, field_aggs: BTreeMap<String, FieldAgg>) -> EpochRow {
        EpochRow {
            start_height: start,
            end_height: start + 143,
            txs,
            defects: 0,
            axis_counts: BTreeMap::new(),
            vectors: BTreeMap::new(),
            vectors_extended: BTreeMap::new(),
            aux_counts: BTreeMap::new(),
            field_aggs,
            template_matches: BTreeMap::new(),
        }
    }

    #[test]
    fn fields_is_empty_when_no_epoch_carries_field_aggs() {
        // The compatibility case: a report built from an old `epochs.jsonl` written
        // before `field_aggs` existed (every row's `field_aggs` is `{}`, exactly what
        // `row()` above already produces) must still succeed, with `fields: []` rather
        // than panicking or fabricating entries.
        let rows = vec![row(1000, &[("Rbf", 90), ("CakeGroupC", 10)], ("cake", 10))];
        let report = build_report(&rows, &[]);
        assert!(report.fields.is_empty());
    }

    #[test]
    fn fields_rollup_merges_categorical_counts_and_numeric_hist_across_epochs() {
        // A Bool-kind column ("op_return"): two epochs' raw "0"/"1" tallies must add.
        let mut fa1: BTreeMap<String, FieldAgg> = BTreeMap::new();
        fa1.insert(
            "op_return".to_string(),
            FieldAgg::Categorical(
                [("0".to_string(), 7u64), ("1".to_string(), 3u64)]
                    .into_iter()
                    .collect(),
            ),
        );
        // A Numeric-kind column ("input_count", edges = EDGES_IO_COUNT, 10 edges -> 11
        // buckets): count/sum/hist must add elementwise; min/max must reduce.
        fa1.insert(
            "input_count".to_string(),
            FieldAgg::Numeric {
                count: 10,
                sum: 20.0,
                min: 1.0,
                max: 5.0,
                hist: vec![2, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0],
            },
        );

        let mut fa2: BTreeMap<String, FieldAgg> = BTreeMap::new();
        fa2.insert(
            "op_return".to_string(),
            FieldAgg::Categorical(
                [("0".to_string(), 4u64), ("1".to_string(), 6u64)]
                    .into_iter()
                    .collect(),
            ),
        );
        fa2.insert(
            "input_count".to_string(),
            FieldAgg::Numeric {
                count: 5,
                sum: 15.0,
                min: 0.5,
                max: 10.0,
                hist: vec![1, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
            },
        );

        let rows = vec![field_row(1000, 10, fa1), field_row(1144, 5, fa2)];
        let report = build_report(&rows, &[]);

        assert_eq!(
            report.fields.len(),
            2,
            "exactly the two columns present in the rows"
        );

        let op_return = report
            .fields
            .iter()
            .find(|f| f.name == "op_return")
            .unwrap();
        assert_eq!(op_return.kind, "bool");
        assert_eq!(op_return.section, "01");
        assert_eq!(
            op_return.total, 20,
            "7+3 from epoch 1 plus 4+6 from epoch 2"
        );
        let v0 = op_return.values.iter().find(|v| v.value == "0").unwrap();
        let v1 = op_return.values.iter().find(|v| v.value == "1").unwrap();
        assert_eq!(v0.count, 11, "7 + 4 across the two epochs");
        assert_eq!(v1.count, 9, "3 + 6 across the two epochs");
        assert!((v0.share - 11.0 / 20.0).abs() < 1e-9);
        // Sorted by count descending: "0" (11) before "1" (9).
        assert_eq!(op_return.values[0].value, "0");
        assert!(op_return.stats.is_none());
        assert!(op_return.hist.is_none());

        let input_count = report
            .fields
            .iter()
            .find(|f| f.name == "input_count")
            .unwrap();
        assert_eq!(input_count.kind, "numeric");
        assert_eq!(input_count.section, "06");
        assert_eq!(input_count.total, 15, "10 + 5 across the two epochs");
        assert!(input_count.values.is_empty());
        let stats = input_count.stats.as_ref().unwrap();
        assert_eq!(stats.count, 15);
        assert_eq!(
            stats.min, 0.5,
            "min must reduce to the smaller of the two epochs' mins"
        );
        assert_eq!(
            stats.max, 10.0,
            "max must reduce to the larger of the two epochs' maxes"
        );
        assert!((stats.mean - 35.0 / 15.0).abs() < 1e-9, "(20+15)/(10+5)");
        let hist = input_count.hist.as_ref().unwrap();
        assert_eq!(hist.len(), 11, "edges.len() + 1 buckets");
        assert_eq!(
            hist.iter().map(|b| b.count).sum::<u64>(),
            15,
            "hist buckets must add elementwise"
        );
        // bucket 0 (underflow) = 2+1=3, bucket 1 = 3+0=3, bucket 2 = 5+0=5, bucket 3 = 0+4=4.
        assert_eq!(hist[0].count, 3);
        assert_eq!(hist[1].count, 3);
        assert_eq!(hist[2].count, 5);
        assert_eq!(hist[3].count, 4);
    }

    #[test]
    fn numeric_stats_from_hist_median_and_p90_land_within_one_bucket_width_of_the_true_value() {
        // A known, evenly-spread distribution: 100 items all in the bucket [10, 20),
        // (bucket width 10) built so the true median and 90th-percentile are both
        // knowable by construction: values 10.0, 10.1, 10.2, ..., 19.9 (uniform steps of
        // 0.1), so the true median is ~14.95 and the true p90 is ~18.91 (linear-spaced).
        let edges = [0.0, 10.0, 20.0, 30.0];
        let hist = [0u64, 0, 100, 0, 0];
        let count = 100u64;
        let sum: f64 = (0..100).map(|i| 10.0 + i as f64 * 0.1).sum();
        let min = 10.0;
        let max = 19.9;

        let stats = numeric_stats_from_hist(&edges, count, sum, min, max, &hist);

        assert_eq!(stats.min, 10.0);
        assert_eq!(stats.max, 19.9);
        assert_eq!(stats.count, 100);
        assert!((stats.mean - sum / 100.0).abs() < 1e-9);

        let bucket_width = 10.0;
        let true_median = 14.95;
        let true_p90 = 18.91;
        assert!(
            (stats.median - true_median).abs() <= bucket_width,
            "median {} must be within one bucket width of the true value {}",
            stats.median,
            true_median
        );
        assert!(
            (stats.p90 - true_p90).abs() <= bucket_width,
            "p90 {} must be within one bucket width of the true value {}",
            stats.p90,
            true_p90
        );
    }

    #[test]
    fn numeric_stats_from_hist_clamps_open_ended_buckets_to_exact_min_max() {
        // A rank landing in the underflow bucket (index 0) or the overflow bucket
        // (index edges.len()) has no finite bucket width to interpolate across, so it
        // must clamp to the field's exact min/max instead.
        let edges = [0.0, 10.0];
        // Bucket 0 (underflow, < 0.0) holds every item: any rank lands there.
        let hist = [10u64, 0, 0];
        let stats = numeric_stats_from_hist(&edges, 10, -50.0, -10.0, -1.0, &hist);
        assert_eq!(stats.median, -10.0, "median must clamp to the exact min");
        assert_eq!(stats.p90, -10.0, "p90 must clamp to the exact min");

        // Bucket 2 (overflow, >= 10.0) holds every item: any rank lands there.
        let hist_over = [0u64, 0, 10];
        let stats_over = numeric_stats_from_hist(&edges, 10, 500.0, 10.0, 90.0, &hist_over);
        assert_eq!(
            stats_over.median, 90.0,
            "median must clamp to the exact max"
        );
        assert_eq!(stats_over.p90, 90.0, "p90 must clamp to the exact max");
    }

    #[test]
    fn every_feature_column_has_a_section_and_a_recognised_kind() {
        // Guards `section_for`'s exhaustiveness and `field_report`'s kind labels against
        // COLUMN_NAMES/COLUMN_KINDS drifting out from under this file: every column in
        // the ~173-entry catalog must resolve to one of the documented section ids and
        // kind labels, with no panic.
        use crate::features::new_field_agg;
        for (name, kind) in COLUMN_NAMES[2..].iter().zip(COLUMN_KINDS.iter()) {
            let section = section_for(name);
            assert!(
                ("00"..="11").contains(&section),
                "column {name} resolved to an out-of-range section {section:?}"
            );
            let agg = new_field_agg(kind);
            let report = field_report(name, kind, &agg);
            assert!(
                matches!(report.kind, "bool" | "onehot" | "ordinal" | "numeric"),
                "column {name} produced an unrecognised kind label {:?}",
                report.kind
            );
        }
    }
}
