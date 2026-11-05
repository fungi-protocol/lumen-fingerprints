use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde::Serialize;

use crate::epoch::EpochRow;
use crate::features::{
    COLUMN_KINDS, COLUMN_NAMES, COLUMN_SECTIONS, ColumnKind, FieldAgg, merge_field,
};
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
    /// The template's anonymity set as a share of the epoch's transactions.
    pub share: f64,
}

/// 0.0 when the epoch never saw the value.
#[derive(Debug, Serialize)]
pub struct AxisSharePoint {
    pub start_height: u32,
    pub share: f64,
}

/// Epoch bounds skip `txs == 0` epochs; absence from any other epoch counts as 0.0.
#[derive(Debug, Serialize)]
pub struct AxisValueShare {
    pub value: String,
    pub count: u64,
    pub share: f64,
    pub min_epoch_share: f64,
    pub max_epoch_share: f64,
}

#[derive(Debug, Serialize)]
pub struct AxisSummary {
    /// Sorted by share descending, ties by value name.
    pub values: Vec<AxisValueShare>,
    pub distinct_values: u64,
    pub largest_value: String,
    pub largest_share: f64,
    pub smallest_value: String,
    pub smallest_share: f64,
    /// Per-epoch spread of `largest_value`: how far `largest_share` is from a stable point.
    pub dominant_amplitude: f64,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub window: Window,
    pub totals: Totals,
    pub axis_marginals: BTreeMap<String, BTreeMap<String, u64>>,
    pub axis_summaries: BTreeMap<String, AxisSummary>,
    /// CORE axis -> value -> anonymity-set distribution of the txs using it.
    pub conditional_anonymity: BTreeMap<String, BTreeMap<String, ConditionalAnon>>,
    /// `Indeterminate` share of `change_position`.
    pub change_heuristic_abstention_rate: f64,
    pub template_series: BTreeMap<String, Vec<TemplatePoint>>,
    /// Only pairs some template constrains.
    pub axis_value_series: BTreeMap<String, BTreeMap<String, Vec<AxisSharePoint>>>,
    /// Current axes only.
    pub template_axes: BTreeMap<String, BTreeMap<String, Vec<String>>>,
    /// Flags co-occur, so shares do not sum to 1.0.
    pub aux_flags: BTreeMap<String, AuxShare>,
    /// Families overlap, so shares can sum past 1.0.
    pub encoding_families: EncodingFamilies,
    /// Empty when no epoch carries `field_aggs` (rows written before that field).
    pub fields: Vec<FieldReport>,
}

/// `value` is the raw, unlabelled accumulator key (e.g. `"0"`/`"1"` or an `Ordinal` index).
#[derive(Debug, Serialize)]
pub struct FieldValueShare {
    pub value: String,
    pub count: u64,
    pub share: f64,
}

/// `median`/`p90` are interpolated from the histogram; `min`/`max`/`mean` are exact.
#[derive(Debug, Serialize, Clone, PartialEq)]
pub struct Stats {
    pub min: f64,
    pub median: f64,
    pub mean: f64,
    pub p90: f64,
    pub max: f64,
    pub count: u64,
}

/// Bucket 0 is `(-inf, edges[0])`, the last is `[edges[last], +inf)`; the infinite
/// ends serialize as JSON `null`.
#[derive(Debug, Serialize)]
pub struct HistBucket {
    pub lo: f64,
    pub hi: f64,
    pub count: u64,
    pub share: f64,
}

/// Categorical columns fill `values`; `Numeric` ones fill `stats` and `hist`.
#[derive(Debug, Serialize)]
pub struct FieldReport {
    pub name: String,
    pub section: &'static str,
    pub kind: &'static str,
    pub total: u64,
    pub values: Vec<FieldValueShare>,
    pub stats: Option<Stats>,
    pub hist: Option<Vec<HistBucket>>,
}

#[derive(Debug, Serialize)]
pub struct AuxShare {
    pub count: u64,
    pub share: f64,
}

/// Classified txs with at least one output in this family.
#[derive(Debug, Serialize)]
pub struct EncodingFamilyShare {
    pub family: String,
    pub count: u64,
    pub share: f64,
}

/// Every transaction lands in `none` XOR in at least one family.
#[derive(Debug, Serialize)]
pub struct EncodingFamilies {
    /// base58, bech32, bech32m, in that order.
    pub families: Vec<EncodingFamilyShare>,
    pub none_count: u64,
    pub none_share: f64,
    pub total: u64,
}

/// Stored rows may carry retired axes; reports skip them, like `vector::strip_retired_axes`.
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

/// A value missing from `bounds` falls back to `(0.0, 0.0)`.
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
    // Ties broken by value name so the report is byte-reproducible.
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

/// A separate pass: a value first seen late must still record 0% for earlier epochs.
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

/// `txs == 0` epochs carry no information and are skipped; in any other epoch an absent
/// value is a real 0% observation.
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

/// No infinity may reach `report.json`, where it would serialize as `null`.
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

/// Exact tokens of `"[P2wpkh, OpReturn]"`, not substrings, so `P2pk` never matches `P2pkh`.
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

/// A tx with no address-encoded output (e.g. only `OpReturn`) counts toward `none`.
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

/// One `EpochRow` resident at a time.
struct Aggregates {
    totals: Totals,
    axis_marginals: BTreeMap<String, BTreeMap<String, u64>>,
    /// CORE joint keys; each count is the anonymity set its txs share.
    vectors: BTreeMap<String, u64>,
    aux_counts: BTreeMap<String, u64>,
    /// One point per template per row, zero when nothing matches: an absent series
    /// would be indistinguishable from a typo in the template.
    template_series: BTreeMap<String, Vec<TemplatePoint>>,
    /// Pre-seeded so a value that never occurs still yields an all-zeros series.
    axis_value_series: BTreeMap<String, BTreeMap<String, Vec<AxisSharePoint>>>,
    template_axes: BTreeMap<String, BTreeMap<String, Vec<String>>>,
    field_aggs: BTreeMap<String, FieldAgg>,
    start_height: Option<u32>,
    end_height: u32,
    epochs: u64,
}

impl Aggregates {
    fn new(templates: &[WalletTemplate]) -> Self {
        let template_series: BTreeMap<String, Vec<TemplatePoint>> = templates
            .iter()
            .map(|t| (t.era_label(), Vec::new()))
            .collect();
        debug_assert_eq!(
            template_series.len(),
            templates.len(),
            "duplicate era labels would silently merge series"
        );
        let mut axis_value_series: BTreeMap<String, BTreeMap<String, Vec<AxisSharePoint>>> =
            BTreeMap::new();
        let mut template_axes: BTreeMap<String, BTreeMap<String, Vec<String>>> = BTreeMap::new();
        for t in templates {
            let mut constrained = BTreeMap::new();
            for (axis, values) in &t.axes {
                if !is_current_axis(axis) {
                    continue;
                }
                for v in values {
                    axis_value_series
                        .entry(axis.clone())
                        .or_default()
                        .entry(v.clone())
                        .or_default();
                }
                constrained.insert(axis.clone(), values.clone());
            }
            template_axes.insert(t.era_label(), constrained);
        }
        Self {
            totals: Totals { txs: 0, defects: 0 },
            axis_marginals: BTreeMap::new(),
            vectors: BTreeMap::new(),
            aux_counts: BTreeMap::new(),
            template_series,
            axis_value_series,
            template_axes,
            field_aggs: BTreeMap::new(),
            start_height: None,
            end_height: 0,
            epochs: 0,
        }
    }
}

/// Matches templates against `vectors_extended`, not the scan-time `template_matches`,
/// so editing `wallets.toml` needs no rescan.
fn fold_row(acc: &mut Aggregates, row: &EpochRow, templates: &[WalletTemplate]) {
    acc.totals.txs += row.txs;
    acc.totals.defects += row.defects;
    acc.epochs += 1;
    if acc.start_height.is_none() {
        acc.start_height = Some(row.start_height);
    }
    acc.end_height = row.end_height;

    for (axis, values) in &row.axis_counts {
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

    // One point per epoch per constrained pair, so series stay aligned with
    // `template_series`.
    for (axis, per_value) in acc.axis_value_series.iter_mut() {
        let counts_this_epoch = row.axis_counts.get(axis);
        for (value, series) in per_value.iter_mut() {
            let count = counts_this_epoch
                .and_then(|c| c.get(value))
                .copied()
                .unwrap_or(0);
            series.push(AxisSharePoint {
                start_height: row.start_height,
                share: if row.txs == 0 {
                    0.0
                } else {
                    count as f64 / row.txs as f64
                },
            });
        }
    }

    for (key, count) in &row.vectors {
        *acc.vectors.entry(key.clone()).or_insert(0) += count;
    }

    for (name, agg) in &row.field_aggs {
        match acc.field_aggs.get_mut(name) {
            Some(existing) => merge_field(existing, agg),
            None => {
                acc.field_aggs.insert(name.clone(), agg.clone());
            }
        }
    }

    // Indexed positionally: two eras of one wallet share a `name`.
    let mut matches_this_row = vec![0u64; templates.len()];
    for (key, count) in &row.vectors_extended {
        let axes = parse_key(key);
        for (i, template) in templates.iter().enumerate() {
            if template.matches_axes(&axes) {
                matches_this_row[i] += count;
            }
        }
    }
    for (i, template) in templates.iter().enumerate() {
        let matches = matches_this_row[i];
        acc.template_series
            .get_mut(&template.era_label())
            .expect("template_series pre-populated with every era label")
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

/// Contiguous over every count >= 1.
const TAIL_MASS_BOUNDS: [(u64, u64, &str); 7] = [
    (1, 1, "exactly 1"),
    (2, 9, "2-9"),
    (10, 99, "10-99"),
    (100, 999, "100-999"),
    (1_000, 9_999, "1,000-9,999"),
    (10_000, 99_999, "10,000-99,999"),
    (100_000, u64::MAX, ">=100,000"),
];

/// CORE-joint anonymity-set sizes among txs using one CORE axis value.
#[derive(Debug, Serialize)]
pub struct ConditionalAnon {
    pub buckets: BTreeMap<String, u64>,
    pub share_lt10: f64,
    pub share_lt100: f64,
}

fn bucket_label_for(count: u64) -> &'static str {
    TAIL_MASS_BOUNDS
        .iter()
        .find(|&&(lo, hi, _)| count >= lo && count <= hi)
        .map(|&(_, _, label)| label)
        .unwrap_or("")
}

/// A joint vector's count is both its anonymity-set size and the number of txs in it,
/// so it is added into the bucket of every CORE value the vector carries.
fn conditional_anonymity(
    vectors: &BTreeMap<String, u64>,
) -> BTreeMap<String, BTreeMap<String, ConditionalAnon>> {
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

fn column_index(name: &str) -> Option<usize> {
    COLUMN_NAMES[2..].iter().position(|n| *n == name)
}

fn column_kind_for(name: &str) -> Option<&'static ColumnKind> {
    column_index(name).map(|i| &COLUMN_KINDS[i])
}

/// Panics rather than mis-file a column missing from the table.
fn section_for(name: &str) -> &'static str {
    match name {
        "txid" | "block_height" => "00",
        _ => column_index(name)
            .map(|i| COLUMN_SECTIONS[i])
            .unwrap_or_else(|| panic!("column {name:?} is not in COLUMN_NAMES")),
    }
}

/// The open-ended end buckets have no width, so ranks there clamp to the exact
/// `min`/`max`.
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
    // Unreachable while `hist` sums to `count`.
    max
}

/// `count == 0` yields all-zero `Stats`, not `NaN` or infinite seeds.
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
        axis_value_series,
        template_axes,
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

    // `change_position`, not `change_type`: it also counts sub-2-output txs as
    // `Indeterminate`.
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
        axis_value_series,
        template_axes,
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

/// Reads the file twice: the bounds pass needs the complete `axis_marginals` first.
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

pub fn epoch_rows(
    epochs_path: &Path,
) -> Result<impl Iterator<Item = Result<EpochRow, ReportError>>, ReportError> {
    let file = std::fs::File::open(epochs_path).map_err(ReportError::Io)?;
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

pub fn read_epoch_rows(epochs_path: &Path) -> Result<Vec<EpochRow>, ReportError> {
    epoch_rows(epochs_path)?.collect()
}

/// RFC 4180 quoting; era labels and `output_types` values can contain commas.
fn csv_field(s: &str) -> Cow<'_, str> {
    if s.contains([',', '"', '\n', '\r']) {
        Cow::Owned(format!("\"{}\"", s.replace('"', "\"\"")))
    } else {
        Cow::Borrowed(s)
    }
}

/// One row per (epoch, era, constrained axis value). `in_set_lt*` are blank for
/// non-CORE axes; `end_height` is the next epoch's start minus one, or the window end.
fn write_cross_csv(report: &Report, out: &Path) -> Result<(), ReportError> {
    let mut w = BufWriter::new(File::create(out).map_err(ReportError::Io)?);
    writeln!(
        w,
        "start_height,end_height,wallet_version,axis,value,share,in_set_lt10,in_set_lt100"
    )
    .map_err(ReportError::Io)?;
    for (era, axes) in &report.template_axes {
        for (axis, value) in axes
            .iter()
            .flat_map(|(axis, vs)| vs.iter().map(move |v| (axis, v)))
        {
            let Some(series) = report
                .axis_value_series
                .get(axis)
                .and_then(|m| m.get(value))
            else {
                continue;
            };
            let (id10, id100) = report
                .conditional_anonymity
                .get(axis)
                .and_then(|m| m.get(value))
                .map(|c| (c.share_lt10.to_string(), c.share_lt100.to_string()))
                .unwrap_or_default();
            for (i, pt) in series.iter().enumerate() {
                let end_height = series
                    .get(i + 1)
                    .map(|next| next.start_height.saturating_sub(1))
                    .unwrap_or(report.window.end_height);
                writeln!(
                    w,
                    "{},{},{},{},{},{},{},{}",
                    pt.start_height,
                    end_height,
                    csv_field(era),
                    csv_field(axis),
                    csv_field(value),
                    pt.share,
                    id10,
                    id100
                )
                .map_err(ReportError::Io)?;
            }
        }
    }
    w.flush().map_err(ReportError::Io)
}

pub fn write_report(
    epochs_path: &Path,
    out_dir: &Path,
    templates: &[WalletTemplate],
) -> Result<(), ReportError> {
    let report = build_report_from_path(epochs_path, templates)?;
    std::fs::create_dir_all(out_dir).map_err(ReportError::Io)?;

    let json = serde_json::to_string_pretty(&report).map_err(ReportError::Parse)?;
    std::fs::write(out_dir.join("report.json"), json).map_err(ReportError::Io)?;
    write_cross_csv(&report, &out_dir.join("wallet-axis-cross.csv"))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::template::Confidence;
    use std::collections::BTreeMap;

    /// Keys are real `nsequence=<value>` pairs so templates match them; `template_matches`
    /// is never read.
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

    fn nsequence_template(name: &str, expected: &str) -> WalletTemplate {
        WalletTemplate {
            name: name.to_string(),
            confidence: Confidence::CodePredicted,
            software: None,
            from_version: None,
            until_version: None,
            axes: [("nsequence".to_string(), vec![expected.to_string()])]
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
    fn axis_value_series_and_template_axes_drive_the_per_axis_wallet_view() {
        let rows = vec![
            row(1000, &[("Rbf", 90), ("CakeGroupC", 10)], ("cake", 10)),
            row(1144, &[("Rbf", 80), ("CakeGroupC", 20)], ("cake", 20)),
        ];
        let cake = nsequence_template("cake", "CakeGroupC");
        let ghost = nsequence_template("ghost", "NeverObservedOnChain");
        let report = build_report(&rows, &[cake, ghost]);

        assert_eq!(report.template_axes["cake"]["nsequence"], ["CakeGroupC"]);
        assert_eq!(
            report.template_axes["ghost"]["nsequence"],
            ["NeverObservedOnChain"]
        );

        let cake_series = &report.axis_value_series["nsequence"]["CakeGroupC"];
        assert_eq!(cake_series.len(), 2);
        assert_eq!(cake_series[0].start_height, 1000);
        assert!((cake_series[0].share - 0.1).abs() < 1e-9);
        assert!((cake_series[1].share - 0.2).abs() < 1e-9);

        let ghost_series = &report.axis_value_series["nsequence"]["NeverObservedOnChain"];
        assert_eq!(ghost_series.len(), 2);
        assert!(ghost_series.iter().all(|p| p.share == 0.0));

        // Observed but unconstrained.
        assert!(!report.axis_value_series["nsequence"].contains_key("Rbf"));
    }

    #[test]
    fn write_cross_csv_emits_one_row_per_epoch_wallet_axis() {
        let rows = vec![
            row(1000, &[("Rbf", 90), ("CakeGroupC", 10)], ("cake", 10)),
            row(1144, &[("Rbf", 80), ("CakeGroupC", 20)], ("cake", 20)),
        ];
        let cake = nsequence_template("cake", "CakeGroupC");
        let report = build_report(&rows, std::slice::from_ref(&cake));

        let out = std::env::temp_dir().join("lumen_cross_csv_test.csv");
        write_cross_csv(&report, &out).unwrap();
        let csv = std::fs::read_to_string(&out).unwrap();
        std::fs::remove_file(&out).ok();

        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(
            lines[0],
            "start_height,end_height,wallet_version,axis,value,share,in_set_lt10,in_set_lt100"
        );
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[1].split(',').count(), 8);
        assert!(lines[1].starts_with("1000,"));
        assert!(lines[1].contains(",cake,nsequence,CakeGroupC,"));
        assert!(lines[1].contains(",0.1,"));
        assert_eq!(lines[1].split(',').nth(1).unwrap(), "1143");
        assert_eq!(lines[2].split(',').nth(1).unwrap(), "1287");
    }

    #[test]
    fn report_time_matching_produces_expected_counts_for_a_known_composition() {
        // Four txs in three shapes, one duplicated, so a key carries count 2: with a
        // single key, counting keys instead of summing counts would go unnoticed.
        use crate::epoch::EpochAccumulator;
        use crate::template::Confidence;
        use crate::vector::tests_support::{
            EPOCH_TEST_HEIGHT, block_with_many, cake_like_tx, legacy_p2pkh_tx,
        };

        // Shape A, x2: Cake-shaped segwit (nsequence=CakeGroupC, nlocktime=Zero).
        let cake_a = cake_like_tx();
        let cake_b = cake_like_tx();

        // Shape B: legacy P2PKH, nsequence=Max, nlocktime=Zero.
        let mut pubkey = vec![0xab; 33];
        pubkey[0] = 0x02;
        let mut legacy_max = legacy_p2pkh_tx(pubkey.clone());
        legacy_max.input[0].previous_output.vout = 100;

        // Shape C: legacy, nsequence=Rbf, nlocktime=AntiFeeSnipe (height - 1 lies in
        // (height - 100, height]).
        let mut legacy_rbf = legacy_p2pkh_tx(pubkey);
        legacy_rbf.input[0].previous_output.vout = 200;
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

        // These partition the fixture by nsequence class.
        let by_nsequence = |name: &str, expected: &str| WalletTemplate {
            name: name.to_string(),
            confidence: Confidence::CodePredicted,
            software: None,
            from_version: None,
            until_version: None,
            axes: [("nsequence".to_string(), vec![expected.to_string()])]
                .into_iter()
                .collect(),
        };
        let templates = vec![
            by_nsequence("cake-class", "first 0x01 rest 0xffffffff"),
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
        let total: u64 = ["cake-class", "max-class", "rbf-class"]
            .iter()
            .map(|name| report.template_series[*name][0].matches)
            .sum();
        assert_eq!(
            total, row.txs,
            "every transaction must land in exactly one nsequence class"
        );

        // The fixture's inputs are in BIP-69 order (ascending vout, shared txid), so
        // only the pre-shuffle Cake Wallet era (`input_order = "Bip69"`) matches.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("templates/wallets.toml");
        let shipped = crate::template::load_templates(&path).unwrap();
        let cake_wallet = shipped
            .iter()
            .find(|t| t.name == "Cake Wallet" && t.until_version.is_some())
            .expect("cake template ships");
        let report_with_shipped = build_report(
            std::slice::from_ref(&row),
            std::slice::from_ref(cake_wallet),
        );
        assert_eq!(
            report_with_shipped.template_series[&cake_wallet.era_label()][0].matches,
            2,
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

        assert!(!report.change_heuristic_abstention_rate.is_nan());
        assert_eq!(report.change_heuristic_abstention_rate, 0.0);
    }

    #[test]
    fn axis_summary_identifies_largest_and_smallest_share_by_known_distribution() {
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

    #[test]
    fn per_value_min_max_reflects_the_swing_between_epochs_without_changing_the_aggregate() {
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

        assert!(
            (summary.dominant_amplitude - 0.4).abs() < 1e-9,
            "dominant_amplitude should be max - min = 0.9 - 0.5 = 0.4, got {}",
            summary.dominant_amplitude
        );
    }

    #[test]
    fn a_zero_tx_epoch_does_not_panic_and_does_not_drag_the_minimum_down() {
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

        let flag = &report.aux_flags["address_reuse"];
        assert_eq!(flag.count, 0);
        assert_eq!(flag.share, 0.0);
        assert!(!flag.share.is_nan());
    }

    #[test]
    fn encoding_families_from_a_hand_built_output_types_distribution() {
        let mut marginal: BTreeMap<String, u64> = BTreeMap::new();
        marginal.insert("[P2pkh]".to_string(), 10);
        marginal.insert("[P2wpkh]".to_string(), 20);
        marginal.insert("[P2tr]".to_string(), 5);
        marginal.insert("[P2pkh, P2tr]".to_string(), 3);
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
        let mut vectors: BTreeMap<String, u64> = BTreeMap::new();
        vectors.insert("nsequence=Rbf|sighash=All".to_string(), 5);
        vectors.insert("nsequence=Final|sighash=All".to_string(), 500);
        let cond = super::conditional_anonymity(&vectors);
        let all = &cond["sighash"]["All"];
        assert!((all.share_lt10 - 5.0 / 505.0).abs() < 1e-9);
        assert!((all.share_lt100 - 5.0 / 505.0).abs() < 1e-9);
        assert_eq!(all.buckets["2-9"], 5);
        assert_eq!(all.buckets["100-999"], 500);
        let rbf = &cond["nsequence"]["Rbf"];
        assert!((rbf.share_lt10 - 1.0).abs() < 1e-9);
        assert!((rbf.share_lt100 - 1.0).abs() < 1e-9);
    }

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
        let rows = vec![row(1000, &[("Rbf", 90), ("CakeGroupC", 10)], ("cake", 10))];
        let report = build_report(&rows, &[]);
        assert!(report.fields.is_empty());
    }

    #[test]
    fn fields_rollup_merges_categorical_counts_and_numeric_hist_across_epochs() {
        let mut fa1: BTreeMap<String, FieldAgg> = BTreeMap::new();
        fa1.insert(
            "op_return".to_string(),
            FieldAgg::Categorical(
                [("0".to_string(), 7u64), ("1".to_string(), 3u64)]
                    .into_iter()
                    .collect(),
            ),
        );
        // `input_count` has 10 edges, so 11 buckets.
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
        assert_eq!(hist[0].count, 3);
        assert_eq!(hist[1].count, 3);
        assert_eq!(hist[2].count, 5);
        assert_eq!(hist[3].count, 4);
    }

    #[test]
    fn numeric_stats_from_hist_median_and_p90_land_within_one_bucket_width_of_the_true_value() {
        // 100 items 10.0, 10.1, ..., 19.9, all in bucket [10, 20).
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
        let edges = [0.0, 10.0];
        let hist = [10u64, 0, 0];
        let stats = numeric_stats_from_hist(&edges, 10, -50.0, -10.0, -1.0, &hist);
        assert_eq!(stats.median, -10.0, "median must clamp to the exact min");
        assert_eq!(stats.p90, -10.0, "p90 must clamp to the exact min");

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

    #[test]
    fn two_eras_of_one_wallet_get_separate_series() {
        let rows = vec![
            row(1000, &[("Rbf", 90), ("CakeGroupC", 10)], ("cake", 10)),
            row(1144, &[("Rbf", 80), ("CakeGroupC", 20)], ("cake", 20)),
        ];
        let old_era = WalletTemplate {
            name: "Cake Wallet".to_string(),
            confidence: Confidence::CodePredicted,
            software: Some("cake_wallet".to_string()),
            from_version: None,
            until_version: Some("4.28.0".to_string()),
            axes: [("nsequence".to_string(), vec!["CakeGroupC".to_string()])]
                .into_iter()
                .collect(),
        };
        let new_era = WalletTemplate {
            name: "Cake Wallet".to_string(),
            confidence: Confidence::CodePredicted,
            software: Some("cake_wallet".to_string()),
            from_version: Some("4.28.0".to_string()),
            until_version: None,
            axes: [("nsequence".to_string(), vec!["Rbf".to_string()])]
                .into_iter()
                .collect(),
        };
        let report = build_report(&rows, &[old_era, new_era]);

        let old_series = &report.template_series["Cake Wallet (cake_wallet <4.28.0)"];
        let new_series = &report.template_series["Cake Wallet (cake_wallet ≥4.28.0)"];
        assert_eq!(old_series.len(), 2, "one point per epoch for the old era");
        assert_eq!(new_series.len(), 2, "one point per epoch for the new era");
        assert_eq!(old_series[0].matches, 10);
        assert_eq!(new_series[0].matches, 90);
        assert_eq!(old_series[1].matches, 20);
        assert_eq!(new_series[1].matches, 80);
    }
}
