use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::OnceLock;

use lumen_primitives::traits::SourcedBlock;
use serde::{Deserialize, Serialize};

use crate::aux::aux_flags;
use crate::features::{
    BlockContext, COLUMN_KINDS, COLUMN_NAMES, FieldAgg, feature_row, fold_field, new_field_agg,
    tx_shape,
};
use crate::vector::{CORE_AXES, EXTENDED_AXES, HEURISTIC_AXES, classify_tx, tx_fee_vsize_weight};

/// Order matters: `ingest` slices the core and core+extended prefixes, so heuristic
/// axes never enter a joint key.
fn all_axes() -> &'static [&'static str] {
    static ALL_AXES: OnceLock<Vec<&'static str>> = OnceLock::new();
    ALL_AXES.get_or_init(|| {
        CORE_AXES
            .iter()
            .chain(EXTENDED_AXES.iter())
            .chain(HEURISTIC_AXES.iter())
            .copied()
            .collect()
    })
}

/// The only thing a scan writes to disk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpochRow {
    pub start_height: u32,
    pub end_height: u32,
    pub txs: u64,
    pub defects: u64,
    pub axis_counts: BTreeMap<String, BTreeMap<String, u64>>,
    pub vectors: BTreeMap<String, u64>,
    pub vectors_extended: BTreeMap<String, u64>,
    pub aux_counts: BTreeMap<String, u64>,
    #[serde(default)]
    pub field_aggs: BTreeMap<String, FieldAgg>,
    /// Always empty and never read; kept so older `epochs.jsonl` rows still deserialize.
    pub template_matches: BTreeMap<String, u64>,
}

/// `&'static str` keys avoid a per-transaction allocation; `flush` converts them.
#[derive(Debug)]
pub struct EpochAccumulator {
    start_height: u32,
    epoch_size: u32,
    last_height: u32,
    txs: u64,
    defects: u64,
    axis_counts: BTreeMap<&'static str, BTreeMap<String, u64>>,
    vectors: BTreeMap<String, u64>,
    vectors_extended: BTreeMap<String, u64>,
    aux_counts: BTreeMap<&'static str, u64>,
    field_aggs: BTreeMap<&'static str, FieldAgg>,
    /// Not reset by `flush`: block adjacency spans epoch boundaries.
    prev_block_min_feerate: Option<f64>,
}

impl EpochAccumulator {
    pub fn new(start_height: u32, epoch_size: u32) -> Self {
        Self {
            start_height,
            epoch_size,
            last_height: start_height,
            txs: 0,
            defects: 0,
            axis_counts: BTreeMap::new(),
            vectors: BTreeMap::new(),
            vectors_extended: BTreeMap::new(),
            aux_counts: BTreeMap::new(),
            field_aggs: BTreeMap::new(),
            prev_block_min_feerate: None,
        }
    }

    fn bump(
        map: &mut BTreeMap<&'static str, BTreeMap<String, u64>>,
        axis: &'static str,
        value: &str,
    ) {
        let values = map.entry(axis).or_default();
        if let Some(count) = values.get_mut(value) {
            *count += 1;
        } else {
            values.insert(value.to_string(), 1);
        }
    }

    pub fn boundary_reached(&self, height: u32) -> bool {
        height >= self.start_height + self.epoch_size - 1
    }

    pub fn ingest(&mut self, block: &SourcedBlock) {
        self.ingest_with(block, &mut |_, _, _, _, _| {});
    }

    /// `on_tx` sees exactly the aggregated population: defects are excluded.
    pub fn ingest_with(
        &mut self,
        block: &SourcedBlock,
        on_tx: &mut impl FnMut(
            &bitcoin::Transaction,
            u32,
            &crate::vector::FingerprintVector,
            &crate::aux::AuxFlags,
            &crate::features::FeatureRow,
        ),
    ) {
        self.last_height = block.height;

        let block_min_feerate = block
            .block
            .txdata
            .iter()
            .filter_map(|t| {
                tx_fee_vsize_weight(t, block).map(|(fee, vsize, _)| fee as f64 / vsize as f64)
            })
            .fold(None, |acc: Option<f64>, r| {
                Some(acc.map_or(r, |a| a.min(r)))
            });
        let ctx = BlockContext {
            block_min_feerate,
            prev_block_min_feerate: self.prev_block_min_feerate,
        };

        for tx in &block.block.txdata {
            if tx.is_coinbase() {
                continue;
            }
            let Some(vector) = classify_tx(tx, block) else {
                self.defects += 1;
                continue;
            };

            self.txs += 1;

            // Per-transaction: entries borrow from this iteration's `vector`.
            let mut axis_values: Vec<(&'static str, Cow<'_, str>)> =
                Vec::with_capacity(all_axes().len());
            for axis in all_axes() {
                if let Some(value) = vector.axis_value(axis) {
                    axis_values.push((*axis, value));
                }
            }
            // The prefix slices below need every axis to yield a value.
            debug_assert_eq!(
                axis_values.len(),
                all_axes().len(),
                "axis_value must return Some for every name in all_axes()"
            );

            for (axis, value) in &axis_values {
                Self::bump(&mut self.axis_counts, axis, value);
            }

            // Heuristic axes stay out of joint keys: their abstention rate reflects the
            // method, not chain diversity.
            *self
                .vectors
                .entry(join_key(&axis_values[..CORE_AXES.len()]))
                .or_insert(0) += 1;
            *self
                .vectors_extended
                .entry(join_key(
                    &axis_values[..CORE_AXES.len() + EXTENDED_AXES.len()],
                ))
                .or_insert(0) += 1;

            let flags = aux_flags(tx, block);

            // `row.values[i]` and `COLUMN_KINDS[i]` belong to `COLUMN_NAMES[i + 2]`.
            let shape = tx_shape(tx, block, &flags, &ctx);
            let row = feature_row(&vector, &shape, tx.compute_txid().to_string(), block.height);
            for (i, &value) in row.values.iter().enumerate() {
                let kind = &COLUMN_KINDS[i];
                let agg = self
                    .field_aggs
                    .entry(COLUMN_NAMES[i + 2])
                    .or_insert_with(|| new_field_agg(kind));
                fold_field(agg, kind, value);
            }

            on_tx(tx, block.height, &vector, &flags, &row);
            for (name, set) in [
                ("round_fee", flags.round_fee),
                ("round_payment", flags.round_payment),
                ("uih1", flags.uih1),
                ("uih2", flags.uih2),
                ("address_reuse", flags.address_reuse),
                ("same_block_parent", flags.same_block_parent),
                ("changeless", flags.changeless),
            ] {
                if set {
                    *self.aux_counts.entry(name).or_insert(0) += 1;
                }
            }
        }

        self.prev_block_min_feerate = block_min_feerate;
    }

    pub fn flush(&mut self, next_start: u32) -> EpochRow {
        let axis_counts = std::mem::take(&mut self.axis_counts)
            .into_iter()
            .map(|(axis, values)| (axis.to_string(), values))
            .collect();
        let aux_counts = std::mem::take(&mut self.aux_counts)
            .into_iter()
            .map(|(name, count)| (name.to_string(), count))
            .collect();
        let field_aggs = std::mem::take(&mut self.field_aggs)
            .into_iter()
            .map(|(name, agg)| (name.to_string(), agg))
            .collect();

        let row = EpochRow {
            start_height: self.start_height,
            end_height: self.last_height,
            txs: self.txs,
            defects: self.defects,
            axis_counts,
            vectors: std::mem::take(&mut self.vectors),
            vectors_extended: std::mem::take(&mut self.vectors_extended),
            aux_counts,
            field_aggs,
            template_matches: BTreeMap::new(),
        };
        self.start_height = next_start;
        self.last_height = next_start;
        self.txs = 0;
        self.defects = 0;
        row
    }
}

/// Same `"axis=value|..."` format as `FingerprintVector::key_for`.
fn join_key(axis_values: &[(&'static str, Cow<'_, str>)]) -> String {
    let mut out = String::new();
    for (i, (axis, value)) in axis_values.iter().enumerate() {
        if i > 0 {
            out.push('|');
        }
        out.push_str(axis);
        out.push('=');
        out.push_str(value);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::tests_support::{EPOCH_TEST_HEIGHT, cake_like_block, cake_like_tx};

    #[test]
    fn counts_axes_and_vectors_for_one_block() {
        let block = cake_like_block(EPOCH_TEST_HEIGHT);
        let mut acc = EpochAccumulator::new(EPOCH_TEST_HEIGHT, 144);
        acc.ingest(&block);
        let row = acc.flush(EPOCH_TEST_HEIGHT + 144);

        assert_eq!(row.start_height, EPOCH_TEST_HEIGHT);
        assert_eq!(row.txs, 1, "coinbase is excluded");
        assert_eq!(row.defects, 0);
        assert_eq!(
            row.axis_counts["nsequence"]["first 0x01 rest 0xffffffff"],
            1
        );
        assert_eq!(row.axis_counts["nlocktime"]["Zero"], 1);
        assert_eq!(row.vectors.values().sum::<u64>(), 1);
    }

    #[test]
    fn unresolvable_prevouts_count_as_defects_not_panics() {
        let mut block = cake_like_block(EPOCH_TEST_HEIGHT);
        block.prevouts.clear();
        let mut acc = EpochAccumulator::new(EPOCH_TEST_HEIGHT, 144);
        acc.ingest(&block);
        let row = acc.flush(EPOCH_TEST_HEIGHT + 144);

        assert_eq!(row.txs, 0);
        assert_eq!(row.defects, 1);
    }

    #[test]
    fn flush_resets_state() {
        let mut block = cake_like_block(EPOCH_TEST_HEIGHT);
        // An unresolvable tx so `defects` is non-zero too.
        let mut defect_tx = cake_like_tx();
        for txin in &mut defect_tx.input {
            txin.previous_output.vout += 1000;
        }
        block.block.txdata.push(defect_tx);

        let mut acc = EpochAccumulator::new(EPOCH_TEST_HEIGHT, 144);
        acc.ingest(&block);
        let first = acc.flush(EPOCH_TEST_HEIGHT + 144);

        assert_eq!(first.txs, 1);
        assert_eq!(
            first.defects, 1,
            "the unresolvable-prevout tx must count as a defect"
        );
        assert!(!first.axis_counts.is_empty());
        assert!(!first.vectors.is_empty());
        assert!(!first.vectors_extended.is_empty());
        assert!(!first.aux_counts.is_empty());
        assert!(
            first.template_matches.is_empty(),
            "template_matches is never populated"
        );

        let second = acc.flush(EPOCH_TEST_HEIGHT + 288);

        assert_eq!(second.txs, 0, "counters cleared on flush");
        assert_eq!(second.defects, 0, "defects cleared on flush");
        assert!(
            second.axis_counts.is_empty(),
            "axis_counts cleared on flush"
        );
        assert!(second.vectors.is_empty(), "vectors cleared on flush");
        assert!(
            second.vectors_extended.is_empty(),
            "vectors_extended cleared on flush"
        );
        assert!(second.aux_counts.is_empty(), "aux_counts cleared on flush");
        assert!(
            second.template_matches.is_empty(),
            "template_matches is never populated"
        );
        assert_eq!(second.start_height, EPOCH_TEST_HEIGHT + 144);
    }

    #[test]
    fn vectors_extended_differs_from_vectors_and_shares_totals() {
        let block = cake_like_block(EPOCH_TEST_HEIGHT);
        let mut acc = EpochAccumulator::new(EPOCH_TEST_HEIGHT, 144);
        acc.ingest(&block);
        let row = acc.flush(EPOCH_TEST_HEIGHT + 144);

        assert_eq!(row.vectors.len(), 1);
        assert_eq!(row.vectors_extended.len(), 1);
        let core_key = row.vectors.keys().next().unwrap();
        let extended_key = row.vectors_extended.keys().next().unwrap();

        assert_ne!(
            core_key, extended_key,
            "core and extended keys must be different strings"
        );
        assert!(
            extended_key.starts_with(core_key.as_str()),
            "extended key {extended_key:?} must contain the core key {core_key:?} as a prefix"
        );
        assert_eq!(
            row.vectors.values().sum::<u64>(),
            row.vectors_extended.values().sum::<u64>(),
            "every classified transaction must land in both maps"
        );
    }

    #[test]
    fn boundary_is_epoch_size_aligned() {
        let acc = EpochAccumulator::new(1000, 144);
        assert!(!acc.boundary_reached(1000));
        assert!(!acc.boundary_reached(1142));
        assert!(acc.boundary_reached(1143), "1000..=1143 is 144 blocks");
    }

    #[test]
    fn field_aggs_accumulate_categorical_and_numeric_columns() {
        let block = cake_like_block(EPOCH_TEST_HEIGHT);
        let mut acc = EpochAccumulator::new(EPOCH_TEST_HEIGHT, 144);
        acc.ingest(&block);
        let row = acc.flush(EPOCH_TEST_HEIGHT + 144);

        // Bool/OneHot/Ordinal columns fold into `Categorical`.
        match row
            .field_aggs
            .get("op_return")
            .expect("op_return column must be present")
        {
            crate::features::FieldAgg::Categorical(m) => {
                assert_eq!(
                    m.values().sum::<u64>(),
                    row.txs,
                    "every classified tx lands in exactly one op_return bucket"
                );
            }
            other => panic!("op_return must be a Categorical aggregate, got {other:?}"),
        }

        match row
            .field_aggs
            .get("input_count")
            .expect("input_count column must be present")
        {
            crate::features::FieldAgg::Numeric { count, .. } => {
                assert_eq!(
                    *count, row.txs,
                    "input_count must be folded exactly once per classified tx"
                );
            }
            other => panic!("input_count must be a Numeric aggregate, got {other:?}"),
        }
    }

    #[test]
    fn epoch_row_round_trips_through_json() {
        let block = cake_like_block(EPOCH_TEST_HEIGHT);
        let mut acc = EpochAccumulator::new(EPOCH_TEST_HEIGHT, 144);
        acc.ingest(&block);
        let row = acc.flush(EPOCH_TEST_HEIGHT + 144);

        let json = serde_json::to_string(&row).unwrap();
        let back: EpochRow = serde_json::from_str(&json).unwrap();
        assert_eq!(row, back);
    }

    #[test]
    fn old_epochs_jsonl_without_field_aggs_deserializes_with_empty_map() {
        let old_json = r#"{
  "start_height": 100,
  "end_height": 243,
  "txs": 42,
  "defects": 0,
  "axis_counts": {},
  "vectors": {},
  "vectors_extended": {},
  "aux_counts": {},
  "template_matches": {}
}"#;

        let row: EpochRow = serde_json::from_str(old_json)
            .expect("should deserialize old epoch without field_aggs");
        assert!(
            row.field_aggs.is_empty(),
            "field_aggs should default to empty map when missing from JSON"
        );
        assert_eq!(row.start_height, 100);
        assert_eq!(row.end_height, 243);
        assert_eq!(row.txs, 42);
    }
}
