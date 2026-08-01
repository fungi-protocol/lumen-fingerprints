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

/// Core + extended + heuristic axis names, in the order `key_for` should join them.
/// Built once and shared across every `EpochAccumulator`, rather than re-collected
/// into a fresh `Vec` for every non-defective transaction — this loop runs on the
/// order of 200 million times over a full scan.
///
/// `HEURISTIC_AXES` is included here — every axis in it must still be computed and
/// counted in `axis_counts` (see `ingest` below) — but `ingest` only ever slices the
/// CORE and CORE+EXTENDED prefixes of this list when building the two joint-vector
/// keys, so a heuristic axis is never part of `vectors` or `vectors_extended`.
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

/// One epoch's aggregate. This is the only thing that reaches disk during a scan.
// `Eq` (not just `PartialEq`) used to be derivable here, but `field_aggs`'s `FieldAgg`
// carries `f64` fields (`Numeric`'s `sum`/`min`/`max`), and `f64` has no `Eq` impl (NaN
// breaks the reflexivity `Eq` requires) — so this struct can only derive `PartialEq` now.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpochRow {
    pub start_height: u32,
    pub end_height: u32,
    pub txs: u64,
    pub defects: u64,
    /// axis name -> value name -> count
    pub axis_counts: BTreeMap<String, BTreeMap<String, u64>>,
    /// canonical core-axis vector key -> count
    pub vectors: BTreeMap<String, u64>,
    /// canonical core+extended vector key -> count
    pub vectors_extended: BTreeMap<String, u64>,
    /// aux flag name -> count of transactions where it is true
    pub aux_counts: BTreeMap<String, u64>,
    /// column name (`COLUMN_NAMES[2..]`) -> its aggregate over every classified transaction
    /// in this epoch, per `ColumnKind`'s classification (see `features.rs`). Populated by
    /// the same single classification pass that feeds `axis_counts`/`vectors`/`aux_counts`
    /// above, so it always covers exactly the epoch's classified population.
    #[serde(default)]
    pub field_aggs: BTreeMap<String, FieldAgg>,
    /// wallet template name -> count of transactions consistent with it, AS JUDGED BY
    /// WHICHEVER TEMPLATES WERE PASSED TO THE SCAN THAT PRODUCED THIS ROW.
    ///
    /// LEGACY / SUPERSEDED — always empty on rows written by the current `ingest`, and
    /// no longer read. `lumen report` does not consult this field: it recomputes
    /// matching at report time from `vectors_extended` (see `report::build_report` and
    /// `template::WalletTemplate::matches_axes`), because this field was a frozen
    /// snapshot that a later correction to `wallets.toml` could not retroactively fix,
    /// and because a template matching zero transactions never got a key here at all —
    /// making "wallet has no anonymity set" indistinguishable from "template name has a
    /// typo". `ingest` no longer computes it at all (that was 9 templates' worth of
    /// matching work per transaction feeding a field report-time code never read). The
    /// field stays in the struct, always serialized as `{}`, purely so rows already
    /// written by a historical scan continue to deserialize; new code must treat
    /// report-time matching as authoritative and must not read this field.
    pub template_matches: BTreeMap<String, u64>,
}

/// Accumulates one epoch in memory, then flushes it as an `EpochRow`.
///
/// The two count maps are keyed by `&'static str` rather than `String` while the scan
/// is live: every axis name (21 of them) and aux flag name (7 of them) is one of a
/// small, known set of string constants, so there is no need to allocate one afresh on
/// every bump — `flush` converts them to the owned `String` `EpochRow` needs on disk,
/// a few dozen times per epoch rather than roughly 1.8 billion times per scan.
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
    /// One `FieldAgg` per `COLUMN_NAMES[2..]` column, keyed by `&'static str` for the same
    /// reason as `axis_counts`/`aux_counts` above: `COLUMN_NAMES` is a small, known set of
    /// string constants, so no per-transaction allocation is needed to look one up. Created
    /// lazily (via `new_field_agg`) the first time a column is touched, rather than eagerly
    /// pre-populating all ~173 entries in `new`.
    field_aggs: BTreeMap<&'static str, FieldAgg>,
    /// The minimum `sat/vB` feerate over the previous block's fee-paying transactions,
    /// carried across `ingest_with` calls so each block's `BlockContext` can reference the
    /// block before it. `None` before the first block, or when that block had no fee-paying
    /// tx. Not reset by `flush`: it describes adjacency between blocks, which spans epoch
    /// boundaries, not a per-epoch quantity.
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

    /// Bump one axis's value counter. `axis` is never allocated here: the outer map is
    /// keyed by the axis's own `&'static str`, one of the 21 known axis names. The inner
    /// map still counts distinct *values*, which `get_mut` bumps with no allocation at
    /// all when the value has been seen before in this epoch — only a value observed
    /// for the first time pays for a `to_string()`.
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

    /// True once `height` is the last block of the current epoch.
    pub fn boundary_reached(&self, height: u32) -> bool {
        height >= self.start_height + self.epoch_size - 1
    }

    pub fn ingest(&mut self, block: &SourcedBlock) {
        self.ingest_with(block, &mut |_, _, _, _, _| {});
    }

    /// Like `ingest`, but calls `on_tx(tx, height, vector, flags, row)` for every classified
    /// transaction — the single classification pass that also feeds the aggregation, so a
    /// feature sink observes exactly the aggregated population (defects are skipped in
    /// both, so the sink never sees a transaction the aggregate did not count). `flags` is
    /// the same `AuxFlags` value this call already computes for `aux_counts` below, handed
    /// to the sink instead of making it call `aux_flags` a second time per transaction.
    /// `row` is the same `FeatureRow` this call builds to fold into `field_aggs`, handed to
    /// the sink instead of making it recompute `tx_shape`/`feature_row` a second time.
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

        // The block-relative feerate context `tx_shape` needs for `feerate_over_block_min`/
        // `feerate_over_prev_block_min`: this block's own minimum sat/vB feerate, and the
        // previous block's (carried in `self.prev_block_min_feerate`, updated below). This
        // used to be computed by the scan engine only when a feature sink was active;
        // `field_aggs` needs it for every block now that per-field aggregation is core, not
        // an optional sink.
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

            // Scratch buffer for this one transaction's axis values, in `all_axes()`
            // order (core axes first, then extended): every axis is evaluated exactly
            // once here, and `axis_counts` plus both joint-vector keys below are all
            // fed from that single pass — replacing the historical triple walk (once
            // for `axis_counts`, once via `vector.key()`, once via
            // `vector.key_for(all_axes())`, the last of which silently repeated the
            // first 12 axes of the second) that added up to roughly 54 `axis_value`
            // calls per transaction.
            //
            // This has to be declared fresh per transaction, not hoisted above the
            // loop and cleared/refilled in place: its entries borrow out of `vector`
            // (via `Cow::Borrowed`), and `vector` is itself a fresh value each
            // iteration, so a buffer declared outside the loop would need one fixed
            // lifetime valid across every iteration's distinct `vector` — which none
            // exists. Reaching for that would mean giving `EpochAccumulator` itself a
            // lifetime parameter purely to save one small Vec's allocation per
            // transaction; that is exactly the kind of trade the brief warns against,
            // clarity for a marginal win.
            let mut axis_values: Vec<(&'static str, Cow<'_, str>)> =
                Vec::with_capacity(all_axes().len());
            for axis in all_axes() {
                if let Some(value) = vector.axis_value(axis) {
                    axis_values.push((*axis, value));
                }
            }
            // Every name in `all_axes()` is one `axis_value` itself matches (see its
            // exhaustive `match`), so this always holds today; asserted rather than
            // assumed so a future axis added to one set but not the other fails loudly
            // here instead of silently truncating the core/extended split below.
            debug_assert_eq!(
                axis_values.len(),
                all_axes().len(),
                "axis_value must return Some for every name in all_axes()"
            );

            for (axis, value) in &axis_values {
                Self::bump(&mut self.axis_counts, axis, value);
            }

            // Two joint keys: core-only (the headline sparsity number) and core+extended
            // (sparsity when per-coin variation is also counted). Built directly from
            // the scratch values above rather than by re-deriving them.
            //
            // Both slices stop before the HEURISTIC_AXES tail of `axis_values` (`all_axes()`
            // orders core, then extended, then heuristic): `change_position`/`change_type`
            // are still computed and counted above in `axis_counts`, but must never enter
            // a joint key (see `HEURISTIC_AXES`'s doc comment in vector.rs) — a heuristic's
            // own abstention rate is a property of the method, not of chain diversity.
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

            // Build the per-tx feature shape/row exactly once, then fold every value into
            // its column's aggregate. `row.values[i]` is `COLUMN_NAMES[i + 2]`'s value;
            // `COLUMN_KINDS[i]` is index-aligned with it (see both const docs in
            // features.rs), so a column's `FieldAgg` is created (lazily, via
            // `new_field_agg`) and updated in the same pass.
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

        // Carried forward for the next call's `BlockContext`; not reset by `flush` (see the
        // field's doc comment).
        self.prev_block_min_feerate = block_min_feerate;
    }

    /// Emit the current epoch and reset for the one starting at `next_start`.
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

/// Joins already-computed `(axis, value)` pairs into the canonical
/// `"axis=value|axis=value|..."` key format — the same format
/// `FingerprintVector::key_for` produces, but built directly from values already
/// computed once per transaction (see `ingest`'s `axis_values` scratch buffer) rather
/// than recomputing `axis_value` for each axis a second or third time.
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
        assert_eq!(row.axis_counts["nsequence"]["CakeGroupC"], 1);
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
        // A second transaction whose prevouts are not resolvable, alongside the
        // classifiable one, so this epoch accumulates a non-zero value in every field
        // `flush` is supposed to clear (including `defects`, which the classifiable-only
        // fixture used below can never produce).
        let mut defect_tx = cake_like_tx();
        for txin in &mut defect_tx.input {
            txin.previous_output.vout += 1000;
        }
        block.block.txdata.push(defect_tx);

        let mut acc = EpochAccumulator::new(EPOCH_TEST_HEIGHT, 144);
        acc.ingest(&block);
        let first = acc.flush(EPOCH_TEST_HEIGHT + 144);

        // Sanity: confirm the fixture actually exercises every field below before
        // relying on it to catch a leaked counter.
        assert_eq!(first.txs, 1);
        assert_eq!(
            first.defects, 1,
            "the unresolvable-prevout tx must count as a defect"
        );
        assert!(!first.axis_counts.is_empty());
        assert!(!first.vectors.is_empty());
        assert!(!first.vectors_extended.is_empty());
        assert!(!first.aux_counts.is_empty());
        // `template_matches` is never populated by `ingest` any more (see its doc
        // comment on `EpochRow`), so it is always empty — asserted here too, not just
        // on `second` below, since "always empty" is the invariant under test, not
        // "cleared by flush".
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

        // A Bool/OneHot/Ordinal column folds into `Categorical`: every classified tx
        // must land in exactly one bucket, so the map's counts sum to `row.txs`.
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

        // A `Numeric` column's running `count` must equal the number of classified txs:
        // one fold per transaction, no more, no fewer.
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
        // This JSON represents an old EpochRow that was serialized before `field_aggs`
        // was added to the struct. It omits the `field_aggs` key entirely, as would
        // appear in a historical epochs.jsonl file.
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
