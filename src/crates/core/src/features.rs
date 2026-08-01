//! The numeric, per-transaction projection of a `FingerprintVector`: one fixed-width
//! row of `f64`s per transaction, for a machine-learning feature matrix. This is the
//! wide, redundancy-tolerant view; the curated, orthogonal view used by the sparsity
//! joint vector stays in `vector.rs`'s `axis_value`. Columns here MUST NEVER be added
//! to `CORE_AXES`/`EXTENDED_AXES`.

use crate::aux::AuxFlags;
use crate::change::{change_index_by_round_number, change_index_by_script_type};
use crate::vector::tx_fee_vsize_weight;
use crate::vector::{
    AgeClass, ChangePosition, EcdsaSigCount, FingerprintVector, InputSubtype, InputTypeClass,
    OrderClass, Tri,
};
use bitcoin::Transaction;
use lumen_fingerprints_lib::input_with_prevout::taproot_keyspend_non_default_sighash as fp_taproot_nondefault_sighash;
use lumen_fingerprints_lib::transaction::nlocktime_optin_without_use as fp_nlocktime_optin;
use lumen_fingerprints_lib::transaction::tx_signals_rbf;
use lumen_fingerprints_lib::{
    LocktimeOffsetType, NLockTimeType, NSequenceType, OutputStructureType, SighashType,
};
use lumen_primitives::OutputType;
use lumen_primitives::classify_script_pubkey;
use lumen_primitives::traits::SourcedBlock;

/// All `OutputType`s in discriminant order, so an index computed as `t as usize` maps
/// back to the type. Kept next to `tx_shape`, which indexes count arrays by `t as usize`.
const OUTPUT_TYPES_IN_ORDER: [OutputType; 8] = [
    OutputType::P2pkh,
    OutputType::P2sh,
    OutputType::P2wpkh,
    OutputType::P2wsh,
    OutputType::P2tr,
    OutputType::OpReturn,
    OutputType::NonStandard,
    OutputType::P2pk,
];

/// The raw per-transaction numeric shape for the ML feature matrix: nSequence bit
/// aggregates, per-type input/output multisets, positional signal, and the aux flags.
/// Computed once per tx by `tx_shape`; consumed by `feature_row`. Never enters the
/// sparsity joint vector.
#[derive(Debug, Clone)]
pub struct TxShape {
    pub bip125_rbf: bool,
    pub all_inputs_final: bool,
    pub bip68_active: bool,
    pub bip68_type_time: bool,
    pub nsequence_reserved_bits_set: bool,
    pub bip68_relative_value_max: u32,
    pub input_type_counts: [u32; 8],
    pub output_type_counts: [u32; 8],
    pub input_count: u32,
    pub output_count: u32,
    pub first_output_type: Option<OutputType>,
    pub last_output_type: Option<OutputType>,
    pub first_output_matches_input_type: bool,
    pub last_output_matches_input_type: bool,
    pub outputs_type_grouped: bool,
    pub first_input_type: Option<OutputType>,
    pub last_input_type: Option<OutputType>,
    pub inputs_type_grouped: bool,
    pub input_types_positional: [Option<OutputType>; 10],
    pub output_types_positional: [Option<OutputType>; 10],
    pub fee_sat: f64,
    pub vsize: f64,
    pub weight: f64,
    pub feerate_sat_per_vb: f64,
    pub feerate_sat_per_kwu: f64,
    pub fee_is_multiple_of_vsize: bool,
    pub fee_is_multiple_of_kwu: bool,
    pub fee_is_multiple_of_weight: bool,
    pub feerate_over_block_min: f64,
    pub feerate_over_prev_block_min: f64,
    pub max_output_value_sat: f64,
    pub hamming_decimal: f64,
    pub hamming_base2: f64,
    pub hamming_base3: f64,
    pub decimal_sig_fig_span: f64,
    pub hamming_decimal_min_over_outputs: f64,
    pub taproot_max_merkle_depth: f64,
    pub taproot_script_path_input_count: f64,
    pub is_standard: bool,
    pub nonstd_version: bool,
    pub nonstd_weight: bool,
    pub nonstd_output_type: bool,
    pub nonstd_dust: bool,
    pub nonstd_multi_op_return: bool,
    pub nonstd_bare_multisig: bool,
    pub has_inscription_envelope: bool,
    pub has_runestone: bool,
    pub multisig_m: u32,
    pub multisig_n: u32,
    pub multisig_mixed: bool,
    pub distinct_output_value_count: u32,
    pub max_equal_value_output_count: u32,
    pub nlocktime_optin_without_use: bool,
    pub taproot_keyspend_non_default_sighash: bool,
    /// Total sig-op cost (legacy ×4 + P2SH + witness) over the resolved prevouts — the
    /// broader, not-ECDSA-scoped signature measure `ecdsa_sigs` is not (Armin).
    pub sigop_count: u32,
    /// The value-based round-number change heuristic identified a change output.
    /// `change_type` (on the vector) already flags the script-type heuristic; this is the
    /// independent second heuristic, so which of them fires is a construction fingerprint.
    pub change_detected_by_round_number: bool,
    /// The script-type and round-number heuristics both fired and picked the same output.
    pub change_heuristics_agree: bool,
    pub aux: AuxFlags,
}

pub fn tx_shape(
    tx: &Transaction,
    block: &SourcedBlock,
    aux: &AuxFlags,
    ctx: &BlockContext,
) -> TxShape {
    let seqs: Vec<u32> = tx
        .input
        .iter()
        .map(|i| i.sequence.to_consensus_u32())
        .collect();
    let bip125_rbf = tx_signals_rbf(&tx.input);
    let all_inputs_final = !seqs.is_empty() && seqs.iter().all(|&s| s == 0xFFFF_FFFF);

    // BIP-68 applies only to version >= 2; read the version the same way classify_tx does.
    let version_ge2 = tx.version.0 >= 2;
    let bip68_inputs: Vec<u32> = if version_ge2 {
        seqs.iter()
            .copied()
            .filter(|&s| s & 0x8000_0000 == 0)
            .collect()
    } else {
        Vec::new()
    };
    let bip68_active = !bip68_inputs.is_empty();
    let bip68_type_time = bip68_inputs.iter().any(|&s| s & 0x0040_0000 != 0);
    let bip68_relative_value_max = bip68_inputs
        .iter()
        .map(|&s| s & 0x0000_FFFF)
        .max()
        .unwrap_or(0);
    // reserved = bits 16-21 (0x003F0000) | bits 23-30 (0x7F800000) = 0x7FBF0000, only
    // meaningful on bit-31-clear inputs (where BIP-68 would interpret the field).
    let nsequence_reserved_bits_set = seqs
        .iter()
        .any(|&s| s & 0x8000_0000 == 0 && s & 0x7FBF_0000 != 0);

    // Ordered input types (mirror of `output_types`). Every prevout resolves on this path
    // (`classify_tx` returns None upstream otherwise; coinbase skipped), so the vec is
    // total; the `NonStandard` fallback is unreachable and only keeps positions aligned
    // with input indices.
    let input_types: Vec<OutputType> = tx
        .input
        .iter()
        .map(|input| {
            block
                .prevout(&input.previous_output)
                .map(|p| classify_script_pubkey(p.script_pubkey.as_bytes()))
                .unwrap_or(OutputType::NonStandard)
        })
        .collect();
    let mut input_type_counts = [0u32; 8];
    for &t in &input_types {
        input_type_counts[t as usize] += 1;
    }
    let output_types: Vec<OutputType> = tx
        .output
        .iter()
        .map(|o| classify_script_pubkey(o.script_pubkey.as_bytes()))
        .collect();
    let mut output_type_counts = [0u32; 8];
    for &t in &output_types {
        output_type_counts[t as usize] += 1;
    }

    // Uniform input type: exactly one distinct input type present.
    let uniform_input_type = {
        let nz: Vec<usize> = (0..8).filter(|&i| input_type_counts[i] > 0).collect();
        if nz.len() == 1 {
            Some(OUTPUT_TYPES_IN_ORDER[nz[0]])
        } else {
            None
        }
    };
    let first_output_type = output_types.first().copied();
    let last_output_type = output_types.last().copied();
    let first_output_matches_input_type =
        uniform_input_type.is_some() && uniform_input_type == first_output_type;
    let last_output_matches_input_type =
        uniform_input_type.is_some() && uniform_input_type == last_output_type;

    // Grouped: no type reappears after a run of a different type.
    let mut outputs_type_grouped = true;
    {
        let mut closed: Vec<OutputType> = Vec::new();
        let mut current: Option<OutputType> = None;
        for &t in &output_types {
            if current != Some(t) {
                if closed.contains(&t) {
                    outputs_type_grouped = false;
                    break;
                }
                if let Some(c) = current {
                    closed.push(c);
                }
                current = Some(t);
            }
        }
    }

    let first_input_type = input_types.first().copied();
    let last_input_type = input_types.last().copied();

    // Grouped: no input type reappears after a run of a different type (same rule as
    // `outputs_type_grouped`).
    let mut inputs_type_grouped = true;
    {
        let mut closed: Vec<OutputType> = Vec::new();
        let mut current: Option<OutputType> = None;
        for &t in &input_types {
            if current != Some(t) {
                if closed.contains(&t) {
                    inputs_type_grouped = false;
                    break;
                }
                if let Some(c) = current {
                    closed.push(c);
                }
                current = Some(t);
            }
        }
    }

    let mut input_types_positional = [None; 10];
    let mut output_types_positional = [None; 10];
    for i in 0..10 {
        input_types_positional[i] = input_types.get(i).copied();
        output_types_positional[i] = output_types.get(i).copied();
    }

    // Feerate
    let (fee, vsize, weight) = tx_fee_vsize_weight(tx, block).unwrap_or((0, 1, 1));
    let feerate_sat_per_vb = fee as f64 / vsize as f64;
    let feerate_sat_per_kwu = fee as f64 * 1000.0 / weight as f64;
    let fee_is_multiple_of_vsize = vsize != 0 && fee % vsize == 0;
    // Parallel to `fee_is_multiple_of_vsize`, but "by weight": the sat/kwu rate is an
    // exact integer. u128 avoids overflow on `fee * 1000` for large fees.
    let fee_is_multiple_of_kwu = weight != 0 && (u128::from(fee) * 1000) % u128::from(weight) == 0;
    // Parallel to `fee_is_multiple_of_kwu`, but for sat/wu directly: the sat/wu rate is
    // an exact integer. Plain u64 modulo suffices (no `* 1000` scaling, so no overflow).
    let fee_is_multiple_of_weight = weight != 0 && fee % weight == 0;
    let ratio = |min: Option<f64>| match min {
        Some(m) if m > 0.0 => feerate_sat_per_vb / m,
        _ => -1.0,
    };
    let feerate_over_block_min = ratio(ctx.block_min_feerate);
    let feerate_over_prev_block_min = ratio(ctx.prev_block_min_feerate);

    // Amounts
    let out_values: Vec<u64> = tx.output.iter().map(|o| o.value.to_sat()).collect();
    let max_output_value_sat = out_values.iter().copied().max().unwrap_or(0);
    let hamming_decimal_min_over_outputs = out_values
        .iter()
        .copied()
        .filter(|&x| x > 0)
        .map(hamming_decimal)
        .min()
        .map_or(-1.0, f64::from);
    let distinct_output_value_count = {
        let s: std::collections::HashSet<u64> = out_values.iter().copied().collect();
        s.len() as u32
    };
    let max_equal_value_output_count = {
        let mut counts: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
        for &v in &out_values {
            *counts.entry(v).or_default() += 1;
        }
        counts.values().copied().max().unwrap_or(0)
    };

    // Sig-op cost over the resolved prevouts — broader than the ECDSA-only `ecdsa_sigs`
    // bucket (Armin): counts legacy (×4), P2SH, and witness sig ops, so Schnorr and
    // multisig are included.
    let sigop_count = tx.total_sigop_cost(|op| block.prevout(op).cloned()) as u32;

    // Which change heuristic fires. The script-type heuristic already feeds `change_type`
    // on the vector; the round-number one is an independent, value-based second heuristic.
    // Which of them identifies change — and whether they agree — is itself a fingerprint.
    let change_by_script_type = change_index_by_script_type(&input_types, &output_types);
    let change_by_round_number = change_index_by_round_number(&out_values);
    let change_detected_by_round_number = change_by_round_number.is_some();
    let change_heuristics_agree = matches!(
        (change_by_script_type, change_by_round_number),
        (Some(a), Some(b)) if a == b
    );

    // Taproot
    let depths: Vec<u32> = tx
        .input
        .iter()
        .filter_map(|i| taproot_script_path_depth(&i.witness))
        .collect();
    let taproot_max_merkle_depth = depths.iter().copied().max().unwrap_or(0);
    let taproot_script_path_input_count = depths.len() as u32;

    // Standardness (Core IsStandardTx subset). CSV-only; see FEATURES.md for the policy caveat.
    let nonstd_version = !(1..=3).contains(&tx.version.0);
    let nonstd_weight = tx.weight().to_wu() > MAX_STANDARD_TX_WEIGHT;
    let mut nonstd_output_type = false;
    let mut nonstd_dust = false;
    let mut nonstd_bare_multisig = false;
    let mut op_return_count = 0u32;
    // Iterate outputs in order so `i` indexes `output_types`, reusing the single
    // classification pass above instead of re-calling `classify_script_pubkey`.
    for (i, o) in tx.output.iter().enumerate() {
        let spk = &o.script_pubkey;
        if spk.is_op_return() {
            op_return_count += 1;
            continue;
        }
        match bare_multisig_n(spk) {
            Some(n) => {
                if n > 3 {
                    nonstd_bare_multisig = true;
                }
            }
            None => {
                if output_types[i] == OutputType::NonStandard {
                    nonstd_output_type = true;
                }
            }
        }
        if o.value.to_sat() < output_dust_threshold(spk) {
            nonstd_dust = true;
        }
    }
    let nonstd_multi_op_return = op_return_count > 1;
    let is_standard = !(nonstd_version
        || nonstd_weight
        || nonstd_output_type
        || nonstd_dust
        || nonstd_multi_op_return
        || nonstd_bare_multisig);

    // Protocol markers (sub-project 3d)
    let tx_has_inscription = tx
        .input
        .iter()
        .any(|i| has_inscription_envelope(&i.witness));
    let has_runestone = tx
        .output
        .iter()
        .any(|o| is_runestone_output(&o.script_pubkey));

    // Multisig configuration (sub-project 3f)
    let multisig_configs: Vec<(u32, u32)> = tx
        .input
        .iter()
        .filter_map(|txin| {
            block
                .prevout(&txin.previous_output)
                .and_then(|p| input_multisig_config(txin, &p.script_pubkey))
        })
        .collect();
    let (multisig_m, multisig_n) = multisig_configs
        .iter()
        .copied()
        .max_by_key(|(_, n)| *n)
        .unwrap_or((0, 0));
    let multisig_mixed = {
        let distinct: std::collections::HashSet<(u32, u32)> =
            multisig_configs.iter().copied().collect();
        distinct.len() > 1
    };

    // Two more upstream fingerprints (sub-project 3g)
    let nlocktime_optin_without_use =
        fp_nlocktime_optin(&tx.input, tx.lock_time.to_consensus_u32());
    let taproot_keyspend_non_default_sighash = tx.input.iter().any(|txin| {
        block
            .prevout(&txin.previous_output)
            .is_some_and(|p| fp_taproot_nondefault_sighash(txin, p))
    });

    TxShape {
        bip125_rbf,
        all_inputs_final,
        bip68_active,
        bip68_type_time,
        nsequence_reserved_bits_set,
        bip68_relative_value_max,
        input_type_counts,
        output_type_counts,
        input_count: tx.input.len() as u32,
        output_count: tx.output.len() as u32,
        first_output_type,
        last_output_type,
        first_output_matches_input_type,
        last_output_matches_input_type,
        outputs_type_grouped,
        first_input_type,
        last_input_type,
        inputs_type_grouped,
        input_types_positional,
        output_types_positional,
        fee_sat: fee as f64,
        vsize: vsize as f64,
        weight: weight as f64,
        feerate_sat_per_vb,
        feerate_sat_per_kwu,
        fee_is_multiple_of_vsize,
        fee_is_multiple_of_kwu,
        fee_is_multiple_of_weight,
        feerate_over_block_min,
        feerate_over_prev_block_min,
        max_output_value_sat: max_output_value_sat as f64,
        hamming_decimal: f64::from(hamming_decimal(max_output_value_sat)),
        hamming_base2: f64::from(max_output_value_sat.count_ones()),
        hamming_base3: f64::from(hamming_base3(max_output_value_sat)),
        decimal_sig_fig_span: f64::from(decimal_sig_fig_span(max_output_value_sat)),
        hamming_decimal_min_over_outputs,
        taproot_max_merkle_depth: f64::from(taproot_max_merkle_depth),
        taproot_script_path_input_count: f64::from(taproot_script_path_input_count),
        is_standard,
        nonstd_version,
        nonstd_weight,
        nonstd_output_type,
        nonstd_dust,
        nonstd_multi_op_return,
        nonstd_bare_multisig,
        has_inscription_envelope: tx_has_inscription,
        has_runestone,
        multisig_m,
        multisig_n,
        multisig_mixed,
        distinct_output_value_count,
        max_equal_value_output_count,
        nlocktime_optin_without_use,
        taproot_keyspend_non_default_sighash,
        sigop_count,
        change_detected_by_round_number,
        change_heuristics_agree,
        aux: *aux,
    }
}

/// One transaction's feature row: two identity fields plus the numeric feature values,
/// in `COLUMN_NAMES[2..]` order.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureRow {
    pub txid: String,
    pub height: u32,
    pub values: Vec<f64>,
}

/// Full CSV header and column order. Index 0/1 are identity; the rest are features.
/// Adding a feature means adding its name here AND pushing its value in `feature_row`
/// in the same position — `row_width_matches_column_count` guards the two stay in sync.
pub const COLUMN_NAMES: &[&str] = &[
    "txid",
    "block_height",
    // booleans + tri-state (Task 1)
    "op_return",
    "round_feerate",
    "low_s_yes",
    "round_fee",
    "round_payment",
    "uih1",
    "uih2",
    "address_reuse",
    "same_block_parent",
    "changeless",
    "low_r_yes",
    "low_r_indeterminate",
    "uncompressed_pubkey_yes",
    "uncompressed_pubkey_indeterminate",
    // version: 1, 2, 3, other (open i32 domain → trailing catch-all)
    "version_1",
    "version_2",
    "version_3",
    "version_other",
    // nsequence (closed enum, exhaustive)
    "nsequence_cake_group_c",
    "nsequence_lone_0x01",
    "nsequence_rbf",
    "nsequence_final",
    "nsequence_max",
    "nsequence_mixed_other",
    // nlocktime
    "nlocktime_zero",
    "nlocktime_anti_fee_snipe",
    "nlocktime_backdated",
    "nlocktime_future",
    "nlocktime_timestamp",
    // input_order
    "input_order_bip69",
    "input_order_value_ascending",
    "input_order_value_descending",
    "input_order_age_ascending",
    "input_order_other",
    "input_order_indeterminate",
    // output_order (same OrderClass variants)
    "output_order_bip69",
    "output_order_value_ascending",
    "output_order_value_descending",
    "output_order_age_ascending",
    "output_order_other",
    "output_order_indeterminate",
    // input_subtype
    "input_subtype_p2sh_p2wpkh",
    "input_subtype_p2sh_multisig",
    "input_subtype_p2sh_other",
    "input_subtype_p2wsh_multisig",
    "input_subtype_p2wsh_other",
    "input_subtype_taproot_key_path",
    "input_subtype_taproot_script_path",
    "input_subtype_bare",
    "input_subtype_mixed",
    "input_subtype_indeterminate",
    // sighash
    "sighash_taproot_default",
    "sighash_taproot_explicit",
    "sighash_all",
    "sighash_none",
    "sighash_single",
    "sighash_acp_all",
    "sighash_acp_none",
    "sighash_acp_single",
    "sighash_other",
    "sighash_mixed",
    "sighash_na",
    // input_types: Uniform(type) collapses to that type's index; Mixed and Unknown are
    // their own columns. The 8 OutputType variants keep OutputType's own order.
    "input_types_uniform_p2pkh",
    "input_types_uniform_p2sh",
    "input_types_uniform_p2wpkh",
    "input_types_uniform_p2wsh",
    "input_types_uniform_p2tr",
    "input_types_uniform_op_return",
    "input_types_uniform_nonstandard",
    "input_types_uniform_p2pk",
    "input_types_mixed",
    "input_types_unknown",
    // ordinal indices (ascending = larger bucket)
    "ecdsa_sigs_index",       // None=0, One=1, Few=2, Many=3
    "input_age_index",        // SameBlock=0, Within6=1, Within144=2, Older=3, Indeterminate=4
    "output_structure_index", // Single=0, Double=1, Multi=2, Unknown=3
    "feerate_bucket_index",   // parsed lower bound rank; see encoder
    "locktime_offset_index",  // NotApplicable→0 with the flag below set
    "locktime_offset_not_applicable",
    // heuristic-derived (in the CSV for clustering; never in the joint key)
    "change_position_first",
    "change_position_last",
    "change_position_middle",
    "change_position_indeterminate",
    "change_type_present",
    // Group A — nSequence bit aggregates (sub-project 2)
    "bip125_rbf",
    "all_inputs_final",
    "bip68_active",
    "bip68_type_time",
    "nsequence_reserved_bits_set",
    "bip68_relative_value_max",
    // Group B — nLockTime coupling
    "nlocktime_dead",
    // Group C — type counts and positional
    "input_count",
    "output_count",
    "input_count_p2pkh",
    "input_count_p2sh",
    "input_count_p2wpkh",
    "input_count_p2wsh",
    "input_count_p2tr",
    "input_count_op_return",
    "input_count_nonstandard",
    "input_count_p2pk",
    "output_count_p2pkh",
    "output_count_p2sh",
    "output_count_p2wpkh",
    "output_count_p2wsh",
    "output_count_p2tr",
    "output_count_op_return",
    "output_count_nonstandard",
    "output_count_p2pk",
    "first_output_type_index",
    "last_output_type_index",
    "first_output_matches_input_type",
    "last_output_matches_input_type",
    "outputs_type_grouped",
    // Positional type encoding (sub-project 3e): input-side mirror + capped per-position arrays
    "first_input_type_index",
    "last_input_type_index",
    "inputs_type_grouped",
    "input_type_at_pos_0",
    "input_type_at_pos_1",
    "input_type_at_pos_2",
    "input_type_at_pos_3",
    "input_type_at_pos_4",
    "input_type_at_pos_5",
    "input_type_at_pos_6",
    "input_type_at_pos_7",
    "input_type_at_pos_8",
    "input_type_at_pos_9",
    "output_type_at_pos_0",
    "output_type_at_pos_1",
    "output_type_at_pos_2",
    "output_type_at_pos_3",
    "output_type_at_pos_4",
    "output_type_at_pos_5",
    "output_type_at_pos_6",
    "output_type_at_pos_7",
    "output_type_at_pos_8",
    "output_type_at_pos_9",
    // Feerate (sub-project 3a)
    "fee_sat",
    "vsize",
    "weight",
    "feerate_sat_per_vb",
    "feerate_sat_per_kwu",
    "fee_is_multiple_of_vsize",
    "feerate_over_block_min",
    "feerate_over_prev_block_min",
    // Amounts
    "max_output_value_sat",
    "hamming_decimal",
    "hamming_base2",
    "hamming_base3",
    "decimal_sig_fig_span",
    "hamming_decimal_min_over_outputs",
    // Taproot
    "taproot_max_merkle_depth",
    "taproot_script_path_input_count",
    // Standardness (sub-project 3b)
    "is_standard",
    "nonstd_version",
    "nonstd_weight",
    "nonstd_output_type",
    "nonstd_dust",
    "nonstd_multi_op_return",
    "nonstd_bare_multisig",
    // Protocol markers (sub-project 3d)
    "has_inscription_envelope",
    "has_runestone",
    // Feerate (appended): "by weight" analogue of fee_is_multiple_of_vsize
    "fee_is_multiple_of_kwu",
    // Feerate (appended): sat/wu analogue, stricter than fee_is_multiple_of_kwu (implies it)
    "fee_is_multiple_of_weight",
    // Multisig configuration (sub-project 3f)
    "multisig_m",
    "multisig_n",
    "multisig_mixed",
    // Output value structure — denomination / coinjoin (sub-project 3f)
    "distinct_output_value_count",
    "max_equal_value_output_count",
    // Two more upstream fingerprints (sub-project 3g)
    "nlocktime_optin_without_use",
    "taproot_keyspend_non_default_sighash",
    // Sig-op cost + which change heuristic fired (Armin's two attribute requests)
    "sigop_count",
    "change_detected_by_round_number",
    "change_heuristics_agree",
];

/// How a feature column's raw `f64` values should be folded into an aggregate for
/// the dashboard. `Bool` is the default for plain 0/1 flags; the other three variants
/// cover the columns explicitly listed in the classification rule (see task brief).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColumnKind {
    /// A plain 0/1 flag.
    Bool,
    /// One member of a one-hot family; `group` is the family's shared name prefix
    /// (e.g. `"version_"`), so aggregation can regroup the family's columns.
    OneHot { group: &'static str },
    /// A small closed set of ordered categories, indexed 0..labels.len(); the raw
    /// value is the index into `labels`.
    Ordinal { labels: &'static [&'static str] },
    /// A continuous/count value bucketed by lower edges (inclusive); `edges[i]` is
    /// the lower bound of bucket `i`.
    Numeric { edges: &'static [f64] },
}

// Numeric bucket-edge tables, named by the group of columns that share them.
const EDGES_BIP68_RELATIVE_VALUE: &[f64] =
    &[0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0, 1024.0, 65536.0];
const EDGES_IO_COUNT: &[f64] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 20.0, 50.0];
const EDGES_IO_TYPE_COUNT: &[f64] = &[0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0];
const EDGES_FEE_SAT: &[f64] = &[0.0, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 1e7];
const EDGES_VSIZE: &[f64] = &[0.0, 110.0, 200.0, 400.0, 800.0, 1500.0, 3000.0, 10000.0];
const EDGES_WEIGHT: &[f64] = &[0.0, 440.0, 800.0, 1600.0, 3200.0, 6000.0, 12000.0, 40000.0];
const EDGES_FEERATE: &[f64] = &[0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0, 100.0, 300.0];
const EDGES_FEERATE_RATIO: &[f64] = &[-1.0, 0.0, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0];
const EDGES_AMOUNT: &[f64] = &[0.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10];
const EDGES_HAMMING: &[f64] = &[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0];
const EDGES_TAPROOT_DEPTH: &[f64] = &[0.0, 1.0, 2.0, 3.0, 4.0, 8.0];
const EDGES_MULTISIG: &[f64] = &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 16.0];
const EDGES_OUTPUT_VALUE_COUNT: &[f64] = &[0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0];
const EDGES_SIGOP: &[f64] = &[0.0, 1.0, 4.0, 8.0, 16.0, 32.0, 80.0, 200.0, 800.0];

// Ordinal label tables, named by the column(s) that share them.
const LABELS_ECDSA_SIGS: &[&str] = &["none", "one", "few", "many"];
const LABELS_INPUT_AGE: &[&str] = &["same_block", "<=6", "<=144", "older", "indeterminate"];
const LABELS_OUTPUT_STRUCTURE: &[&str] = &["single", "double", "multi", "unknown"];
/// `feerate_bucket_index` is `feerate_rank`'s lower-bound rank (see that function):
/// 0..9 for buckets "0".."9", then the tens boundaries 10..100 for "10-19".."100+",
/// and `u16::MAX` for "Unknown". The label for a rank is `"r" + rank`, per the task
/// brief ("numeric index; label = the value itself"); "unknown" covers the sentinel.
const LABELS_FEERATE_BUCKET: &[&str] = &[
    "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r20", "r30", "r40", "r50",
    "r60", "r70", "r80", "r90", "r100", "unknown",
];
const LABELS_LOCKTIME_OFFSET: &[&str] = &[
    "0", "1", "2", "3", "4-6", "7-12", "13-25", "26-50", "51-100",
];
/// Shared by every `*_type_index` column (`first_output_type_index`,
/// `last_output_type_index`, `first_input_type_index`, `last_input_type_index`,
/// `input_type_at_pos_0..9`, `output_type_at_pos_0..9`): index `-1` (absent input/output)
/// maps to `"absent"`, else the index follows `OUTPUT_TYPES_IN_ORDER`.
const LABELS_TYPE_INDEX: &[&str] = &[
    "absent",
    "p2pkh",
    "p2sh",
    "p2wpkh",
    "p2wsh",
    "p2tr",
    "op_return",
    "nonstandard",
    "p2pk",
];

/// One `ColumnKind` per `COLUMN_NAMES[2..]` column, same order — index-aligned so
/// `COLUMN_KINDS[i]` describes `COLUMN_NAMES[i + 2]`. `Bool` is the default; only the
/// columns explicitly classified as `Numeric`, `Ordinal`, or `OneHot` in the task
/// brief deviate from it. Later tasks fold `feature_row`'s per-tx values into
/// per-column aggregates using this classification.
pub const COLUMN_KINDS: &[ColumnKind] = &[
    // booleans + tri-state (Task 1)
    ColumnKind::Bool, // op_return
    ColumnKind::Bool, // round_feerate
    ColumnKind::Bool, // low_s_yes
    ColumnKind::Bool, // round_fee
    ColumnKind::Bool, // round_payment
    ColumnKind::Bool, // uih1
    ColumnKind::Bool, // uih2
    ColumnKind::Bool, // address_reuse
    ColumnKind::Bool, // same_block_parent
    ColumnKind::Bool, // changeless
    ColumnKind::Bool, // low_r_yes
    ColumnKind::Bool, // low_r_indeterminate
    ColumnKind::Bool, // uncompressed_pubkey_yes
    ColumnKind::Bool, // uncompressed_pubkey_indeterminate
    // version: 1, 2, 3, other (open i32 domain → trailing catch-all)
    ColumnKind::OneHot { group: "version_" }, // version_1
    ColumnKind::OneHot { group: "version_" }, // version_2
    ColumnKind::OneHot { group: "version_" }, // version_3
    ColumnKind::OneHot { group: "version_" }, // version_other
    // nsequence (closed enum, exhaustive)
    ColumnKind::OneHot {
        group: "nsequence_",
    }, // nsequence_cake_group_c
    ColumnKind::OneHot {
        group: "nsequence_",
    }, // nsequence_lone_0x01
    ColumnKind::OneHot {
        group: "nsequence_",
    }, // nsequence_rbf
    ColumnKind::OneHot {
        group: "nsequence_",
    }, // nsequence_final
    ColumnKind::OneHot {
        group: "nsequence_",
    }, // nsequence_max
    ColumnKind::OneHot {
        group: "nsequence_",
    }, // nsequence_mixed_other
    // nlocktime
    ColumnKind::OneHot {
        group: "nlocktime_",
    }, // nlocktime_zero
    ColumnKind::OneHot {
        group: "nlocktime_",
    }, // nlocktime_anti_fee_snipe
    ColumnKind::OneHot {
        group: "nlocktime_",
    }, // nlocktime_backdated
    ColumnKind::OneHot {
        group: "nlocktime_",
    }, // nlocktime_future
    ColumnKind::OneHot {
        group: "nlocktime_",
    }, // nlocktime_timestamp
    // input_order
    ColumnKind::OneHot {
        group: "input_order_",
    }, // input_order_bip69
    ColumnKind::OneHot {
        group: "input_order_",
    }, // input_order_value_ascending
    ColumnKind::OneHot {
        group: "input_order_",
    }, // input_order_value_descending
    ColumnKind::OneHot {
        group: "input_order_",
    }, // input_order_age_ascending
    ColumnKind::OneHot {
        group: "input_order_",
    }, // input_order_other
    ColumnKind::OneHot {
        group: "input_order_",
    }, // input_order_indeterminate
    // output_order (same OrderClass variants)
    ColumnKind::OneHot {
        group: "output_order_",
    }, // output_order_bip69
    ColumnKind::OneHot {
        group: "output_order_",
    }, // output_order_value_ascending
    ColumnKind::OneHot {
        group: "output_order_",
    }, // output_order_value_descending
    ColumnKind::OneHot {
        group: "output_order_",
    }, // output_order_age_ascending
    ColumnKind::OneHot {
        group: "output_order_",
    }, // output_order_other
    ColumnKind::OneHot {
        group: "output_order_",
    }, // output_order_indeterminate
    // input_subtype
    ColumnKind::OneHot {
        group: "input_subtype_",
    }, // input_subtype_p2sh_p2wpkh
    ColumnKind::OneHot {
        group: "input_subtype_",
    }, // input_subtype_p2sh_multisig
    ColumnKind::OneHot {
        group: "input_subtype_",
    }, // input_subtype_p2sh_other
    ColumnKind::OneHot {
        group: "input_subtype_",
    }, // input_subtype_p2wsh_multisig
    ColumnKind::OneHot {
        group: "input_subtype_",
    }, // input_subtype_p2wsh_other
    ColumnKind::OneHot {
        group: "input_subtype_",
    }, // input_subtype_taproot_key_path
    ColumnKind::OneHot {
        group: "input_subtype_",
    }, // input_subtype_taproot_script_path
    ColumnKind::OneHot {
        group: "input_subtype_",
    }, // input_subtype_bare
    ColumnKind::OneHot {
        group: "input_subtype_",
    }, // input_subtype_mixed
    ColumnKind::OneHot {
        group: "input_subtype_",
    }, // input_subtype_indeterminate
    // sighash
    ColumnKind::OneHot { group: "sighash_" }, // sighash_taproot_default
    ColumnKind::OneHot { group: "sighash_" }, // sighash_taproot_explicit
    ColumnKind::OneHot { group: "sighash_" }, // sighash_all
    ColumnKind::OneHot { group: "sighash_" }, // sighash_none
    ColumnKind::OneHot { group: "sighash_" }, // sighash_single
    ColumnKind::OneHot { group: "sighash_" }, // sighash_acp_all
    ColumnKind::OneHot { group: "sighash_" }, // sighash_acp_none
    ColumnKind::OneHot { group: "sighash_" }, // sighash_acp_single
    ColumnKind::OneHot { group: "sighash_" }, // sighash_other
    ColumnKind::OneHot { group: "sighash_" }, // sighash_mixed
    ColumnKind::OneHot { group: "sighash_" }, // sighash_na
    // input_types: Uniform(type) collapses to that type's index; Mixed and Unknown are
    // their own columns. The 8 OutputType variants keep OutputType's own order.
    ColumnKind::OneHot {
        group: "input_types_",
    }, // input_types_uniform_p2pkh
    ColumnKind::OneHot {
        group: "input_types_",
    }, // input_types_uniform_p2sh
    ColumnKind::OneHot {
        group: "input_types_",
    }, // input_types_uniform_p2wpkh
    ColumnKind::OneHot {
        group: "input_types_",
    }, // input_types_uniform_p2wsh
    ColumnKind::OneHot {
        group: "input_types_",
    }, // input_types_uniform_p2tr
    ColumnKind::OneHot {
        group: "input_types_",
    }, // input_types_uniform_op_return
    ColumnKind::OneHot {
        group: "input_types_",
    }, // input_types_uniform_nonstandard
    ColumnKind::OneHot {
        group: "input_types_",
    }, // input_types_uniform_p2pk
    ColumnKind::OneHot {
        group: "input_types_",
    }, // input_types_mixed
    ColumnKind::OneHot {
        group: "input_types_",
    }, // input_types_unknown
    // ordinal indices (ascending = larger bucket)
    ColumnKind::Ordinal {
        labels: LABELS_ECDSA_SIGS,
    }, // ecdsa_sigs_index
    ColumnKind::Ordinal {
        labels: LABELS_INPUT_AGE,
    }, // input_age_index
    ColumnKind::Ordinal {
        labels: LABELS_OUTPUT_STRUCTURE,
    }, // output_structure_index
    ColumnKind::Ordinal {
        labels: LABELS_FEERATE_BUCKET,
    }, // feerate_bucket_index
    ColumnKind::Ordinal {
        labels: LABELS_LOCKTIME_OFFSET,
    }, // locktime_offset_index
    ColumnKind::Bool, // locktime_offset_not_applicable
    // heuristic-derived (in the CSV for clustering; never in the joint key)
    ColumnKind::OneHot {
        group: "change_position_",
    }, // change_position_first
    ColumnKind::OneHot {
        group: "change_position_",
    }, // change_position_last
    ColumnKind::OneHot {
        group: "change_position_",
    }, // change_position_middle
    ColumnKind::OneHot {
        group: "change_position_",
    }, // change_position_indeterminate
    ColumnKind::Bool, // change_type_present
    // Group A — nSequence bit aggregates (sub-project 2)
    ColumnKind::Bool, // bip125_rbf
    ColumnKind::Bool, // all_inputs_final
    ColumnKind::Bool, // bip68_active
    ColumnKind::Bool, // bip68_type_time
    ColumnKind::Bool, // nsequence_reserved_bits_set
    ColumnKind::Numeric {
        edges: EDGES_BIP68_RELATIVE_VALUE,
    }, // bip68_relative_value_max
    // Group B — nLockTime coupling
    ColumnKind::Bool, // nlocktime_dead
    // Group C — type counts and positional
    ColumnKind::Numeric {
        edges: EDGES_IO_COUNT,
    }, // input_count
    ColumnKind::Numeric {
        edges: EDGES_IO_COUNT,
    }, // output_count
    ColumnKind::Numeric {
        edges: EDGES_IO_TYPE_COUNT,
    }, // input_count_p2pkh
    ColumnKind::Numeric {
        edges: EDGES_IO_TYPE_COUNT,
    }, // input_count_p2sh
    ColumnKind::Numeric {
        edges: EDGES_IO_TYPE_COUNT,
    }, // input_count_p2wpkh
    ColumnKind::Numeric {
        edges: EDGES_IO_TYPE_COUNT,
    }, // input_count_p2wsh
    ColumnKind::Numeric {
        edges: EDGES_IO_TYPE_COUNT,
    }, // input_count_p2tr
    ColumnKind::Numeric {
        edges: EDGES_IO_TYPE_COUNT,
    }, // input_count_op_return
    ColumnKind::Numeric {
        edges: EDGES_IO_TYPE_COUNT,
    }, // input_count_nonstandard
    ColumnKind::Numeric {
        edges: EDGES_IO_TYPE_COUNT,
    }, // input_count_p2pk
    ColumnKind::Numeric {
        edges: EDGES_IO_TYPE_COUNT,
    }, // output_count_p2pkh
    ColumnKind::Numeric {
        edges: EDGES_IO_TYPE_COUNT,
    }, // output_count_p2sh
    ColumnKind::Numeric {
        edges: EDGES_IO_TYPE_COUNT,
    }, // output_count_p2wpkh
    ColumnKind::Numeric {
        edges: EDGES_IO_TYPE_COUNT,
    }, // output_count_p2wsh
    ColumnKind::Numeric {
        edges: EDGES_IO_TYPE_COUNT,
    }, // output_count_p2tr
    ColumnKind::Numeric {
        edges: EDGES_IO_TYPE_COUNT,
    }, // output_count_op_return
    ColumnKind::Numeric {
        edges: EDGES_IO_TYPE_COUNT,
    }, // output_count_nonstandard
    ColumnKind::Numeric {
        edges: EDGES_IO_TYPE_COUNT,
    }, // output_count_p2pk
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // first_output_type_index
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // last_output_type_index
    ColumnKind::Bool, // first_output_matches_input_type
    ColumnKind::Bool, // last_output_matches_input_type
    ColumnKind::Bool, // outputs_type_grouped
    // Positional type encoding (sub-project 3e): input-side mirror + capped per-position arrays
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // first_input_type_index
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // last_input_type_index
    ColumnKind::Bool, // inputs_type_grouped
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // input_type_at_pos_0
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // input_type_at_pos_1
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // input_type_at_pos_2
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // input_type_at_pos_3
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // input_type_at_pos_4
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // input_type_at_pos_5
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // input_type_at_pos_6
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // input_type_at_pos_7
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // input_type_at_pos_8
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // input_type_at_pos_9
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // output_type_at_pos_0
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // output_type_at_pos_1
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // output_type_at_pos_2
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // output_type_at_pos_3
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // output_type_at_pos_4
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // output_type_at_pos_5
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // output_type_at_pos_6
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // output_type_at_pos_7
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // output_type_at_pos_8
    ColumnKind::Ordinal {
        labels: LABELS_TYPE_INDEX,
    }, // output_type_at_pos_9
    // Feerate (sub-project 3a)
    ColumnKind::Numeric {
        edges: EDGES_FEE_SAT,
    }, // fee_sat
    ColumnKind::Numeric { edges: EDGES_VSIZE }, // vsize
    ColumnKind::Numeric {
        edges: EDGES_WEIGHT,
    }, // weight
    ColumnKind::Numeric {
        edges: EDGES_FEERATE,
    }, // feerate_sat_per_vb
    ColumnKind::Numeric {
        edges: EDGES_FEERATE,
    }, // feerate_sat_per_kwu
    ColumnKind::Bool,                           // fee_is_multiple_of_vsize
    ColumnKind::Numeric {
        edges: EDGES_FEERATE_RATIO,
    }, // feerate_over_block_min
    ColumnKind::Numeric {
        edges: EDGES_FEERATE_RATIO,
    }, // feerate_over_prev_block_min
    // Amounts
    ColumnKind::Numeric {
        edges: EDGES_AMOUNT,
    }, // max_output_value_sat
    ColumnKind::Numeric {
        edges: EDGES_HAMMING,
    }, // hamming_decimal
    ColumnKind::Numeric {
        edges: EDGES_HAMMING,
    }, // hamming_base2
    ColumnKind::Numeric {
        edges: EDGES_HAMMING,
    }, // hamming_base3
    ColumnKind::Numeric {
        edges: EDGES_HAMMING,
    }, // decimal_sig_fig_span
    ColumnKind::Numeric {
        edges: EDGES_AMOUNT,
    }, // hamming_decimal_min_over_outputs
    // Taproot
    ColumnKind::Numeric {
        edges: EDGES_TAPROOT_DEPTH,
    }, // taproot_max_merkle_depth
    ColumnKind::Numeric {
        edges: EDGES_TAPROOT_DEPTH,
    }, // taproot_script_path_input_count
    // Standardness (sub-project 3b)
    ColumnKind::Bool, // is_standard
    ColumnKind::Bool, // nonstd_version
    ColumnKind::Bool, // nonstd_weight
    ColumnKind::Bool, // nonstd_output_type
    ColumnKind::Bool, // nonstd_dust
    ColumnKind::Bool, // nonstd_multi_op_return
    ColumnKind::Bool, // nonstd_bare_multisig
    // Protocol markers (sub-project 3d)
    ColumnKind::Bool, // has_inscription_envelope
    ColumnKind::Bool, // has_runestone
    // Feerate (appended): "by weight" analogue of fee_is_multiple_of_vsize
    ColumnKind::Bool, // fee_is_multiple_of_kwu
    // Feerate (appended): sat/wu analogue, stricter than fee_is_multiple_of_kwu (implies it)
    ColumnKind::Bool, // fee_is_multiple_of_weight
    // Multisig configuration (sub-project 3f)
    ColumnKind::Numeric {
        edges: EDGES_MULTISIG,
    }, // multisig_m
    ColumnKind::Numeric {
        edges: EDGES_MULTISIG,
    }, // multisig_n
    ColumnKind::Bool, // multisig_mixed
    // Output value structure — denomination / coinjoin (sub-project 3f)
    ColumnKind::Numeric {
        edges: EDGES_OUTPUT_VALUE_COUNT,
    }, // distinct_output_value_count
    ColumnKind::Numeric {
        edges: EDGES_OUTPUT_VALUE_COUNT,
    }, // max_equal_value_output_count
    // Two more upstream fingerprints (sub-project 3g)
    ColumnKind::Bool, // nlocktime_optin_without_use
    ColumnKind::Bool, // taproot_keyspend_non_default_sighash
    // Sig-op cost + which change heuristic fired (Armin's two attribute requests)
    ColumnKind::Numeric { edges: EDGES_SIGOP }, // sigop_count
    ColumnKind::Bool,                           // change_detected_by_round_number
    ColumnKind::Bool,                           // change_heuristics_agree
];

/// One column's aggregated value across many transactions: `Categorical` tallies raw
/// values (see `fold_field`'s Categorical branch for the map-key convention), `Numeric`
/// keeps running count/sum/min/max plus a histogram over `ColumnKind::Numeric`'s edges.
/// Built by `new_field_agg`, updated per-tx by `fold_field`, combined across shards/epochs
/// by `merge_field`. Serialized in epoch/report artifacts (Tasks 3-4).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum FieldAgg {
    Categorical(std::collections::BTreeMap<String, u64>),
    Numeric {
        count: u64,
        sum: f64,
        min: f64,
        max: f64,
        hist: Vec<u64>,
    },
}

/// A fresh, zeroed aggregate matching `kind`: `Bool`/`OneHot`/`Ordinal` all get an empty
/// `Categorical` map (the map's keys are populated lazily by `fold_field`); `Numeric` gets
/// a `hist` sized `edges.len() + 1` (one bucket per edge plus one overflow bucket for
/// values below the first edge — see `bucket_index`), with `min`/`max` seeded at the
/// identity values for their reductions so the first `fold_field` call always wins.
pub fn new_field_agg(kind: &ColumnKind) -> FieldAgg {
    match kind {
        ColumnKind::Bool | ColumnKind::OneHot { .. } | ColumnKind::Ordinal { .. } => {
            FieldAgg::Categorical(std::collections::BTreeMap::new())
        }
        ColumnKind::Numeric { edges } => FieldAgg::Numeric {
            count: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            hist: vec![0; edges.len() + 1],
        },
    }
}

/// Folds one raw column value into `agg`. Categorical kinds (`Bool`, `OneHot`, `Ordinal`)
/// all key the map by the value rendered as a compact integer string — `"0"`/`"1"` for
/// `Bool`/`OneHot`, `format!("{}", value as i64)` for `Ordinal` (e.g. `"-1"`, `"0"`, `"3"`).
/// This is deliberately NOT `labels[value as usize]`: two ordinal families store values
/// `labels` can't index directly — the `*_type_index` columns use `-1` for "absent" (an
/// implicit `+1` offset from `labels`), and `feerate_bucket_index` stores a sparse rank
/// (10, 20, ..., 100) rather than a dense `labels` index. Keying by the raw integer is
/// uniform across every `Ordinal` column; `labels` is applied later, at report/display
/// time, once the raw tallies are already keyed and merged.
pub fn fold_field(agg: &mut FieldAgg, kind: &ColumnKind, value: f64) {
    match (agg, kind) {
        (FieldAgg::Categorical(m), ColumnKind::Bool | ColumnKind::OneHot { .. }) => {
            let key = if value != 0.0 { "1" } else { "0" };
            *m.entry(key.to_string()).or_insert(0) += 1;
        }
        (FieldAgg::Categorical(m), ColumnKind::Ordinal { .. }) => {
            let key = format!("{}", value as i64);
            *m.entry(key).or_insert(0) += 1;
        }
        (
            FieldAgg::Numeric {
                count,
                sum,
                min,
                max,
                hist,
            },
            ColumnKind::Numeric { edges },
        ) => {
            *count += 1;
            *sum += value;
            if value < *min {
                *min = value;
            }
            if value > *max {
                *max = value;
            }
            hist[bucket_index(edges, value)] += 1;
        }
        _ => unreachable!("FieldAgg variant must match the ColumnKind it was created from"),
    }
}

/// Combines `other` into `into` in place: categorical maps add matching keys (union of
/// keys otherwise); numeric `count`/`sum`/`hist` add elementwise, `min`/`max` reduce.
pub fn merge_field(into: &mut FieldAgg, other: &FieldAgg) {
    match (into, other) {
        (FieldAgg::Categorical(a), FieldAgg::Categorical(b)) => {
            for (k, v) in b {
                *a.entry(k.clone()).or_insert(0) += v;
            }
        }
        (
            FieldAgg::Numeric {
                count,
                sum,
                min,
                max,
                hist,
            },
            FieldAgg::Numeric {
                count: oc,
                sum: os,
                min: omin,
                max: omax,
                hist: ohist,
            },
        ) => {
            *count += oc;
            *sum += os;
            *min = min.min(*omin);
            *max = max.max(*omax);
            for (h, oh) in hist.iter_mut().zip(ohist) {
                *h += oh;
            }
        }
        _ => unreachable!("merge_field requires both aggregates to be the same variant"),
    }
}

/// The histogram bucket for `v`: the count of edges `<= v` (equivalently, `edges`'s
/// partition point on predicate `e <= v`, since `edges` is sorted ascending). Bucket 0
/// catches `v` below every edge (underflow — there's no bucket for "less than the first
/// lower bound"); bucket `i` for `0 < i < edges.len()` is `[edges[i-1], edges[i])`; bucket
/// `edges.len()` catches `v >= ` the last edge (open-ended overflow above the highest
/// bound). This is why `hist` is sized `edges.len() + 1`, not `edges.len()`.
fn bucket_index(edges: &[f64], v: f64) -> usize {
    edges.partition_point(|&e| e <= v)
}

fn push_bool(out: &mut Vec<f64>, b: bool) {
    out.push(if b { 1.0 } else { 0.0 });
}

/// Tri as two booleans: `<axis>_yes` then `<axis>_indeterminate`. Both zero means No.
fn push_tri(out: &mut Vec<f64>, t: Tri) {
    push_bool(out, t == Tri::Yes);
    push_bool(out, t == Tri::Indeterminate);
}

/// One-hot: push `count` values, 1.0 at `active`, 0.0 elsewhere.
fn push_onehot(out: &mut Vec<f64>, active: usize, count: usize) {
    for i in 0..count {
        out.push(if i == active { 1.0 } else { 0.0 });
    }
}

pub fn feature_row(
    v: &FingerprintVector,
    shape: &TxShape,
    txid: String,
    height: u32,
) -> FeatureRow {
    let mut values = Vec::with_capacity(COLUMN_NAMES.len() - 2);

    // `op_return`, `round_feerate`, `low_s`, `low_r`, `uncompressed_pubkey` are all
    // fields on the vector already.
    push_bool(&mut values, v.op_return);
    push_bool(&mut values, v.round_feerate);
    push_bool(&mut values, v.low_s == Tri::Yes);
    // aux-flag columns flow from `shape.aux`, in COLUMN_NAMES order.
    push_bool(&mut values, shape.aux.round_fee);
    push_bool(&mut values, shape.aux.round_payment);
    push_bool(&mut values, shape.aux.uih1);
    push_bool(&mut values, shape.aux.uih2);
    push_bool(&mut values, shape.aux.address_reuse);
    push_bool(&mut values, shape.aux.same_block_parent);
    push_bool(&mut values, shape.aux.changeless);
    push_tri(&mut values, v.low_r);
    push_tri(&mut values, v.uncompressed_pubkey);

    // version — one-hot with open-domain catch-all
    let version_active = match v.version {
        1 => 0,
        2 => 1,
        3 => 2,
        _ => 3,
    };
    push_onehot(&mut values, version_active, 4);

    push_onehot(
        &mut values,
        match v.nsequence {
            NSequenceType::CakeGroupC => 0,
            NSequenceType::Lone0x01 => 1,
            NSequenceType::Rbf => 2,
            NSequenceType::Final => 3,
            NSequenceType::Max => 4,
            NSequenceType::MixedOther => 5,
        },
        6,
    );

    push_onehot(
        &mut values,
        match v.nlocktime {
            NLockTimeType::Zero => 0,
            NLockTimeType::AntiFeeSnipe => 1,
            NLockTimeType::Backdated => 2,
            NLockTimeType::Future => 3,
            NLockTimeType::Timestamp => 4,
        },
        5,
    );

    let order_index = |o: OrderClass| match o {
        OrderClass::Bip69 => 0,
        OrderClass::ValueAscending => 1,
        OrderClass::ValueDescending => 2,
        OrderClass::AgeAscending => 3,
        OrderClass::Other => 4,
        OrderClass::Indeterminate => 5,
    };
    push_onehot(&mut values, order_index(v.input_order), 6);
    push_onehot(&mut values, order_index(v.output_order), 6);

    push_onehot(
        &mut values,
        match v.input_subtype {
            InputSubtype::P2shP2wpkh => 0,
            InputSubtype::P2shMultisig => 1,
            InputSubtype::P2shOther => 2,
            InputSubtype::P2wshMultisig => 3,
            InputSubtype::P2wshOther => 4,
            InputSubtype::TaprootKeyPath => 5,
            InputSubtype::TaprootScriptPath => 6,
            InputSubtype::Bare => 7,
            InputSubtype::Mixed => 8,
            InputSubtype::Indeterminate => 9,
        },
        10,
    );

    push_onehot(
        &mut values,
        match v.sighash {
            SighashType::TaprootDefault => 0,
            SighashType::TaprootExplicit => 1,
            SighashType::All => 2,
            SighashType::None => 3,
            SighashType::Single => 4,
            SighashType::AcpAll => 5,
            SighashType::AcpNone => 6,
            SighashType::AcpSingle => 7,
            SighashType::Other => 8,
            SighashType::Mixed => 9,
            SighashType::Na => 10,
        },
        11,
    );

    // input_types: 8 uniform-type columns + Mixed + Unknown = 10.
    let ot_index = |t: OutputType| match t {
        OutputType::P2pkh => 0,
        OutputType::P2sh => 1,
        OutputType::P2wpkh => 2,
        OutputType::P2wsh => 3,
        OutputType::P2tr => 4,
        OutputType::OpReturn => 5,
        OutputType::NonStandard => 6,
        OutputType::P2pk => 7,
    };
    let input_types_active = match v.input_types {
        InputTypeClass::Uniform(t) => ot_index(t),
        InputTypeClass::Mixed => 8,
        InputTypeClass::Unknown => 9,
    };
    push_onehot(&mut values, input_types_active, 10);

    values.push(match v.ecdsa_sigs {
        EcdsaSigCount::None => 0.0,
        EcdsaSigCount::One => 1.0,
        EcdsaSigCount::Few => 2.0,
        EcdsaSigCount::Many => 3.0,
    });
    values.push(match v.input_age {
        AgeClass::SameBlock => 0.0,
        AgeClass::Within6 => 1.0,
        AgeClass::Within144 => 2.0,
        AgeClass::Older => 3.0,
        AgeClass::Indeterminate => 4.0,
    });
    values.push(match v.output_structure {
        OutputStructureType::Single => 0.0,
        OutputStructureType::Double => 1.0,
        OutputStructureType::Multi => 2.0,
        OutputStructureType::Unknown => 3.0,
    });
    // feerate_bucket is a String like "0".."9", "10-19", "20-29", ..., "100+", or
    // "Unknown". Rank by the integer lower bound; Unknown sorts last.
    values.push(feerate_rank(&v.feerate_bucket));
    // locktime_offset ordinal + NotApplicable flag
    let (lo_index, lo_na) = match v.locktime_offset {
        LocktimeOffsetType::NotApplicable => (0.0, true),
        LocktimeOffsetType::Zero => (0.0, false),
        LocktimeOffsetType::One => (1.0, false),
        LocktimeOffsetType::Two => (2.0, false),
        LocktimeOffsetType::Three => (3.0, false),
        LocktimeOffsetType::FourToSix => (4.0, false),
        LocktimeOffsetType::SevenToTwelve => (5.0, false),
        LocktimeOffsetType::ThirteenToTwentyFive => (6.0, false),
        LocktimeOffsetType::TwentySixToFifty => (7.0, false),
        LocktimeOffsetType::FiftyOneToHundred => (8.0, false),
    };
    values.push(lo_index);
    push_bool(&mut values, lo_na);

    push_onehot(
        &mut values,
        match v.change_position {
            ChangePosition::First => 0,
            ChangePosition::Last => 1,
            ChangePosition::Middle => 2,
            ChangePosition::Indeterminate => 3,
        },
        4,
    );
    push_bool(&mut values, v.change_type.is_some());

    // Group A — nSequence bit aggregates (sub-project 2)
    push_bool(&mut values, shape.bip125_rbf);
    push_bool(&mut values, shape.all_inputs_final);
    push_bool(&mut values, shape.bip68_active);
    push_bool(&mut values, shape.bip68_type_time);
    push_bool(&mut values, shape.nsequence_reserved_bits_set);
    values.push(f64::from(shape.bip68_relative_value_max));
    // Group B: dead = non-zero locktime that consensus ignores because all inputs final.
    let nlocktime_dead = v.nlocktime != NLockTimeType::Zero && shape.all_inputs_final;
    push_bool(&mut values, nlocktime_dead);
    // Group C — type counts and positional
    values.push(f64::from(shape.input_count));
    values.push(f64::from(shape.output_count));
    for c in shape.input_type_counts {
        values.push(f64::from(c));
    }
    for c in shape.output_type_counts {
        values.push(f64::from(c));
    }
    values.push(shape.first_output_type.map_or(-1.0, |t| t as usize as f64));
    values.push(shape.last_output_type.map_or(-1.0, |t| t as usize as f64));
    push_bool(&mut values, shape.first_output_matches_input_type);
    push_bool(&mut values, shape.last_output_matches_input_type);
    push_bool(&mut values, shape.outputs_type_grouped);
    values.push(shape.first_input_type.map_or(-1.0, |t| t as usize as f64));
    values.push(shape.last_input_type.map_or(-1.0, |t| t as usize as f64));
    push_bool(&mut values, shape.inputs_type_grouped);
    for t in shape.input_types_positional {
        values.push(t.map_or(-1.0, |t| t as usize as f64));
    }
    for t in shape.output_types_positional {
        values.push(t.map_or(-1.0, |t| t as usize as f64));
    }

    // Feerate / amounts / taproot (sub-project 3a)
    values.push(shape.fee_sat);
    values.push(shape.vsize);
    values.push(shape.weight);
    values.push(shape.feerate_sat_per_vb);
    values.push(shape.feerate_sat_per_kwu);
    push_bool(&mut values, shape.fee_is_multiple_of_vsize);
    values.push(shape.feerate_over_block_min);
    values.push(shape.feerate_over_prev_block_min);
    values.push(shape.max_output_value_sat);
    values.push(shape.hamming_decimal);
    values.push(shape.hamming_base2);
    values.push(shape.hamming_base3);
    values.push(shape.decimal_sig_fig_span);
    values.push(shape.hamming_decimal_min_over_outputs);
    values.push(shape.taproot_max_merkle_depth);
    values.push(shape.taproot_script_path_input_count);

    // Standardness (sub-project 3b)
    push_bool(&mut values, shape.is_standard);
    push_bool(&mut values, shape.nonstd_version);
    push_bool(&mut values, shape.nonstd_weight);
    push_bool(&mut values, shape.nonstd_output_type);
    push_bool(&mut values, shape.nonstd_dust);
    push_bool(&mut values, shape.nonstd_multi_op_return);
    push_bool(&mut values, shape.nonstd_bare_multisig);

    // Protocol markers (sub-project 3d)
    push_bool(&mut values, shape.has_inscription_envelope);
    push_bool(&mut values, shape.has_runestone);

    // Feerate (appended): "by weight" analogue of fee_is_multiple_of_vsize
    push_bool(&mut values, shape.fee_is_multiple_of_kwu);

    // Feerate (appended): sat/wu analogue, stricter than fee_is_multiple_of_kwu (implies it)
    push_bool(&mut values, shape.fee_is_multiple_of_weight);

    // Multisig configuration (sub-project 3f)
    values.push(shape.multisig_m as f64);
    values.push(shape.multisig_n as f64);
    push_bool(&mut values, shape.multisig_mixed);

    // Output value structure — denomination / coinjoin (sub-project 3f)
    values.push(shape.distinct_output_value_count as f64);
    values.push(shape.max_equal_value_output_count as f64);

    // Two more upstream fingerprints (sub-project 3g)
    push_bool(&mut values, shape.nlocktime_optin_without_use);
    push_bool(&mut values, shape.taproot_keyspend_non_default_sighash);

    // Sig-op cost (broader than the ECDSA-only `ecdsa_sigs`) and which change heuristic
    // fired — Armin's two attribute requests.
    values.push(shape.sigop_count as f64);
    push_bool(&mut values, shape.change_detected_by_round_number);
    push_bool(&mut values, shape.change_heuristics_agree);

    FeatureRow {
        txid,
        height,
        values,
    }
}

/// Rank a `feerate_bucket` string by its integer lower bound. "0".."9" → 0..9,
/// "10-19" → 10, "100+" → 100, "Unknown" → f64::from(u16::MAX) so it sorts last.
/// This is a monotonic rank for ML, not the feerate itself (that is a later sub-project).
fn feerate_rank(bucket: &str) -> f64 {
    if bucket == "Unknown" {
        return f64::from(u16::MAX);
    }
    let head = bucket.split(['-', '+']).next().unwrap_or("");
    head.parse::<u32>()
        .map(f64::from)
        .unwrap_or(f64::from(u16::MAX))
}

/// Feerate context for the block a transaction sits in and the one before it. Both
/// minima are the smallest `sat/vB` over that block's fee-paying transactions, or `None`
/// when the block has none. Computed once per block by the scan engine and consumed by
/// `tx_shape` to derive the block-relative feerate ratios.
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockContext {
    pub block_min_feerate: Option<f64>,
    pub prev_block_min_feerate: Option<f64>,
}

/// Count of non-zero decimal digits of `v` (its base-10 Hamming weight).
fn hamming_decimal(mut v: u64) -> u32 {
    let mut c = 0;
    while v > 0 {
        if !v.is_multiple_of(10) {
            c += 1;
        }
        v /= 10;
    }
    c
}

/// Count of non-zero base-3 digits of `v`.
fn hamming_base3(mut v: u64) -> u32 {
    let mut c = 0;
    while v > 0 {
        if !v.is_multiple_of(3) {
            c += 1;
        }
        v /= 3;
    }
    c
}

/// Bounded-precision measure: decimal positions spanned from the most-significant to the
/// least-significant non-zero digit, inclusive. Separates `110000000` (span 2) from
/// `100000001` (span 9) though both have decimal Hamming weight 2. Zero → 0.
fn decimal_sig_fig_span(v: u64) -> u32 {
    if v == 0 {
        return 0;
    }
    let mut lsd = 0u32;
    let mut x = v;
    while x.is_multiple_of(10) {
        x /= 10;
        lsd += 1;
    }
    let mut digits = 0u32;
    let mut y = v;
    while y > 0 {
        digits += 1;
        y /= 10;
    }
    (digits - 1) - lsd + 1
}

/// A taproot annex is a trailing witness element that starts with `0x50`.
fn is_annex(item: &[u8]) -> bool {
    item.first() == Some(&0x50)
}

/// For a taproot **script-path** spend, return `(tapscript, control_block)`: the control
/// block is the last non-annex witness element, and the tapscript the element immediately
/// before it, after skipping a trailing annex (an element starting with `0x50`). `None`
/// for fewer than 2 witness elements (key-path or empty), or an annex-only witness with
/// fewer than 3. Shared by both taproot detectors so the annex handling lives in one place.
fn tapscript_and_control(witness: &bitcoin::Witness) -> Option<(&[u8], &[u8])> {
    let items: Vec<&[u8]> = witness.iter().collect();
    if items.len() < 2 {
        return None;
    }
    let has_annex = items.last().is_some_and(|e| is_annex(e));
    let control_idx = if has_annex {
        if items.len() < 3 {
            return None;
        }
        items.len() - 2
    } else {
        items.len() - 1
    };
    Some((items[control_idx - 1], items[control_idx]))
}

/// Taproot script-path merkle depth from a witness: `Some(k)` when the spend is
/// script-path (control block of length `33 + 32k`, after skipping a trailing annex that
/// starts with `0x50`); `None` for key-path spends or malformed control blocks.
fn taproot_script_path_depth(witness: &bitcoin::Witness) -> Option<u32> {
    let (_tapscript, control) = tapscript_and_control(witness)?;
    let len = control.len();
    if len < 33 || (len - 33) % 32 != 0 {
        return None;
    }
    Some(((len - 33) / 32) as u32)
}

/// True when a taproot script-path witness's revealed tapscript contains the canonical
/// Ordinals inscription envelope `OP_FALSE OP_IF OP_PUSHBYTES_3 "ord"` — the 6-byte marker
/// `00 63 03 6f 72 64`. The tapscript is the witness element immediately before the control
/// block (the last non-annex element; an annex is a trailing element starting with `0x50`).
/// A non-script-path witness (fewer than 2 elements) has no tapscript. High-certainty on the
/// positive; not exhaustive (a fragmented or non-canonical envelope escapes).
fn has_inscription_envelope(witness: &bitcoin::Witness) -> bool {
    const ORD: [u8; 6] = [0x00, 0x63, 0x03, 0x6f, 0x72, 0x64];
    let Some((tapscript, _control)) = tapscript_and_control(witness) else {
        return false;
    };
    tapscript.windows(ORD.len()).any(|w| w == ORD)
}

/// True when an output is a Runestone: an OP_RETURN script whose second byte is
/// `OP_PUSHNUM_13` (`0x5d`) — the Runes protocol marker `6a 5d ...`.
fn is_runestone_output(spk: &bitcoin::Script) -> bool {
    spk.is_op_return() && spk.as_bytes().get(1) == Some(&0x5d)
}

/// Core's standard-transaction weight ceiling (`MAX_STANDARD_TX_WEIGHT`). A heavier tx is
/// non-standard even though it can be mined.
const MAX_STANDARD_TX_WEIGHT: u64 = 400_000;

/// Core's `GetDustThreshold` at the default `dustRelayFee` (3000 sat/kvB): the value below
/// which an output is dust. `nSize` is the output's serialized size (`8` value + `1`
/// compact-size length + script, valid for scripts under 253 bytes) plus the cost of the
/// input that would later spend it (67 for a witness program, 148 otherwise). The fee is
/// `3000 * nSize / 1000 == 3 * nSize`. P2WPKH → 294, P2PKH → 546 (Core's known values).
fn output_dust_threshold(spk: &bitcoin::Script) -> u64 {
    let spend_cost = if spk.is_witness_program() { 67 } else { 148 };
    let n_size = 9 + spk.len() as u64 + spend_cost;
    3 * n_size
}

/// A `OP_PUSHNUM_1..=16` opcode's numeric value (`0x51..=0x60` → 1..=16), else `None`.
fn op_pushnum(instr: &bitcoin::script::Instruction<'_>) -> Option<u32> {
    if let bitcoin::script::Instruction::Op(opcode) = instr {
        let code = opcode.to_u8();
        if (0x51..=0x60).contains(&code) {
            return Some(u32::from(code - 0x50));
        }
    }
    None
}

/// The `(m, n)` of a multisig script (`OP_m <pubkeys> OP_n OP_CHECKMULTISIG`): the first
/// opcode is `OP_m`, the second-to-last is `OP_n`, both `OP_PUSHNUM_1..=16` (`0x51..=0x60`).
/// `None` when the script is not multisig or the threshold opcodes are out of range.
fn multisig_m_n(spk: &bitcoin::Script) -> Option<(u32, u32)> {
    if !spk.is_multisig() {
        return None;
    }
    let instrs: Vec<_> = spk.instructions().collect::<Result<_, _>>().ok()?;
    let m = op_pushnum(instrs.first()?)?;
    let n = op_pushnum(instrs.get(instrs.len().checked_sub(2)?)?)?;
    Some((m, n))
}

/// The `(m, n)` of a taproot script-path CHECKSIGADD threshold multisig (BIP-342 `multi_a`):
/// `<pk1> OP_CHECKSIG (<pk_i> OP_CHECKSIGADD)* <m> OP_NUMEQUAL[VERIFY]`, each `<pk_i>` a
/// 32-byte x-only key. `None` when the tapscript is not this canonical shape (or the
/// threshold is `> 16`, pushed as data rather than `OP_PUSHNUM`).
fn checksigadd_m_n(tapscript: &bitcoin::Script) -> Option<(u32, u32)> {
    use bitcoin::opcodes::all::{OP_CHECKSIG, OP_CHECKSIGADD, OP_NUMEQUAL, OP_NUMEQUALVERIFY};
    use bitcoin::script::Instruction;
    let instrs: Vec<Instruction<'_>> = tapscript.instructions().collect::<Result<_, _>>().ok()?;
    if instrs.len() < 4 {
        return None; // minimum is <pk> CHECKSIG <m> NUMEQUAL
    }
    let mut n = 0u32;
    let mut i = 0;
    while i + 1 < instrs.len() {
        let Instruction::PushBytes(pk) = &instrs[i] else {
            break;
        };
        if pk.len() != 32 {
            break;
        }
        let Instruction::Op(op) = instrs[i + 1] else {
            return None;
        };
        if n == 0 {
            if op != OP_CHECKSIG {
                return None;
            }
        } else if op != OP_CHECKSIGADD {
            return None;
        }
        n += 1;
        i += 2;
    }
    if n == 0 || i + 2 != instrs.len() {
        return None;
    }
    let m = op_pushnum(&instrs[i])?;
    let Instruction::Op(end) = instrs[i + 1] else {
        return None;
    };
    if end != OP_NUMEQUAL && end != OP_NUMEQUALVERIFY {
        return None;
    }
    Some((m, n))
}

/// The `n` of a multisig script — the threshold-of count. Delegates to `multisig_m_n`.
fn bare_multisig_n(spk: &bitcoin::Script) -> Option<u32> {
    multisig_m_n(spk).map(|(_, n)| n)
}

/// The `(m, n)` config of one input's multisig, if it spends one: the prevout script itself
/// (bare multisig), else the P2SH redeem script (last scriptSig push), else the P2WSH
/// witness script (last witness item), else — for P2TR script-path spends — the revealed
/// tapscript decoded as a BIP-342 CHECKSIGADD `multi_a` threshold (see `checksigadd_m_n`).
/// Mirrors `input_subtype_class`'s navigation.
fn input_multisig_config(
    txin: &bitcoin::TxIn,
    prevout_spk: &bitcoin::Script,
) -> Option<(u32, u32)> {
    if let Some(mn) = multisig_m_n(prevout_spk) {
        return Some(mn);
    }
    if prevout_spk.is_p2sh()
        && let Some(Ok(bitcoin::script::Instruction::PushBytes(redeem))) =
            txin.script_sig.instructions().last()
        && let Some(mn) = multisig_m_n(bitcoin::Script::from_bytes(redeem.as_bytes()))
    {
        return Some(mn);
    }
    if prevout_spk.is_p2wsh()
        && let Some(ws) = txin.witness.last()
        && let Some(mn) = multisig_m_n(bitcoin::Script::from_bytes(ws))
    {
        return Some(mn);
    }
    if prevout_spk.is_p2tr()
        && let Some((tapscript, _control)) = tapscript_and_control(&txin.witness)
        && let Some(mn) = checksigadd_m_n(bitcoin::Script::from_bytes(tapscript))
    {
        return Some(mn);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aux::AuxFlags;
    use crate::vector::tests_support::classifiable_vector;
    use bitcoin::hashes::Hash;
    use bitcoin::{Amount, OutPoint, ScriptBuf, Sequence, TxIn, TxOut, Txid, Witness};
    use lumen_primitives::traits::SourcedBlock;
    use lumen_primitives::traits::block_source::SpentOutput;
    use std::collections::HashMap;

    /// A scriptPubKey that `classify_script_pubkey` maps to `t`. Total over all eight
    /// `OutputType`s so `tx_with_io` can request any type; the two the tests actually
    /// exercise (P2WPKH, P2TR) match the shapes `tests_support` already uses.
    fn spk_for(t: OutputType) -> ScriptBuf {
        let hex = match t {
            OutputType::P2pkh => "76a914000102030405060708090a0b0c0d0e0f1011121388ac".to_string(),
            OutputType::P2sh => "a914000102030405060708090a0b0c0d0e0f1011121387".to_string(),
            OutputType::P2wpkh => "0014000102030405060708090a0b0c0d0e0f10111213".to_string(),
            OutputType::P2wsh => format!("0020{}", "00".repeat(32)),
            OutputType::P2tr => {
                "5120000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f".to_string()
            }
            OutputType::OpReturn => "6a04deadbeef".to_string(),
            OutputType::P2pk => format!("21{}ac", "02".to_string() + &"11".repeat(32)),
            OutputType::NonStandard => "6351".to_string(),
        };
        let spk = ScriptBuf::from_hex(&hex).expect("valid test scriptPubKey hex");
        assert_eq!(
            classify_script_pubkey(spk.as_bytes()),
            t,
            "spk_for produced a script that classifies as the wrong type"
        );
        spk
    }

    /// A default `TxShape` for tests that exercise only the `FingerprintVector`-driven
    /// columns; its aux flags are all false and its shape columns take the cake fixture's
    /// values, neither of which those tests assert on.
    fn default_shape() -> TxShape {
        tx_shape(
            &crate::vector::tests_support::cake_like_tx(),
            &crate::vector::tests_support::cake_like_block(800_000),
            &AuxFlags::default(),
            &BlockContext::default(),
        )
    }

    fn dummy_txin(vout: u32, seq: u32) -> TxIn {
        TxIn {
            previous_output: OutPoint::new(Txid::from_raw_hash(Hash::all_zeros()), vout),
            script_sig: ScriptBuf::new(),
            sequence: Sequence(seq),
            witness: Witness::new(),
        }
    }

    /// Build a tx whose inputs have the given sequences (all spending a P2WPKH prevout)
    /// at version `ver`. Returns `(tx, block)` with every prevout resolvable.
    fn tx_with_sequences(seqs: &[u32], ver: i32) -> (bitcoin::Transaction, SourcedBlock) {
        let mut prevouts = HashMap::new();
        let mut input = Vec::new();
        for (idx, &seq) in seqs.iter().enumerate() {
            let txin = dummy_txin(idx as u32, seq);
            prevouts.insert(
                txin.previous_output,
                SpentOutput {
                    txout: TxOut {
                        value: Amount::from_sat(100_000),
                        script_pubkey: spk_for(OutputType::P2wpkh),
                    },
                    creation_height: 0,
                },
            );
            input.push(txin);
        }
        let tx = bitcoin::Transaction {
            version: bitcoin::transaction::Version(ver),
            lock_time: bitcoin::absolute::LockTime::ZERO,
            input,
            output: Vec::new(),
        };
        let mut block = bitcoin::constants::genesis_block(bitcoin::Network::Bitcoin);
        block.txdata.push(tx.clone());
        (
            tx,
            SourcedBlock {
                height: 1,
                block,
                prevouts,
            },
        )
    }

    /// Build a tx whose inputs' prevouts have `in_types` and whose outputs have
    /// `out_types`. Returns `(tx, block)` with every prevout resolvable.
    fn tx_with_io(
        in_types: &[OutputType],
        out_types: &[OutputType],
    ) -> (bitcoin::Transaction, SourcedBlock) {
        let mut prevouts = HashMap::new();
        let mut input = Vec::new();
        for (idx, &t) in in_types.iter().enumerate() {
            let txin = dummy_txin(idx as u32, 0xFFFF_FFFF);
            prevouts.insert(
                txin.previous_output,
                SpentOutput {
                    txout: TxOut {
                        value: Amount::from_sat(100_000),
                        script_pubkey: spk_for(t),
                    },
                    creation_height: 0,
                },
            );
            input.push(txin);
        }
        let output = out_types
            .iter()
            .map(|&t| TxOut {
                value: Amount::from_sat(50_000),
                script_pubkey: spk_for(t),
            })
            .collect();
        let tx = bitcoin::Transaction {
            version: bitcoin::transaction::Version(2),
            lock_time: bitcoin::absolute::LockTime::ZERO,
            input,
            output,
        };
        let mut block = bitcoin::constants::genesis_block(bitcoin::Network::Bitcoin);
        block.txdata.push(tx.clone());
        (
            tx,
            SourcedBlock {
                height: 1,
                block,
                prevouts,
            },
        )
    }

    #[test]
    fn tx_shape_sigop_count_counts_all_sig_ops_over_prevouts() {
        // Two P2WPKH-spending inputs: witness sig ops are counted (not just legacy ECDSA).
        let (tx, block) = tx_with_sequences(&[0xFFFF_FFFF, 0xFFFF_FFFF], 2);
        let s = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        let expected = tx.total_sigop_cost(|op| block.prevout(op).cloned()) as u32;
        assert_eq!(
            s.sigop_count, expected,
            "column must be the crate's total_sigop_cost"
        );
        assert!(
            s.sigop_count > 0,
            "a witness-spending tx has non-zero sig-op cost"
        );
    }

    #[test]
    fn tx_shape_change_heuristics_disagree_when_type_and_value_pick_different_outputs() {
        // Inputs unanimous P2WPKH; outputs [P2WPKH round, P2TR non-round].
        // script-type -> the single P2WPKH output (index 0); round-number -> non-round (index 1).
        let (mut tx, block) = tx_with_io(
            &[OutputType::P2wpkh, OutputType::P2wpkh],
            &[OutputType::P2wpkh, OutputType::P2tr],
        );
        tx.output[0].value = bitcoin::Amount::from_sat(50_000); // round
        tx.output[1].value = bitcoin::Amount::from_sat(49_999); // not round
        let s = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        assert!(s.change_detected_by_round_number);
        assert!(
            !s.change_heuristics_agree,
            "type picks out#0, round-number picks out#1"
        );
    }

    #[test]
    fn tx_shape_change_heuristics_agree_when_both_pick_the_same_output() {
        // Inputs unanimous P2TR; outputs [P2WPKH round, P2TR non-round].
        // script-type -> the single P2TR output (index 1); round-number -> non-round (index 1).
        let (mut tx, block) = tx_with_io(
            &[OutputType::P2tr, OutputType::P2tr],
            &[OutputType::P2wpkh, OutputType::P2tr],
        );
        tx.output[0].value = bitcoin::Amount::from_sat(50_000); // round
        tx.output[1].value = bitcoin::Amount::from_sat(49_999); // not round
        let s = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        assert!(s.change_detected_by_round_number);
        assert!(s.change_heuristics_agree, "both point to out#1");
    }

    #[test]
    fn tx_shape_change_round_number_abstains_when_both_outputs_round() {
        // tx_with_io gives every output value 50_000 (round) -> round-number heuristic abstains.
        let (tx, block) = tx_with_io(
            &[OutputType::P2wpkh],
            &[OutputType::P2wpkh, OutputType::P2tr],
        );
        let s = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        assert!(!s.change_detected_by_round_number);
        assert!(!s.change_heuristics_agree);
    }

    #[test]
    fn tx_shape_nsequence_bits() {
        // Two inputs, both RBF (0xFFFFFFFD): bip125 yes, not all-final, bip68 inactive.
        let (tx, block) = tx_with_sequences(&[0xFFFF_FFFD, 0xFFFF_FFFD], 2);
        let s = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        assert!(s.bip125_rbf);
        assert!(!s.all_inputs_final);
        assert!(!s.bip68_active);
        assert!(!s.nsequence_reserved_bits_set);

        // All inputs final.
        let (tx, block) = tx_with_sequences(&[0xFFFF_FFFF, 0xFFFF_FFFF], 2);
        let s = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        assert!(s.all_inputs_final);
        assert!(!s.bip125_rbf);

        // BIP-68 relative timelock, time units, value 5, on a v2 tx: 0x00400005.
        let (tx, block) = tx_with_sequences(&[0x0040_0005], 2);
        let s = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        assert!(s.bip68_active);
        assert!(s.bip68_type_time);
        assert_eq!(s.bip68_relative_value_max, 5);

        // Reserved bit set (bit 16 = 0x00010000), bit 31 clear, v2 → reserved flag.
        let (tx, block) = tx_with_sequences(&[0x0001_0000], 2);
        let s = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        assert!(s.nsequence_reserved_bits_set);

        // Same sequence on a v1 tx: BIP-68 does not apply, so bip68_active is false.
        let (tx, block) = tx_with_sequences(&[0x0040_0005], 1);
        let s = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        assert!(!s.bip68_active);
    }

    #[test]
    fn tx_shape_type_counts_and_positional() {
        // 2 P2WPKH inputs; outputs [P2WPKH, P2TR] (payment then change-ish).
        let (tx, block) = tx_with_io(
            &[OutputType::P2wpkh, OutputType::P2wpkh],
            &[OutputType::P2wpkh, OutputType::P2tr],
        );
        let s = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        assert_eq!(s.input_count, 2);
        assert_eq!(s.output_count, 2);
        assert_eq!(s.input_type_counts[OutputType::P2wpkh as usize], 2);
        assert_eq!(s.output_type_counts[OutputType::P2tr as usize], 1);
        assert_eq!(s.first_output_type, Some(OutputType::P2wpkh));
        assert_eq!(s.last_output_type, Some(OutputType::P2tr));
        assert!(s.first_output_matches_input_type); // first output P2WPKH == uniform input
        assert!(!s.last_output_matches_input_type); // last output P2TR != input
        assert!(s.outputs_type_grouped); // [P2WPKH, P2TR] is grouped

        // Interleaved outputs [P2WPKH, P2TR, P2WPKH] → not grouped.
        let (tx, block) = tx_with_io(
            &[OutputType::P2wpkh],
            &[OutputType::P2wpkh, OutputType::P2tr, OutputType::P2wpkh],
        );
        let s = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        assert!(!s.outputs_type_grouped);
    }

    #[test]
    fn row_width_matches_column_count() {
        let v = classifiable_vector();
        let row = feature_row(&v, &default_shape(), "deadbeef".to_string(), 800_000);
        assert_eq!(row.values.len(), COLUMN_NAMES.len() - 2);
        assert_eq!(COLUMN_NAMES[0], "txid");
        assert_eq!(COLUMN_NAMES[1], "block_height");
    }

    #[test]
    fn booleans_and_tristate_encode_as_expected() {
        let mut v = classifiable_vector();
        v.op_return = true;
        v.low_r = Tri::Yes;
        v.uncompressed_pubkey = Tri::Indeterminate;
        let row = feature_row(&v, &default_shape(), "x".to_string(), 1);
        let col =
            |name: &str| row.values[COLUMN_NAMES[2..].iter().position(|c| *c == name).unwrap()];
        assert_eq!(col("op_return"), 1.0);
        assert_eq!(col("low_r_yes"), 1.0);
        assert_eq!(col("low_r_indeterminate"), 0.0);
        assert_eq!(col("uncompressed_pubkey_yes"), 0.0);
        assert_eq!(col("uncompressed_pubkey_indeterminate"), 1.0);
    }

    #[test]
    fn nominal_axes_one_hot_exactly_one_column() {
        use crate::vector::OrderClass;
        use lumen_fingerprints_lib::NSequenceType;
        let mut v = classifiable_vector();
        v.nsequence = NSequenceType::Rbf;
        v.input_order = OrderClass::Bip69;
        let row = feature_row(&v, &default_shape(), "x".to_string(), 1);
        let col =
            |name: &str| row.values[COLUMN_NAMES[2..].iter().position(|c| *c == name).unwrap()];
        assert_eq!(col("nsequence_rbf"), 1.0);
        assert_eq!(col("nsequence_max"), 0.0);
        assert_eq!(col("input_order_bip69"), 1.0);
        assert_eq!(col("input_order_indeterminate"), 0.0);
    }

    #[test]
    fn version_other_catches_nonstandard() {
        let mut v = classifiable_vector();
        v.version = 2_771_273;
        let row = feature_row(&v, &default_shape(), "x".to_string(), 1);
        let col =
            |name: &str| row.values[COLUMN_NAMES[2..].iter().position(|c| *c == name).unwrap()];
        assert_eq!(col("version_other"), 1.0);
        assert_eq!(col("version_2"), 0.0);
    }

    #[test]
    fn ordinal_presence_and_heuristic_columns() {
        let mut v = classifiable_vector();
        v.ecdsa_sigs = EcdsaSigCount::Few; // index 2
        v.input_age = AgeClass::Within144; // index 2
        v.output_structure = OutputStructureType::Double; // index 1
        v.locktime_offset = LocktimeOffsetType::NotApplicable;
        v.feerate_bucket = "10-19".to_string();
        v.input_types = InputTypeClass::Uniform(OutputType::P2wpkh);
        v.output_types = vec![OutputType::P2wpkh, OutputType::OpReturn];
        v.change_position = ChangePosition::Last;
        v.change_type = Some(OutputType::P2wpkh);
        let row = feature_row(&v, &default_shape(), "x".to_string(), 1);
        let col =
            |name: &str| row.values[COLUMN_NAMES[2..].iter().position(|c| *c == name).unwrap()];
        assert_eq!(col("ecdsa_sigs_index"), 2.0);
        assert_eq!(col("input_age_index"), 2.0);
        assert_eq!(col("output_structure_index"), 1.0);
        assert_eq!(col("locktime_offset_index"), 0.0);
        assert_eq!(col("locktime_offset_not_applicable"), 1.0);
        assert_eq!(col("change_position_last"), 1.0);
        assert_eq!(col("change_type_present"), 1.0);
    }

    #[test]
    fn feature_row_new_columns_and_aux_from_shape() {
        let v = classifiable_vector();
        let mut shape = tx_shape(
            &crate::vector::tests_support::cake_like_tx(),
            &crate::vector::tests_support::cake_like_block(800_000),
            &AuxFlags {
                address_reuse: true,
                ..AuxFlags::default()
            },
            &BlockContext::default(),
        );
        shape.bip125_rbf = true;
        shape.all_inputs_final = false;
        let row = feature_row(&v, &shape, "x".to_string(), 1);
        let col =
            |name: &str| row.values[COLUMN_NAMES[2..].iter().position(|c| *c == name).unwrap()];
        // new nSequence columns present and set
        assert_eq!(col("bip125_rbf"), 1.0);
        assert_eq!(col("all_inputs_final"), 0.0);
        // aux now flows from shape.aux, not a placeholder 0.0
        assert_eq!(col("address_reuse"), 1.0);
        // removed presence columns are gone
        assert!(!COLUMN_NAMES.contains(&"input_has_p2wpkh"));
        assert!(!COLUMN_NAMES.contains(&"output_has_p2tr"));
        // width invariant holds
        assert_eq!(row.values.len(), COLUMN_NAMES.len() - 2);
    }

    #[test]
    fn feature_row_feerate_amount_taproot_columns() {
        let v = classifiable_vector();
        let ctx = BlockContext {
            block_min_feerate: Some(2.0),
            prev_block_min_feerate: Some(4.0),
        };
        let shape = tx_shape(
            &crate::vector::tests_support::cake_like_tx(),
            &crate::vector::tests_support::cake_like_block(800_000),
            &AuxFlags::default(),
            &ctx,
        );
        let row = feature_row(&v, &shape, "x".to_string(), 1);
        let col =
            |name: &str| row.values[COLUMN_NAMES[2..].iter().position(|c| *c == name).unwrap()];
        // presence + wiring: fee/vsize/weight positive, ratios computed from ctx
        assert!(col("fee_sat") > 0.0);
        assert!(col("vsize") > 0.0);
        assert!(col("weight") > 0.0);
        assert!((col("feerate_sat_per_vb") - shape.feerate_sat_per_vb).abs() < 1e-9);
        // block-min ratio = rate / 2.0 ; prev-min ratio = rate / 4.0
        assert!((col("feerate_over_block_min") - shape.feerate_sat_per_vb / 2.0).abs() < 1e-9);
        assert!((col("feerate_over_prev_block_min") - shape.feerate_sat_per_vb / 4.0).abs() < 1e-9);
        assert_eq!(row.values.len(), COLUMN_NAMES.len() - 2);

        // absent block min → -1.0 sentinel
        let ctx0 = BlockContext {
            block_min_feerate: None,
            prev_block_min_feerate: None,
        };
        let shape0 = tx_shape(
            &crate::vector::tests_support::cake_like_tx(),
            &crate::vector::tests_support::cake_like_block(800_000),
            &AuxFlags::default(),
            &ctx0,
        );
        let row0 = feature_row(&v, &shape0, "x".to_string(), 1);
        assert_eq!(
            row0.values[COLUMN_NAMES[2..]
                .iter()
                .position(|c| *c == "feerate_over_block_min")
                .unwrap()],
            -1.0
        );
    }

    #[test]
    fn fee_is_multiple_of_kwu_matches_formula_and_a_known_case() {
        use crate::vector::tx_fee_vsize_weight;

        // Wired to the documented "sat/kwu is an exact integer" formula, computed over
        // the same integer fee/weight `tx_shape` uses (u128 to mirror the field).
        let tx = crate::vector::tests_support::cake_like_tx();
        let block = crate::vector::tests_support::cake_like_block(800_000);
        let shape = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        let (fee, _vsize, weight) = tx_fee_vsize_weight(&tx, &block).expect("cake fee resolves");
        let expected = weight != 0 && (u128::from(fee) * 1000) % u128::from(weight) == 0;
        assert_eq!(shape.fee_is_multiple_of_kwu, expected);

        // Known positive: tune the single output so fee == weight, which makes
        // fee*1000 an exact multiple of weight. `tx_with_io` gives two 100_000-sat
        // P2WPKH prevouts (input_sum 200_000); the output value does not affect weight,
        // so weight is stable across the edit, and the prevout map (keyed by outpoint)
        // is untouched by an output-value change.
        let (mut tx2, block2) = tx_with_io(
            &[OutputType::P2wpkh, OutputType::P2wpkh],
            &[OutputType::P2wpkh],
        );
        let w = tx2.weight().to_wu();
        tx2.output[0].value = bitcoin::Amount::from_sat(200_000 - w);
        let shape2 = tx_shape(
            &tx2,
            &block2,
            &AuxFlags::default(),
            &BlockContext::default(),
        );
        let (fee2, _v2, w2) = tx_fee_vsize_weight(&tx2, &block2).expect("fee resolves");
        assert_eq!(fee2, w2); // fee equals weight by construction
        assert!(shape2.fee_is_multiple_of_kwu);
    }

    #[test]
    fn fee_is_multiple_of_weight_column() {
        use crate::vector::tx_fee_vsize_weight;

        // Wired to the documented "sat/wu is an exact integer" formula, computed over
        // the same integer fee/weight `tx_shape` uses.
        let tx = crate::vector::tests_support::cake_like_tx();
        let block = crate::vector::tests_support::cake_like_block(800_000);
        let shape = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        let (fee, _vsize, weight) = tx_fee_vsize_weight(&tx, &block).expect("cake fee resolves");
        let expected = weight != 0 && fee % weight == 0;
        assert_eq!(shape.fee_is_multiple_of_weight, expected);

        // Known positive: reuse the fee == weight construction from the kwu test above.
        // fee == weight trivially satisfies fee % weight == 0, so this is also a
        // fee_is_multiple_of_weight positive (and, since fee*1000 % weight == 0 follows
        // from fee % weight == 0, it demonstrates the "stricter, implies kwu" relationship).
        let (mut tx2, block2) = tx_with_io(
            &[OutputType::P2wpkh, OutputType::P2wpkh],
            &[OutputType::P2wpkh],
        );
        let w = tx2.weight().to_wu();
        tx2.output[0].value = bitcoin::Amount::from_sat(200_000 - w);
        let shape2 = tx_shape(
            &tx2,
            &block2,
            &AuxFlags::default(),
            &BlockContext::default(),
        );
        let (fee2, _v2, w2) = tx_fee_vsize_weight(&tx2, &block2).expect("fee resolves");
        assert_eq!(fee2, w2); // fee equals weight by construction
        assert!(shape2.fee_is_multiple_of_weight);
        assert!(shape2.fee_is_multiple_of_kwu); // implication holds

        // Row-level check: the column value matches the shape field.
        let v = classifiable_vector();
        let row = feature_row(&v, &shape2, "y".to_string(), 2);
        let col_value = row.values[COLUMN_NAMES[2..]
            .iter()
            .position(|c| *c == "fee_is_multiple_of_weight")
            .unwrap()];
        assert_eq!(col_value, 1.0);
    }

    #[test]
    fn feerate_rank_parses_buckets_and_falls_back_on_unknown() {
        assert_eq!(feerate_rank("0"), 0.0);
        assert_eq!(feerate_rank("10-19"), 10.0);
        assert_eq!(feerate_rank("100+"), 100.0);
        assert_eq!(feerate_rank("Unknown"), f64::from(u16::MAX));
        assert_eq!(feerate_rank("garbage"), f64::from(u16::MAX));
    }

    #[test]
    fn hamming_and_sig_fig_span() {
        assert_eq!(super::hamming_decimal(100_000_001), 2);
        assert_eq!(super::hamming_decimal(110_000_000), 2);
        assert_eq!(super::hamming_decimal(0), 0);
        assert_eq!(super::hamming_decimal(100_000_000), 1); // a round power of ten
        assert_eq!(super::hamming_decimal(34_567_891), 8); // a high-precision amount
        assert_eq!(super::hamming_base3(0), 0);
        assert_eq!(super::hamming_base3(9), 1); // 9 = 100_3
        assert_eq!(super::hamming_base3(5), 2); // 5 = 12_3
        assert_eq!(super::decimal_sig_fig_span(110_000_000), 2);
        assert_eq!(super::decimal_sig_fig_span(100_000_001), 9);
        assert_eq!(super::decimal_sig_fig_span(0), 0);
        assert_eq!(super::decimal_sig_fig_span(7), 1);
    }

    #[test]
    fn taproot_depth_from_control_block() {
        use bitcoin::Witness;
        // key-path: single 64-byte element → not script-path.
        let mut kp = Witness::new();
        kp.push(vec![0u8; 64]);
        assert_eq!(super::taproot_script_path_depth(&kp), None);
        // script-path depth 0: [script, control(33)].
        let mut sp0 = Witness::new();
        sp0.push(vec![0xacu8; 1]);
        sp0.push(vec![0xc0u8; 33]);
        assert_eq!(super::taproot_script_path_depth(&sp0), Some(0));
        // depth 3: control length 33 + 32*3 = 129.
        let mut sp3 = Witness::new();
        sp3.push(vec![0xacu8; 1]);
        sp3.push(vec![0xc0u8; 129]);
        assert_eq!(super::taproot_script_path_depth(&sp3), Some(3));
        // annex present (last element starts 0x50): control is second-to-last.
        let mut ann = Witness::new();
        ann.push(vec![0xacu8; 1]);
        ann.push(vec![0xc0u8; 33]);
        ann.push(vec![0x50u8, 1, 2]); // annex
        assert_eq!(super::taproot_script_path_depth(&ann), Some(0));
        // malformed control length (not 33+32k) → None.
        let mut bad = Witness::new();
        bad.push(vec![0xacu8; 1]);
        bad.push(vec![0xc0u8; 40]);
        assert_eq!(super::taproot_script_path_depth(&bad), None);
    }

    #[test]
    fn inscription_envelope_detection() {
        use bitcoin::Witness;
        let envelope = [0x00u8, 0x63, 0x03, 0x6f, 0x72, 0x64, 0x00, 0x01, 0x02]; // OP_FALSE OP_IF "ord" ...
        // script-path spend: [tapscript, control_block]
        let mut sp = Witness::new();
        sp.push(envelope);
        sp.push(vec![0xc0u8; 33]);
        assert!(super::has_inscription_envelope(&sp));

        // script-path without the envelope
        let mut plain = Witness::new();
        plain.push(vec![0xacu8; 10]);
        plain.push(vec![0xc0u8; 33]);
        assert!(!super::has_inscription_envelope(&plain));

        // key-path (single element) → no tapscript
        let mut kp = Witness::new();
        kp.push(vec![0u8; 64]);
        assert!(!super::has_inscription_envelope(&kp));

        // annex present: [tapscript(envelope), control, annex] → tapscript is items[len-3]
        let mut ann = Witness::new();
        ann.push(envelope);
        ann.push(vec![0xc0u8; 33]);
        ann.push(vec![0x50u8, 1, 2]);
        assert!(super::has_inscription_envelope(&ann));
    }

    #[test]
    fn runestone_detection() {
        use bitcoin::ScriptBuf;
        // OP_RETURN OP_PUSHNUM_13 <data> = 0x6a 0x5d ...
        let runestone = ScriptBuf::from_bytes(vec![0x6a, 0x5d, 0x01, 0x02]);
        assert!(super::is_runestone_output(&runestone));
        // plain OP_RETURN (no 0x5d second byte)
        let plain = ScriptBuf::from_bytes(vec![0x6a, 0x04, 1, 2, 3, 4]);
        assert!(!super::is_runestone_output(&plain));
        // non-OP_RETURN
        let p2wpkh = ScriptBuf::new_p2wpkh(&bitcoin::WPubkeyHash::from_byte_array([0u8; 20]));
        assert!(!super::is_runestone_output(&p2wpkh));
    }

    #[test]
    fn dust_threshold_matches_core() {
        use bitcoin::ScriptBuf;
        // P2WPKH: OP_0 <20 bytes> → 22-byte spk, witness program → 3*(9+22+67) = 294.
        let p2wpkh = ScriptBuf::new_p2wpkh(&bitcoin::WPubkeyHash::from_byte_array([0u8; 20]));
        assert_eq!(super::output_dust_threshold(&p2wpkh), 294);
        // P2PKH: 25-byte spk, non-witness → 3*(9+25+148) = 546.
        let p2pkh = ScriptBuf::new_p2pkh(&bitcoin::PubkeyHash::from_byte_array([0u8; 20]));
        assert_eq!(super::output_dust_threshold(&p2pkh), 546);
    }

    #[test]
    fn bare_multisig_arity() {
        use bitcoin::ScriptBuf;
        // Build a 2-of-3 bare multisig: OP_2 <33><33><33> OP_3 OP_CHECKMULTISIG.
        let key = [0x02u8; 33];
        let ms_2of3 = bitcoin::blockdata::script::Builder::new()
            .push_opcode(bitcoin::opcodes::all::OP_PUSHNUM_2)
            .push_slice(key)
            .push_slice(key)
            .push_slice(key)
            .push_opcode(bitcoin::opcodes::all::OP_PUSHNUM_3)
            .push_opcode(bitcoin::opcodes::all::OP_CHECKMULTISIG)
            .into_script();
        assert_eq!(super::bare_multisig_n(&ms_2of3), Some(3));
        // 4-of-5 → Some(5).
        let ms_4of5 = bitcoin::blockdata::script::Builder::new()
            .push_opcode(bitcoin::opcodes::all::OP_PUSHNUM_4)
            .push_slice(key)
            .push_slice(key)
            .push_slice(key)
            .push_slice(key)
            .push_slice(key)
            .push_opcode(bitcoin::opcodes::all::OP_PUSHNUM_5)
            .push_opcode(bitcoin::opcodes::all::OP_CHECKMULTISIG)
            .into_script();
        assert_eq!(super::bare_multisig_n(&ms_4of5), Some(5));
        // P2WPKH is not bare multisig.
        let p2wpkh = ScriptBuf::new_p2wpkh(&bitcoin::WPubkeyHash::from_byte_array([0u8; 20]));
        assert_eq!(super::bare_multisig_n(&p2wpkh), None);
    }

    #[test]
    fn multisig_m_n_extracts_both() {
        use bitcoin::opcodes::all::{OP_CHECKMULTISIG, OP_PUSHNUM_2, OP_PUSHNUM_3};
        use bitcoin::script::Builder;
        // OP_2 <3 * 33-byte pushes> OP_3 OP_CHECKMULTISIG
        let mut b = Builder::new().push_opcode(OP_PUSHNUM_2);
        for _ in 0..3 {
            b = b.push_slice([0x02u8; 33]);
        }
        let script = b
            .push_opcode(OP_PUSHNUM_3)
            .push_opcode(OP_CHECKMULTISIG)
            .into_script();
        assert_eq!(super::multisig_m_n(&script), Some((2, 3)));
        // bare_multisig_n still returns n after the refactor.
        assert_eq!(super::bare_multisig_n(&script), Some(3));
        // Non-multisig.
        let p2wpkh =
            bitcoin::ScriptBuf::new_p2wpkh(&bitcoin::WPubkeyHash::from_byte_array([0u8; 20]));
        assert_eq!(super::multisig_m_n(&p2wpkh), None);
    }

    #[test]
    fn checksigadd_m_n_parses_taproot_multisig() {
        use bitcoin::opcodes::all::{OP_CHECKSIG, OP_CHECKSIGADD, OP_NUMEQUAL, OP_PUSHNUM_2};
        use bitcoin::script::Builder;
        // 2-of-3 multi_a: <pk1> CHECKSIG <pk2> CHECKSIGADD <pk3> CHECKSIGADD OP_2 NUMEQUAL
        let s = Builder::new()
            .push_slice([0x01u8; 32])
            .push_opcode(OP_CHECKSIG)
            .push_slice([0x02u8; 32])
            .push_opcode(OP_CHECKSIGADD)
            .push_slice([0x03u8; 32])
            .push_opcode(OP_CHECKSIGADD)
            .push_opcode(OP_PUSHNUM_2)
            .push_opcode(OP_NUMEQUAL)
            .into_script();
        assert_eq!(super::checksigadd_m_n(&s), Some((2, 3)));
        // Not a multi_a script (an inscription-ish tapscript) → None.
        let not = Builder::new().push_slice([0x00u8; 5]).into_script();
        assert_eq!(super::checksigadd_m_n(&not), None);
    }

    #[test]
    fn multisig_config_columns() {
        use bitcoin::opcodes::all::{OP_CHECKMULTISIG, OP_PUSHNUM_2, OP_PUSHNUM_3};
        use bitcoin::script::Builder;
        use bitcoin::{Amount, OutPoint, ScriptBuf, Sequence, TxIn, TxOut, Witness};
        use lumen_primitives::traits::SourcedBlock;
        use lumen_primitives::traits::block_source::SpentOutput;
        use std::collections::HashMap;

        // Bare 2-of-3 multisig prevout spent by input 0.
        let mut b = Builder::new().push_opcode(OP_PUSHNUM_2);
        for _ in 0..3 {
            b = b.push_slice([0x02u8; 33]);
        }
        let ms = b
            .push_opcode(OP_PUSHNUM_3)
            .push_opcode(OP_CHECKMULTISIG)
            .into_script();

        let txin = TxIn {
            previous_output: OutPoint::new(
                bitcoin::Txid::from_raw_hash(bitcoin::hashes::Hash::all_zeros()),
                0,
            ),
            script_sig: ScriptBuf::new(),
            sequence: Sequence(0xffffffff),
            witness: Witness::new(),
        };
        let tx = Transaction {
            version: bitcoin::transaction::Version(2),
            lock_time: bitcoin::absolute::LockTime::ZERO,
            input: vec![txin],
            output: vec![TxOut {
                value: Amount::from_sat(1000),
                script_pubkey: ScriptBuf::new(),
            }],
        };
        let mut prevouts = HashMap::new();
        prevouts.insert(
            tx.input[0].previous_output,
            SpentOutput {
                txout: TxOut {
                    value: Amount::from_sat(5000),
                    script_pubkey: ms.clone(),
                },
                creation_height: 799_990,
            },
        );
        let mut block = bitcoin::constants::genesis_block(bitcoin::Network::Bitcoin);
        block.txdata.push(tx.clone());
        let block = SourcedBlock {
            height: 800_000,
            block,
            prevouts,
        };

        let shape = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        let row = feature_row(
            &crate::vector::tests_support::classifiable_vector(),
            &shape,
            "x".to_string(),
            1,
        );
        let col =
            |name: &str| row.values[COLUMN_NAMES[2..].iter().position(|c| *c == name).unwrap()];
        assert_eq!(col("multisig_m"), 2.0);
        assert_eq!(col("multisig_n"), 3.0);
        assert_eq!(col("multisig_mixed"), 0.0);
        assert_eq!(row.values.len(), COLUMN_NAMES.len() - 2);
    }

    /// Exercises the P2SH-redeem and P2WSH-witness navigation branches of
    /// `input_multisig_config` (the bare-prevout branch is covered above), plus the
    /// `multisig_mixed` path and dominant-by-largest-`n` selection.
    #[test]
    fn input_multisig_config_p2sh_p2wsh_and_mixed() {
        use bitcoin::opcodes::all::{OP_CHECKMULTISIG, OP_PUSHNUM_1, OP_PUSHNUM_2, OP_PUSHNUM_3};
        use bitcoin::script::{Builder, PushBytesBuf};
        use bitcoin::{Amount, OutPoint, ScriptBuf, Sequence, TxIn, TxOut, Witness};
        use lumen_primitives::traits::SourcedBlock;
        use lumen_primitives::traits::block_source::SpentOutput;
        use std::collections::HashMap;

        // Build an m-of-n bare multisig script (n dummy 33-byte pubkey pushes).
        let msig = |m_op, n: usize, n_op| {
            let mut b = Builder::new().push_opcode(m_op);
            for _ in 0..n {
                b = b.push_slice([0x02u8; 33]);
            }
            b.push_opcode(n_op)
                .push_opcode(OP_CHECKMULTISIG)
                .into_script()
        };
        let redeem_2of3 = msig(OP_PUSHNUM_2, 3, OP_PUSHNUM_3);
        let wscript_1of2 = msig(OP_PUSHNUM_1, 2, OP_PUSHNUM_2);

        let outpoint = |v: u32| {
            OutPoint::new(
                bitcoin::Txid::from_raw_hash(bitcoin::hashes::Hash::all_zeros()),
                v,
            )
        };

        // P2SH input: prevout is P2SH(redeem); scriptSig's last push is the redeem script.
        let p2sh_spk = ScriptBuf::new_p2sh(&redeem_2of3.script_hash());
        let p2sh_in = TxIn {
            previous_output: outpoint(0),
            script_sig: Builder::new()
                .push_slice(PushBytesBuf::try_from(redeem_2of3.to_bytes()).unwrap())
                .into_script(),
            sequence: Sequence(0xffffffff),
            witness: Witness::new(),
        };
        assert_eq!(
            super::input_multisig_config(&p2sh_in, &p2sh_spk),
            Some((2, 3))
        );

        // P2WSH input: prevout is P2WSH(wscript); witness's last item is the witness script.
        let p2wsh_spk = ScriptBuf::new_p2wsh(&wscript_1of2.wscript_hash());
        let p2wsh_in = TxIn {
            previous_output: outpoint(1),
            script_sig: ScriptBuf::new(),
            sequence: Sequence(0xffffffff),
            witness: Witness::from_slice(&[wscript_1of2.to_bytes()]),
        };
        assert_eq!(
            super::input_multisig_config(&p2wsh_in, &p2wsh_spk),
            Some((1, 2))
        );

        // A plain P2WPKH input resolves to no multisig config.
        let p2wpkh_spk = ScriptBuf::new_p2wpkh(&bitcoin::WPubkeyHash::from_byte_array([0u8; 20]));
        let plain_in = TxIn {
            previous_output: outpoint(2),
            script_sig: ScriptBuf::new(),
            sequence: Sequence(0xffffffff),
            witness: Witness::from_slice(&[vec![0u8; 64]]),
        };
        assert_eq!(super::input_multisig_config(&plain_in, &p2wpkh_spk), None);

        // tx_shape over the 2-of-3 (P2SH) and 1-of-2 (P2WSH) inputs → mixed, dominant (2,3).
        let tx = Transaction {
            version: bitcoin::transaction::Version(2),
            lock_time: bitcoin::absolute::LockTime::ZERO,
            input: vec![p2sh_in, p2wsh_in],
            output: vec![TxOut {
                value: Amount::from_sat(1000),
                script_pubkey: ScriptBuf::new(),
            }],
        };
        let mut prevouts = HashMap::new();
        prevouts.insert(
            outpoint(0),
            SpentOutput {
                txout: TxOut {
                    value: Amount::from_sat(5000),
                    script_pubkey: p2sh_spk,
                },
                creation_height: 799_990,
            },
        );
        prevouts.insert(
            outpoint(1),
            SpentOutput {
                txout: TxOut {
                    value: Amount::from_sat(5000),
                    script_pubkey: p2wsh_spk,
                },
                creation_height: 799_990,
            },
        );
        let mut block = bitcoin::constants::genesis_block(bitcoin::Network::Bitcoin);
        block.txdata.push(tx.clone());
        let block = SourcedBlock {
            height: 800_000,
            block,
            prevouts,
        };

        let shape = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        assert_eq!(shape.multisig_m, 2); // dominant = the input with the largest n (3)
        assert_eq!(shape.multisig_n, 3);
        assert!(shape.multisig_mixed); // (2,3) and (1,2) are distinct configs
    }

    #[test]
    fn input_multisig_config_decodes_taproot_checksigadd() {
        use bitcoin::opcodes::all::{OP_CHECKSIG, OP_CHECKSIGADD, OP_NUMEQUAL, OP_PUSHNUM_2};
        use bitcoin::script::Builder;
        use bitcoin::{Amount, OutPoint, ScriptBuf, Sequence, TxIn, TxOut, Witness};
        // 2-of-3 multi_a tapscript.
        let tapscript = Builder::new()
            .push_slice([0x01u8; 32])
            .push_opcode(OP_CHECKSIG)
            .push_slice([0x02u8; 32])
            .push_opcode(OP_CHECKSIGADD)
            .push_slice([0x03u8; 32])
            .push_opcode(OP_CHECKSIGADD)
            .push_opcode(OP_PUSHNUM_2)
            .push_opcode(OP_NUMEQUAL)
            .into_script();
        // Script-path witness: [signatures..., tapscript, control_block]. Control block is
        // 33 bytes (leaf version + internal key), no annex.
        let mut witness = Witness::new();
        witness.push(vec![0u8; 64]); // a signature
        witness.push(tapscript.as_bytes());
        witness.push(vec![0xc0u8; 33]); // control block
        let p2tr = crate::vector::tests_support::p2tr_spk();
        let txin = TxIn {
            previous_output: OutPoint::new(
                bitcoin::Txid::from_raw_hash(bitcoin::hashes::Hash::all_zeros()),
                0,
            ),
            script_sig: ScriptBuf::new(),
            sequence: Sequence(0xffffffff),
            witness,
        };
        assert_eq!(super::input_multisig_config(&txin, &p2tr), Some((2, 3)));

        // Full tx_shape path: multisig_m == 2, multisig_n == 3.
        let tx = Transaction {
            version: bitcoin::transaction::Version(2),
            lock_time: bitcoin::absolute::LockTime::ZERO,
            input: vec![txin],
            output: vec![TxOut {
                value: Amount::from_sat(1000),
                script_pubkey: ScriptBuf::new(),
            }],
        };
        let mut prevouts = std::collections::HashMap::new();
        prevouts.insert(
            tx.input[0].previous_output,
            lumen_primitives::traits::block_source::SpentOutput {
                txout: TxOut {
                    value: Amount::from_sat(5000),
                    script_pubkey: p2tr,
                },
                creation_height: 799_990,
            },
        );
        let mut block = bitcoin::constants::genesis_block(bitcoin::Network::Bitcoin);
        block.txdata.push(tx.clone());
        let block = lumen_primitives::traits::SourcedBlock {
            height: 800_000,
            block,
            prevouts,
        };
        let shape = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        assert_eq!(shape.multisig_m, 2);
        assert_eq!(shape.multisig_n, 3);
    }

    #[test]
    fn output_value_structure_columns() {
        use bitcoin::{Amount, OutPoint, ScriptBuf, Sequence, TxIn, TxOut, Witness};
        let mk_out = |v: u64| TxOut {
            value: Amount::from_sat(v),
            script_pubkey: ScriptBuf::new(),
        };
        let tx = Transaction {
            version: bitcoin::transaction::Version(2),
            lock_time: bitcoin::absolute::LockTime::ZERO,
            input: vec![TxIn {
                previous_output: OutPoint::new(
                    bitcoin::Txid::from_raw_hash(bitcoin::hashes::Hash::all_zeros()),
                    0,
                ),
                script_sig: ScriptBuf::new(),
                sequence: Sequence(0xffffffff),
                witness: Witness::from_slice(&[vec![0u8; 71], vec![0u8; 33]]),
            }],
            output: vec![mk_out(1000), mk_out(1000), mk_out(1000), mk_out(500)],
        };
        let block = crate::vector::tests_support::block_with(tx.clone(), 800_000);
        let shape = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        let row = feature_row(
            &crate::vector::tests_support::classifiable_vector(),
            &shape,
            "x".to_string(),
            1,
        );
        let col =
            |name: &str| row.values[COLUMN_NAMES[2..].iter().position(|c| *c == name).unwrap()];
        // values [1000,1000,1000,500] → 2 distinct, largest equal-group = 3
        assert_eq!(col("distinct_output_value_count"), 2.0);
        assert_eq!(col("max_equal_value_output_count"), 3.0);
        assert_eq!(row.values.len(), COLUMN_NAMES.len() - 2);
    }

    #[test]
    fn standardness_fields() {
        // A normal cake-like tx is standard.
        let normal = tx_shape(
            &crate::vector::tests_support::cake_like_tx(),
            &crate::vector::tests_support::cake_like_block(800_000),
            &AuxFlags::default(),
            &BlockContext::default(),
        );
        assert!(normal.is_standard);
        assert!(!normal.nonstd_version);
        assert!(!normal.nonstd_weight);
        assert!(!normal.nonstd_dust);
        assert!(!normal.nonstd_multi_op_return);

        // Below-dust P2WPKH output (100 sat < 294) → nonstd_dust, not standard.
        let mut dusty = crate::vector::tests_support::cake_like_tx();
        if let Some(o) = dusty
            .output
            .iter_mut()
            .find(|o| !o.script_pubkey.is_op_return())
        {
            o.value = Amount::from_sat(100);
        }
        let block = crate::vector::tests_support::block_with(dusty.clone(), 800_000);
        let s = tx_shape(
            &dusty,
            &block,
            &AuxFlags::default(),
            &BlockContext::default(),
        );
        assert!(s.nonstd_dust);
        assert!(!s.is_standard);

        // Non-standard version (5 ∉ {1,2,3}) → nonstd_version, not standard.
        let mut oddver = crate::vector::tests_support::cake_like_tx();
        oddver.version = bitcoin::transaction::Version(5);
        let block = crate::vector::tests_support::block_with(oddver.clone(), 800_000);
        let s = tx_shape(
            &oddver,
            &block,
            &AuxFlags::default(),
            &BlockContext::default(),
        );
        assert!(s.nonstd_version);
        assert!(!s.is_standard);

        // Two OP_RETURN outputs → nonstd_multi_op_return, not standard. OP_RETURN
        // outputs are exempt from the dust check, so nonstd_dust stays false.
        let mut multi = crate::vector::tests_support::cake_like_tx();
        let op_ret = ScriptBuf::from_hex("6a04deadbeef").expect("valid op_return hex");
        multi.output = vec![
            TxOut {
                value: Amount::from_sat(0),
                script_pubkey: op_ret.clone(),
            },
            TxOut {
                value: Amount::from_sat(0),
                script_pubkey: op_ret,
            },
        ];
        let block = crate::vector::tests_support::block_with(multi.clone(), 800_000);
        let s = tx_shape(
            &multi,
            &block,
            &AuxFlags::default(),
            &BlockContext::default(),
        );
        assert!(s.nonstd_multi_op_return);
        assert!(!s.nonstd_dust);
        assert!(!s.is_standard);
    }

    #[test]
    fn protocol_columns() {
        // A tx with a Runestone output → has_runestone true, has_inscription_envelope false.
        let mut tx = crate::vector::tests_support::cake_like_tx();
        tx.output = vec![bitcoin::TxOut {
            value: bitcoin::Amount::from_sat(0),
            script_pubkey: bitcoin::ScriptBuf::from_bytes(vec![0x6a, 0x5d, 0x01, 0x02]),
        }];
        let block = crate::vector::tests_support::block_with(tx.clone(), 800_000);
        let shape = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        let row = feature_row(&classifiable_vector(), &shape, "x".to_string(), 1);
        let col =
            |name: &str| row.values[COLUMN_NAMES[2..].iter().position(|c| *c == name).unwrap()];
        assert_eq!(col("has_runestone"), 1.0);
        assert_eq!(col("has_inscription_envelope"), 0.0);
        assert_eq!(row.values.len(), COLUMN_NAMES.len() - 2);
    }

    #[test]
    fn standardness_bare_multisig_routing() {
        // A standard 2-of-3 bare multisig output (well above dust) must be routed
        // through `bare_multisig_n`'s Some(3) arm, not the generic output-type check:
        // is_standard true, nonstd_output_type false, nonstd_bare_multisig false.
        let key = [0x02u8; 33];
        let ms_2of3 = bitcoin::blockdata::script::Builder::new()
            .push_opcode(bitcoin::opcodes::all::OP_PUSHNUM_2)
            .push_slice(key)
            .push_slice(key)
            .push_slice(key)
            .push_opcode(bitcoin::opcodes::all::OP_PUSHNUM_3)
            .push_opcode(bitcoin::opcodes::all::OP_CHECKMULTISIG)
            .into_script();
        let mut tx = crate::vector::tests_support::cake_like_tx();
        tx.output = vec![TxOut {
            value: Amount::from_sat(100_000),
            script_pubkey: ms_2of3,
        }];
        let block = crate::vector::tests_support::block_with(tx.clone(), 800_000);
        let s = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        assert!(s.is_standard);
        assert!(!s.nonstd_output_type);
        assert!(!s.nonstd_bare_multisig);

        // A 4-of-5 bare multisig output (n > 3, still well above dust) must set
        // nonstd_bare_multisig, must NOT also set nonstd_output_type, and must fail
        // is_standard.
        let ms_4of5 = bitcoin::blockdata::script::Builder::new()
            .push_opcode(bitcoin::opcodes::all::OP_PUSHNUM_4)
            .push_slice(key)
            .push_slice(key)
            .push_slice(key)
            .push_slice(key)
            .push_slice(key)
            .push_opcode(bitcoin::opcodes::all::OP_PUSHNUM_5)
            .push_opcode(bitcoin::opcodes::all::OP_CHECKMULTISIG)
            .into_script();
        let mut tx2 = crate::vector::tests_support::cake_like_tx();
        tx2.output = vec![TxOut {
            value: Amount::from_sat(100_000),
            script_pubkey: ms_4of5,
        }];
        let block2 = crate::vector::tests_support::block_with(tx2.clone(), 800_000);
        let s2 = tx_shape(
            &tx2,
            &block2,
            &AuxFlags::default(),
            &BlockContext::default(),
        );
        assert!(s2.nonstd_bare_multisig);
        assert!(!s2.nonstd_output_type);
        assert!(!s2.is_standard);
    }

    #[test]
    fn positional_type_columns() {
        use crate::vector::tests_support::{p2tr_spk, p2wpkh_spk};
        use bitcoin::{Amount, OutPoint, ScriptBuf, Sequence, TxIn, TxOut, Witness};
        use lumen_primitives::traits::SourcedBlock;
        use lumen_primitives::traits::block_source::SpentOutput;
        use std::collections::HashMap;

        // Two inputs: prevout 0 is P2WPKH, prevout 1 is P2TR (distinct types, known order).
        let mk_in = |vout: u32| TxIn {
            previous_output: OutPoint::new(
                bitcoin::Txid::from_raw_hash(bitcoin::hashes::Hash::all_zeros()),
                vout,
            ),
            script_sig: ScriptBuf::new(),
            sequence: Sequence(0xffffffff),
            witness: Witness::from_slice(&[vec![0u8; 71], vec![0u8; 33]]),
        };
        // Three outputs: P2WPKH, P2WPKH, P2TR (grouped: no type reappears after a run).
        let tx = Transaction {
            version: bitcoin::transaction::Version(2),
            lock_time: bitcoin::absolute::LockTime::ZERO,
            input: vec![mk_in(0), mk_in(1)],
            output: vec![
                TxOut {
                    value: Amount::from_sat(1000),
                    script_pubkey: p2wpkh_spk(),
                },
                TxOut {
                    value: Amount::from_sat(1000),
                    script_pubkey: p2wpkh_spk(),
                },
                TxOut {
                    value: Amount::from_sat(1000),
                    script_pubkey: p2tr_spk(),
                },
            ],
        };
        let mut prevouts = HashMap::new();
        prevouts.insert(
            tx.input[0].previous_output,
            SpentOutput {
                txout: TxOut {
                    value: Amount::from_sat(5000),
                    script_pubkey: p2wpkh_spk(),
                },
                creation_height: 799_990,
            },
        );
        prevouts.insert(
            tx.input[1].previous_output,
            SpentOutput {
                txout: TxOut {
                    value: Amount::from_sat(5000),
                    script_pubkey: p2tr_spk(),
                },
                creation_height: 799_990,
            },
        );
        let mut block = bitcoin::constants::genesis_block(bitcoin::Network::Bitcoin);
        block.txdata.push(tx.clone());
        let block = SourcedBlock {
            height: 800_000,
            block,
            prevouts,
        };

        let shape = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        let row = feature_row(
            &crate::vector::tests_support::classifiable_vector(),
            &shape,
            "x".to_string(),
            1,
        );
        let col =
            |name: &str| row.values[COLUMN_NAMES[2..].iter().position(|c| *c == name).unwrap()];

        // p2wpkh_spk() is a correctly-shaped 22-byte P2WPKH script, so it classifies as
        // P2wpkh (not NonStandard); p2tr_spk() is P2tr.
        let p2wpkh = OutputType::P2wpkh as usize as f64;
        let p2tr = OutputType::P2tr as usize as f64;

        assert_eq!(col("first_input_type_index"), p2wpkh);
        assert_eq!(col("last_input_type_index"), p2tr);
        assert_eq!(col("inputs_type_grouped"), 1.0); // [P2wpkh, P2tr] is grouped
        assert_eq!(col("input_type_at_pos_0"), p2wpkh);
        assert_eq!(col("input_type_at_pos_1"), p2tr);
        assert_eq!(col("input_type_at_pos_2"), -1.0); // only 2 inputs
        assert_eq!(col("output_type_at_pos_0"), p2wpkh);
        assert_eq!(col("output_type_at_pos_1"), p2wpkh);
        assert_eq!(col("output_type_at_pos_2"), p2tr);
        assert_eq!(col("output_type_at_pos_3"), -1.0); // only 3 outputs
        assert_eq!(col("outputs_type_grouped"), 1.0);
        assert_eq!(row.values.len(), COLUMN_NAMES.len() - 2);

        // DRY regression: counts derived from the ordered vec still total correctly.
        assert_eq!(shape.input_type_counts.iter().sum::<u32>(), 2);
        assert_eq!(shape.input_type_counts[OutputType::P2tr as usize], 1);
    }

    /// The per-position arrays are capped at 10: a transaction with more than 10
    /// inputs/outputs populates positions 0..=9 and DISCARDS everything past 9 (there is
    /// no `*_at_pos_10` column at all), while the per-type COUNTS remain uncapped.
    #[test]
    fn positional_arrays_cap_at_ten() {
        use crate::vector::tests_support::{p2tr_spk, p2wpkh_spk};
        use bitcoin::{Amount, OutPoint, ScriptBuf, Sequence, TxIn, TxOut, Witness};
        use lumen_primitives::traits::SourcedBlock;
        use lumen_primitives::traits::block_source::SpentOutput;
        use std::collections::HashMap;

        // 11 inputs (all P2WPKH prevouts) and 12 outputs (all P2TR): both sides exceed 10.
        let mk_in = |vout: u32| TxIn {
            previous_output: OutPoint::new(
                bitcoin::Txid::from_raw_hash(bitcoin::hashes::Hash::all_zeros()),
                vout,
            ),
            script_sig: ScriptBuf::new(),
            sequence: Sequence(0xffffffff),
            witness: Witness::from_slice(&[vec![0u8; 71], vec![0u8; 33]]),
        };
        let input: Vec<TxIn> = (0..11).map(mk_in).collect();
        let output: Vec<TxOut> = (0..12)
            .map(|_| TxOut {
                value: Amount::from_sat(1000),
                script_pubkey: p2tr_spk(),
            })
            .collect();
        let tx = Transaction {
            version: bitcoin::transaction::Version(2),
            lock_time: bitcoin::absolute::LockTime::ZERO,
            input,
            output,
        };
        let mut prevouts = HashMap::new();
        for txin in &tx.input {
            prevouts.insert(
                txin.previous_output,
                SpentOutput {
                    txout: TxOut {
                        value: Amount::from_sat(5000),
                        script_pubkey: p2wpkh_spk(),
                    },
                    creation_height: 799_990,
                },
            );
        }
        let mut block = bitcoin::constants::genesis_block(bitcoin::Network::Bitcoin);
        block.txdata.push(tx.clone());
        let block = SourcedBlock {
            height: 800_000,
            block,
            prevouts,
        };

        let shape = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        let row = feature_row(
            &crate::vector::tests_support::classifiable_vector(),
            &shape,
            "x".to_string(),
            1,
        );
        let col =
            |name: &str| row.values[COLUMN_NAMES[2..].iter().position(|c| *c == name).unwrap()];

        let p2wpkh = OutputType::P2wpkh as usize as f64;
        let p2tr = OutputType::P2tr as usize as f64;

        // Position 9 (the 10th, last capped slot) is populated on both sides...
        assert_eq!(col("input_type_at_pos_9"), p2wpkh);
        assert_eq!(col("output_type_at_pos_9"), p2tr);
        // ...and position 10 is DISCARDED: the column does not exist at all.
        assert!(!COLUMN_NAMES.contains(&"input_type_at_pos_10"));
        assert!(!COLUMN_NAMES.contains(&"output_type_at_pos_10"));

        // The COUNTS are NOT capped — they reflect all 11 inputs / 12 outputs.
        assert_eq!(shape.input_type_counts[OutputType::P2wpkh as usize], 11);
        assert_eq!(shape.output_type_counts[OutputType::P2tr as usize], 12);
        assert_eq!(shape.input_count, 11);
        assert_eq!(shape.output_count, 12);
        assert_eq!(row.values.len(), COLUMN_NAMES.len() - 2);
    }

    #[test]
    fn upstream_optin_and_taproot_sighash_columns() {
        use bitcoin::{Amount, OutPoint, ScriptBuf, Sequence, TxIn, TxOut, Witness};
        use lumen_primitives::traits::SourcedBlock;
        use lumen_primitives::traits::block_source::SpentOutput;
        use std::collections::HashMap;

        // P2TR prevout spent key-path with a 65-byte (explicit-sighash) signature, and
        // locktime 0 with an RBF-signaling sequence 0xFFFFFFFD.
        let p2tr = crate::vector::tests_support::p2tr_spk();
        let txin = TxIn {
            previous_output: OutPoint::new(
                bitcoin::Txid::from_raw_hash(bitcoin::hashes::Hash::all_zeros()),
                0,
            ),
            script_sig: ScriptBuf::new(),
            sequence: Sequence(0xFFFFFFFD),
            witness: Witness::from_slice(&[vec![0u8; 65]]),
        };
        let tx = Transaction {
            version: bitcoin::transaction::Version(2),
            lock_time: bitcoin::absolute::LockTime::ZERO,
            input: vec![txin],
            output: vec![TxOut {
                value: Amount::from_sat(1000),
                script_pubkey: ScriptBuf::new(),
            }],
        };
        let mut prevouts = HashMap::new();
        prevouts.insert(
            tx.input[0].previous_output,
            SpentOutput {
                txout: TxOut {
                    value: Amount::from_sat(5000),
                    script_pubkey: p2tr,
                },
                creation_height: 799_990,
            },
        );
        let mut block = bitcoin::constants::genesis_block(bitcoin::Network::Bitcoin);
        block.txdata.push(tx.clone());
        let block = SourcedBlock {
            height: 800_000,
            block,
            prevouts,
        };

        let shape = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        let row = feature_row(
            &crate::vector::tests_support::classifiable_vector(),
            &shape,
            "x".to_string(),
            1,
        );
        let col =
            |name: &str| row.values[COLUMN_NAMES[2..].iter().position(|c| *c == name).unwrap()];
        assert_eq!(col("nlocktime_optin_without_use"), 1.0); // locktime 0 + seq 0xFFFFFFFD
        assert_eq!(col("taproot_keyspend_non_default_sighash"), 1.0); // P2TR key-path, 65-byte sig
        assert_eq!(row.values.len(), COLUMN_NAMES.len() - 2);

        // A 64-byte (default-sighash) key-path signature → taproot column false.
        let mut tx2 = tx.clone();
        tx2.input[0].witness = Witness::from_slice(&[vec![0u8; 64]]);
        let block2 = crate::vector::tests_support::block_with_prevout_script(
            tx2.clone(),
            800_000,
            crate::vector::tests_support::p2tr_spk(),
        );
        let shape2 = tx_shape(
            &tx2,
            &block2,
            &AuxFlags::default(),
            &BlockContext::default(),
        );
        let row2 = feature_row(
            &crate::vector::tests_support::classifiable_vector(),
            &shape2,
            "x".to_string(),
            1,
        );
        assert_eq!(
            row2.values[COLUMN_NAMES[2..]
                .iter()
                .position(|c| *c == "taproot_keyspend_non_default_sighash")
                .unwrap()],
            0.0
        );
    }

    #[test]
    fn output_types_in_order_index_matches_discriminant() {
        for (i, t) in OUTPUT_TYPES_IN_ORDER.iter().enumerate() {
            assert_eq!(
                *t as usize, i,
                "OUTPUT_TYPES_IN_ORDER[{i}] discriminant must equal its index — the count/positional columns index by `t as usize`"
            );
        }
    }

    #[test]
    fn feature_columns_share_only_the_intentional_dual_names_with_axes() {
        use crate::vector::{CORE_AXES, EXTENDED_AXES, HEURISTIC_AXES};
        let axes: std::collections::HashSet<&str> = CORE_AXES
            .iter()
            .chain(EXTENDED_AXES)
            .chain(HEURISTIC_AXES)
            .copied()
            .collect();
        let shared: std::collections::BTreeSet<&str> = COLUMN_NAMES
            .iter()
            .copied()
            .filter(|c| axes.contains(c))
            .collect();
        assert_eq!(
            shared,
            std::collections::BTreeSet::from(["op_return", "round_feerate"]),
            "the only names shared between the feature matrix and the sparsity axes must be the intentional duals; any other overlap breaks the two-objectives separation"
        );
    }

    #[test]
    fn column_kinds_align_with_names() {
        assert_eq!(COLUMN_KINDS.len(), COLUMN_NAMES.len() - 2);
        // spot-checks by name→kind
        let kind = |n: &str| &COLUMN_KINDS[COLUMN_NAMES[2..].iter().position(|c| *c == n).unwrap()];
        assert!(matches!(kind("op_return"), ColumnKind::Bool));
        assert!(matches!(kind("version_1"), ColumnKind::OneHot { group } if *group == "version_"));
        assert!(matches!(
            kind("ecdsa_sigs_index"),
            ColumnKind::Ordinal { .. }
        ));
        assert!(matches!(
            kind("feerate_sat_per_vb"),
            ColumnKind::Numeric { .. }
        ));
    }

    #[test]
    fn fold_categorical_bool() {
        let k = ColumnKind::Bool;
        let mut a = new_field_agg(&k);
        fold_field(&mut a, &k, 1.0);
        fold_field(&mut a, &k, 0.0);
        fold_field(&mut a, &k, 1.0);
        let FieldAgg::Categorical(m) = a else {
            panic!()
        };
        assert_eq!(m["1"], 2);
        assert_eq!(m["0"], 1);
    }

    #[test]
    fn fold_and_merge_numeric() {
        let k = ColumnKind::Numeric {
            edges: &[0.0, 10.0, 100.0],
        };
        let mut a = new_field_agg(&k);
        for v in [5.0, 50.0, 500.0] {
            fold_field(&mut a, &k, v);
        }
        let mut b = new_field_agg(&k);
        fold_field(&mut b, &k, 5.0);
        merge_field(&mut a, &b);
        let FieldAgg::Numeric {
            count,
            sum,
            min,
            max,
            hist,
        } = a
        else {
            panic!()
        };
        assert_eq!(count, 4);
        assert_eq!(sum, 560.0);
        assert_eq!(min, 5.0);
        assert_eq!(max, 500.0);
        assert_eq!(hist, vec![0, 2, 1, 1]); // buckets: <0, [0,10), [10,100), [100,∞)
    }
}
