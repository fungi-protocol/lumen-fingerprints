//! Wide per-tx numeric feature row for ML. These columns must never join
//! `CORE_AXES`/`EXTENDED_AXES`: the sparsity joint vector uses `vector.rs`'s `axis_value`.

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

/// Indexed by `OutputType as usize`.
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

/// Never enters the sparsity joint vector.
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
    /// Legacy ×4 + P2SH + witness, so unlike `ecdsa_sigs` not limited to ECDSA.
    pub sigop_count: u32,
    /// Independent of `change_type`'s script-type heuristic; which one fires is a fingerprint.
    pub change_detected_by_round_number: bool,
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

    // BIP-68 applies only to version >= 2.
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
    // Reserved bits 16-21 and 23-30; meaningful only when bit 31 is clear (BIP-68).
    let nsequence_reserved_bits_set = seqs
        .iter()
        .any(|&s| s & 0x8000_0000 == 0 && s & 0x7FBF_0000 != 0);

    // Prevouts always resolve here; the NonStandard fallback only keeps indices aligned.
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

    let (fee, vsize, weight) = tx_fee_vsize_weight(tx, block).unwrap_or((0, 1, 1));
    let feerate_sat_per_vb = fee as f64 / vsize as f64;
    let feerate_sat_per_kwu = fee as f64 * 1000.0 / weight as f64;
    let fee_is_multiple_of_vsize = vsize != 0 && fee % vsize == 0;
    // u128: `fee * 1000` can overflow u64.
    let fee_is_multiple_of_kwu = weight != 0 && (u128::from(fee) * 1000) % u128::from(weight) == 0;
    let fee_is_multiple_of_weight = weight != 0 && fee % weight == 0;
    let ratio = |min: Option<f64>| match min {
        Some(m) if m > 0.0 => feerate_sat_per_vb / m,
        _ => -1.0,
    };
    let feerate_over_block_min = ratio(ctx.block_min_feerate);
    let feerate_over_prev_block_min = ratio(ctx.prev_block_min_feerate);

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

    let sigop_count = tx.total_sigop_cost(|op| block.prevout(op).cloned()) as u32;

    let change_by_script_type = change_index_by_script_type(&input_types, &output_types);
    let change_by_round_number = change_index_by_round_number(&out_values);
    let change_detected_by_round_number = change_by_round_number.is_some();
    let change_heuristics_agree = matches!(
        (change_by_script_type, change_by_round_number),
        (Some(a), Some(b)) if a == b
    );

    let depths: Vec<u32> = tx
        .input
        .iter()
        .filter_map(|i| taproot_script_path_depth(&i.witness))
        .collect();
    let taproot_max_merkle_depth = depths.iter().copied().max().unwrap_or(0);
    let taproot_script_path_input_count = depths.len() as u32;

    // Subset of Core's IsStandardTx; see FEATURES.md for the policy caveat.
    let nonstd_version = !(1..=3).contains(&tx.version.0);
    let nonstd_weight = tx.weight().to_wu() > MAX_STANDARD_TX_WEIGHT;
    let mut nonstd_output_type = false;
    let mut nonstd_dust = false;
    let mut nonstd_bare_multisig = false;
    let mut op_return_count = 0u32;
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

    let tx_has_inscription = tx
        .input
        .iter()
        .any(|i| has_inscription_envelope(&i.witness));
    let has_runestone = tx
        .output
        .iter()
        .any(|o| is_runestone_output(&o.script_pubkey));

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

/// `values` follows `COLUMN_NAMES[2..]`.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureRow {
    pub txid: String,
    pub height: u32,
    pub values: Vec<f64>,
}

/// How a feature column's raw `f64` values fold into a dashboard aggregate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColumnKind {
    /// Any non-zero value folds as 1.
    Bool,
    /// `group` is the family's shared name prefix, e.g. `"version_"`.
    OneHot { group: &'static str },
    /// The raw value indexes `labels`.
    Ordinal { labels: &'static [&'static str] },
    /// `edges[i]` is the inclusive lower bound of bucket `i`.
    Numeric { edges: &'static [f64] },
}

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

const LABELS_ECDSA_SIGS: &[&str] = &["none", "one", "few", "many"];
const LABELS_INPUT_AGE: &[&str] = &["same_block", "<=6", "<=144", "older", "indeterminate"];
const LABELS_OUTPUT_STRUCTURE: &[&str] = &["single", "double", "multi", "unknown"];
/// Sparse ranks from `feerate_rank`: 0..9, then tens 10..100; `u16::MAX` maps to "unknown".
const LABELS_FEERATE_BUCKET: &[&str] = &[
    "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r20", "r30", "r40", "r50",
    "r60", "r70", "r80", "r90", "r100", "unknown",
];
const LABELS_LOCKTIME_OFFSET: &[&str] = &[
    "0", "1", "2", "3", "4-6", "7-12", "13-25", "26-50", "51-100",
];
/// Shared by every `*_type_index` and `*_type_at_pos_*` column; `-1` (absent) maps to
/// "absent", else the index follows `OUTPUT_TYPES_IN_ORDER`.
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

/// Declares every feature column once, generating the index-aligned `COLUMN_NAMES`
/// (after the two identity columns), `COLUMN_KINDS` and `COLUMN_SECTIONS`.
macro_rules! columns {
    ($($section:literal: [$($name:literal => $kind:expr),* $(,)?]),* $(,)?) => {
        pub const COLUMN_NAMES: &[&str] = &["txid", "block_height", $($($name),*),*];
        pub const COLUMN_KINDS: &[ColumnKind] = &[$($({
            use ColumnKind::*;
            $kind
        }),*),*];
        pub(crate) const COLUMN_SECTIONS: &[&str] = &[$($(second!($name, $section)),*),*];
    };
}

// Repeats `$section` once per `$name`: macro_rules can't repeat it at that depth alone.
macro_rules! second {
    ($_first:expr, $second:expr) => {
        $second
    };
}

// Append-only: new columns go at the end, under their catalog section.
columns! {
    "01": [
        "op_return" => Bool,
        "round_feerate" => Bool,
        "low_s_yes" => Bool,
        "round_fee" => Bool,
        "round_payment" => Bool,
        "uih1" => Bool,
        "uih2" => Bool,
        "address_reuse" => Bool,
        "same_block_parent" => Bool,
        "changeless" => Bool,
        "low_r_yes" => Bool,
        "low_r_indeterminate" => Bool,
        "uncompressed_pubkey_yes" => Bool,
        "uncompressed_pubkey_indeterminate" => Bool,
    ],
    "02": [
        "version_1" => OneHot { group: "version_" },
        "version_2" => OneHot { group: "version_" },
        "version_3" => OneHot { group: "version_" },
        "version_other" => OneHot { group: "version_" },
        "nsequence_cake_group_c" => OneHot { group: "nsequence_" },
        "nsequence_lone_0x01" => OneHot { group: "nsequence_" },
        "nsequence_rbf" => OneHot { group: "nsequence_" },
        "nsequence_final" => OneHot { group: "nsequence_" },
        "nsequence_max" => OneHot { group: "nsequence_" },
        "nsequence_mixed_other" => OneHot { group: "nsequence_" },
        "nlocktime_zero" => OneHot { group: "nlocktime_" },
        "nlocktime_anti_fee_snipe" => OneHot { group: "nlocktime_" },
        "nlocktime_backdated" => OneHot { group: "nlocktime_" },
        "nlocktime_future" => OneHot { group: "nlocktime_" },
        "nlocktime_timestamp" => OneHot { group: "nlocktime_" },
        "input_order_bip69" => OneHot { group: "input_order_" },
        "input_order_value_ascending" => OneHot { group: "input_order_" },
        "input_order_value_descending" => OneHot { group: "input_order_" },
        "input_order_age_ascending" => OneHot { group: "input_order_" },
        "input_order_other" => OneHot { group: "input_order_" },
        "input_order_indeterminate" => OneHot { group: "input_order_" },
        "output_order_bip69" => OneHot { group: "output_order_" },
        "output_order_value_ascending" => OneHot { group: "output_order_" },
        "output_order_value_descending" => OneHot { group: "output_order_" },
        "output_order_age_ascending" => OneHot { group: "output_order_" },
        "output_order_other" => OneHot { group: "output_order_" },
        "output_order_indeterminate" => OneHot { group: "output_order_" },
        "input_subtype_p2sh_p2wpkh" => OneHot { group: "input_subtype_" },
        "input_subtype_p2sh_multisig" => OneHot { group: "input_subtype_" },
        "input_subtype_p2sh_other" => OneHot { group: "input_subtype_" },
        "input_subtype_p2wsh_multisig" => OneHot { group: "input_subtype_" },
        "input_subtype_p2wsh_other" => OneHot { group: "input_subtype_" },
        "input_subtype_taproot_key_path" => OneHot { group: "input_subtype_" },
        "input_subtype_taproot_script_path" => OneHot { group: "input_subtype_" },
        "input_subtype_bare" => OneHot { group: "input_subtype_" },
        "input_subtype_mixed" => OneHot { group: "input_subtype_" },
        "input_subtype_indeterminate" => OneHot { group: "input_subtype_" },
        "sighash_taproot_default" => OneHot { group: "sighash_" },
        "sighash_taproot_explicit" => OneHot { group: "sighash_" },
        "sighash_all" => OneHot { group: "sighash_" },
        "sighash_none" => OneHot { group: "sighash_" },
        "sighash_single" => OneHot { group: "sighash_" },
        "sighash_acp_all" => OneHot { group: "sighash_" },
        "sighash_acp_none" => OneHot { group: "sighash_" },
        "sighash_acp_single" => OneHot { group: "sighash_" },
        "sighash_other" => OneHot { group: "sighash_" },
        "sighash_mixed" => OneHot { group: "sighash_" },
        "sighash_na" => OneHot { group: "sighash_" },
        "input_types_uniform_p2pkh" => OneHot { group: "input_types_" },
        "input_types_uniform_p2sh" => OneHot { group: "input_types_" },
        "input_types_uniform_p2wpkh" => OneHot { group: "input_types_" },
        "input_types_uniform_p2wsh" => OneHot { group: "input_types_" },
        "input_types_uniform_p2tr" => OneHot { group: "input_types_" },
        "input_types_uniform_op_return" => OneHot { group: "input_types_" },
        "input_types_uniform_nonstandard" => OneHot { group: "input_types_" },
        "input_types_uniform_p2pk" => OneHot { group: "input_types_" },
        "input_types_mixed" => OneHot { group: "input_types_" },
        "input_types_unknown" => OneHot { group: "input_types_" },
    ],
    "03": [
        // ordinal: larger index = larger bucket
        "ecdsa_sigs_index" => Ordinal { labels: LABELS_ECDSA_SIGS },
        "input_age_index" => Ordinal { labels: LABELS_INPUT_AGE },
        "output_structure_index" => Ordinal { labels: LABELS_OUTPUT_STRUCTURE },
        "feerate_bucket_index" => Ordinal { labels: LABELS_FEERATE_BUCKET },
        "locktime_offset_index" => Ordinal { labels: LABELS_LOCKTIME_OFFSET },
        "locktime_offset_not_applicable" => Bool,
    ],
    "04": [
        // heuristic-derived: CSV only, never in the joint key
        "change_position_first" => OneHot { group: "change_position_" },
        "change_position_last" => OneHot { group: "change_position_" },
        "change_position_middle" => OneHot { group: "change_position_" },
        "change_position_indeterminate" => OneHot { group: "change_position_" },
        "change_type_present" => Bool,
    ],
    "05": [
        "bip125_rbf" => Bool,
        "all_inputs_final" => Bool,
        "bip68_active" => Bool,
        "bip68_type_time" => Bool,
        "nsequence_reserved_bits_set" => Bool,
        "bip68_relative_value_max" => Numeric { edges: EDGES_BIP68_RELATIVE_VALUE },
        "nlocktime_dead" => Bool,
    ],
    "06": [
        "input_count" => Numeric { edges: EDGES_IO_COUNT },
        "output_count" => Numeric { edges: EDGES_IO_COUNT },
        "input_count_p2pkh" => Numeric { edges: EDGES_IO_TYPE_COUNT },
        "input_count_p2sh" => Numeric { edges: EDGES_IO_TYPE_COUNT },
        "input_count_p2wpkh" => Numeric { edges: EDGES_IO_TYPE_COUNT },
        "input_count_p2wsh" => Numeric { edges: EDGES_IO_TYPE_COUNT },
        "input_count_p2tr" => Numeric { edges: EDGES_IO_TYPE_COUNT },
        "input_count_op_return" => Numeric { edges: EDGES_IO_TYPE_COUNT },
        "input_count_nonstandard" => Numeric { edges: EDGES_IO_TYPE_COUNT },
        "input_count_p2pk" => Numeric { edges: EDGES_IO_TYPE_COUNT },
        "output_count_p2pkh" => Numeric { edges: EDGES_IO_TYPE_COUNT },
        "output_count_p2sh" => Numeric { edges: EDGES_IO_TYPE_COUNT },
        "output_count_p2wpkh" => Numeric { edges: EDGES_IO_TYPE_COUNT },
        "output_count_p2wsh" => Numeric { edges: EDGES_IO_TYPE_COUNT },
        "output_count_p2tr" => Numeric { edges: EDGES_IO_TYPE_COUNT },
        "output_count_op_return" => Numeric { edges: EDGES_IO_TYPE_COUNT },
        "output_count_nonstandard" => Numeric { edges: EDGES_IO_TYPE_COUNT },
        "output_count_p2pk" => Numeric { edges: EDGES_IO_TYPE_COUNT },
        "first_output_type_index" => Ordinal { labels: LABELS_TYPE_INDEX },
        "last_output_type_index" => Ordinal { labels: LABELS_TYPE_INDEX },
        "first_output_matches_input_type" => Bool,
        "last_output_matches_input_type" => Bool,
        "outputs_type_grouped" => Bool,
        "first_input_type_index" => Ordinal { labels: LABELS_TYPE_INDEX },
        "last_input_type_index" => Ordinal { labels: LABELS_TYPE_INDEX },
        "inputs_type_grouped" => Bool,
        "input_type_at_pos_0" => Ordinal { labels: LABELS_TYPE_INDEX },
        "input_type_at_pos_1" => Ordinal { labels: LABELS_TYPE_INDEX },
        "input_type_at_pos_2" => Ordinal { labels: LABELS_TYPE_INDEX },
        "input_type_at_pos_3" => Ordinal { labels: LABELS_TYPE_INDEX },
        "input_type_at_pos_4" => Ordinal { labels: LABELS_TYPE_INDEX },
        "input_type_at_pos_5" => Ordinal { labels: LABELS_TYPE_INDEX },
        "input_type_at_pos_6" => Ordinal { labels: LABELS_TYPE_INDEX },
        "input_type_at_pos_7" => Ordinal { labels: LABELS_TYPE_INDEX },
        "input_type_at_pos_8" => Ordinal { labels: LABELS_TYPE_INDEX },
        "input_type_at_pos_9" => Ordinal { labels: LABELS_TYPE_INDEX },
        "output_type_at_pos_0" => Ordinal { labels: LABELS_TYPE_INDEX },
        "output_type_at_pos_1" => Ordinal { labels: LABELS_TYPE_INDEX },
        "output_type_at_pos_2" => Ordinal { labels: LABELS_TYPE_INDEX },
        "output_type_at_pos_3" => Ordinal { labels: LABELS_TYPE_INDEX },
        "output_type_at_pos_4" => Ordinal { labels: LABELS_TYPE_INDEX },
        "output_type_at_pos_5" => Ordinal { labels: LABELS_TYPE_INDEX },
        "output_type_at_pos_6" => Ordinal { labels: LABELS_TYPE_INDEX },
        "output_type_at_pos_7" => Ordinal { labels: LABELS_TYPE_INDEX },
        "output_type_at_pos_8" => Ordinal { labels: LABELS_TYPE_INDEX },
        "output_type_at_pos_9" => Ordinal { labels: LABELS_TYPE_INDEX },
    ],
    "07": [
        "fee_sat" => Numeric { edges: EDGES_FEE_SAT },
        "vsize" => Numeric { edges: EDGES_VSIZE },
        "weight" => Numeric { edges: EDGES_WEIGHT },
        "feerate_sat_per_vb" => Numeric { edges: EDGES_FEERATE },
        "feerate_sat_per_kwu" => Numeric { edges: EDGES_FEERATE },
        "fee_is_multiple_of_vsize" => Bool,
        "feerate_over_block_min" => Numeric { edges: EDGES_FEERATE_RATIO },
        "feerate_over_prev_block_min" => Numeric { edges: EDGES_FEERATE_RATIO },
    ],
    "08": [
        "max_output_value_sat" => Numeric { edges: EDGES_AMOUNT },
        "hamming_decimal" => Numeric { edges: EDGES_HAMMING },
        "hamming_base2" => Numeric { edges: EDGES_HAMMING },
        "hamming_base3" => Numeric { edges: EDGES_HAMMING },
        "decimal_sig_fig_span" => Numeric { edges: EDGES_HAMMING },
        "hamming_decimal_min_over_outputs" => Numeric { edges: EDGES_AMOUNT },
    ],
    "09": [
        "taproot_max_merkle_depth" => Numeric { edges: EDGES_TAPROOT_DEPTH },
        "taproot_script_path_input_count" => Numeric { edges: EDGES_TAPROOT_DEPTH },
    ],
    "10": [
        "is_standard" => Bool,
        "nonstd_version" => Bool,
        "nonstd_weight" => Bool,
        "nonstd_output_type" => Bool,
        "nonstd_dust" => Bool,
        "nonstd_multi_op_return" => Bool,
        "nonstd_bare_multisig" => Bool,
    ],
    "11": [
        "has_inscription_envelope" => Bool,
        "has_runestone" => Bool,
    ],
    "07": [
        "fee_is_multiple_of_kwu" => Bool,
        "fee_is_multiple_of_weight" => Bool,
    ],
    "06": [
        "multisig_m" => Numeric { edges: EDGES_MULTISIG },
        "multisig_n" => Numeric { edges: EDGES_MULTISIG },
        "multisig_mixed" => Bool,
    ],
    "08": [
        "distinct_output_value_count" => Numeric { edges: EDGES_OUTPUT_VALUE_COUNT },
        "max_equal_value_output_count" => Numeric { edges: EDGES_OUTPUT_VALUE_COUNT },
    ],
    "05": [
        "nlocktime_optin_without_use" => Bool,
    ],
    "09": [
        "taproot_keyspend_non_default_sighash" => Bool,
    ],
    "06": [
        "sigop_count" => Numeric { edges: EDGES_SIGOP },
    ],
    "04": [
        "change_detected_by_round_number" => Bool,
        "change_heuristics_agree" => Bool,
    ],
}

/// Mergeable across shards and epochs, so `Numeric` keeps sums rather than means.
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

/// Categorical keys are the raw integer, not `labels[value]`: `-1` (absent) and sparse
/// feerate ranks don't index `labels`, so the explorer labels them.
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

/// Bucket 0 is underflow below the first edge, hence `edges.len() + 1` buckets.
fn bucket_index(edges: &[f64], v: f64) -> usize {
    edges.partition_point(|&e| e <= v)
}

const fn name_is(name: &str, prefix: &str, suffix: &str) -> bool {
    let (n, p, s) = (name.as_bytes(), prefix.as_bytes(), suffix.as_bytes());
    if n.len() != p.len() + s.len() {
        return false;
    }
    let mut i = 0;
    while i < n.len() {
        let c = if i < p.len() { p[i] } else { s[i - p.len()] };
        if n[i] != c {
            return false;
        }
        i += 1;
    }
    true
}

/// Position in a row's values of the column named `prefix` + `suffix`. Called in `const`
/// blocks, so a misspelled name is a compile error.
const fn column_index(prefix: &str, suffix: &str) -> usize {
    let mut i = 2;
    while i < COLUMN_NAMES.len() {
        if name_is(COLUMN_NAMES[i], prefix, suffix) {
            return i - 2;
        }
        i += 1;
    }
    panic!("no feature column with this name");
}

const fn is_bool(i: usize) -> bool {
    matches!(COLUMN_KINDS[i], ColumnKind::Bool)
}

const fn is_scalar(i: usize) -> bool {
    matches!(
        COLUMN_KINDS[i],
        ColumnKind::Ordinal { .. } | ColumnKind::Numeric { .. }
    )
}

const fn in_group(i: usize, group: &str) -> bool {
    match COLUMN_KINDS[i] {
        ColumnKind::OneHot { group: g } => name_is(g, group, ""),
        _ => false,
    }
}

const fn group_range(group: &str) -> (usize, usize) {
    let mut start = 0;
    while start < COLUMN_KINDS.len() && !in_group(start, group) {
        start += 1;
    }
    assert!(
        start < COLUMN_KINDS.len(),
        "no one-hot group with this name"
    );
    let mut end = start;
    while end < COLUMN_KINDS.len() && in_group(end, group) {
        end += 1;
    }
    (start, end)
}

const fn type_name(t: OutputType) -> &'static str {
    match t {
        OutputType::P2pkh => "p2pkh",
        OutputType::P2sh => "p2sh",
        OutputType::P2wpkh => "p2wpkh",
        OutputType::P2wsh => "p2wsh",
        OutputType::P2tr => "p2tr",
        OutputType::OpReturn => "op_return",
        OutputType::NonStandard => "nonstandard",
        OutputType::P2pk => "p2pk",
    }
}

/// A Bool column; only `RowWriter::flag` accepts it.
#[derive(Clone, Copy)]
struct FlagCol(usize);

/// An Ordinal or Numeric column; only `RowWriter::num` accepts it.
#[derive(Clone, Copy)]
struct ScalarCol(usize);

/// `prefix<type>` columns, indexed by `OutputType as usize`.
const fn by_type(prefix: &str) -> [ScalarCol; 8] {
    let mut cols = [ScalarCol(0); 8];
    let mut k = 0;
    while k < 8 {
        let t = OUTPUT_TYPES_IN_ORDER[k];
        let i = column_index(prefix, type_name(t));
        assert!(is_scalar(i), "per-type columns must be Ordinal or Numeric");
        cols[t as usize] = ScalarCol(i);
        k += 1;
    }
    cols
}

/// `prefix<position>` columns for positions 0 to 9.
const fn by_position(prefix: &str) -> [ScalarCol; 10] {
    const DIGITS: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
    let mut cols = [ScalarCol(0); 10];
    let mut k = 0;
    while k < 10 {
        let i = column_index(prefix, DIGITS[k]);
        assert!(
            is_scalar(i),
            "positional columns must be Ordinal or Numeric"
        );
        cols[k] = ScalarCol(i);
        k += 1;
    }
    cols
}

macro_rules! flag_col {
    ($name:literal) => {
        const {
            let i = column_index($name, "");
            assert!(is_bool(i), concat!($name, " is not a Bool column"));
            FlagCol(i)
        }
    };
}

macro_rules! num_col {
    ($name:literal) => {
        const {
            let i = column_index($name, "");
            assert!(
                is_scalar(i),
                concat!($name, " is not an Ordinal or Numeric column")
            );
            ScalarCol(i)
        }
    };
}

macro_rules! member {
    ($group:literal, $suffix:literal) => {
        const {
            let i = column_index($group, $suffix);
            assert!(
                in_group(i, $group),
                concat!($group, $suffix, " is not in ", $group)
            );
            i
        }
    };
}

/// Sets the one column of `$group` whose suffix the matching arm names; the group is
/// written once, so no arm can reach into another group.
macro_rules! onehot {
    ($row:ident, $group:literal, $value:expr, { $($pat:pat => $suffix:literal),* $(,)? }) => {
        $row.onehot(
            const { group_range($group) },
            match $value {
                $($pat => member!($group, $suffix)),*
            },
        )
    };
}

/// Each value lands at the position its column name resolves to at compile time, so
/// write order is irrelevant; debug builds also check every column is written once.
struct RowWriter {
    values: Vec<f64>,
    #[cfg(debug_assertions)]
    written: Vec<bool>,
}

impl RowWriter {
    fn new() -> Self {
        Self {
            values: vec![0.0; COLUMN_KINDS.len()],
            #[cfg(debug_assertions)]
            written: vec![false; COLUMN_KINDS.len()],
        }
    }

    fn mark(&mut self, _i: usize) {
        #[cfg(debug_assertions)]
        {
            assert!(
                !self.written[_i],
                "column {} written twice",
                COLUMN_NAMES[_i + 2]
            );
            self.written[_i] = true;
        }
    }

    fn set(&mut self, i: usize, v: f64) {
        self.mark(i);
        self.values[i] = v;
    }

    fn num(&mut self, ScalarCol(i): ScalarCol, v: f64) {
        self.set(i, v);
    }

    fn flag(&mut self, FlagCol(i): FlagCol, b: bool) {
        self.set(i, if b { 1.0 } else { 0.0 });
    }

    fn onehot(&mut self, (start, end): (usize, usize), active: usize) {
        debug_assert!((start..end).contains(&active));
        for i in start..end {
            self.mark(i);
        }
        self.values[active] = 1.0;
    }

    fn finish(self) -> Vec<f64> {
        #[cfg(debug_assertions)]
        if let Some(i) = self.written.iter().position(|w| !w) {
            panic!("column {} never written", COLUMN_NAMES[i + 2]);
        }
        self.values
    }
}

/// `-1` means no type at that position.
fn type_index(t: Option<OutputType>) -> f64 {
    t.map_or(-1.0, |t| t as usize as f64)
}

pub fn feature_row(
    v: &FingerprintVector,
    shape: &TxShape,
    txid: String,
    height: u32,
) -> FeatureRow {
    let mut row = RowWriter::new();

    row.flag(flag_col!("op_return"), v.op_return);
    row.flag(flag_col!("round_feerate"), v.round_feerate);
    row.flag(flag_col!("low_s_yes"), v.low_s == Tri::Yes);
    row.flag(flag_col!("round_fee"), shape.aux.round_fee);
    row.flag(flag_col!("round_payment"), shape.aux.round_payment);
    row.flag(flag_col!("uih1"), shape.aux.uih1);
    row.flag(flag_col!("uih2"), shape.aux.uih2);
    row.flag(flag_col!("address_reuse"), shape.aux.address_reuse);
    row.flag(flag_col!("same_block_parent"), shape.aux.same_block_parent);
    row.flag(flag_col!("changeless"), shape.aux.changeless);
    // Tri as two flags; both zero means No.
    row.flag(flag_col!("low_r_yes"), v.low_r == Tri::Yes);
    row.flag(
        flag_col!("low_r_indeterminate"),
        v.low_r == Tri::Indeterminate,
    );
    let uncompressed = v.uncompressed_pubkey;
    row.flag(
        flag_col!("uncompressed_pubkey_yes"),
        uncompressed == Tri::Yes,
    );
    row.flag(
        flag_col!("uncompressed_pubkey_indeterminate"),
        uncompressed == Tri::Indeterminate,
    );

    onehot!(row, "version_", v.version, {
        1 => "1",
        2 => "2",
        3 => "3",
        _ => "other",
    });
    onehot!(row, "nsequence_", v.nsequence, {
        NSequenceType::CakeGroupC => "cake_group_c",
        NSequenceType::Lone0x01 => "lone_0x01",
        NSequenceType::Rbf => "rbf",
        NSequenceType::Final => "final",
        NSequenceType::Max => "max",
        NSequenceType::MixedOther => "mixed_other",
    });
    onehot!(row, "nlocktime_", v.nlocktime, {
        NLockTimeType::Zero => "zero",
        NLockTimeType::AntiFeeSnipe => "anti_fee_snipe",
        NLockTimeType::Backdated => "backdated",
        NLockTimeType::Future => "future",
        NLockTimeType::Timestamp => "timestamp",
    });
    macro_rules! order {
        ($group:literal, $o:expr) => {
            onehot!(row, $group, $o, {
                OrderClass::Bip69 => "bip69",
                OrderClass::ValueAscending => "value_ascending",
                OrderClass::ValueDescending => "value_descending",
                OrderClass::AgeAscending => "age_ascending",
                OrderClass::Other => "other",
                OrderClass::Indeterminate => "indeterminate",
            })
        };
    }
    order!("input_order_", v.input_order);
    order!("output_order_", v.output_order);
    onehot!(row, "input_subtype_", v.input_subtype, {
        InputSubtype::P2shP2wpkh => "p2sh_p2wpkh",
        InputSubtype::P2shMultisig => "p2sh_multisig",
        InputSubtype::P2shOther => "p2sh_other",
        InputSubtype::P2wshMultisig => "p2wsh_multisig",
        InputSubtype::P2wshOther => "p2wsh_other",
        InputSubtype::TaprootKeyPath => "taproot_key_path",
        InputSubtype::TaprootScriptPath => "taproot_script_path",
        InputSubtype::Bare => "bare",
        InputSubtype::Mixed => "mixed",
        InputSubtype::Indeterminate => "indeterminate",
    });
    onehot!(row, "sighash_", v.sighash, {
        SighashType::TaprootDefault => "taproot_default",
        SighashType::TaprootExplicit => "taproot_explicit",
        SighashType::All => "all",
        SighashType::None => "none",
        SighashType::Single => "single",
        SighashType::AcpAll => "acp_all",
        SighashType::AcpNone => "acp_none",
        SighashType::AcpSingle => "acp_single",
        SighashType::Other => "other",
        SighashType::Mixed => "mixed",
        SighashType::Na => "na",
    });
    onehot!(row, "input_types_", v.input_types, {
        InputTypeClass::Uniform(OutputType::P2pkh) => "uniform_p2pkh",
        InputTypeClass::Uniform(OutputType::P2sh) => "uniform_p2sh",
        InputTypeClass::Uniform(OutputType::P2wpkh) => "uniform_p2wpkh",
        InputTypeClass::Uniform(OutputType::P2wsh) => "uniform_p2wsh",
        InputTypeClass::Uniform(OutputType::P2tr) => "uniform_p2tr",
        InputTypeClass::Uniform(OutputType::OpReturn) => "uniform_op_return",
        InputTypeClass::Uniform(OutputType::NonStandard) => "uniform_nonstandard",
        InputTypeClass::Uniform(OutputType::P2pk) => "uniform_p2pk",
        InputTypeClass::Mixed => "mixed",
        InputTypeClass::Unknown => "unknown",
    });

    row.num(
        num_col!("ecdsa_sigs_index"),
        match v.ecdsa_sigs {
            EcdsaSigCount::None => 0.0,
            EcdsaSigCount::One => 1.0,
            EcdsaSigCount::Few => 2.0,
            EcdsaSigCount::Many => 3.0,
        },
    );
    row.num(
        num_col!("input_age_index"),
        match v.input_age {
            AgeClass::SameBlock => 0.0,
            AgeClass::Within6 => 1.0,
            AgeClass::Within144 => 2.0,
            AgeClass::Older => 3.0,
            AgeClass::Indeterminate => 4.0,
        },
    );
    row.num(
        num_col!("output_structure_index"),
        match v.output_structure {
            OutputStructureType::Single => 0.0,
            OutputStructureType::Double => 1.0,
            OutputStructureType::Multi => 2.0,
            OutputStructureType::Unknown => 3.0,
        },
    );
    row.num(
        num_col!("feerate_bucket_index"),
        feerate_rank(&v.feerate_bucket),
    );
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
    row.num(num_col!("locktime_offset_index"), lo_index);
    row.flag(flag_col!("locktime_offset_not_applicable"), lo_na);

    onehot!(row, "change_position_", v.change_position, {
        ChangePosition::First => "first",
        ChangePosition::Last => "last",
        ChangePosition::Middle => "middle",
        ChangePosition::Indeterminate => "indeterminate",
    });
    row.flag(flag_col!("change_type_present"), v.change_type.is_some());

    row.flag(flag_col!("bip125_rbf"), shape.bip125_rbf);
    row.flag(flag_col!("all_inputs_final"), shape.all_inputs_final);
    row.flag(flag_col!("bip68_active"), shape.bip68_active);
    row.flag(flag_col!("bip68_type_time"), shape.bip68_type_time);
    row.flag(
        flag_col!("nsequence_reserved_bits_set"),
        shape.nsequence_reserved_bits_set,
    );
    row.num(
        num_col!("bip68_relative_value_max"),
        f64::from(shape.bip68_relative_value_max),
    );
    // Consensus ignores nLockTime when every input is final.
    let nlocktime_dead = v.nlocktime != NLockTimeType::Zero && shape.all_inputs_final;
    row.flag(flag_col!("nlocktime_dead"), nlocktime_dead);
    row.num(num_col!("input_count"), f64::from(shape.input_count));
    row.num(num_col!("output_count"), f64::from(shape.output_count));
    let cols = const { by_type("input_count_") };
    for (t, c) in shape.input_type_counts.into_iter().enumerate() {
        row.num(cols[t], f64::from(c));
    }
    let cols = const { by_type("output_count_") };
    for (t, c) in shape.output_type_counts.into_iter().enumerate() {
        row.num(cols[t], f64::from(c));
    }
    row.num(
        num_col!("first_output_type_index"),
        type_index(shape.first_output_type),
    );
    row.num(
        num_col!("last_output_type_index"),
        type_index(shape.last_output_type),
    );
    row.flag(
        flag_col!("first_output_matches_input_type"),
        shape.first_output_matches_input_type,
    );
    row.flag(
        flag_col!("last_output_matches_input_type"),
        shape.last_output_matches_input_type,
    );
    row.flag(
        flag_col!("outputs_type_grouped"),
        shape.outputs_type_grouped,
    );
    row.num(
        num_col!("first_input_type_index"),
        type_index(shape.first_input_type),
    );
    row.num(
        num_col!("last_input_type_index"),
        type_index(shape.last_input_type),
    );
    row.flag(flag_col!("inputs_type_grouped"), shape.inputs_type_grouped);
    let cols = const { by_position("input_type_at_pos_") };
    for (k, t) in shape.input_types_positional.into_iter().enumerate() {
        row.num(cols[k], type_index(t));
    }
    let cols = const { by_position("output_type_at_pos_") };
    for (k, t) in shape.output_types_positional.into_iter().enumerate() {
        row.num(cols[k], type_index(t));
    }

    row.num(num_col!("fee_sat"), shape.fee_sat);
    row.num(num_col!("vsize"), shape.vsize);
    row.num(num_col!("weight"), shape.weight);
    row.num(num_col!("feerate_sat_per_vb"), shape.feerate_sat_per_vb);
    row.num(num_col!("feerate_sat_per_kwu"), shape.feerate_sat_per_kwu);
    row.flag(
        flag_col!("fee_is_multiple_of_vsize"),
        shape.fee_is_multiple_of_vsize,
    );
    row.num(
        num_col!("feerate_over_block_min"),
        shape.feerate_over_block_min,
    );
    row.num(
        num_col!("feerate_over_prev_block_min"),
        shape.feerate_over_prev_block_min,
    );
    row.num(num_col!("max_output_value_sat"), shape.max_output_value_sat);
    row.num(num_col!("hamming_decimal"), shape.hamming_decimal);
    row.num(num_col!("hamming_base2"), shape.hamming_base2);
    row.num(num_col!("hamming_base3"), shape.hamming_base3);
    row.num(num_col!("decimal_sig_fig_span"), shape.decimal_sig_fig_span);
    row.num(
        num_col!("hamming_decimal_min_over_outputs"),
        shape.hamming_decimal_min_over_outputs,
    );
    row.num(
        num_col!("taproot_max_merkle_depth"),
        shape.taproot_max_merkle_depth,
    );
    row.num(
        num_col!("taproot_script_path_input_count"),
        shape.taproot_script_path_input_count,
    );

    row.flag(flag_col!("is_standard"), shape.is_standard);
    row.flag(flag_col!("nonstd_version"), shape.nonstd_version);
    row.flag(flag_col!("nonstd_weight"), shape.nonstd_weight);
    row.flag(flag_col!("nonstd_output_type"), shape.nonstd_output_type);
    row.flag(flag_col!("nonstd_dust"), shape.nonstd_dust);
    row.flag(
        flag_col!("nonstd_multi_op_return"),
        shape.nonstd_multi_op_return,
    );
    row.flag(
        flag_col!("nonstd_bare_multisig"),
        shape.nonstd_bare_multisig,
    );
    row.flag(
        flag_col!("has_inscription_envelope"),
        shape.has_inscription_envelope,
    );
    row.flag(flag_col!("has_runestone"), shape.has_runestone);
    row.flag(
        flag_col!("fee_is_multiple_of_kwu"),
        shape.fee_is_multiple_of_kwu,
    );
    row.flag(
        flag_col!("fee_is_multiple_of_weight"),
        shape.fee_is_multiple_of_weight,
    );
    row.num(num_col!("multisig_m"), shape.multisig_m as f64);
    row.num(num_col!("multisig_n"), shape.multisig_n as f64);
    row.flag(flag_col!("multisig_mixed"), shape.multisig_mixed);
    row.num(
        num_col!("distinct_output_value_count"),
        shape.distinct_output_value_count as f64,
    );
    row.num(
        num_col!("max_equal_value_output_count"),
        shape.max_equal_value_output_count as f64,
    );
    row.flag(
        flag_col!("nlocktime_optin_without_use"),
        shape.nlocktime_optin_without_use,
    );
    row.flag(
        flag_col!("taproot_keyspend_non_default_sighash"),
        shape.taproot_keyspend_non_default_sighash,
    );
    row.num(num_col!("sigop_count"), shape.sigop_count as f64);
    row.flag(
        flag_col!("change_detected_by_round_number"),
        shape.change_detected_by_round_number,
    );
    row.flag(
        flag_col!("change_heuristics_agree"),
        shape.change_heuristics_agree,
    );

    FeatureRow {
        txid,
        height,
        values: row.finish(),
    }
}

/// Monotonic rank by integer lower bound; "Unknown" maps to `u16::MAX` so it sorts last.
fn feerate_rank(bucket: &str) -> f64 {
    if bucket == "Unknown" {
        return f64::from(u16::MAX);
    }
    let head = bucket.split(['-', '+']).next().unwrap_or("");
    head.parse::<u32>()
        .map(f64::from)
        .unwrap_or(f64::from(u16::MAX))
}

/// Minimum sat/vB over fee-paying txs of this block and the previous; `None` if none.
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockContext {
    pub block_min_feerate: Option<f64>,
    pub prev_block_min_feerate: Option<f64>,
}

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

/// Separates `110000000` (span 2) from `100000001` (span 9), both decimal Hamming weight 2.
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

/// BIP-341: an annex is a trailing witness element starting with `0x50`.
fn is_annex(item: &[u8]) -> bool {
    item.first() == Some(&0x50)
}

/// Script-path only: the control block is the last non-annex element, the tapscript
/// the one before it.
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

/// A control block is `33 + 32k` bytes for merkle depth `k`.
fn taproot_script_path_depth(witness: &bitcoin::Witness) -> Option<u32> {
    let (_tapscript, control) = tapscript_and_control(witness)?;
    let len = control.len();
    if len < 33 || (len - 33) % 32 != 0 {
        return None;
    }
    Some(((len - 33) / 32) as u32)
}

/// Canonical Ordinals envelope `OP_FALSE OP_IF OP_PUSHBYTES_3 "ord"` only; fragmented or
/// non-canonical envelopes escape.
fn has_inscription_envelope(witness: &bitcoin::Witness) -> bool {
    const ORD: [u8; 6] = [0x00, 0x63, 0x03, 0x6f, 0x72, 0x64];
    let Some((tapscript, _control)) = tapscript_and_control(witness) else {
        return false;
    };
    tapscript.windows(ORD.len()).any(|w| w == ORD)
}

/// Runes marker: `OP_RETURN OP_PUSHNUM_13` (`6a 5d`).
fn is_runestone_output(spk: &bitcoin::Script) -> bool {
    spk.is_op_return() && spk.as_bytes().get(1) == Some(&0x5d)
}

/// Core policy limit: heavier txs are non-standard but still minable.
const MAX_STANDARD_TX_WEIGHT: u64 = 400_000;

/// Core's `GetDustThreshold` at 3000 sat/kvB: 3 * (output size + spending input size,
/// 67 witness / 148 legacy). Assumes scripts under 253 bytes. P2WPKH 294, P2PKH 546.
fn output_dust_threshold(spk: &bitcoin::Script) -> u64 {
    let spend_cost = if spk.is_witness_program() { 67 } else { 148 };
    let n_size = 9 + spk.len() as u64 + spend_cost;
    3 * n_size
}

fn op_pushnum(instr: &bitcoin::script::Instruction<'_>) -> Option<u32> {
    if let bitcoin::script::Instruction::Op(opcode) = instr {
        let code = opcode.to_u8();
        if (0x51..=0x60).contains(&code) {
            return Some(u32::from(code - 0x50));
        }
    }
    None
}

/// `OP_m <pubkeys> OP_n OP_CHECKMULTISIG`.
fn multisig_m_n(spk: &bitcoin::Script) -> Option<(u32, u32)> {
    if !spk.is_multisig() {
        return None;
    }
    let instrs: Vec<_> = spk.instructions().collect::<Result<_, _>>().ok()?;
    let m = op_pushnum(instrs.first()?)?;
    let n = op_pushnum(instrs.get(instrs.len().checked_sub(2)?)?)?;
    Some((m, n))
}

/// BIP-342 `multi_a`: `<pk1> OP_CHECKSIG (<pk_i> OP_CHECKSIGADD)* <m> OP_NUMEQUAL[VERIFY]`.
/// Thresholds above 16 are pushed as data and not recognized.
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

fn bare_multisig_n(spk: &bitcoin::Script) -> Option<u32> {
    multisig_m_n(spk).map(|(_, n)| n)
}

/// Tries bare, P2SH redeem, P2WSH witness script, then P2TR `multi_a`; mirrors
/// `input_subtype_class`.
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

    /// Every variant of every axis, so each one-hot suffix and ordinal arm is exercised.
    #[test]
    fn every_axis_variant_writes_a_checked_row() {
        use crate::vector::{AgeClass, ChangePosition, EcdsaSigCount, InputSubtype, OrderClass};
        use lumen_fingerprints_lib::LocktimeOffsetType as L;
        use lumen_fingerprints_lib::OutputStructureType as O;
        use lumen_fingerprints_lib::SighashType as H;
        let base = classifiable_vector();
        let mut vs = vec![base.clone()];
        macro_rules! vary {
            ($f:ident: $($x:expr),*) => {
                $({
                    let mut v = base.clone();
                    v.$f = $x;
                    vs.push(v);
                })*
            };
        }
        vary!(version: 1, 2, 3, 7);
        vary!(nsequence: NSequenceType::CakeGroupC, NSequenceType::Lone0x01, NSequenceType::Rbf,
            NSequenceType::Final, NSequenceType::Max, NSequenceType::MixedOther);
        vary!(nlocktime: NLockTimeType::Zero, NLockTimeType::AntiFeeSnipe, NLockTimeType::Backdated,
            NLockTimeType::Future, NLockTimeType::Timestamp);
        vary!(locktime_offset: L::NotApplicable, L::Zero, L::One, L::Two, L::Three, L::FourToSix,
            L::SevenToTwelve, L::ThirteenToTwentyFive, L::TwentySixToFifty, L::FiftyOneToHundred);
        for o in [
            OrderClass::Bip69,
            OrderClass::ValueAscending,
            OrderClass::ValueDescending,
            OrderClass::AgeAscending,
            OrderClass::Other,
            OrderClass::Indeterminate,
        ] {
            vary!(input_order: o);
            vary!(output_order: o);
        }
        vary!(output_structure: O::Single, O::Double, O::Multi, O::Unknown);
        for t in OUTPUT_TYPES_IN_ORDER {
            vary!(input_types: InputTypeClass::Uniform(t));
            vary!(change_type: Some(t));
        }
        vary!(input_types: InputTypeClass::Mixed, InputTypeClass::Unknown);
        for t in [Tri::Yes, Tri::No, Tri::Indeterminate] {
            vary!(low_r: t);
            vary!(uncompressed_pubkey: t);
            vary!(low_s: t);
        }
        vary!(sighash: H::TaprootDefault, H::TaprootExplicit, H::All, H::None, H::Single, H::AcpAll,
            H::AcpNone, H::AcpSingle, H::Other, H::Mixed, H::Na);
        vary!(input_subtype: InputSubtype::P2shP2wpkh, InputSubtype::P2shMultisig,
            InputSubtype::P2shOther, InputSubtype::P2wshMultisig, InputSubtype::P2wshOther,
            InputSubtype::TaprootKeyPath, InputSubtype::TaprootScriptPath, InputSubtype::Bare,
            InputSubtype::Mixed, InputSubtype::Indeterminate);
        vary!(ecdsa_sigs: EcdsaSigCount::None, EcdsaSigCount::One, EcdsaSigCount::Few,
            EcdsaSigCount::Many);
        vary!(input_age: AgeClass::SameBlock, AgeClass::Within6, AgeClass::Within144,
            AgeClass::Older, AgeClass::Indeterminate);
        vary!(change_position: ChangePosition::First, ChangePosition::Last, ChangePosition::Middle,
            ChangePosition::Indeterminate);
        for v in &vs {
            let row = feature_row(v, &default_shape(), String::new(), 0);
            assert_eq!(row.values.len(), COLUMN_KINDS.len());
        }
    }

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

    /// For tests asserting only on `FingerprintVector`-driven columns.
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
        let (mut tx, block) = tx_with_io(
            &[OutputType::P2wpkh, OutputType::P2wpkh],
            &[OutputType::P2wpkh, OutputType::P2tr],
        );
        tx.output[0].value = bitcoin::Amount::from_sat(50_000);
        tx.output[1].value = bitcoin::Amount::from_sat(49_999);
        let s = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        assert!(s.change_detected_by_round_number);
        assert!(
            !s.change_heuristics_agree,
            "type picks out#0, round-number picks out#1"
        );
    }

    #[test]
    fn tx_shape_change_heuristics_agree_when_both_pick_the_same_output() {
        let (mut tx, block) = tx_with_io(
            &[OutputType::P2tr, OutputType::P2tr],
            &[OutputType::P2wpkh, OutputType::P2tr],
        );
        tx.output[0].value = bitcoin::Amount::from_sat(50_000);
        tx.output[1].value = bitcoin::Amount::from_sat(49_999);
        let s = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        assert!(s.change_detected_by_round_number);
        assert!(s.change_heuristics_agree, "both point to out#1");
    }

    #[test]
    fn tx_shape_change_round_number_abstains_when_both_outputs_round() {
        // All tx_with_io outputs are 50_000 (round), so the round-number heuristic abstains.
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
        let (tx, block) = tx_with_sequences(&[0xFFFF_FFFD, 0xFFFF_FFFD], 2);
        let s = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        assert!(s.bip125_rbf);
        assert!(!s.all_inputs_final);
        assert!(!s.bip68_active);
        assert!(!s.nsequence_reserved_bits_set);

        let (tx, block) = tx_with_sequences(&[0xFFFF_FFFF, 0xFFFF_FFFF], 2);
        let s = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        assert!(s.all_inputs_final);
        assert!(!s.bip125_rbf);

        let (tx, block) = tx_with_sequences(&[0x0040_0005], 2);
        let s = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        assert!(s.bip68_active);
        assert!(s.bip68_type_time);
        assert_eq!(s.bip68_relative_value_max, 5);

        let (tx, block) = tx_with_sequences(&[0x0001_0000], 2);
        let s = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        assert!(s.nsequence_reserved_bits_set);

        // BIP-68 does not apply to v1.
        let (tx, block) = tx_with_sequences(&[0x0040_0005], 1);
        let s = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        assert!(!s.bip68_active);
    }

    #[test]
    fn tx_shape_type_counts_and_positional() {
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
        assert!(s.first_output_matches_input_type);
        assert!(!s.last_output_matches_input_type);
        assert!(s.outputs_type_grouped);

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
        v.ecdsa_sigs = EcdsaSigCount::Few;
        v.input_age = AgeClass::Within144;
        v.output_structure = OutputStructureType::Double;
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
        assert_eq!(col("bip125_rbf"), 1.0);
        assert_eq!(col("all_inputs_final"), 0.0);
        assert_eq!(col("address_reuse"), 1.0);
        assert!(!COLUMN_NAMES.contains(&"input_has_p2wpkh"));
        assert!(!COLUMN_NAMES.contains(&"output_has_p2tr"));
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
        assert!(col("fee_sat") > 0.0);
        assert!(col("vsize") > 0.0);
        assert!(col("weight") > 0.0);
        assert!((col("feerate_sat_per_vb") - shape.feerate_sat_per_vb).abs() < 1e-9);
        assert!((col("feerate_over_block_min") - shape.feerate_sat_per_vb / 2.0).abs() < 1e-9);
        assert!((col("feerate_over_prev_block_min") - shape.feerate_sat_per_vb / 4.0).abs() < 1e-9);
        assert_eq!(row.values.len(), COLUMN_NAMES.len() - 2);

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

        let tx = crate::vector::tests_support::cake_like_tx();
        let block = crate::vector::tests_support::cake_like_block(800_000);
        let shape = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        let (fee, _vsize, weight) = tx_fee_vsize_weight(&tx, &block).expect("cake fee resolves");
        let expected = weight != 0 && (u128::from(fee) * 1000) % u128::from(weight) == 0;
        assert_eq!(shape.fee_is_multiple_of_kwu, expected);

        // fee == weight: the output value doesn't change weight, and inputs total 200_000.
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
        assert_eq!(fee2, w2);
        assert!(shape2.fee_is_multiple_of_kwu);
    }

    #[test]
    fn fee_is_multiple_of_weight_column() {
        use crate::vector::tx_fee_vsize_weight;

        let tx = crate::vector::tests_support::cake_like_tx();
        let block = crate::vector::tests_support::cake_like_block(800_000);
        let shape = tx_shape(&tx, &block, &AuxFlags::default(), &BlockContext::default());
        let (fee, _vsize, weight) = tx_fee_vsize_weight(&tx, &block).expect("cake fee resolves");
        let expected = weight != 0 && fee % weight == 0;
        assert_eq!(shape.fee_is_multiple_of_weight, expected);

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
        assert_eq!(fee2, w2);
        assert!(shape2.fee_is_multiple_of_weight);
        assert!(shape2.fee_is_multiple_of_kwu);

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
        assert_eq!(super::hamming_decimal(100_000_000), 1);
        assert_eq!(super::hamming_decimal(34_567_891), 8);
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
        let mut kp = Witness::new();
        kp.push(vec![0u8; 64]);
        assert_eq!(super::taproot_script_path_depth(&kp), None);
        let mut sp0 = Witness::new();
        sp0.push(vec![0xacu8; 1]);
        sp0.push(vec![0xc0u8; 33]);
        assert_eq!(super::taproot_script_path_depth(&sp0), Some(0));
        // 33 + 32*3 = 129
        let mut sp3 = Witness::new();
        sp3.push(vec![0xacu8; 1]);
        sp3.push(vec![0xc0u8; 129]);
        assert_eq!(super::taproot_script_path_depth(&sp3), Some(3));
        // annex: the control block is second-to-last
        let mut ann = Witness::new();
        ann.push(vec![0xacu8; 1]);
        ann.push(vec![0xc0u8; 33]);
        ann.push(vec![0x50u8, 1, 2]);
        assert_eq!(super::taproot_script_path_depth(&ann), Some(0));
        let mut bad = Witness::new();
        bad.push(vec![0xacu8; 1]);
        bad.push(vec![0xc0u8; 40]);
        assert_eq!(super::taproot_script_path_depth(&bad), None);
    }

    #[test]
    fn inscription_envelope_detection() {
        use bitcoin::Witness;
        let envelope = [0x00u8, 0x63, 0x03, 0x6f, 0x72, 0x64, 0x00, 0x01, 0x02]; // OP_FALSE OP_IF "ord"
        let mut sp = Witness::new();
        sp.push(envelope);
        sp.push(vec![0xc0u8; 33]);
        assert!(super::has_inscription_envelope(&sp));

        let mut plain = Witness::new();
        plain.push(vec![0xacu8; 10]);
        plain.push(vec![0xc0u8; 33]);
        assert!(!super::has_inscription_envelope(&plain));

        let mut kp = Witness::new();
        kp.push(vec![0u8; 64]);
        assert!(!super::has_inscription_envelope(&kp));

        let mut ann = Witness::new();
        ann.push(envelope);
        ann.push(vec![0xc0u8; 33]);
        ann.push(vec![0x50u8, 1, 2]);
        assert!(super::has_inscription_envelope(&ann));
    }

    #[test]
    fn runestone_detection() {
        use bitcoin::ScriptBuf;
        // OP_RETURN OP_PUSHNUM_13
        let runestone = ScriptBuf::from_bytes(vec![0x6a, 0x5d, 0x01, 0x02]);
        assert!(super::is_runestone_output(&runestone));
        let plain = ScriptBuf::from_bytes(vec![0x6a, 0x04, 1, 2, 3, 4]);
        assert!(!super::is_runestone_output(&plain));
        let p2wpkh = ScriptBuf::new_p2wpkh(&bitcoin::WPubkeyHash::from_byte_array([0u8; 20]));
        assert!(!super::is_runestone_output(&p2wpkh));
    }

    #[test]
    fn dust_threshold_matches_core() {
        use bitcoin::ScriptBuf;
        // 3 * (9 + 22 + 67)
        let p2wpkh = ScriptBuf::new_p2wpkh(&bitcoin::WPubkeyHash::from_byte_array([0u8; 20]));
        assert_eq!(super::output_dust_threshold(&p2wpkh), 294);
        // 3 * (9 + 25 + 148)
        let p2pkh = ScriptBuf::new_p2pkh(&bitcoin::PubkeyHash::from_byte_array([0u8; 20]));
        assert_eq!(super::output_dust_threshold(&p2pkh), 546);
    }

    #[test]
    fn bare_multisig_arity() {
        use bitcoin::ScriptBuf;
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
        let p2wpkh = ScriptBuf::new_p2wpkh(&bitcoin::WPubkeyHash::from_byte_array([0u8; 20]));
        assert_eq!(super::bare_multisig_n(&p2wpkh), None);
    }

    #[test]
    fn multisig_m_n_extracts_both() {
        use bitcoin::opcodes::all::{OP_CHECKMULTISIG, OP_PUSHNUM_2, OP_PUSHNUM_3};
        use bitcoin::script::Builder;
        let mut b = Builder::new().push_opcode(OP_PUSHNUM_2);
        for _ in 0..3 {
            b = b.push_slice([0x02u8; 33]);
        }
        let script = b
            .push_opcode(OP_PUSHNUM_3)
            .push_opcode(OP_CHECKMULTISIG)
            .into_script();
        assert_eq!(super::multisig_m_n(&script), Some((2, 3)));
        assert_eq!(super::bare_multisig_n(&script), Some(3));
        let p2wpkh =
            bitcoin::ScriptBuf::new_p2wpkh(&bitcoin::WPubkeyHash::from_byte_array([0u8; 20]));
        assert_eq!(super::multisig_m_n(&p2wpkh), None);
    }

    #[test]
    fn checksigadd_m_n_parses_taproot_multisig() {
        use bitcoin::opcodes::all::{OP_CHECKSIG, OP_CHECKSIGADD, OP_NUMEQUAL, OP_PUSHNUM_2};
        use bitcoin::script::Builder;
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

    #[test]
    fn input_multisig_config_p2sh_p2wsh_and_mixed() {
        use bitcoin::opcodes::all::{OP_CHECKMULTISIG, OP_PUSHNUM_1, OP_PUSHNUM_2, OP_PUSHNUM_3};
        use bitcoin::script::{Builder, PushBytesBuf};
        use bitcoin::{Amount, OutPoint, ScriptBuf, Sequence, TxIn, TxOut, Witness};
        use lumen_primitives::traits::SourcedBlock;
        use lumen_primitives::traits::block_source::SpentOutput;
        use std::collections::HashMap;

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

        let p2wpkh_spk = ScriptBuf::new_p2wpkh(&bitcoin::WPubkeyHash::from_byte_array([0u8; 20]));
        let plain_in = TxIn {
            previous_output: outpoint(2),
            script_sig: ScriptBuf::new(),
            sequence: Sequence(0xffffffff),
            witness: Witness::from_slice(&[vec![0u8; 64]]),
        };
        assert_eq!(super::input_multisig_config(&plain_in, &p2wpkh_spk), None);

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
        assert_eq!(shape.multisig_m, 2); // dominant = the input with the largest n
        assert_eq!(shape.multisig_n, 3);
        assert!(shape.multisig_mixed);
    }

    #[test]
    fn input_multisig_config_decodes_taproot_checksigadd() {
        use bitcoin::opcodes::all::{OP_CHECKSIG, OP_CHECKSIGADD, OP_NUMEQUAL, OP_PUSHNUM_2};
        use bitcoin::script::Builder;
        use bitcoin::{Amount, OutPoint, ScriptBuf, Sequence, TxIn, TxOut, Witness};
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
        let mut witness = Witness::new();
        witness.push(vec![0u8; 64]);
        witness.push(tapscript.as_bytes());
        witness.push(vec![0xc0u8; 33]);
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
        assert_eq!(col("distinct_output_value_count"), 2.0);
        assert_eq!(col("max_equal_value_output_count"), 3.0);
        assert_eq!(row.values.len(), COLUMN_NAMES.len() - 2);
    }

    #[test]
    fn standardness_fields() {
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

        // OP_RETURN outputs are exempt from the dust check.
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
        // n <= 3 bare multisig is standard and must bypass the generic output-type check.
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

        let mk_in = |vout: u32| TxIn {
            previous_output: OutPoint::new(
                bitcoin::Txid::from_raw_hash(bitcoin::hashes::Hash::all_zeros()),
                vout,
            ),
            script_sig: ScriptBuf::new(),
            sequence: Sequence(0xffffffff),
            witness: Witness::from_slice(&[vec![0u8; 71], vec![0u8; 33]]),
        };
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

        let p2wpkh = OutputType::P2wpkh as usize as f64;
        let p2tr = OutputType::P2tr as usize as f64;

        assert_eq!(col("first_input_type_index"), p2wpkh);
        assert_eq!(col("last_input_type_index"), p2tr);
        assert_eq!(col("inputs_type_grouped"), 1.0);
        assert_eq!(col("input_type_at_pos_0"), p2wpkh);
        assert_eq!(col("input_type_at_pos_1"), p2tr);
        assert_eq!(col("input_type_at_pos_2"), -1.0);
        assert_eq!(col("output_type_at_pos_0"), p2wpkh);
        assert_eq!(col("output_type_at_pos_1"), p2wpkh);
        assert_eq!(col("output_type_at_pos_2"), p2tr);
        assert_eq!(col("output_type_at_pos_3"), -1.0);
        assert_eq!(col("outputs_type_grouped"), 1.0);
        assert_eq!(row.values.len(), COLUMN_NAMES.len() - 2);

        assert_eq!(shape.input_type_counts.iter().sum::<u32>(), 2);
        assert_eq!(shape.input_type_counts[OutputType::P2tr as usize], 1);
    }

    #[test]
    fn positional_arrays_cap_at_ten() {
        use crate::vector::tests_support::{p2tr_spk, p2wpkh_spk};
        use bitcoin::{Amount, OutPoint, ScriptBuf, Sequence, TxIn, TxOut, Witness};
        use lumen_primitives::traits::SourcedBlock;
        use lumen_primitives::traits::block_source::SpentOutput;
        use std::collections::HashMap;

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

        assert_eq!(col("input_type_at_pos_9"), p2wpkh);
        assert_eq!(col("output_type_at_pos_9"), p2tr);
        assert!(!COLUMN_NAMES.contains(&"input_type_at_pos_10"));
        assert!(!COLUMN_NAMES.contains(&"output_type_at_pos_10"));

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

        // 65-byte key-path sig = explicit sighash; locktime 0 + RBF sequence = opt-in without use.
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
        assert_eq!(col("nlocktime_optin_without_use"), 1.0);
        assert_eq!(col("taproot_keyspend_non_default_sighash"), 1.0);
        assert_eq!(row.values.len(), COLUMN_NAMES.len() - 2);

        // 64-byte sig = default sighash.
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

    /// The published header, so reordering, renaming or removing a column fails here;
    /// appending one means adding it to the file too, keeping FEATURES.md honest.
    #[test]
    fn columns_match_the_published_header() {
        let published: Vec<&str> = include_str!("../feature_columns.txt").lines().collect();
        assert_eq!(COLUMN_NAMES, published.as_slice());
    }

    #[test]
    fn column_names_are_unique() {
        let mut seen = std::collections::BTreeSet::new();
        for name in COLUMN_NAMES {
            assert!(seen.insert(name), "duplicate column {name}");
        }
    }

    #[test]
    fn column_kinds_align_with_names() {
        assert_eq!(COLUMN_KINDS.len(), COLUMN_NAMES.len() - 2);
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
        assert_eq!(hist, vec![0, 2, 1, 1]); // <0, [0,10), [10,100), >=100
    }
}
