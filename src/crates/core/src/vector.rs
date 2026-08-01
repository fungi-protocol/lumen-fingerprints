use std::borrow::Cow;
use std::collections::BTreeMap;

use crate::change::change_index_by_script_type;
use bitcoin::Transaction;
use lumen_fingerprints_lib::classify::{
    extract_signatures_from_script_sig, extract_signatures_from_witness, has_low_r_signature,
};
use lumen_fingerprints_lib::transaction::{
    input_order, is_bip69_sorted, locktime_offset_class, nlocktime_class, nsequence_class,
    output_structure, sighash_class,
};
use lumen_fingerprints_lib::{
    InputSortingType, LocktimeOffsetType, NLockTimeType, NSequenceType, OutputStructureType,
    SighashType,
};
use lumen_primitives::OutputType;
use lumen_primitives::traits::SourcedBlock;
use lumen_primitives::traits::block_source::SpentOutput;
use serde::{Deserialize, Serialize};

/// A three-valued axis. `Indeterminate` means the transaction carries no signal on this
/// axis, and therefore matches every wallet template (Ishaana's "consistent-with").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tri {
    Yes,
    No,
    Indeterminate,
}

/// Ordering signal of an input or output vector. `Other` means ordered by none of the
/// recognised rules — which is what a shuffling wallet looks like.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderClass {
    Bip69,
    ValueAscending,
    ValueDescending,
    /// Oldest coin first — Ishaana's "oldest-first" sub-case, measurable only
    /// because `SpentOutput` carries `creation_height`.
    AgeAscending,
    Other,
    /// Fewer than two elements: trivially sorted, so no signal.
    Indeterminate,
}

/// The script types of the outputs a transaction spends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InputTypeClass {
    Uniform(OutputType),
    Mixed,
    Unknown,
}

/// What a P2SH/P2WSH/P2TR input actually resolves to — Ishaana separates
/// "P2SH (multisig)" from "P2SH-P2WPKH", which the outer script type cannot tell apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InputSubtype {
    P2shP2wpkh,
    P2shMultisig,
    P2shOther,
    P2wshMultisig,
    P2wshOther,
    TaprootKeyPath,
    TaprootScriptPath,
    /// Plain P2PK/P2PKH/P2WPKH: no wrapping to resolve.
    Bare,
    Mixed,
    Indeterminate,
}

/// Age of the youngest coin a transaction spends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgeClass {
    /// Spends an output created in this same block — the strongest temporal tell.
    SameBlock,
    Within6,
    Within144,
    Older,
    Indeterminate,
}

/// How many ECDSA signatures a transaction carries, bucketed so the axis stays
/// low-cardinality. Exists so `low_r` can be read in context: low-R grinding is
/// probabilistic — an un-grinding wallet still produces a low-R signature about half
/// the time — so `low_r=Yes` is close to a coin flip on a `One`-signature transaction
/// and near-proof of grinding on a `Many`-signature one. Deliberately a separate axis
/// rather than folding the count into `low_r` itself: that would overload a boolean-ish
/// axis with a cardinality it was never meant to carry, where a reader can instead
/// cross-reference two low-cardinality axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EcdsaSigCount {
    /// No ECDSA signature at all: a pure taproot key-path spend, or nothing
    /// parseable as an ECDSA signature.
    None,
    One,
    /// 2 to 4 signatures.
    Few,
    /// 5 or more signatures.
    Many,
}

/// Where, among a transaction's outputs, the change-identification heuristic placed the
/// change output. Answers the checklist's "random / first / last index is change" rows
/// at once, instead of three separate boolean axes.
///
/// `Indeterminate` covers two distinct situations that both mean "no position to
/// report": fewer than 2 outputs (there is no change to place at all), and the
/// change-identification heuristic itself being inconclusive (see
/// `crate::change::change_index_by_script_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChangePosition {
    First,
    Last,
    Middle,
    Indeterminate,
}

/// One transaction's observable construction fingerprint.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FingerprintVector {
    pub version: i32,
    pub nsequence: NSequenceType,
    pub nlocktime: NLockTimeType,
    /// `block_height - locktime`, bucketed; `NotApplicable` unless `nlocktime` is
    /// `AntiFeeSnipe`. See `LocktimeOffsetType`'s doc comment: this is what actually
    /// distinguishes Core's exact-tip anti-fee-sniping from its backdating tail, since
    /// `nlocktime` cannot make that distinction per transaction. Varies with
    /// confirmation timing rather than wallet policy, which is why it lives in
    /// `EXTENDED_AXES` rather than `CORE_AXES`.
    pub locktime_offset: LocktimeOffsetType,
    pub input_order: OrderClass,
    pub output_order: OrderClass,
    pub output_structure: OutputStructureType,
    pub input_types: InputTypeClass,
    /// Sorted, deduplicated output script types — covers "sending to base58 / bech32 /
    /// bech32m" without needing an address encoder.
    pub output_types: Vec<OutputType>,
    pub low_r: Tri,
    pub sighash: SighashType,
    pub uncompressed_pubkey: Tri,
    pub op_return: bool,
    pub input_subtype: InputSubtype,
    pub low_s: Tri,
    /// How many ECDSA signatures this transaction carries, bucketed. Read together
    /// with `low_r` (see `EcdsaSigCount`'s doc comment) — on its own it describes the
    /// coins being spent, not a wallet policy choice, which is why it lives in
    /// `EXTENDED_AXES` rather than `CORE_AXES`.
    pub ecdsa_sigs: EcdsaSigCount,
    pub input_age: AgeClass,
    /// Feerate in sat/vB, bucketed: floor for <10, then 10s up to 100, then "100+".
    pub feerate_bucket: String,
    pub round_feerate: bool,
    /// Position, among this transaction's outputs, of the output that
    /// `change_index_by_script_type` identifies as change. Answers the "random / first
    /// / last index is change" checklist rows at once.
    ///
    /// HEURISTIC-derived (see `HEURISTIC_AXES`), unlike every other field in this
    /// struct: read via `axis_value` and counted as a marginal, but never part of
    /// either joint-vector key (see `epoch.rs::ingest`).
    pub change_position: ChangePosition,
    /// The script type of the identified change output, or `None` when change
    /// identification was inconclusive (rendered as the axis value `Indeterminate`).
    /// Reading the `P2wpkh`/`P2wsh` share of this axis directly answers "is change
    /// always bech32."
    ///
    /// HEURISTIC-derived (see `HEURISTIC_AXES`); same caveat as `change_position` above.
    pub change_type: Option<OutputType>,
}

/// Wallet-policy axes: the construction choices a wallet makes the same way every time.
pub const CORE_AXES: &[&str] = &[
    "version",
    "nsequence",
    "nlocktime",
    "input_order",
    "output_order",
    "output_structure",
    "input_types",
    "output_types",
    "low_r",
    "sighash",
    "uncompressed_pubkey",
    "op_return",
];

/// Axes that vary with the coins at hand rather than with wallet policy.
///
/// `change_position` and `change_type` used to live here but were moved out to
/// `HEURISTIC_AXES` (see its doc comment): both are derived from a fallible
/// heuristic rather than read directly off a transaction field, and letting a
/// heuristic's own abstentions inflate the joint-vector sparsity count would count
/// the tool's uncertainty as chain diversity.
pub const EXTENDED_AXES: &[&str] = &[
    "input_subtype",
    "low_s",
    "ecdsa_sigs",
    "input_age",
    "feerate_bucket",
    "round_feerate",
    "locktime_offset",
];

/// Axes derived from the change-identification heuristic
/// (`crate::change::change_index_by_script_type`) rather than read directly off a
/// transaction field, unlike every axis in `CORE_AXES`/`EXTENDED_AXES`. They are
/// still computed on every classified transaction, still reachable through
/// `axis_value`, and still counted as per-axis marginals in `EpochRow::axis_counts`
/// (see `epoch.rs`'s `all_axes`) — but they must NEVER enter `vectors` or
/// `vectors_extended`: the heuristic's own abstention rate (`Indeterminate`, 34.4% of
/// transactions on the production dataset) is a property of the method, not of the
/// transactions it's measuring, and folding it into the joint key would corrupt the
/// sparsity number these keys exist to answer. See
/// `docs/superpowers/specs/2026-07-29-remove-heuristic-axes-from-sparsity-design.md`.
pub const HEURISTIC_AXES: &[&str] = &["change_position", "change_type"];

/// The axis set emitted, in order, by `lumen scan --vectors`. Every name is a valid
/// argument to `FingerprintVector::axis_value`; the export is the categorical
/// (pre-collapse) representation decluster's value-keyed scorer consumes. The order is
/// stable — it defines the CSV column order.
pub const VECTOR_AXES: &[&str] = &[
    "version",
    "nsequence",
    "nlocktime",
    "locktime_offset",
    "input_order",
    "output_order",
    "output_structure",
    "input_types",
    "output_types",
    "low_r",
    "sighash",
    "uncompressed_pubkey",
    "op_return",
    "input_subtype",
    "low_s",
    "ecdsa_sigs",
    "input_age",
    "feerate_bucket",
    "round_feerate",
    "change_position",
    "change_type",
];

impl FingerprintVector {
    /// The single place that maps an axis name to its observed value. The joint key,
    /// template matching, and the epoch counters all go through here, so a new axis is
    /// added once and cannot disagree between them.
    ///
    /// Returns `Cow` rather than an owned `String` because 16 of the 21 axes are
    /// fieldless enums whose text is a compile-time constant: for those, `Cow::Borrowed`
    /// hands back a `&'static str` with no allocation at all, which matters because this
    /// is called on the order of 1.8 billion times over a full scan. `feerate_bucket` is
    /// already an owned `String` field, so it borrows from `&self` instead of cloning.
    /// Only `version`, `input_types`, `output_types`, `op_return`, and `round_feerate`
    /// still need to build a fresh string (a number, a `Vec` rendering, or a bool).
    ///
    /// The fieldless-enum arms are hand-written matches to `&'static str`, not derived
    /// `Debug` output, so that this function is unambiguously the one contract for these
    /// strings — the `parse_key` round-trip test (`parse_key_round_trips_against_a_real_key_for_output`)
    /// holds the two to the same values.
    pub fn axis_value(&self, axis: &str) -> Option<Cow<'_, str>> {
        Some(match axis {
            "version" => Cow::Owned(self.version.to_string()),
            "nsequence" => Cow::Borrowed(nsequence_str(self.nsequence)),
            "nlocktime" => Cow::Borrowed(nlocktime_str(self.nlocktime)),
            "locktime_offset" => Cow::Borrowed(locktime_offset_str(self.locktime_offset)),
            "input_order" => Cow::Borrowed(order_class_str(self.input_order)),
            "output_order" => Cow::Borrowed(order_class_str(self.output_order)),
            "output_structure" => Cow::Borrowed(output_structure_str(self.output_structure)),
            "input_types" => Cow::Owned(format!("{:?}", self.input_types)),
            "output_types" => Cow::Owned(format!("{:?}", self.output_types)),
            "low_r" => Cow::Borrowed(tri_str(self.low_r)),
            "sighash" => Cow::Borrowed(sighash_str(self.sighash)),
            "uncompressed_pubkey" => Cow::Borrowed(tri_str(self.uncompressed_pubkey)),
            "op_return" => Cow::Owned(self.op_return.to_string()),
            "input_subtype" => Cow::Borrowed(input_subtype_str(self.input_subtype)),
            "low_s" => Cow::Borrowed(tri_str(self.low_s)),
            "ecdsa_sigs" => Cow::Borrowed(ecdsa_sig_count_str(self.ecdsa_sigs)),
            "input_age" => Cow::Borrowed(age_class_str(self.input_age)),
            "feerate_bucket" => Cow::Borrowed(self.feerate_bucket.as_str()),
            "round_feerate" => Cow::Owned(self.round_feerate.to_string()),
            "change_position" => Cow::Borrowed(change_position_str(self.change_position)),
            "change_type" => Cow::Borrowed(match self.change_type {
                Some(t) => output_type_str(t),
                None => "Indeterminate",
            }),
            _ => return None,
        })
    }

    /// Canonical key over an explicit axis set, so sparsity can be reported as a
    /// function of which axes are measured. Writes directly into one `String` rather
    /// than collecting a `Vec<String>` and joining it, since the joined form is all
    /// this ever produces.
    pub fn key_for(&self, axes: &[&str]) -> String {
        let mut out = String::new();
        for (i, axis) in axes.iter().enumerate() {
            if i > 0 {
                out.push('|');
            }
            out.push_str(axis);
            out.push('=');
            match self.axis_value(axis) {
                Some(value) => out.push_str(&value),
                None => out.push('?'),
            }
        }
        out
    }

    /// The default joint key: core axes only.
    pub fn key(&self) -> String {
        self.key_for(CORE_AXES)
    }

    /// CSV header for `--vectors`: `txid` then `VECTOR_AXES`.
    pub fn vector_csv_header() -> String {
        let mut s = String::from("txid");
        for axis in VECTOR_AXES {
            s.push(',');
            s.push_str(axis);
        }
        s
    }

    /// One CSV line: `txid` then each `VECTOR_AXES` value in order. Fields are
    /// RFC-4180-escaped: any field containing comma, quote, newline, or carriage return
    /// is wrapped in double quotes with embedded quotes doubled. This ensures
    /// Vec-valued axes like `output_types` (`[P2wpkh, P2tr]`) stay a single CSV column,
    /// preserving the fixed arity. An unknown axis renders empty.
    pub fn to_vector_csv_line(&self, txid: &str) -> String {
        let mut s = csv_escape(txid);
        for axis in VECTOR_AXES {
            s.push(',');
            let value = self
                .axis_value(axis)
                .map(|v| v.into_owned())
                .unwrap_or_default();
            s.push_str(&csv_escape(&value));
        }
        s
    }
}

/// Hand-written, exhaustive mappings from each fieldless axis enum to its `&'static str`
/// spelling. These deliberately duplicate what `#[derive(Debug)]` would already produce,
/// rather than calling `format!("{:?}", ...)`, so that `axis_value` never allocates for
/// these 15 axes — and because each match is exhaustive (no wildcard arm), adding a new
/// variant to any of these enums without updating its string here is a compile error,
/// not a silent behavior change.
fn tri_str(v: Tri) -> &'static str {
    match v {
        Tri::Yes => "Yes",
        Tri::No => "No",
        Tri::Indeterminate => "Indeterminate",
    }
}

fn order_class_str(v: OrderClass) -> &'static str {
    match v {
        OrderClass::Bip69 => "Bip69",
        OrderClass::ValueAscending => "ValueAscending",
        OrderClass::ValueDescending => "ValueDescending",
        OrderClass::AgeAscending => "AgeAscending",
        OrderClass::Other => "Other",
        OrderClass::Indeterminate => "Indeterminate",
    }
}

fn input_subtype_str(v: InputSubtype) -> &'static str {
    match v {
        InputSubtype::P2shP2wpkh => "P2shP2wpkh",
        InputSubtype::P2shMultisig => "P2shMultisig",
        InputSubtype::P2shOther => "P2shOther",
        InputSubtype::P2wshMultisig => "P2wshMultisig",
        InputSubtype::P2wshOther => "P2wshOther",
        InputSubtype::TaprootKeyPath => "TaprootKeyPath",
        InputSubtype::TaprootScriptPath => "TaprootScriptPath",
        InputSubtype::Bare => "Bare",
        InputSubtype::Mixed => "Mixed",
        InputSubtype::Indeterminate => "Indeterminate",
    }
}

fn ecdsa_sig_count_str(v: EcdsaSigCount) -> &'static str {
    match v {
        EcdsaSigCount::None => "None",
        EcdsaSigCount::One => "One",
        EcdsaSigCount::Few => "Few",
        EcdsaSigCount::Many => "Many",
    }
}

fn age_class_str(v: AgeClass) -> &'static str {
    match v {
        AgeClass::SameBlock => "SameBlock",
        AgeClass::Within6 => "Within6",
        AgeClass::Within144 => "Within144",
        AgeClass::Older => "Older",
        AgeClass::Indeterminate => "Indeterminate",
    }
}

fn change_position_str(v: ChangePosition) -> &'static str {
    match v {
        ChangePosition::First => "First",
        ChangePosition::Last => "Last",
        ChangePosition::Middle => "Middle",
        ChangePosition::Indeterminate => "Indeterminate",
    }
}

fn nsequence_str(v: NSequenceType) -> &'static str {
    match v {
        NSequenceType::CakeGroupC => "CakeGroupC",
        NSequenceType::Lone0x01 => "Lone0x01",
        NSequenceType::Rbf => "Rbf",
        NSequenceType::Final => "Final",
        NSequenceType::Max => "Max",
        NSequenceType::MixedOther => "MixedOther",
    }
}

fn nlocktime_str(v: NLockTimeType) -> &'static str {
    match v {
        NLockTimeType::Zero => "Zero",
        NLockTimeType::AntiFeeSnipe => "AntiFeeSnipe",
        NLockTimeType::Backdated => "Backdated",
        NLockTimeType::Future => "Future",
        NLockTimeType::Timestamp => "Timestamp",
    }
}

fn locktime_offset_str(v: LocktimeOffsetType) -> &'static str {
    match v {
        LocktimeOffsetType::NotApplicable => "NotApplicable",
        LocktimeOffsetType::Zero => "0",
        LocktimeOffsetType::One => "1",
        LocktimeOffsetType::Two => "2",
        LocktimeOffsetType::Three => "3",
        LocktimeOffsetType::FourToSix => "4-6",
        LocktimeOffsetType::SevenToTwelve => "7-12",
        LocktimeOffsetType::ThirteenToTwentyFive => "13-25",
        LocktimeOffsetType::TwentySixToFifty => "26-50",
        LocktimeOffsetType::FiftyOneToHundred => "51-100",
    }
}

fn sighash_str(v: SighashType) -> &'static str {
    match v {
        SighashType::TaprootDefault => "TaprootDefault",
        SighashType::TaprootExplicit => "TaprootExplicit",
        SighashType::All => "All",
        SighashType::None => "None",
        SighashType::Single => "Single",
        SighashType::AcpAll => "AcpAll",
        SighashType::AcpNone => "AcpNone",
        SighashType::AcpSingle => "AcpSingle",
        SighashType::Other => "Other",
        SighashType::Mixed => "Mixed",
        SighashType::Na => "Na",
    }
}

fn output_structure_str(v: OutputStructureType) -> &'static str {
    match v {
        OutputStructureType::Single => "Single",
        OutputStructureType::Double => "Double",
        OutputStructureType::Multi => "Multi",
        OutputStructureType::Unknown => "Unknown",
    }
}

fn output_type_str(v: OutputType) -> &'static str {
    match v {
        OutputType::P2pkh => "P2pkh",
        OutputType::P2sh => "P2sh",
        OutputType::P2wpkh => "P2wpkh",
        OutputType::P2wsh => "P2wsh",
        OutputType::P2tr => "P2tr",
        OutputType::OpReturn => "OpReturn",
        OutputType::NonStandard => "NonStandard",
        OutputType::P2pk => "P2pk",
    }
}

/// RFC-4180 CSV field escaping: wraps the field in double quotes if it contains
/// comma, double-quote, newline, or carriage return, and doubles any embedded quotes.
fn csv_escape(field: &str) -> String {
    if !field.contains(',')
        && !field.contains('"')
        && !field.contains('\n')
        && !field.contains('\r')
    {
        // No special characters: return as-is
        field.to_string()
    } else {
        // Needs escaping: wrap in quotes and double any embedded quotes
        let mut escaped = String::from("\"");
        for ch in field.chars() {
            if ch == '"' {
                escaped.push_str("\"\"");
            } else {
                escaped.push(ch);
            }
        }
        escaped.push('"');
        escaped
    }
}

/// Parse a joint-vector key produced by `key_for`/`key` back into an axis -> value map.
/// This is what makes report-time template matching possible: `EpochRow::vectors_extended`
/// only survives on disk as these strings, never as a `FingerprintVector`.
///
/// Splits into `axis=value` segments on `|`, then splits each segment on the FIRST `=`
/// only. Axis names never contain `=`, but axis values legitimately contain
/// `=`-free-but-structured text such as `[P2wpkh, P2tr]`, `Uniform(P2wpkh)`, `10-19`, or
/// `100+` — splitting on the first `=` is the simplest rule that is guaranteed correct
/// for those shapes, since it never looks past the axis-name/value boundary.
pub fn parse_key(key: &str) -> BTreeMap<String, String> {
    key.split('|')
        .filter_map(|segment| segment.split_once('='))
        .map(|(axis, value)| (axis.to_string(), value.to_string()))
        .collect()
}

/// Compatibility shim for a joint-vector key stored on disk under a DIFFERENT axis set
/// than the one currently defined in this file. `out/survey/epochs-v2.jsonl` was
/// written while the two now-deleted circular axes (`change_type_matches_inputs` /
/// `change_type_matches_outputs`) and the two axes since moved to `HEURISTIC_AXES`
/// (`change_position` / `change_type`) were still inside `EXTENDED_AXES`, so every
/// `vectors_extended` key on disk carries all four as extra `axis=value` segments.
/// Re-scanning the whole dataset over an axis-set change alone would cost hours; this
/// filter costs nothing extra at report time.
///
/// Strips any `axis=value` segment whose axis is not currently in
/// `CORE_AXES ∪ EXTENDED_AXES`. This is exact, not heuristic: the key format is
/// `axis=value` segments joined by `|`, with axis names that never contain `=`, so
/// membership-testing the axis name on each segment cannot misfire. The same filter
/// makes the report forward-compatible with any FUTURE axis-set change too — an axis
/// added, removed, or moved into `HEURISTIC_AXES` later needs no re-scan, only this
/// function staying in sync with the current `CORE_AXES`/`EXTENDED_AXES` definitions.
///
/// Deliberately excludes `HEURISTIC_AXES` from the surviving set: those axes must
/// never enter a joint key under the current scheme either, so a stored key's
/// `change_position=...`/`change_type=...` segments are stripped right along with the
/// two fully-deleted axes.
///
/// This can — and on the production dataset does — merge previously-distinct keys:
/// two stored keys differing only in a stripped axis collapse into the same key once
/// filtered. That is the point, not a defect: it is what makes `distinct_vectors_extended`
/// reflect the CURRENT axis set rather than whichever one happened to be in force when
/// the data was scanned (507,966 → ~452,754 distinct on `epochs-v2.jsonl`).
pub fn strip_retired_axes(key: &str) -> String {
    key.split('|')
        .filter(|segment| {
            segment
                .split_once('=')
                .map(|(axis, _)| CORE_AXES.contains(&axis) || EXTENDED_AXES.contains(&axis))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>()
        .join("|")
}

/// Input ordering, delegating to the pre-existing `fingerprints::input_order` (which
/// already covers BIP-69 / ascending / descending) and adding the age case that only
/// becomes measurable now that prevouts carry `creation_height`.
fn input_order_class(tx: &Transaction, block: &SourcedBlock) -> OrderClass {
    if tx.input.len() < 2 {
        return OrderClass::Indeterminate;
    }

    let Some(spent): Option<Vec<&SpentOutput>> = tx
        .input
        .iter()
        .map(|i| block.spent(&i.previous_output))
        .collect()
    else {
        return OrderClass::Indeterminate;
    };

    let values: Vec<bitcoin::TxOut> = spent.iter().map(|s| s.txout.clone()).collect();
    let sorts = input_order(&tx.input, &values);

    // Age first: an age-ordered vector is the more specific claim, and Core/BDK-style
    // shuffles will not satisfy it by accident on more than a couple of inputs.
    let heights: Vec<u32> = spent.iter().map(|s| s.creation_height).collect();
    let ages_distinct = heights.windows(2).any(|w| w[0] != w[1]);
    if ages_distinct && heights.windows(2).all(|w| w[0] <= w[1]) {
        return OrderClass::AgeAscending;
    }

    if sorts.contains(&InputSortingType::Bip69) {
        OrderClass::Bip69
    } else if sorts.contains(&InputSortingType::Ascending) {
        OrderClass::ValueAscending
    } else if sorts.contains(&InputSortingType::Descending) {
        OrderClass::ValueDescending
    } else {
        OrderClass::Other
    }
}

/// Output ordering. BIP-69 first, then plain value ordering — "change last" shows up
/// as `Other`, and is only decisive over many transactions (see #1597 comment 03).
fn output_order_class(tx: &Transaction) -> OrderClass {
    if tx.output.len() < 2 {
        return OrderClass::Indeterminate;
    }
    if is_bip69_sorted(&tx.output) {
        return OrderClass::Bip69;
    }
    let values: Vec<bitcoin::Amount> = tx.output.iter().map(|o| o.value).collect();
    let distinct = values.windows(2).any(|w| w[0] != w[1]);
    if distinct && values.windows(2).all(|w| w[0] <= w[1]) {
        OrderClass::ValueAscending
    } else if distinct && values.windows(2).all(|w| w[0] >= w[1]) {
        OrderClass::ValueDescending
    } else {
        OrderClass::Other
    }
}

/// Resolve what a wrapped input actually is, by looking past the outer script type.
fn input_subtype_class(tx: &Transaction, block: &SourcedBlock) -> InputSubtype {
    let mut seen: Option<InputSubtype> = None;

    for txin in &tx.input {
        let Some(prevout) = block.prevout(&txin.previous_output) else {
            return InputSubtype::Indeterminate;
        };
        let spk = &prevout.script_pubkey;

        let subtype = if spk.is_p2sh() {
            // The redeem script is the last push of the scriptSig.
            match txin.script_sig.instructions().last() {
                Some(Ok(bitcoin::script::Instruction::PushBytes(redeem))) => {
                    let redeem = bitcoin::Script::from_bytes(redeem.as_bytes());
                    if redeem.is_p2wpkh() {
                        InputSubtype::P2shP2wpkh
                    } else if redeem.is_multisig() {
                        InputSubtype::P2shMultisig
                    } else {
                        InputSubtype::P2shOther
                    }
                }
                _ => InputSubtype::P2shOther,
            }
        } else if spk.is_p2wsh() {
            // The witness script is the last witness item.
            match txin.witness.last() {
                Some(script) if bitcoin::Script::from_bytes(script).is_multisig() => {
                    InputSubtype::P2wshMultisig
                }
                _ => InputSubtype::P2wshOther,
            }
        } else if spk.is_p2tr() {
            // Key-path spends carry exactly one witness item (the signature), ignoring
            // an optional annex. An empty witness is unreachable on consensus-valid
            // data (a P2TR input must carry at least a signature or script-path
            // witness), but it carries no signal either way, so it must not default
            // to `TaprootScriptPath`.
            let items = txin.witness.len();
            let has_annex = txin
                .witness
                .last()
                .is_some_and(|i| i.first() == Some(&0x50) && items > 1);
            if items == 0 {
                InputSubtype::Indeterminate
            } else if items - usize::from(has_annex) == 1 {
                InputSubtype::TaprootKeyPath
            } else {
                InputSubtype::TaprootScriptPath
            }
        } else {
            InputSubtype::Bare
        };

        match seen {
            None => seen = Some(subtype),
            Some(prev) if prev == subtype => {}
            Some(_) => return InputSubtype::Mixed,
        }
    }

    seen.unwrap_or(InputSubtype::Indeterminate)
}

/// Age of the youngest coin spent. `SameBlock` is the temporal tell from #1597
/// comment 10 (spending unconfirmed outputs), visible on-chain as a parent→child
/// spend inside one block.
fn input_age_class(tx: &Transaction, block: &SourcedBlock) -> AgeClass {
    let youngest = tx
        .input
        .iter()
        .filter_map(|i| block.confirmations(&i.previous_output))
        .min();
    match youngest {
        None => AgeClass::Indeterminate,
        Some(0) => AgeClass::SameBlock,
        Some(c) if c < 6 => AgeClass::Within6,
        Some(c) if c < 144 => AgeClass::Within144,
        Some(_) => AgeClass::Older,
    }
}

/// Candidate ECDSA signature byte strings for one input, drawn from both places a
/// signature can live: the scriptSig (legacy P2PKH / bare P2PK / P2SH-multisig) and
/// the witness (segwit). Delegates to `lumen_fingerprints_lib::classify`'s
/// extractors — the same ones `fingerprints::input::low_r_grinding` uses — so this
/// crate does not grow a second, subtly different copy of signature extraction.
fn candidate_signatures(txin: &bitcoin::TxIn) -> Vec<Vec<u8>> {
    let mut sigs = extract_signatures_from_script_sig(txin.script_sig.as_bytes());
    let witness_items: Vec<Vec<u8>> = txin.witness.iter().map(|item| item.to_vec()).collect();
    sigs.extend(extract_signatures_from_witness(&witness_items));
    sigs
}

/// BIP-146 low-S, low-R grinding (every ECDSA signature has its R below 2^255, i.e. a
/// 71-byte DER encoding), and the bucketed ECDSA signature count (`ecdsa_sigs`),
/// computed together in one pass.
///
/// `low_r_class` and `low_s_class` used to be separate functions, each independently
/// calling `candidate_signatures` (which extracts from both scriptSig and witness) and
/// DER-parsing every signature it found — meaning every signature in a transaction was
/// extracted and parsed twice. All three values are derived from the exact same set of
/// parsed ECDSA signatures, so this computes that set once and derives all three from
/// it — `ecdsa_sigs` is a counter over the same loop, not a second traversal.
///
/// Scans both scriptSig and witness (via `candidate_signatures`): a legacy input's
/// signature lives in scriptSig with an empty witness, so witness-only scanning
/// silently returned `Indeterminate` for every non-segwit input on both `Tri` axes. A
/// transaction with no ECDSA signatures (pure taproot key-path) is indeterminate on
/// both `low_r`/`low_s` and `EcdsaSigCount::None` on `ecdsa_sigs`.
fn low_r_and_low_s_class(tx: &Transaction) -> (Tri, Tri, EcdsaSigCount) {
    let mut sig_count: u32 = 0;
    let mut all_low_r = true;
    let mut all_low_s = true;
    for txin in &tx.input {
        for sig_bytes in candidate_signatures(txin) {
            if let Ok(sig) = bitcoin::ecdsa::Signature::from_slice(&sig_bytes) {
                sig_count += 1;
                if !has_low_r_signature(&sig) {
                    all_low_r = false;
                }
                // serialize_compact is (r || s); S is high when its top bit is set and
                // it exceeds half the curve order. libsecp256k1 normalises, so the
                // top-bit test is the practical discriminator.
                if sig.signature.serialize_compact()[32] >= 0x80 {
                    all_low_s = false;
                }
            }
        }
    }
    let ecdsa_sigs = match sig_count {
        0 => EcdsaSigCount::None,
        1 => EcdsaSigCount::One,
        2..=4 => EcdsaSigCount::Few,
        _ => EcdsaSigCount::Many,
    };
    if sig_count == 0 {
        (Tri::Indeterminate, Tri::Indeterminate, ecdsa_sigs)
    } else {
        let low_r = if all_low_r { Tri::Yes } else { Tri::No };
        let low_s = if all_low_s { Tri::Yes } else { Tri::No };
        (low_r, low_s, ecdsa_sigs)
    }
}

/// Fee (sat), vsize (vB), and weight (wu) for a transaction, or `None` when the fee is
/// not computable — a coinbase (its null-prevout input is absent from `block`, so
/// `input_sum < output_sum`) or a zero-vsize tx. The shared primitive behind `feerate`
/// and the survey's block-minimum feerate.
pub(crate) fn tx_fee_vsize_weight(
    tx: &Transaction,
    block: &SourcedBlock,
) -> Option<(u64, u64, u64)> {
    let input_sum: u64 = tx
        .input
        .iter()
        .filter_map(|i| block.prevout(&i.previous_output))
        .map(|o| o.value.to_sat())
        .sum();
    let output_sum: u64 = tx.output.iter().map(|o| o.value.to_sat()).sum();
    let fee = input_sum.checked_sub(output_sum)?;
    let vsize = tx.vsize() as u64;
    if vsize == 0 {
        return None;
    }
    Some((fee, vsize, tx.weight().to_wu()))
}

/// Feerate in sat/vB, bucketed, plus whether it is a round number — the chain-visible
/// half of the "feerate source" and "manual feerate" axes (#1597 comments 12 and 21).
/// The *source* of the feerate is not chain-observable; only the value is.
fn feerate(tx: &Transaction, block: &SourcedBlock) -> (String, bool) {
    let Some((fee, vsize, _weight)) = tx_fee_vsize_weight(tx, block) else {
        return ("Unknown".to_string(), false);
    };

    let rate = fee as f64 / vsize as f64;
    let bucket = if rate < 10.0 {
        format!("{}", rate.floor() as u64)
    } else if rate < 100.0 {
        format!(
            "{}-{}",
            (rate as u64 / 10) * 10,
            (rate as u64 / 10) * 10 + 9
        )
    } else {
        "100+".to_string()
    };

    // A user typing "5" or "20" sat/vB lands within a hair of an integer; a fee
    // estimator's output essentially never does.
    let round = (rate - rate.round()).abs() < 0.01 && rate >= 1.0;
    (bucket, round)
}

/// Change-identification-derived axes: which output (if any) the shared
/// `change_index_by_script_type` heuristic identifies as change, and what that implies
/// about its position and script type.
///
/// This used to also compute `change_type_matches_inputs` / `change_type_matches_outputs`,
/// two axes deleted entirely (see `docs/superpowers/specs/2026-07-29-remove-heuristic-axes-from-sparsity-design.md`):
/// `change_index_by_script_type` identifies change output BY its type equalling the
/// unanimous input type, so `change_type_matches_inputs` was provably `Yes` (or
/// `Indeterminate`) by construction and `change_type_matches_outputs` provably never
/// `Yes` when there was exactly one matching output — both measured the heuristic's own
/// definition, not a fact about the chain.
///
/// The heuristic returns `None` both when it is confident an output is not change and
/// when it simply cannot tell (mixed input types, no unique matching output); this
/// function must not conflate the two. `None` here always means "inconclusive," and
/// both returned axes are `Indeterminate` in that case — never a guessed value.
/// Publishing `NotChange` as if it were a measurement would misrepresent an abstention
/// as a result.
///
/// Callers must have already required every input's prevout to resolve (as
/// `classify_tx` does before calling this), so `input_types` is never partial here.
fn change_axes(
    tx: &Transaction,
    input_types: &[OutputType],
    output_types_ordered: &[OutputType],
) -> (ChangePosition, Option<OutputType>) {
    let Some(change_index) = change_index_by_script_type(input_types, output_types_ordered) else {
        return (ChangePosition::Indeterminate, None);
    };

    // Fewer than 2 outputs: there is no change to place, regardless of what the
    // heuristic's index computation returned.
    let position = if tx.output.len() < 2 {
        ChangePosition::Indeterminate
    } else if change_index == 0 {
        ChangePosition::First
    } else if change_index == tx.output.len() - 1 {
        ChangePosition::Last
    } else {
        ChangePosition::Middle
    };

    let change_type = output_types_ordered[change_index];

    (position, Some(change_type))
}

fn input_type_class(tx: &Transaction, block: &SourcedBlock) -> InputTypeClass {
    let mut seen: Option<OutputType> = None;
    for txin in &tx.input {
        let Some(prevout) = block.prevout(&txin.previous_output) else {
            return InputTypeClass::Unknown;
        };
        let ty = lumen_primitives::classify_script_pubkey(prevout.script_pubkey.as_bytes());
        match seen {
            None => seen = Some(ty),
            Some(prev) if prev == ty => {}
            Some(_) => return InputTypeClass::Mixed,
        }
    }
    seen.map(InputTypeClass::Uniform)
        .unwrap_or(InputTypeClass::Unknown)
}

/// Candidate public-key pushes for one input, from both scriptSig (P2PKH/P2PK's
/// pubkey push, or a multisig redeem script's embedded pubkeys) and witness
/// (P2WPKH/P2WSH). There is no shared pubkey-extraction helper in
/// `lumen_fingerprints_lib` to delegate to (only signature extraction exists there),
/// so this scans pushes directly rather than duplicating signature-extraction logic.
///
/// Borrows every push and witness item rather than copying it into a fresh `Vec<u8>`:
/// the caller only ever reads a length and a first byte, so there is nothing for an
/// owned copy to buy here, and this runs on the order of 1.8 billion times over a
/// full scan.
fn candidate_pubkey_bytes(txin: &bitcoin::TxIn) -> impl Iterator<Item = &[u8]> + '_ {
    let script_sig_pushes = txin
        .script_sig
        .instructions()
        .filter_map(|instr| match instr {
            Ok(bitcoin::script::Instruction::PushBytes(bytes)) => Some(bytes.as_bytes()),
            _ => None,
        });
    script_sig_pushes.chain(txin.witness.iter())
}

/// Uncompressed pubkeys are 65 bytes starting with 0x04. Indeterminate when the
/// transaction exposes no pubkey at all (e.g. taproot key-path).
///
/// Scans both scriptSig and witness pushes (via `candidate_pubkey_bytes`): uncompressed
/// keys are specifically a legacy-era phenomenon, so witness-only scanning missed
/// exactly the transactions this axis exists to measure.
fn uncompressed_pubkey_class(tx: &Transaction) -> Tri {
    let mut saw_pubkey = false;
    let mut saw_uncompressed = false;
    for txin in &tx.input {
        for item in candidate_pubkey_bytes(txin) {
            if item.len() == 33 && (item[0] == 0x02 || item[0] == 0x03) {
                saw_pubkey = true;
            }
            if item.len() == 65 && item[0] == 0x04 {
                saw_pubkey = true;
                saw_uncompressed = true;
            }
        }
    }
    if !saw_pubkey {
        Tri::Indeterminate
    } else if saw_uncompressed {
        Tri::Yes
    } else {
        Tri::No
    }
}

/// Classify one transaction. Returns `None` for coinbase transactions and for any
/// transaction whose prevouts the source could not resolve.
pub fn classify_tx(tx: &Transaction, block: &SourcedBlock) -> Option<FingerprintVector> {
    if tx.is_coinbase() {
        return None;
    }
    if tx
        .input
        .iter()
        .any(|i| block.prevout(&i.previous_output).is_none())
    {
        return None;
    }

    let (feerate_bucket, round_feerate) = feerate(tx, block);

    // Raw, positional output types (NOT the sorted/deduped `output_types` field below):
    // `change_axes` needs to know which specific output index matches, which the
    // deduped set cannot answer.
    let output_types_ordered: Vec<OutputType> = tx
        .output
        .iter()
        .map(|o| lumen_primitives::classify_script_pubkey(o.script_pubkey.as_bytes()))
        .collect();

    let mut output_types = output_types_ordered.clone();
    output_types.sort_by_key(|t| t.as_u32());
    output_types.dedup();

    // Every input's prevout is resolved by this point (checked above), so this is
    // never partial.
    let input_types_ordered: Vec<OutputType> = tx
        .input
        .iter()
        .map(|i| {
            let prevout = block
                .prevout(&i.previous_output)
                .expect("prevout resolution checked above");
            lumen_primitives::classify_script_pubkey(prevout.script_pubkey.as_bytes())
        })
        .collect();

    let (change_position, change_type) =
        change_axes(tx, &input_types_ordered, &output_types_ordered);

    let (low_r, low_s, ecdsa_sigs) = low_r_and_low_s_class(tx);

    Some(FingerprintVector {
        version: tx.version.0,
        nsequence: nsequence_class(&tx.input),
        nlocktime: nlocktime_class(tx.lock_time.to_consensus_u32(), block.height),
        locktime_offset: locktime_offset_class(tx.lock_time.to_consensus_u32(), block.height),
        input_order: input_order_class(tx, block),
        output_order: output_order_class(tx),
        output_structure: output_structure(&tx.output),
        input_types: input_type_class(tx, block),
        output_types,
        low_r,
        sighash: sighash_class(&tx.input),
        uncompressed_pubkey: uncompressed_pubkey_class(tx),
        op_return: tx.output.iter().any(|o| o.script_pubkey.is_op_return()),
        input_subtype: input_subtype_class(tx, block),
        low_s,
        ecdsa_sigs,
        input_age: input_age_class(tx, block),
        feerate_bucket,
        round_feerate,
        change_position,
        change_type,
    })
}

/// Shared test fixtures. Defined here (not inside `mod tests`) because Tasks 6, 7 and 8
/// build their tests on the same Cake-shaped block.
#[cfg(test)]
pub(crate) mod tests_support {
    use bitcoin::{Amount, OutPoint, ScriptBuf, Sequence, Transaction, TxIn, TxOut, Witness};
    use lumen_primitives::traits::SourcedBlock;
    use lumen_primitives::traits::block_source::SpentOutput;
    use std::collections::HashMap;

    pub(crate) const EPOCH_TEST_HEIGHT: u32 = 960_000;

    pub(crate) fn p2wpkh_spk() -> ScriptBuf {
        ScriptBuf::from_hex("0014000102030405060708090a0b0c0d0e0f10111213").unwrap()
    }

    /// A P2TR scriptPubKey (`OP_1 <32-byte-program>`) — used as a script type distinct
    /// from `p2wpkh_spk()` so change-identification fixtures can have a payment output
    /// that is unambiguously NOT the same type as change.
    pub(crate) fn p2tr_spk() -> ScriptBuf {
        ScriptBuf::from_hex("5120000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
            .unwrap()
    }

    /// A correctly-shaped P2WPKH scriptPubKey (`OP_0 <20-byte-program>`, 22 bytes
    /// total) that really does classify as `OutputType::P2wpkh`. `p2wpkh_spk()` above
    /// is one byte too long (a pre-existing fixture quirk that classifies as
    /// `NonStandard`); every current test built on it only checks "uniform"/"same
    /// script", never the specific `OutputType`, so it is left as-is. The
    /// change-identification tests below assert the exact identified type, so they use
    /// this one instead.
    pub(crate) fn real_p2wpkh_spk() -> ScriptBuf {
        ScriptBuf::from_hex("0014000102030405060708090a0b0c0d0e0f10111213").unwrap()
    }

    /// Like `block_with`, but the prevout scriptPubKey for every input is the
    /// caller-supplied `prevout_spk` rather than the hardcoded `p2wpkh_spk()` — needed
    /// by tests that must control the exact `OutputType` the inputs classify as.
    pub(crate) fn block_with_prevout_script(
        tx: Transaction,
        height: u32,
        prevout_spk: ScriptBuf,
    ) -> SourcedBlock {
        let mut prevouts = HashMap::new();
        for txin in &tx.input {
            prevouts.insert(
                txin.previous_output,
                SpentOutput {
                    txout: TxOut {
                        value: Amount::from_sat(440_337),
                        script_pubkey: prevout_spk.clone(),
                    },
                    creation_height: height - 10,
                },
            );
        }
        let mut block = bitcoin::constants::genesis_block(bitcoin::Network::Bitcoin);
        block.txdata.push(tx);
        SourcedBlock {
            height,
            block,
            prevouts,
        }
    }

    /// Two inputs whose sequences are the Cake group-C bug, locktime 0, two p2wpkh
    /// outputs — the shape of the chain-proven Cake prior 9ecd77ab from payjoin#1597.
    pub(crate) fn cake_like_tx() -> Transaction {
        let input = |vout: u32, seq: u32| TxIn {
            previous_output: OutPoint::new(
                bitcoin::Txid::from_raw_hash(bitcoin::hashes::Hash::all_zeros()),
                vout,
            ),
            script_sig: ScriptBuf::new(),
            sequence: Sequence(seq),
            witness: Witness::from_slice(&[vec![0u8; 71], vec![0u8; 33]]),
        };
        Transaction {
            version: bitcoin::transaction::Version(2),
            lock_time: bitcoin::absolute::LockTime::ZERO,
            input: vec![input(0, 0x01), input(1, 0xffffffff)],
            output: vec![
                TxOut {
                    value: Amount::from_sat(29_358),
                    script_pubkey: p2wpkh_spk(),
                },
                TxOut {
                    value: Amount::from_sat(429_919),
                    script_pubkey: p2wpkh_spk(),
                },
            ],
        }
    }

    /// One legacy (P2PKH-shaped) input: `script_sig` carries a real DER-encoded ECDSA
    /// signature push (parseable by `bitcoin::ecdsa::Signature::from_slice`, which is
    /// what the low_r/low_s/sighash extractors use) followed by a pubkey push, and the
    /// witness is empty. This is the fixture the pre-existing `cake_like_tx` fixture
    /// could not provide: every prior fixture was segwit-only, which is exactly why
    /// the witness-only scanning defect in `low_r_class`, `low_s_class`,
    /// `uncompressed_pubkey_class`, and `sighash_class` went uncaught.
    ///
    /// `pubkey_bytes` is the caller-supplied pubkey push — its *shape* (33 bytes
    /// starting 0x02/0x03 vs. 65 bytes starting 0x04) is all `uncompressed_pubkey_class`
    /// inspects, so it need not be a genuine point on the curve.
    pub(crate) fn legacy_p2pkh_tx(pubkey_bytes: Vec<u8>) -> Transaction {
        use bitcoin::script::{Builder, PushBytesBuf};
        use bitcoin::secp256k1::{Message, Secp256k1, SecretKey};

        let secp = Secp256k1::new();
        let sk = SecretKey::from_slice(&[0x11; 32]).expect("valid secret key");
        let msg = Message::from_digest([0x22; 32]);
        let sig = secp.sign_ecdsa(&msg, &sk);
        let mut sig_bytes = sig.serialize_der().to_vec();
        sig_bytes.push(0x01); // SIGHASH_ALL

        let script_sig = Builder::new()
            .push_slice(PushBytesBuf::try_from(sig_bytes).expect("sig fits a push"))
            .push_slice(PushBytesBuf::try_from(pubkey_bytes).expect("pubkey fits a push"))
            .into_script();

        let input = TxIn {
            previous_output: OutPoint::new(
                bitcoin::Txid::from_raw_hash(bitcoin::hashes::Hash::all_zeros()),
                0,
            ),
            script_sig,
            sequence: Sequence(0xffffffff),
            witness: Witness::new(),
        };
        Transaction {
            version: bitcoin::transaction::Version(2),
            lock_time: bitcoin::absolute::LockTime::ZERO,
            input: vec![input],
            output: vec![TxOut {
                value: Amount::from_sat(50_000),
                script_pubkey: p2wpkh_spk(),
            }],
        }
    }

    /// One input whose `script_sig` carries `n` real, independently-generated
    /// DER-encoded ECDSA signature pushes back-to-back (multisig-scriptSig-shaped,
    /// though there is no redeem script — nothing here inspects one). Built for the
    /// `ecdsa_sigs` axis: `low_r_and_low_s_class`'s `candidate_signatures` extraction
    /// finds every push of at least 9 bytes starting `0x30` in the scriptSig, and each
    /// signature here is genuinely `bitcoin::ecdsa::Signature::from_slice`-parseable,
    /// so this produces exactly `n` counted signatures regardless of how the bucketing
    /// in `EcdsaSigCount` groups them.
    pub(crate) fn tx_with_n_ecdsa_sigs(n: u8) -> Transaction {
        use bitcoin::script::{Builder, PushBytesBuf};
        use bitcoin::secp256k1::{Message, Secp256k1, SecretKey};

        let secp = Secp256k1::new();
        let mut builder = Builder::new();
        for i in 0..n {
            let sk = SecretKey::from_slice(&[0x11 + i; 32]).expect("valid secret key");
            let msg = Message::from_digest([0x22 + i; 32]);
            let sig = secp.sign_ecdsa(&msg, &sk);
            let mut sig_bytes = sig.serialize_der().to_vec();
            sig_bytes.push(0x01); // SIGHASH_ALL
            builder =
                builder.push_slice(PushBytesBuf::try_from(sig_bytes).expect("sig fits a push"));
        }

        let input = TxIn {
            previous_output: OutPoint::new(
                bitcoin::Txid::from_raw_hash(bitcoin::hashes::Hash::all_zeros()),
                0,
            ),
            script_sig: builder.into_script(),
            sequence: Sequence(0xffffffff),
            witness: Witness::new(),
        };
        Transaction {
            version: bitcoin::transaction::Version(2),
            lock_time: bitcoin::absolute::LockTime::ZERO,
            input: vec![input],
            output: vec![TxOut {
                value: Amount::from_sat(50_000),
                script_pubkey: p2wpkh_spk(),
            }],
        }
    }

    /// A pure taproot key-path spend: one input whose prevout is P2TR and whose
    /// witness carries a single 64-byte Schnorr-shaped item (not ECDSA-DER-parseable,
    /// so `candidate_signatures`/`from_slice` finds nothing). Used to exercise
    /// `ecdsa_sigs`'s `None` bucket on a transaction with no ECDSA signature at all.
    pub(crate) fn taproot_key_path_tx() -> Transaction {
        let input = TxIn {
            previous_output: OutPoint::new(
                bitcoin::Txid::from_raw_hash(bitcoin::hashes::Hash::all_zeros()),
                0,
            ),
            script_sig: ScriptBuf::new(),
            sequence: Sequence(0xffffffff),
            witness: Witness::from_slice(&[vec![0u8; 64]]),
        };
        Transaction {
            version: bitcoin::transaction::Version(2),
            lock_time: bitcoin::absolute::LockTime::ZERO,
            input: vec![input],
            output: vec![TxOut {
                value: Amount::from_sat(50_000),
                script_pubkey: p2wpkh_spk(),
            }],
        }
    }

    pub(crate) fn block_with(tx: Transaction, height: u32) -> SourcedBlock {
        let mut prevouts = HashMap::new();
        for txin in &tx.input {
            prevouts.insert(
                txin.previous_output,
                SpentOutput {
                    txout: TxOut {
                        value: Amount::from_sat(440_337),
                        script_pubkey: p2wpkh_spk(),
                    },
                    // 10 blocks back: old enough that input_age is not SameBlock.
                    creation_height: height - 10,
                },
            );
        }
        let mut block = bitcoin::constants::genesis_block(bitcoin::Network::Bitcoin);
        block.txdata.push(tx);
        SourcedBlock {
            height,
            block,
            prevouts,
        }
    }

    /// One block containing exactly one classifiable Cake-shaped transaction.
    pub(crate) fn cake_like_block(height: u32) -> SourcedBlock {
        block_with(cake_like_tx(), height)
    }

    /// A fully-populated, classifiable `FingerprintVector` for tests in other modules
    /// (e.g. `features.rs`) that just need *some* valid vector to exercise, without
    /// caring about its specific axis values.
    pub(crate) fn classifiable_vector() -> super::FingerprintVector {
        let tx = cake_like_tx();
        let block = block_with(tx.clone(), EPOCH_TEST_HEIGHT);
        super::classify_tx(&tx, &block).expect("cake_like tx classifies")
    }

    /// Like `block_with`, but for several non-coinbase transactions at once. Needed by
    /// tests that must exercise more than one distinct fingerprint shape in a single
    /// epoch row — `block_with` only ever carries one transaction, so every key it can
    /// produce is trivially "all 1 of 1" or "none of 1". Every input's prevout resolves
    /// the same way `block_with` resolves it (a uniform p2wpkh coin, 10 blocks old), so
    /// callers only need to vary the transactions themselves to get different shapes.
    pub(crate) fn block_with_many(txs: Vec<Transaction>, height: u32) -> SourcedBlock {
        let mut prevouts = HashMap::new();
        for tx in &txs {
            for txin in &tx.input {
                prevouts.insert(
                    txin.previous_output,
                    SpentOutput {
                        txout: TxOut {
                            value: Amount::from_sat(440_337),
                            script_pubkey: p2wpkh_spk(),
                        },
                        creation_height: height - 10,
                    },
                );
            }
        }
        let mut block = bitcoin::constants::genesis_block(bitcoin::Network::Bitcoin);
        block.txdata.extend(txs);
        SourcedBlock {
            height,
            block,
            prevouts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::*;
    use super::*;
    use bitcoin::{Amount, TxOut};
    use std::collections::HashMap;

    #[test]
    fn classifies_the_cake_group_c_shape() {
        let tx = cake_like_tx();
        let block = block_with(tx.clone(), 960_000);
        let v = classify_tx(&tx, &block).expect("classifiable");

        assert_eq!(v.version, 2);
        assert_eq!(v.nsequence, NSequenceType::CakeGroupC);
        assert_eq!(v.nlocktime, NLockTimeType::Zero);
        // Assert uniformity, not the variant spelling: `OutputType`'s p2wpkh variant name
        // is defined in lumen-primitives and must not be guessed here.
        assert!(matches!(v.input_types, InputTypeClass::Uniform(_)));
        assert!(!v.op_return);
    }

    #[test]
    fn coinbase_is_not_classified() {
        let block = bitcoin::constants::genesis_block(bitcoin::Network::Bitcoin);
        let coinbase = block.txdata[0].clone();
        let sourced = SourcedBlock {
            height: 0,
            block,
            prevouts: HashMap::new(),
        };
        assert!(classify_tx(&coinbase, &sourced).is_none());
    }

    #[test]
    fn missing_prevout_is_not_classified() {
        let tx = cake_like_tx();
        let mut block = block_with(tx.clone(), 960_000);
        block.prevouts.clear();
        assert!(classify_tx(&tx, &block).is_none());
    }

    #[test]
    fn single_input_order_is_indeterminate() {
        let mut tx = cake_like_tx();
        tx.input.truncate(1);
        let block = block_with(tx.clone(), 960_000);
        let v = classify_tx(&tx, &block).expect("classifiable");
        assert_eq!(v.input_order, OrderClass::Indeterminate);
    }

    #[test]
    fn parse_key_round_trips_against_a_real_key_for_output() {
        // Deliberately built against `key_for`'s real output, not a hand-written
        // string, so the parser cannot silently drift from the formatter it inverts.
        let tx = cake_like_tx();
        let block = block_with(tx.clone(), EPOCH_TEST_HEIGHT);
        let v = classify_tx(&tx, &block).unwrap();

        // Includes HEURISTIC_AXES too: `change_position`/`change_type` still need to
        // round-trip through `axis_value`/`parse_key` even though they no longer enter
        // either joint-vector key (see `HEURISTIC_AXES`'s doc comment).
        let all_axes: Vec<&str> = CORE_AXES
            .iter()
            .chain(EXTENDED_AXES.iter())
            .chain(HEURISTIC_AXES.iter())
            .copied()
            .collect();
        let key = v.key_for(&all_axes);

        let parsed = parse_key(&key);

        assert_eq!(parsed.len(), all_axes.len(), "every axis must round-trip");
        for axis in &all_axes {
            assert_eq!(
                parsed.get(*axis).map(String::as_str),
                v.axis_value(axis).as_deref(),
                "axis {axis} must parse back to the value key_for wrote for it"
            );
        }
    }

    #[test]
    fn key_is_stable_and_distinguishes_vectors() {
        let tx = cake_like_tx();
        let block = block_with(tx.clone(), 960_000);
        let v = classify_tx(&tx, &block).unwrap();
        assert_eq!(v.key(), v.key(), "key must be deterministic");

        let mut other_tx = tx.clone();
        other_tx.lock_time = bitcoin::absolute::LockTime::from_height(959_999).unwrap();
        let other_block = block_with(other_tx.clone(), 960_000);
        let other = classify_tx(&other_tx, &other_block).unwrap();
        assert_ne!(v.key(), other.key(), "differing locktime class must differ");
    }

    // Regression coverage for the "witness-only scanning" defect: `low_r_class`,
    // `low_s_class`, `uncompressed_pubkey_class`, and `sighash_class` used to look
    // only at `txin.witness`, so a legacy (empty-witness, signature-in-script_sig)
    // input silently reported `Indeterminate` / `Na` on every one of these axes —
    // exactly the transactions `uncompressed_pubkey` and `low_r` exist to measure.

    #[test]
    fn legacy_input_low_r_is_not_indeterminate() {
        let mut pubkey = vec![0xab; 33];
        pubkey[0] = 0x02;
        let tx = legacy_p2pkh_tx(pubkey);
        let block = block_with(tx.clone(), EPOCH_TEST_HEIGHT);
        let v = classify_tx(&tx, &block).expect("classifiable");
        assert_ne!(
            v.low_r,
            Tri::Indeterminate,
            "a legacy input's scriptSig-borne signature must be seen"
        );
    }

    #[test]
    fn legacy_input_low_s_is_not_indeterminate() {
        let mut pubkey = vec![0xab; 33];
        pubkey[0] = 0x02;
        let tx = legacy_p2pkh_tx(pubkey);
        let block = block_with(tx.clone(), EPOCH_TEST_HEIGHT);
        let v = classify_tx(&tx, &block).expect("classifiable");
        assert_ne!(
            v.low_s,
            Tri::Indeterminate,
            "a legacy input's scriptSig-borne signature must be seen"
        );
    }

    #[test]
    fn legacy_input_uncompressed_pubkey_is_yes_for_65_byte_0x04_pubkey() {
        let mut pubkey = vec![0xab; 65];
        pubkey[0] = 0x04;
        let tx = legacy_p2pkh_tx(pubkey);
        let block = block_with(tx.clone(), EPOCH_TEST_HEIGHT);
        let v = classify_tx(&tx, &block).expect("classifiable");
        assert_eq!(v.uncompressed_pubkey, Tri::Yes);
    }

    #[test]
    fn legacy_input_uncompressed_pubkey_is_no_for_33_byte_compressed_pubkey() {
        let mut pubkey = vec![0xab; 33];
        pubkey[0] = 0x03;
        let tx = legacy_p2pkh_tx(pubkey);
        let block = block_with(tx.clone(), EPOCH_TEST_HEIGHT);
        let v = classify_tx(&tx, &block).expect("classifiable");
        assert_eq!(v.uncompressed_pubkey, Tri::No);
    }

    #[test]
    fn legacy_input_sighash_is_not_na() {
        let mut pubkey = vec![0xab; 33];
        pubkey[0] = 0x02;
        let tx = legacy_p2pkh_tx(pubkey);
        let block = block_with(tx.clone(), EPOCH_TEST_HEIGHT);
        let v = classify_tx(&tx, &block).expect("classifiable");
        assert_ne!(
            v.sighash,
            SighashType::Na,
            "a legacy input's scriptSig-borne signature must be seen"
        );
    }

    // ------------------------------------------------------------------------------
    // ecdsa_sigs: bucketed ECDSA signature count, computed alongside low_r/low_s.
    // ------------------------------------------------------------------------------

    #[test]
    fn ecdsa_sigs_is_one_for_a_single_signature() {
        let tx = tx_with_n_ecdsa_sigs(1);
        let block = block_with(tx.clone(), EPOCH_TEST_HEIGHT);
        let v = classify_tx(&tx, &block).expect("classifiable");
        assert_eq!(v.ecdsa_sigs, EcdsaSigCount::One);
    }

    #[test]
    fn ecdsa_sigs_is_few_for_two_signatures() {
        let tx = tx_with_n_ecdsa_sigs(2);
        let block = block_with(tx.clone(), EPOCH_TEST_HEIGHT);
        let v = classify_tx(&tx, &block).expect("classifiable");
        assert_eq!(v.ecdsa_sigs, EcdsaSigCount::Few);
    }

    #[test]
    fn ecdsa_sigs_is_many_for_five_signatures() {
        let tx = tx_with_n_ecdsa_sigs(5);
        let block = block_with(tx.clone(), EPOCH_TEST_HEIGHT);
        let v = classify_tx(&tx, &block).expect("classifiable");
        assert_eq!(v.ecdsa_sigs, EcdsaSigCount::Many);
    }

    #[test]
    fn ecdsa_sigs_is_none_for_a_pure_taproot_key_path_spend() {
        let tx = taproot_key_path_tx();
        let block = block_with_prevout_script(tx.clone(), EPOCH_TEST_HEIGHT, p2tr_spk());
        let v = classify_tx(&tx, &block).expect("classifiable");
        assert_eq!(v.ecdsa_sigs, EcdsaSigCount::None);
        // No ECDSA signature also means both Tri axes stay Indeterminate, not just
        // ecdsa_sigs — this must not alter low_r/low_s's existing semantics.
        assert_eq!(v.low_r, Tri::Indeterminate);
        assert_eq!(v.low_s, Tri::Indeterminate);
    }

    #[test]
    fn ecdsa_sigs_appears_in_axis_value_and_extended_axes() {
        let tx = tx_with_n_ecdsa_sigs(1);
        let block = block_with(tx.clone(), EPOCH_TEST_HEIGHT);
        let v = classify_tx(&tx, &block).expect("classifiable");

        assert!(
            EXTENDED_AXES.contains(&"ecdsa_sigs"),
            "must be an extended, not core, axis"
        );
        assert!(!CORE_AXES.contains(&"ecdsa_sigs"));
        assert_eq!(v.axis_value("ecdsa_sigs").as_deref(), Some("One"));
    }

    // ------------------------------------------------------------------------------
    // locktime_offset: the observable half of exact-tip-vs-backdating that `nlocktime`
    // cannot capture per transaction (see `NLockTimeType`'s doc comment in
    // `lumen_fingerprints_lib::types`).
    // ------------------------------------------------------------------------------

    #[test]
    fn locktime_offset_appears_in_axis_value_and_extended_axes() {
        let mut tx = cake_like_tx();
        // locktime == block_height: offset 0, the exact-tip case.
        tx.lock_time = bitcoin::absolute::LockTime::from_height(EPOCH_TEST_HEIGHT).unwrap();
        let block = block_with(tx.clone(), EPOCH_TEST_HEIGHT);
        let v = classify_tx(&tx, &block).expect("classifiable");

        assert!(
            EXTENDED_AXES.contains(&"locktime_offset"),
            "must be an extended, not core, axis"
        );
        assert!(!CORE_AXES.contains(&"locktime_offset"));
        assert_eq!(v.nlocktime, NLockTimeType::AntiFeeSnipe);
        assert_eq!(v.locktime_offset, LocktimeOffsetType::Zero);
        assert_eq!(v.axis_value("locktime_offset").as_deref(), Some("0"));

        // Round-trips through the joint-key parser too.
        let all_axes: Vec<&str> = CORE_AXES
            .iter()
            .chain(EXTENDED_AXES.iter())
            .copied()
            .collect();
        let key = v.key_for(&all_axes);
        let parsed = parse_key(&key);
        assert_eq!(parsed.get("locktime_offset").map(String::as_str), Some("0"));
    }

    #[test]
    fn locktime_offset_is_not_applicable_for_zero_locktime() {
        let tx = cake_like_tx(); // lock_time is ZERO by construction
        let block = block_with(tx.clone(), EPOCH_TEST_HEIGHT);
        let v = classify_tx(&tx, &block).expect("classifiable");

        assert_eq!(v.nlocktime, NLockTimeType::Zero);
        assert_eq!(v.locktime_offset, LocktimeOffsetType::NotApplicable);
        assert_eq!(
            v.axis_value("locktime_offset").as_deref(),
            Some("NotApplicable")
        );
    }

    // ------------------------------------------------------------------------------
    // Change-identification-derived axes: change_position / change_type. Both are
    // driven by this crate's own `crate::change::change_index_by_script_type`, so
    // these tests exercise the axis-mapping in `change_axes`/`classify_tx`, not the
    // heuristic itself (that lives in `change.rs`'s own tests).
    //
    // `change_type_matches_inputs` / `change_type_matches_outputs` used to be tested
    // here too, but both axes were deleted entirely (see
    // `docs/superpowers/specs/2026-07-29-remove-heuristic-axes-from-sparsity-design.md`):
    // they measured the change heuristic's own definition, not a fact about the chain.
    // ------------------------------------------------------------------------------

    #[test]
    fn change_axes_are_computed_when_change_is_identifiable() {
        // Two real-p2wpkh inputs, one p2tr payment output followed by one real-p2wpkh
        // change output: input types are unanimous P2wpkh, and exactly one output
        // (index 1) matches — an unambiguous, identifiable case, with change in the
        // LAST position.
        let mut tx = cake_like_tx();
        tx.output = vec![
            TxOut {
                value: Amount::from_sat(120_000),
                script_pubkey: p2tr_spk(),
            },
            TxOut {
                value: Amount::from_sat(430_000),
                script_pubkey: real_p2wpkh_spk(),
            },
        ];
        let block = block_with_prevout_script(tx.clone(), EPOCH_TEST_HEIGHT, real_p2wpkh_spk());
        let v = classify_tx(&tx, &block).expect("classifiable");

        assert_eq!(v.change_position, ChangePosition::Last);
        assert_eq!(v.change_type, Some(OutputType::P2wpkh));
    }

    #[test]
    fn change_axes_first_position_when_change_is_the_first_output() {
        let mut tx = cake_like_tx();
        tx.output = vec![
            TxOut {
                value: Amount::from_sat(430_000),
                script_pubkey: real_p2wpkh_spk(),
            },
            TxOut {
                value: Amount::from_sat(120_000),
                script_pubkey: p2tr_spk(),
            },
        ];
        let block = block_with_prevout_script(tx.clone(), EPOCH_TEST_HEIGHT, real_p2wpkh_spk());
        let v = classify_tx(&tx, &block).expect("classifiable");

        assert_eq!(v.change_position, ChangePosition::First);
        assert_eq!(v.change_type, Some(OutputType::P2wpkh));
    }

    #[test]
    fn change_axes_middle_position_with_three_outputs() {
        let mut tx = cake_like_tx();
        tx.output = vec![
            TxOut {
                value: Amount::from_sat(50_000),
                script_pubkey: p2tr_spk(),
            },
            TxOut {
                value: Amount::from_sat(430_000),
                script_pubkey: p2wpkh_spk(),
            },
            TxOut {
                value: Amount::from_sat(70_000),
                script_pubkey: p2tr_spk(),
            },
        ];
        let block = block_with(tx.clone(), EPOCH_TEST_HEIGHT);
        let v = classify_tx(&tx, &block).expect("classifiable");

        assert_eq!(v.change_position, ChangePosition::Middle);
    }

    #[test]
    fn change_axes_are_indeterminate_when_change_identification_is_inconclusive() {
        // cake_like_tx: two p2wpkh inputs, two p2wpkh outputs — both outputs match the
        // unanimous input type, so the heuristic cannot uniquely identify change. This
        // must NOT surface as `NotChange`-flavoured false values on either axis; both
        // must be `Indeterminate`.
        let tx = cake_like_tx();
        let block = block_with(tx.clone(), EPOCH_TEST_HEIGHT);
        let v = classify_tx(&tx, &block).expect("classifiable");

        assert_eq!(v.change_position, ChangePosition::Indeterminate);
        assert_eq!(v.change_type, None);

        // And the axis_value/key machinery must render that as the literal string
        // "Indeterminate", not silently drop the axis or print "None".
        assert_eq!(
            v.axis_value("change_position").as_deref(),
            Some("Indeterminate")
        );
        assert_eq!(
            v.axis_value("change_type").as_deref(),
            Some("Indeterminate")
        );
    }

    #[test]
    fn change_position_is_indeterminate_for_a_single_output_transaction() {
        let mut pubkey = vec![0xab; 33];
        pubkey[0] = 0x02;
        let tx = legacy_p2pkh_tx(pubkey);
        assert_eq!(
            tx.output.len(),
            1,
            "fixture is single-output by construction"
        );
        let block = block_with(tx.clone(), EPOCH_TEST_HEIGHT);
        let v = classify_tx(&tx, &block).expect("classifiable");

        assert_eq!(
            v.change_position,
            ChangePosition::Indeterminate,
            "fewer than 2 outputs means there is no change to place"
        );
    }

    // ------------------------------------------------------------------------------
    // `strip_retired_axes`: the `epochs-v2.jsonl` compatibility shim.
    // ------------------------------------------------------------------------------

    #[test]
    fn strip_retired_axes_removes_a_deleted_circular_axis_segment() {
        let with_retired = "nsequence=Rbf|change_type_matches_inputs=Yes|low_s=Yes";
        let without_retired = "nsequence=Rbf|low_s=Yes";

        assert_eq!(strip_retired_axes(with_retired), without_retired);
    }

    #[test]
    fn strip_retired_axes_removes_axes_moved_to_heuristic_axes() {
        // change_position/change_type are still valid axes (HEURISTIC_AXES), but a
        // stored key containing them must still be stripped down to the current
        // joint-key axis set, since HEURISTIC_AXES never belongs in a joint key.
        let with_heuristic = "nsequence=Rbf|change_position=Last|change_type=P2wpkh|low_s=Yes";
        let without_heuristic = "nsequence=Rbf|low_s=Yes";

        assert_eq!(strip_retired_axes(with_heuristic), without_heuristic);
    }

    #[test]
    fn strip_retired_axes_is_a_no_op_on_a_key_with_only_current_axes() {
        let key = "nsequence=Rbf|low_s=Yes|input_age=Older";
        assert_eq!(strip_retired_axes(key), key);
    }

    #[test]
    fn strip_retired_axes_merges_previously_distinct_keys() {
        // The whole point: two stored keys that differ only in a retired axis must
        // collapse to the identical stripped string, so aggregating by the stripped
        // key merges their counts.
        let a = "nsequence=Rbf|change_type_matches_inputs=Yes";
        let b = "nsequence=Rbf|change_type_matches_inputs=Indeterminate";

        assert_eq!(strip_retired_axes(a), strip_retired_axes(b));
    }

    // ------------------------------------------------------------------------------
    // `tx_fee_vsize_weight`: the shared fee/size primitive behind `feerate` and the
    // survey's block-minimum feerate.
    // ------------------------------------------------------------------------------

    #[test]
    fn fee_vsize_weight_and_coinbase() {
        // Reuse an existing fixture that has resolvable prevouts and a positive fee.
        let tx = cake_like_tx();
        let block = cake_like_block(800_000);
        let got = super::tx_fee_vsize_weight(&tx, &block);
        assert!(got.is_some());
        let (fee, vsize, weight) = got.unwrap();
        assert!(fee > 0 && vsize > 0 && weight > 0);

        // A coinbase-like tx (null prevout, no resolvable input) yields None. Build a
        // minimal coinbase: one input spending OutPoint::null(). `block_with` would
        // still insert that null outpoint into the prevouts map, so clear the map to
        // genuinely exercise the unresolved-input path: input absent ⇒ input_sum <
        // output_sum ⇒ checked_sub None ⇒ None.
        let mut cb = tx.clone();
        cb.input.truncate(1);
        cb.input[0].previous_output = bitcoin::OutPoint::null();
        let mut empty_block = block_with(cb.clone(), 800_000);
        empty_block.prevouts.clear();
        assert_eq!(super::tx_fee_vsize_weight(&cb, &empty_block), None);
    }

    #[test]
    fn vector_csv_line_is_txid_then_every_axis_value_in_order() {
        // classify a known fixture tx (reuse existing tests_support helpers)
        let mut pubkey = vec![0xab; 33];
        pubkey[0] = 0x02;
        let tx = legacy_p2pkh_tx(pubkey);
        let block = block_with(tx.clone(), EPOCH_TEST_HEIGHT);
        let v = classify_tx(&tx, &block).expect("classifiable");
        let header = FingerprintVector::vector_csv_header();
        let line = v.to_vector_csv_line("deadbeef");

        // header: "txid" + VECTOR_AXES
        assert!(header.starts_with("txid,"));
        assert_eq!(header.split(',').count(), 1 + VECTOR_AXES.len());
        // line: txid + one value per axis, same arity as header
        assert!(line.starts_with("deadbeef,"));
        assert_eq!(line.split(',').count(), 1 + VECTOR_AXES.len());
        // each field equals axis_value for that axis
        let fields: Vec<&str> = line.split(',').collect();
        for (i, axis) in VECTOR_AXES.iter().enumerate() {
            let expected = v.axis_value(axis).unwrap();
            assert_eq!(fields[i + 1], expected.as_ref(), "axis {axis} mismatched");
        }
    }

    #[test]
    fn vector_csv_line_rfc4180_escapes_fields_with_commas() {
        // Construct a tx with multiple distinct output types (P2TR + P2WPKH) so
        // output_types renders as `[P2tr, P2wpkh]` with an internal comma.
        // RFC-4180 escaping must wrap that field in double quotes so it stays
        // a single CSV column, preserving the fixed 1 + VECTOR_AXES.len() arity.
        let mut tx = cake_like_tx();
        tx.output = vec![
            TxOut {
                value: Amount::from_sat(50_000),
                script_pubkey: p2tr_spk(),
            },
            TxOut {
                value: Amount::from_sat(430_000),
                script_pubkey: real_p2wpkh_spk(),
            },
        ];
        let block = block_with_prevout_script(tx.clone(), EPOCH_TEST_HEIGHT, real_p2wpkh_spk());
        let v = classify_tx(&tx, &block).expect("classifiable");

        let line = v.to_vector_csv_line("txid");

        // RFC-4180 parse: split on unquoted commas only
        let mut fields = Vec::new();
        let mut current = String::new();
        let mut in_quotes = false;
        for ch in line.chars() {
            match ch {
                '"' => {
                    in_quotes = !in_quotes;
                    current.push(ch);
                }
                ',' if !in_quotes => {
                    fields.push(current.clone());
                    current.clear();
                }
                _ => current.push(ch),
            }
        }
        fields.push(current);

        // Exact arity: txid + one field per axis
        assert_eq!(
            fields.len(),
            1 + VECTOR_AXES.len(),
            "CSV must have exactly 1 + VECTOR_AXES.len() fields after RFC-4180 parsing"
        );

        // The output_types field must be quoted and contain the comma
        let output_types_index = VECTOR_AXES
            .iter()
            .position(|&a| a == "output_types")
            .expect("output_types in VECTOR_AXES");
        let output_types_field = &fields[1 + output_types_index];
        assert!(
            output_types_field.starts_with('"') && output_types_field.ends_with('"'),
            "output_types field with comma must be quoted: {output_types_field}"
        );
        assert!(
            output_types_field.contains(','),
            "output_types field must contain a comma (the type separator): {output_types_field}"
        );
    }
}
