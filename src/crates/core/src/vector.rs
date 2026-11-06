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

/// `Indeterminate` carries no signal, so it matches every template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tri {
    Yes,
    No,
    Indeterminate,
}

/// `Other` is what a shuffling wallet looks like.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderClass {
    Bip69,
    ValueAscending,
    ValueDescending,
    AgeAscending,
    Other,
    /// Fewer than two elements: no signal.
    Indeterminate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InputTypeClass {
    Uniform(OutputType),
    Mixed,
    Unknown,
}

/// The outer script type cannot tell, e.g., P2SH multisig from P2SH-P2WPKH.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InputSubtype {
    P2shP2wpkh,
    P2shMultisig,
    P2shOther,
    P2wshMultisig,
    P2wshOther,
    TaprootKeyPath,
    TaprootScriptPath,
    /// P2PK/P2PKH/P2WPKH: nothing to resolve.
    Bare,
    Mixed,
    Indeterminate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgeClass {
    SameBlock,
    Within6,
    Within144,
    Older,
    Indeterminate,
}

/// Low-R grinding is probabilistic (non-grinders hit low-R about half the time), so
/// `low_r=Yes` is a coin flip on `One` and near-proof of grinding on `Many`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EcdsaSigCount {
    /// Pure taproot key path, or nothing parseable as ECDSA.
    None,
    One,
    /// 2-4.
    Few,
    /// 5+.
    Many,
}

/// `Indeterminate` also covers fewer than 2 outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChangePosition {
    First,
    Last,
    Middle,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FingerprintVector {
    pub version: i32,
    pub nsequence: NSequenceType,
    pub nlocktime: NLockTimeType,
    /// Separates exact tip from backdating, which `nlocktime` cannot; varies with
    /// confirmation timing, hence `EXTENDED_AXES`.
    pub locktime_offset: LocktimeOffsetType,
    pub input_order: OrderClass,
    pub output_order: OrderClass,
    pub output_structure: OutputStructureType,
    pub input_types: InputTypeClass,
    /// Sorted and deduplicated: stands in for base58/bech32/bech32m without an encoder.
    pub output_types: Vec<OutputType>,
    pub low_r: Tri,
    pub sighash: SighashType,
    pub uncompressed_pubkey: Tri,
    pub op_return: bool,
    pub input_subtype: InputSubtype,
    pub low_s: Tri,
    /// Describes the coins spent, not wallet policy, hence `EXTENDED_AXES`.
    pub ecdsa_sigs: EcdsaSigCount,
    pub input_age: AgeClass,
    /// sat/vB: floor below 10, then 10s up to 100, then "100+".
    pub feerate_bucket: String,
    pub round_feerate: bool,
    pub change_position: ChangePosition,
    pub change_type: Option<OutputType>,
}

/// Wallet policy: chosen the same way on every transaction.
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

/// Vary with the coins at hand rather than wallet policy.
pub const EXTENDED_AXES: &[&str] = &[
    "input_subtype",
    "low_s",
    "ecdsa_sigs",
    "input_age",
    "feerate_bucket",
    "round_feerate",
    "locktime_offset",
];

/// Never in joint keys: the heuristic's abstention rate is a property of the method,
/// not the chain, and would distort the measured sparsity.
pub const HEURISTIC_AXES: &[&str] = &["change_position", "change_type"];

/// Order is stable: it defines the `--vectors` CSV column order.
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

    /// Explicit axis set so sparsity can be reported per set of measured axes.
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

    pub fn key(&self) -> String {
        self.key_for(CORE_AXES)
    }

    pub fn vector_csv_header() -> String {
        let mut s = String::from("txid");
        for axis in VECTOR_AXES {
            s.push(',');
            s.push_str(axis);
        }
        s
    }

    /// RFC-4180-escaped so Vec-valued axes like `output_types` stay one column.
    pub fn to_vector_csv_line(&self, txid: &str) -> String {
        let mut s = csv_escape(txid);
        for axis in VECTOR_AXES {
            s.push(',');
            let value = self.axis_value(axis);
            s.push_str(&csv_escape(value.as_deref().unwrap_or("")));
        }
        s
    }
}

/// No wildcard arms, so a new variant without a spelling here fails to compile.
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
        NSequenceType::CakeGroupC => "first 0x01 rest 0xffffffff",
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

fn csv_escape(field: &str) -> String {
    if !field.contains(',')
        && !field.contains('"')
        && !field.contains('\n')
        && !field.contains('\r')
    {
        field.to_string()
    } else {
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

/// Splits on the first `=` only; axis names never contain `=`.
pub fn parse_key(key: &str) -> BTreeMap<String, String> {
    key.split('|')
        .filter_map(|segment| segment.split_once('='))
        .map(|(axis, value)| (axis.to_string(), value.to_string()))
        .collect()
}

/// Lets keys written under an older axis set report against the current one; keys
/// differing only in a stripped axis collapse into one.
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

    // Age first: the more specific claim, and shuffles rarely satisfy it by accident.
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

/// "Change last" shows up as `Other`; it is only decisive over many transactions.
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
            // Key path is one item plus an optional annex; an empty witness carries no
            // signal, so it must not default to `TaprootScriptPath`.
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

/// Same extractors as `fingerprints::input::low_r_grinding`.
fn candidate_signatures(txin: &bitcoin::TxIn) -> Vec<Vec<u8>> {
    let mut sigs = extract_signatures_from_script_sig(txin.script_sig.as_bytes());
    let witness_items: Vec<Vec<u8>> = txin.witness.iter().map(|item| item.to_vec()).collect();
    sigs.extend(extract_signatures_from_witness(&witness_items));
    sigs
}

/// Low-R: every ECDSA signature's R is below 2^255, i.e. 71-byte DER.
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
                // libsecp256k1 normalises S, so the top bit of s is the practical high-S test.
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

/// `None` for coinbase: its null-prevout input is absent from `block`, so inputs < outputs.
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

    // Typed feerates land within a hair of an integer; estimator output essentially never does.
    let round = (rate - rate.round()).abs() < 0.01 && rate >= 1.0;
    (bucket, round)
}

/// The heuristic's `None` means inconclusive, never "not change". Callers must have
/// resolved every input's prevout.
fn change_axes(
    tx: &Transaction,
    input_types: &[OutputType],
    output_types_ordered: &[OutputType],
) -> (ChangePosition, Option<OutputType>) {
    let Some(change_index) = change_index_by_script_type(input_types, output_types_ordered) else {
        return (ChangePosition::Indeterminate, None);
    };

    // One output has no change to place, whatever index the heuristic returned.
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

/// Scans scriptSig too: uncompressed keys are a legacy-era phenomenon.
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

    // Positional, not the deduped `output_types`: `change_axes` needs the matching index.
    let output_types_ordered: Vec<OutputType> = tx
        .output
        .iter()
        .map(|o| lumen_primitives::classify_script_pubkey(o.script_pubkey.as_bytes()))
        .collect();

    let mut output_types = output_types_ordered.clone();
    output_types.sort_by_key(|t| t.as_u32());
    output_types.dedup();

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

    pub(crate) fn p2tr_spk() -> ScriptBuf {
        ScriptBuf::from_hex("5120000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
            .unwrap()
    }

    pub(crate) fn real_p2wpkh_spk() -> ScriptBuf {
        ScriptBuf::from_hex("0014000102030405060708090a0b0c0d0e0f10111213").unwrap()
    }

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

    /// Only the pubkey's shape matters, so it need not be a point on the curve.
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
                    // Old enough that input_age is not SameBlock.
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

    pub(crate) fn cake_like_block(height: u32) -> SourcedBlock {
        block_with(cake_like_tx(), height)
    }

    pub(crate) fn classifiable_vector() -> super::FingerprintVector {
        let tx = cake_like_tx();
        let block = block_with(tx.clone(), EPOCH_TEST_HEIGHT);
        super::classify_tx(&tx, &block).expect("cake_like tx classifies")
    }

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
        // Built from `key_for`'s output so the parser cannot drift from the formatter.
        let tx = cake_like_tx();
        let block = block_with(tx.clone(), EPOCH_TEST_HEIGHT);
        let v = classify_tx(&tx, &block).unwrap();

        // Heuristic axes are not in either joint key but must still round-trip.
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

    #[test]
    fn locktime_offset_appears_in_axis_value_and_extended_axes() {
        let mut tx = cake_like_tx();
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

    #[test]
    fn change_axes_are_computed_when_change_is_identifiable() {
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
        // Both outputs match the unanimous input type, so change is not identifiable.
        let tx = cake_like_tx();
        let block = block_with(tx.clone(), EPOCH_TEST_HEIGHT);
        let v = classify_tx(&tx, &block).expect("classifiable");

        assert_eq!(v.change_position, ChangePosition::Indeterminate);
        assert_eq!(v.change_type, None);

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

    #[test]
    fn strip_retired_axes_removes_a_deleted_circular_axis_segment() {
        let with_retired = "nsequence=Rbf|change_type_matches_inputs=Yes|low_s=Yes";
        let without_retired = "nsequence=Rbf|low_s=Yes";

        assert_eq!(strip_retired_axes(with_retired), without_retired);
    }

    #[test]
    fn strip_retired_axes_removes_axes_moved_to_heuristic_axes() {
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
        let a = "nsequence=Rbf|change_type_matches_inputs=Yes";
        let b = "nsequence=Rbf|change_type_matches_inputs=Indeterminate";

        assert_eq!(strip_retired_axes(a), strip_retired_axes(b));
    }

    #[test]
    fn fee_vsize_weight_and_coinbase() {
        let tx = cake_like_tx();
        let block = cake_like_block(800_000);
        let got = super::tx_fee_vsize_weight(&tx, &block);
        assert!(got.is_some());
        let (fee, vsize, weight) = got.unwrap();
        assert!(fee > 0 && vsize > 0 && weight > 0);

        // Prevouts cleared so the null-outpoint input is unresolved and the fee underflows.
        let mut cb = tx.clone();
        cb.input.truncate(1);
        cb.input[0].previous_output = bitcoin::OutPoint::null();
        let mut empty_block = block_with(cb.clone(), 800_000);
        empty_block.prevouts.clear();
        assert_eq!(super::tx_fee_vsize_weight(&cb, &empty_block), None);
    }

    #[test]
    fn vector_csv_line_is_txid_then_every_axis_value_in_order() {
        let mut pubkey = vec![0xab; 33];
        pubkey[0] = 0x02;
        let tx = legacy_p2pkh_tx(pubkey);
        let block = block_with(tx.clone(), EPOCH_TEST_HEIGHT);
        let v = classify_tx(&tx, &block).expect("classifiable");
        let header = FingerprintVector::vector_csv_header();
        let line = v.to_vector_csv_line("deadbeef");

        assert!(header.starts_with("txid,"));
        assert_eq!(header.split(',').count(), 1 + VECTOR_AXES.len());
        assert!(line.starts_with("deadbeef,"));
        assert_eq!(line.split(',').count(), 1 + VECTOR_AXES.len());
        let fields: Vec<&str> = line.split(',').collect();
        for (i, axis) in VECTOR_AXES.iter().enumerate() {
            let expected = v.axis_value(axis).unwrap();
            assert_eq!(fields[i + 1], expected.as_ref(), "axis {axis} mismatched");
        }
    }

    #[test]
    fn vector_csv_line_rfc4180_escapes_fields_with_commas() {
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

        assert_eq!(
            fields.len(),
            1 + VECTOR_AXES.len(),
            "CSV must have exactly 1 + VECTOR_AXES.len() fields after RFC-4180 parsing"
        );

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
