use std::collections::HashSet;

use bitcoin::Amount;
use lumen_primitives::{
    HasPrevOutpoint, HasScriptPubkey, HasScriptSig, HasSequence, HasValue, HasWitness, OutputType,
};

use crate::classify::extract_signatures_from_script_sig;
use crate::types::{
    InputSortingType, LocktimeOffsetType, NLockTimeType, NSequenceType, OutputStructureType,
    SighashType,
};

/// Returns true if any input signals RBF.
pub fn tx_signals_rbf(inputs: &[impl HasSequence]) -> bool {
    inputs.iter().any(|input| input.sequence() < 0xfffffffe)
}

/// Returns true if locktime is non-zero (heuristic for anti-fee-sniping).
pub fn anti_fee_snipe(locktime: u32) -> bool {
    locktime != 0
}

/// Returns true if any output scriptPubKey matches any prevout scriptPubKey.
pub fn address_reuse(outputs: &[impl HasScriptPubkey], prevouts: &[impl HasScriptPubkey]) -> bool {
    let input_scripts: HashSet<Vec<u8>> =
        prevouts.iter().map(|p| p.script_pubkey_bytes()).collect();
    let output_scripts: HashSet<Vec<u8>> =
        outputs.iter().map(|o| o.script_pubkey_bytes()).collect();

    !input_scripts.is_disjoint(&output_scripts)
}

/// Returns true if prevouts have more than one distinct scriptPubKey type.
pub fn mixed_input_types(prevouts: &[impl HasScriptPubkey]) -> bool {
    let types: HashSet<OutputType> = prevouts.iter().map(|p| p.output_type()).collect();
    types.len() > 1
}

/// Returns the input sorting types detected in the transaction.
///
/// `inputs` provides outpoint data for BIP69 checking.
/// `prevout_values` provides the value of each prevout (in input order) for
/// ascending/descending checking.
// TODO: should only have one argument. A rich input type that has prevout values and outpoint.
pub fn input_order<I, P>(inputs: &[I], prevout_values: &[P]) -> Vec<InputSortingType>
where
    I: HasPrevOutpoint,
    P: HasValue,
{
    if inputs.len() == 1 {
        return vec![InputSortingType::Single];
    }

    let mut sorting_types = Vec::new();
    let amounts: Vec<Amount> = prevout_values.iter().map(|p| p.value()).collect();

    // Only check ascending/descending when amounts are not all equal — equal amounts
    // trivially satisfy both orderings and reveal nothing about sorting intent.
    let all_equal = amounts.windows(2).all(|w| w[0] == w[1]);
    if !amounts.is_empty() && !all_equal {
        let mut sorted_amounts = amounts.clone();
        sorted_amounts.sort();
        if amounts == sorted_amounts {
            sorting_types.push(InputSortingType::Ascending);
        }

        sorted_amounts.reverse();
        if amounts == sorted_amounts {
            sorting_types.push(InputSortingType::Descending);
        }
    }

    // Check BIP69 sorting.
    // BIP69 sorts by txid as a little-endian uint256 (MSB = wire byte[31] = display byte[0]),
    // which is equivalent to lexicographic comparison of the reversed wire bytes.
    let outpoints: Vec<([u8; 32], u32)> = inputs
        .iter()
        .map(|i| (i.prev_outpoint_txid_bytes(), i.prev_outpoint_vout()))
        .collect();

    let mut sorted_outpoints = outpoints.clone();
    sorted_outpoints.sort_by(|a, b| {
        a.0.iter()
            .rev()
            .cmp(b.0.iter().rev())
            .then_with(|| a.1.cmp(&b.1))
    });

    if outpoints == sorted_outpoints {
        sorting_types.push(InputSortingType::Bip69);
    }

    if sorting_types.is_empty() {
        sorting_types.push(InputSortingType::Unknown);
    }

    sorting_types
}

/// Returns true if any input opts in to nLockTime enforcement (nSequence < 0xFFFFFFFE)
/// but the transaction sets nLockTime = 0.
///
/// This is a wallet fingerprint: the wallet enables nLockTime enforcement via nSequence
/// but doesn't actually use it (e.g. RBF signaling without anti-fee-sniping).
/// No well-implemented wallet does this intentionally.
pub fn nlocktime_optin_without_use(inputs: &[impl HasSequence], locktime: u32) -> bool {
    locktime == 0 && inputs.iter().any(|input| input.sequence() < 0xfffffffe)
}

/// Returns true if any input enables BIP 68 relative timelocks (nSequence < 0x80000000)
/// while the transaction also sets a non-zero absolute nLockTime.
///
/// Consensus-valid but semantically contradictory: wallets doing anti-fee-sniping don't enter BIP 68 range.
/// Strongly fingerprints a specific protocol or wallet composing these fields incorrectly.
pub fn bip68_with_absolute_locktime(inputs: &[impl HasSequence], locktime: u32) -> bool {
    locktime > 0 && inputs.iter().any(|input| input.sequence() < 0x80000000)
}

pub fn is_bip69_sorted(outputs: &[impl HasValue + HasScriptPubkey]) -> bool {
    let pairs: Vec<(Amount, Vec<u8>)> = outputs
        .iter()
        .map(|o| (o.value(), o.script_pubkey_bytes()))
        .collect();
    let amounts: Vec<Amount> = pairs.iter().map(|(v, _)| *v).collect();
    let unique_amounts: HashSet<Amount> = amounts.iter().copied().collect();

    if unique_amounts.len() != amounts.len() {
        // Duplicate amounts — check both value and scriptPubKey are sorted
        let mut sorted_pairs = pairs.clone();
        sorted_pairs.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        pairs == sorted_pairs
    } else {
        // Unique amounts — just check amounts are sorted
        amounts.windows(2).all(|w| w[0] <= w[1])
    }
}

/// Returns the output structure types detected in the transaction.
pub fn output_structure<O>(outputs: &[O]) -> OutputStructureType
where
    O: HasValue + HasScriptPubkey,
{
    if outputs.len() == 1 {
        return OutputStructureType::Single;
    }

    if outputs.len() == 2 {
        return OutputStructureType::Double;
    }
    OutputStructureType::Multi
}

/// Returns true if the transaction fee appears to be a round number,
/// suggesting manual fee entry rather than automatic fee estimation.
///
/// Checks whether the fee (in satoshis) is divisible by 1000.
/// Round satoshi values are common with manual entry,
/// while automatic fee estimation produces non-round amounts.
///
/// Returns `None` if the fee cannot be computed (e.g. input sum < output sum).
pub fn round_fee(input_values: &[impl HasValue], outputs: &[impl HasValue]) -> Option<bool> {
    let input_sum: u64 = input_values.iter().map(|v| v.value().to_sat()).sum();
    let output_sum: u64 = outputs.iter().map(|v| v.value().to_sat()).sum();
    let fee = input_sum.checked_sub(output_sum)?;
    Some(fee > 0 && fee % 1000 == 0)
}

/// Classify the transaction's nSequence pattern (mirrors decluster x_nsequence exactly).
pub fn nsequence_class(inputs: &[impl HasSequence]) -> NSequenceType {
    let s: Vec<u32> = inputs.iter().map(|i| i.sequence()).collect();
    if s.is_empty() {
        return NSequenceType::MixedOther; // no inputs (e.g. coinbase): no signal
    }
    if s.len() >= 2 && s[0] == 0x01 && s[1..].iter().all(|&v| v == 0xffffffff) {
        return NSequenceType::CakeGroupC;
    }
    if s.contains(&0x01) {
        return NSequenceType::Lone0x01;
    }
    if s.iter().all(|&v| v == 0xfffffffd) {
        return NSequenceType::Rbf;
    }
    if s.iter().all(|&v| v == 0xfffffffe) {
        return NSequenceType::Final;
    }
    if s.iter().all(|&v| v == 0xffffffff) {
        return NSequenceType::Max;
    }
    NSequenceType::MixedOther
}

/// Classify the transaction's sighash signature from the first signature-shaped witness
/// item of each input, falling back to the scriptSig for legacy inputs that carry no
/// signature in the witness — P2PKH, bare P2PK, and P2SH-wrapped multisig all carry
/// their signature there rather than in the witness. Scanning for the first
/// signature-shaped item (rather than just the first item) is what lets native P2WSH /
/// bare multisig be classified: their witness leads with the empty `OP_CHECKMULTISIG`
/// dummy push, so `first()` alone would see a zero-length item and miss the signatures
/// behind it. Returns Na when no signature-shaped data is found on any input, Mixed if
/// inputs disagree.
pub fn sighash_class(inputs: &[impl HasWitness + HasScriptSig]) -> SighashType {
    let mut seen: HashSet<SighashType> = HashSet::new();
    for input in inputs {
        let items = input.witness_items();
        let witness_sig = items
            .iter()
            .find(|item| matches!(item.len(), 64 | 65 | 68..=73))
            .cloned();
        let sig = match witness_sig {
            Some(sig) => Some(sig),
            None => {
                let script_sig_bytes = input.script_sig_bytes();
                extract_signatures_from_script_sig(&script_sig_bytes)
                    .into_iter()
                    .next()
            }
        };
        let Some(sig) = sig else { continue };
        let t = match sig.len() {
            64 => SighashType::TaprootDefault,
            65 => SighashType::TaprootExplicit,
            68..=73 => match sig.last() {
                Some(0x01) => SighashType::All,
                Some(0x02) => SighashType::None,
                Some(0x03) => SighashType::Single,
                Some(0x81) => SighashType::AcpAll,
                Some(0x82) => SighashType::AcpNone,
                Some(0x83) => SighashType::AcpSingle,
                _ => SighashType::Other,
            },
            _ => continue,
        };
        seen.insert(t);
    }
    match seen.len() {
        0 => SighashType::Na,
        1 => *seen.iter().next().expect("len checked"),
        _ => SighashType::Mixed,
    }
}

/// Interpret nLockTime against the height of the block containing the transaction.
///
/// The anti-fee-sniping window is `(height-100, height]`: Core sets the locktime to
/// the current height, but ~10% of the time picks a height up to 100 blocks back.
/// Both cases land in the single `AntiFeeSnipe` bucket — see `NLockTimeType`'s doc
/// comment for why a per-transaction split between them is not just hard, but
/// impossible. `locktime_offset_class` below is where that behaviour is measurable.
pub fn nlocktime_class(locktime: u32, block_height: u32) -> NLockTimeType {
    const LOCKTIME_THRESHOLD: u32 = 500_000_000;
    if locktime == 0 {
        return NLockTimeType::Zero;
    }
    if locktime >= LOCKTIME_THRESHOLD {
        return NLockTimeType::Timestamp;
    }
    if locktime > block_height {
        return NLockTimeType::Future;
    }
    if locktime > block_height.saturating_sub(100) {
        return NLockTimeType::AntiFeeSnipe;
    }
    NLockTimeType::Backdated
}

/// Bucket a raw `block_height - locktime` offset into `LocktimeOffsetType`'s nine
/// closed-interval buckets: `0`, `1`, `2`, `3`, `4-6`, `7-12`, `13-25`, `26-50`,
/// `51-100`. A pure numeric mapping (never `NotApplicable`) — separated from
/// `locktime_offset_class` below so the bucket boundaries can be exercised directly by
/// offset value, independent of whether that offset could actually arise from a real
/// `(locktime, block_height)` pair in the `AntiFeeSnipe` window (which tops out at 99,
/// not 100 — see `locktime_offset_class`).
pub fn locktime_offset_bucket(offset: u32) -> LocktimeOffsetType {
    match offset {
        0 => LocktimeOffsetType::Zero,
        1 => LocktimeOffsetType::One,
        2 => LocktimeOffsetType::Two,
        3 => LocktimeOffsetType::Three,
        4..=6 => LocktimeOffsetType::FourToSix,
        7..=12 => LocktimeOffsetType::SevenToTwelve,
        13..=25 => LocktimeOffsetType::ThirteenToTwentyFive,
        26..=50 => LocktimeOffsetType::TwentySixToFifty,
        _ => LocktimeOffsetType::FiftyOneToHundred,
    }
}

/// The observable half of the exact-tip-vs-backdating question `NLockTimeType` cannot
/// answer per transaction (see its doc comment): `block_height - locktime`, bucketed,
/// but only when `nlocktime_class` places the transaction in the `AntiFeeSnipe` window.
/// Every other `NLockTimeType` value carries no interpretable offset — `Zero`,
/// `Backdated`, and `Future` have no tip-height signal, and `Timestamp` is not a height
/// at all — so those all map to `NotApplicable`.
pub fn locktime_offset_class(locktime: u32, block_height: u32) -> LocktimeOffsetType {
    if nlocktime_class(locktime, block_height) != NLockTimeType::AntiFeeSnipe {
        return LocktimeOffsetType::NotApplicable;
    }
    // Safe: `AntiFeeSnipe` is only reached when `locktime <= block_height`.
    locktime_offset_bucket(block_height - locktime)
}

#[cfg(test)]
mod class_tests {
    use super::{
        locktime_offset_bucket, locktime_offset_class, nlocktime_class, nsequence_class,
        sighash_class,
    };
    use crate::types::{LocktimeOffsetType, NLockTimeType, NSequenceType, SighashType};
    use lumen_primitives::{HasScriptSig, HasSequence, HasWitness};

    struct Seq(u32);
    impl HasSequence for Seq {
        fn sequence(&self) -> u32 {
            self.0
        }
    }

    /// A synthetic input carrying both witness items and scriptSig bytes, so tests can
    /// exercise the legacy (empty-witness, non-empty-scriptSig) fallback path.
    struct Wit {
        witness: Vec<Vec<u8>>,
        script_sig: Vec<u8>,
    }
    impl Wit {
        fn witness(items: Vec<Vec<u8>>) -> Self {
            Wit {
                witness: items,
                script_sig: Vec::new(),
            }
        }
    }
    impl HasWitness for Wit {
        fn witness_items(&self) -> Vec<Vec<u8>> {
            self.witness.clone()
        }
    }
    impl HasScriptSig for Wit {
        fn script_sig_bytes(&self) -> Vec<u8> {
            self.script_sig.clone()
        }
    }

    #[test]
    fn classifies_nsequence_patterns() {
        let c = |seqs: &[u32]| nsequence_class(&seqs.iter().map(|&s| Seq(s)).collect::<Vec<_>>());
        assert_eq!(
            c(&[0x01, 0xffffffff, 0xffffffff]),
            NSequenceType::CakeGroupC
        );
        assert_eq!(c(&[0x01, 0xfffffffd]), NSequenceType::Lone0x01);
        assert_eq!(c(&[0xfffffffd, 0xfffffffd]), NSequenceType::Rbf);
        assert_eq!(c(&[0xfffffffe]), NSequenceType::Final);
        assert_eq!(c(&[0xffffffff, 0xffffffff]), NSequenceType::Max);
        assert_eq!(c(&[0xfffffffd, 0xffffffff]), NSequenceType::MixedOther);
    }

    #[test]
    fn classifies_sighash_from_witness() {
        let taproot_default = Wit::witness(vec![vec![0u8; 64]]);
        assert_eq!(
            sighash_class(&[taproot_default]),
            SighashType::TaprootDefault
        );

        let taproot_explicit = Wit::witness(vec![vec![0u8; 65]]);
        assert_eq!(
            sighash_class(&[taproot_explicit]),
            SighashType::TaprootExplicit
        );

        let mut ecdsa_all = vec![0u8; 71];
        *ecdsa_all.last_mut().unwrap() = 0x01;
        assert_eq!(
            sighash_class(&[Wit::witness(vec![ecdsa_all])]),
            SighashType::All
        );

        assert_eq!(sighash_class(&[Wit::witness(vec![])]), SighashType::Na);

        let mut acp = vec![0u8; 71];
        *acp.last_mut().unwrap() = 0x81;
        assert_eq!(
            sighash_class(&[Wit::witness(vec![vec![0u8; 64]]), Wit::witness(vec![acp])]),
            SighashType::Mixed
        );
    }

    /// The legacy blindness this fix addresses: a P2PKH-style input has an empty
    /// witness and carries its signature in the scriptSig instead. Before this fix,
    /// `sighash_class` looked only at `witness_items()` and such an input silently
    /// classified as `Na` — indistinguishable from "no signature at all".
    #[test]
    fn classifies_sighash_from_script_sig_when_witness_is_empty() {
        // scriptSig-shaped push: OP_PUSHBYTES_71 <0x30 ...68 more bytes... SIGHASH_ALL>.
        let mut der_sig = vec![0x30];
        der_sig.extend(std::iter::repeat_n(0u8, 69));
        der_sig.push(0x01); // SIGHASH_ALL
        assert_eq!(der_sig.len(), 71);

        let mut script_sig = vec![0x47]; // push 71 bytes
        script_sig.extend(&der_sig);
        script_sig.push(0x21); // push 33 bytes (a pubkey push, ignored by the extractor)
        script_sig.extend(std::iter::repeat_n(0u8, 33));

        let legacy = Wit {
            witness: Vec::new(),
            script_sig,
        };
        assert_eq!(sighash_class(&[legacy]), SighashType::All);
    }

    /// A native P2WSH multisig input's witness is `[<empty CHECKMULTISIG dummy>, <sig1>,
    /// ..., <witnessScript>]` — the first stack item is the empty `OP_0` push the
    /// `OP_CHECKMULTISIG` off-by-one consumes, not a signature. Before this fix,
    /// `sighash_class` inspected only `witness_items().first()`, hit that empty (len-0)
    /// item, fell through `_ => continue`, and silently classified the input as `Na` —
    /// even though its signatures carry real sighash bytes.
    #[test]
    fn classifies_sighash_from_p2wsh_multisig_witness() {
        // A DER ECDSA signature ending in SIGHASH_ALL, length inside the 68..=73 window.
        let mut der_sig = vec![0x30];
        der_sig.extend(std::iter::repeat_n(0u8, 69));
        der_sig.push(0x01); // SIGHASH_ALL
        assert_eq!(der_sig.len(), 71);

        // The trailing witnessScript is not signature-shaped, so it must be ignored.
        let witness_script = vec![0xaeu8; 40]; // ends in OP_CHECKMULTISIG

        let p2wsh_multisig = Wit::witness(vec![
            Vec::new(),      // OP_0 CHECKMULTISIG dummy
            der_sig.clone(), // sig1
            der_sig,         // sig2
            witness_script,  // 2-of-2 witnessScript
        ]);
        assert_eq!(sighash_class(&[p2wsh_multisig]), SighashType::All);
    }

    #[test]
    fn classifies_nlocktime_against_height() {
        let height = 960_000;
        assert_eq!(nlocktime_class(0, height), NLockTimeType::Zero);
        // The whole (height-100, height] window is one bucket: exact tip and
        // backdating are not distinguishable per transaction (see `NLockTimeType`'s
        // doc comment) — only `locktime_offset_class` can tell them apart, and only
        // distributionally.
        assert_eq!(
            nlocktime_class(960_000, height),
            NLockTimeType::AntiFeeSnipe
        );
        assert_eq!(
            nlocktime_class(959_999, height),
            NLockTimeType::AntiFeeSnipe
        );
        // The last value still inside the window: height - 99.
        assert_eq!(
            nlocktime_class(959_901, height),
            NLockTimeType::AntiFeeSnipe
        );
        // One block further back falls outside the window.
        assert_eq!(nlocktime_class(959_900, height), NLockTimeType::Backdated);
        assert_eq!(nlocktime_class(1, height), NLockTimeType::Backdated);
        // Above the tip but below the 500M timestamp threshold.
        assert_eq!(nlocktime_class(960_001, height), NLockTimeType::Future);
        // >= 500_000_000 is a unix timestamp, not a height.
        assert_eq!(
            nlocktime_class(1_700_000_000, height),
            NLockTimeType::Timestamp
        );
    }

    #[test]
    fn nlocktime_type_discriminants_unchanged() {
        // Persisted via `as_u32()` (e.g. joint-vector keys derived transitively through
        // `nlocktime_str`/consumers keyed off these values). `AntiFeeSnipe` reclaims
        // discriminant 1 — the original single bucket's value, freed up when the
        // (since-reverted) `ExactTip`/`RecentBackdated` split moved it aside — and
        // every other variant's discriminant is unchanged.
        assert_eq!(NLockTimeType::Zero.as_u32(), 0);
        assert_eq!(NLockTimeType::AntiFeeSnipe.as_u32(), 1);
        assert_eq!(NLockTimeType::Backdated.as_u32(), 2);
        assert_eq!(NLockTimeType::Future.as_u32(), 3);
        assert_eq!(NLockTimeType::Timestamp.as_u32(), 4);
    }

    // ------------------------------------------------------------------------------
    // locktime_offset: the observable half of exact-tip vs. backdating.
    // ------------------------------------------------------------------------------

    #[test]
    fn locktime_offset_buckets_every_boundary() {
        assert_eq!(locktime_offset_bucket(0), LocktimeOffsetType::Zero);
        assert_eq!(locktime_offset_bucket(1), LocktimeOffsetType::One);
        assert_eq!(locktime_offset_bucket(2), LocktimeOffsetType::Two);
        assert_eq!(locktime_offset_bucket(3), LocktimeOffsetType::Three);
        assert_eq!(locktime_offset_bucket(4), LocktimeOffsetType::FourToSix);
        assert_eq!(locktime_offset_bucket(6), LocktimeOffsetType::FourToSix);
        assert_eq!(locktime_offset_bucket(7), LocktimeOffsetType::SevenToTwelve);
        assert_eq!(
            locktime_offset_bucket(12),
            LocktimeOffsetType::SevenToTwelve
        );
        assert_eq!(
            locktime_offset_bucket(13),
            LocktimeOffsetType::ThirteenToTwentyFive
        );
        assert_eq!(
            locktime_offset_bucket(25),
            LocktimeOffsetType::ThirteenToTwentyFive
        );
        assert_eq!(
            locktime_offset_bucket(26),
            LocktimeOffsetType::TwentySixToFifty
        );
        assert_eq!(
            locktime_offset_bucket(50),
            LocktimeOffsetType::TwentySixToFifty
        );
        assert_eq!(
            locktime_offset_bucket(51),
            LocktimeOffsetType::FiftyOneToHundred
        );
        assert_eq!(
            locktime_offset_bucket(100),
            LocktimeOffsetType::FiftyOneToHundred
        );
    }

    #[test]
    fn locktime_offset_is_not_applicable_outside_the_anti_fee_snipe_window() {
        let height = 960_000;
        // Zero.
        assert_eq!(
            locktime_offset_class(0, height),
            LocktimeOffsetType::NotApplicable
        );
        // Timestamp.
        assert_eq!(
            locktime_offset_class(1_700_000_000, height),
            LocktimeOffsetType::NotApplicable
        );
        // Backdated (one block outside the window).
        assert_eq!(
            locktime_offset_class(959_900, height),
            LocktimeOffsetType::NotApplicable
        );
    }

    #[test]
    fn locktime_offset_is_applicable_inside_the_anti_fee_snipe_window() {
        let height = 960_000;
        // Exact tip: offset 0.
        assert_eq!(
            locktime_offset_class(960_000, height),
            LocktimeOffsetType::Zero
        );
        // One block of confirmation delay: offset 1.
        assert_eq!(
            locktime_offset_class(959_999, height),
            LocktimeOffsetType::One
        );
        // The last value still inside the window: offset 99.
        assert_eq!(
            locktime_offset_class(959_901, height),
            LocktimeOffsetType::FiftyOneToHundred
        );
    }
}
