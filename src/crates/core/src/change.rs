use lumen_primitives::OutputType;

/// Returns the index of the single output that the unanimous-input-type
/// heuristic identifies as change, or `None` when inconclusive.
///
/// An output is change when the transaction's input script types are
/// unanimous (all inputs resolved and of the same type) and exactly one
/// output matches that type. Mixed input types, an empty input list, no
/// matching output, or more than one matching output are all inconclusive
/// and yield `None` — never a guessed index.
///
/// This is a deliberate, documented duplicate of
/// `tx_indexer_heuristics::change_identification::ScriptTypesMatchingChangeIdentification`,
/// which implements the identical rule against the indexing pipeline's
/// `TxHandle`-based types. The two are kept separate on purpose: this survey
/// operates on plain `bitcoin::Transaction`/`bitcoin::TxOut` data with no
/// pipeline storage behind it, so it cannot call the handle-based heuristic
/// directly, and the shared crate is not the survey's to modify. Because
/// this is a real duplication and not merely two callers of one
/// implementation, a future change to the rule (e.g. a new tie-breaking
/// condition) must be applied in BOTH places, or the two will silently
/// drift apart.
pub(crate) fn change_index_by_script_type(
    input_types: &[OutputType],
    output_types: &[OutputType],
) -> Option<usize> {
    let (first_input_type, rest) = input_types.split_first()?;
    if rest.iter().any(|candidate| candidate != first_input_type) {
        return None;
    }

    let mut matching = output_types
        .iter()
        .enumerate()
        .filter(|(_, output_type)| *output_type == first_input_type);

    let (change_index, _) = matching.next()?;
    if matching.next().is_some() {
        return None;
    }

    Some(change_index)
}

/// The sat granularity at or above which an output value is treated as a deliberately
/// "round" (human-entered) amount by `change_index_by_round_number`.
const ROUND_NUMBER_UNIT: u64 = 1_000;

/// Returns the index of the output the round-number heuristic identifies as change, for
/// the canonical two-output payment-plus-change shape: when exactly one of the two outputs
/// is a non-zero round multiple of `ROUND_NUMBER_UNIT` sat (the deliberately entered
/// payment) and the other is not (the computed leftover), the non-round output is change.
/// `None` for any other output count, or when both or neither output is round.
///
/// Independent of `change_index_by_script_type`, which decides by script *type*, not value.
/// The two can fire together, apart, or on different outputs — which heuristic identifies
/// the change output, and whether they agree, is itself a construction fingerprint.
pub(crate) fn change_index_by_round_number(output_values: &[u64]) -> Option<usize> {
    let [a, b] = output_values else { return None };
    let round = |v: u64| v != 0 && v.is_multiple_of(ROUND_NUMBER_UNIT);
    match (round(*a), round(*b)) {
        (true, false) => Some(1),
        (false, true) => Some(0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------------
    // change_index_by_script_type: the plain, handle-free entry point used by the
    // fingerprint survey's `change_axes` (see `vector.rs`). These tests exercise it
    // directly, with no `TxHandle`/storage machinery, since that is exactly the point
    // of the survey having its own copy.
    // ------------------------------------------------------------------------------

    #[test]
    fn change_index_unanimous_inputs_one_matching_output_returns_that_index() {
        let input_types = [OutputType::P2pkh, OutputType::P2pkh];
        let output_types = [OutputType::P2tr, OutputType::P2pkh];
        assert_eq!(
            change_index_by_script_type(&input_types, &output_types),
            Some(1)
        );
    }

    #[test]
    fn change_index_mixed_inputs_returns_none() {
        let input_types = [OutputType::P2pkh, OutputType::P2tr];
        let output_types = [OutputType::P2pkh, OutputType::P2tr];
        assert_eq!(
            change_index_by_script_type(&input_types, &output_types),
            None
        );
    }

    #[test]
    fn change_index_two_matching_outputs_returns_none() {
        let input_types = [OutputType::P2pkh, OutputType::P2pkh];
        let output_types = [OutputType::P2pkh, OutputType::P2pkh];
        assert_eq!(
            change_index_by_script_type(&input_types, &output_types),
            None
        );
    }

    #[test]
    fn change_index_no_matching_output_returns_none() {
        let input_types = [OutputType::P2pkh, OutputType::P2pkh];
        let output_types = [OutputType::P2tr, OutputType::P2wpkh];
        assert_eq!(
            change_index_by_script_type(&input_types, &output_types),
            None
        );
    }

    #[test]
    fn change_index_empty_inputs_returns_none() {
        let output_types = [OutputType::P2pkh];
        assert_eq!(change_index_by_script_type(&[], &output_types), None);
    }

    // ------------------------------------------------------------------------------
    // change_index_by_round_number: the value-based second heuristic.
    // ------------------------------------------------------------------------------

    #[test]
    fn round_number_picks_the_non_round_output_as_change() {
        // out#0 round (50_000), out#1 not (49_999) -> change is out#1.
        assert_eq!(change_index_by_round_number(&[50_000, 49_999]), Some(1));
        // mirrored.
        assert_eq!(change_index_by_round_number(&[49_999, 50_000]), Some(0));
    }

    #[test]
    fn round_number_abstains_when_both_or_neither_is_round() {
        assert_eq!(change_index_by_round_number(&[50_000, 20_000]), None); // both round
        assert_eq!(change_index_by_round_number(&[49_999, 49_998]), None); // neither round
        assert_eq!(change_index_by_round_number(&[0, 49_999]), None); // zero is not "round"
    }

    #[test]
    fn round_number_only_applies_to_the_two_output_shape() {
        assert_eq!(change_index_by_round_number(&[50_000]), None);
        assert_eq!(change_index_by_round_number(&[50_000, 49_999, 1_000]), None);
        assert_eq!(change_index_by_round_number(&[]), None);
    }
}
