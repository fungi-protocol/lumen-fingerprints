use lumen_primitives::OutputType;

/// Mirror of tx-indexer's `ScriptTypesMatchingChangeIdentification`; change both.
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

const ROUND_NUMBER_UNIT: u64 = 1_000;

/// A round amount is human-entered (the payment), so the computed leftover is change.
/// Independent of the script-type heuristic; their agreement is itself a fingerprint.
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

    #[test]
    fn round_number_picks_the_non_round_output_as_change() {
        assert_eq!(change_index_by_round_number(&[50_000, 49_999]), Some(1));
        assert_eq!(change_index_by_round_number(&[49_999, 50_000]), Some(0));
    }

    #[test]
    fn round_number_abstains_when_both_or_neither_is_round() {
        assert_eq!(change_index_by_round_number(&[50_000, 20_000]), None);
        assert_eq!(change_index_by_round_number(&[49_999, 49_998]), None);
        assert_eq!(change_index_by_round_number(&[0, 49_999]), None);
    }

    #[test]
    fn round_number_only_applies_to_the_two_output_shape() {
        assert_eq!(change_index_by_round_number(&[50_000]), None);
        assert_eq!(change_index_by_round_number(&[50_000, 49_999, 1_000]), None);
        assert_eq!(change_index_by_round_number(&[]), None);
    }
}
