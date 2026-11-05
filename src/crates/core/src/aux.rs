use bitcoin::Transaction;
use lumen_fingerprints_lib::transaction::{
    address_reuse as fp_address_reuse, round_fee as fp_round_fee,
};
use lumen_primitives::traits::SourcedBlock;
use serde::{Deserialize, Serialize};

/// Signals recorded alongside the joint vector but kept out of it, so the vector's
/// cardinality stays interpretable as "distinct wallet-construction shapes".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuxFlags {
    pub round_fee: bool,
    pub round_payment: bool,
    pub uih1: bool,
    /// An input smaller than the smallest output, as a payjoin receiver's contribution leaves.
    pub uih2: bool,
    pub address_reuse: bool,
    pub same_block_parent: bool,
    pub changeless: bool,
}

pub fn aux_flags(tx: &Transaction, block: &SourcedBlock) -> AuxFlags {
    let input_txouts: Vec<bitcoin::TxOut> = tx
        .input
        .iter()
        .filter_map(|i| block.prevout(&i.previous_output))
        .cloned()
        .collect();

    let round_fee = fp_round_fee(&input_txouts, &tx.output).unwrap_or(false);
    let round_payment = tx
        .output
        .iter()
        .any(|o| o.value.to_sat() > 0 && o.value.to_sat() % 100_000 == 0);

    let uih1 = tx.input.len() >= 2
        && match (
            tx.input
                .iter()
                .filter_map(|i| block.prevout(&i.previous_output))
                .map(|o| o.value)
                .max(),
            tx.output.iter().map(|o| o.value).max(),
        ) {
            (Some(max_in), Some(max_out)) => max_in >= max_out,
            _ => false,
        };

    // No uniform wallet behavior removes UIH2: it is a property of the values.
    let uih2 = match (
        tx.input
            .iter()
            .filter_map(|i| block.prevout(&i.previous_output))
            .map(|o| o.value)
            .min(),
        tx.output.iter().map(|o| o.value).min(),
    ) {
        (Some(min_in), Some(min_out)) => min_in < min_out,
        _ => false,
    };

    let changeless = tx.output.len() < 2;
    let address_reuse = fp_address_reuse(&tx.output, &input_txouts);

    let same_block_parent = tx
        .input
        .iter()
        .any(|i| block.confirmations(&i.previous_output) == Some(0));

    AuxFlags {
        round_fee,
        round_payment,
        uih1,
        uih2,
        address_reuse,
        same_block_parent,
        changeless,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::tests_support::{
        EPOCH_TEST_HEIGHT, block_with, cake_like_block, cake_like_tx, p2wpkh_spk,
    };
    use bitcoin::hashes::Hash;
    use bitcoin::{Amount, OutPoint, ScriptBuf, Sequence, TxIn, TxOut, Txid, Witness};
    use lumen_primitives::traits::block_source::SpentOutput;
    use std::collections::HashMap;

    fn other_spk() -> ScriptBuf {
        ScriptBuf::from_hex(&format!("0014{}", "11".repeat(20))).unwrap()
    }

    /// `input_specs` are (value, script, creation_height); unlike `block_with`, they may differ.
    fn custom_block(
        input_specs: &[(u64, ScriptBuf, u32)],
        outputs: Vec<TxOut>,
        height: u32,
    ) -> (Transaction, SourcedBlock) {
        let mut prevouts = HashMap::new();
        let mut txins = Vec::new();
        for (idx, (value, spk, creation_height)) in input_specs.iter().enumerate() {
            let outpoint = OutPoint::new(Txid::from_raw_hash(Hash::all_zeros()), idx as u32);
            prevouts.insert(
                outpoint,
                SpentOutput {
                    txout: TxOut {
                        value: Amount::from_sat(*value),
                        script_pubkey: spk.clone(),
                    },
                    creation_height: *creation_height,
                },
            );
            txins.push(TxIn {
                previous_output: outpoint,
                script_sig: ScriptBuf::new(),
                sequence: Sequence(0xffffffff),
                witness: Witness::from_slice(&[vec![0u8; 71], vec![0u8; 33]]),
            });
        }
        let tx = Transaction {
            version: bitcoin::transaction::Version(2),
            lock_time: bitcoin::absolute::LockTime::ZERO,
            input: txins,
            output: outputs,
        };
        let mut block = bitcoin::constants::genesis_block(bitcoin::Network::Bitcoin);
        block.txdata.push(tx.clone());
        (
            tx,
            SourcedBlock {
                height,
                block,
                prevouts,
            },
        )
    }

    #[test]
    fn uih1_true_when_one_input_alone_covers_the_largest_output() {
        let (tx, block) = custom_block(
            &[
                (100_000, p2wpkh_spk(), EPOCH_TEST_HEIGHT - 10),
                (5_000, p2wpkh_spk(), EPOCH_TEST_HEIGHT - 10),
            ],
            vec![TxOut {
                value: Amount::from_sat(90_000),
                script_pubkey: p2wpkh_spk(),
            }],
            EPOCH_TEST_HEIGHT,
        );
        assert!(aux_flags(&tx, &block).uih1);
    }

    #[test]
    fn uih1_false_when_no_single_input_covers_the_largest_output() {
        let (tx, block) = custom_block(
            &[
                (40_000, p2wpkh_spk(), EPOCH_TEST_HEIGHT - 10),
                (40_000, p2wpkh_spk(), EPOCH_TEST_HEIGHT - 10),
            ],
            vec![TxOut {
                value: Amount::from_sat(70_000),
                script_pubkey: p2wpkh_spk(),
            }],
            EPOCH_TEST_HEIGHT,
        );
        assert!(!aux_flags(&tx, &block).uih1);
    }

    #[test]
    fn uih2_true_when_an_input_is_smaller_than_the_smallest_output() {
        let (tx, block) = custom_block(
            &[
                (1_000, p2wpkh_spk(), EPOCH_TEST_HEIGHT - 10),
                (100_000, p2wpkh_spk(), EPOCH_TEST_HEIGHT - 10),
            ],
            vec![
                TxOut {
                    value: Amount::from_sat(50_000),
                    script_pubkey: p2wpkh_spk(),
                },
                TxOut {
                    value: Amount::from_sat(60_000),
                    script_pubkey: p2wpkh_spk(),
                },
            ],
            EPOCH_TEST_HEIGHT,
        );
        assert!(aux_flags(&tx, &block).uih2);
    }

    #[test]
    fn uih2_false_when_no_input_is_smaller_than_the_smallest_output() {
        let (tx, block) = custom_block(
            &[(100_000, p2wpkh_spk(), EPOCH_TEST_HEIGHT - 10)],
            vec![TxOut {
                value: Amount::from_sat(50_000),
                script_pubkey: p2wpkh_spk(),
            }],
            EPOCH_TEST_HEIGHT,
        );
        assert!(!aux_flags(&tx, &block).uih2);
    }

    #[test]
    fn round_fee_true_for_a_fee_of_exactly_1000_sat() {
        let (tx, block) = custom_block(
            &[(101_000, p2wpkh_spk(), EPOCH_TEST_HEIGHT - 10)],
            vec![TxOut {
                value: Amount::from_sat(100_000),
                script_pubkey: p2wpkh_spk(),
            }],
            EPOCH_TEST_HEIGHT,
        );
        assert!(aux_flags(&tx, &block).round_fee);
    }

    #[test]
    fn round_fee_false_when_the_fee_is_not_a_multiple_of_1000() {
        // The Cake-shaped fixture pays 421_397 sat.
        let block = block_with(cake_like_tx(), EPOCH_TEST_HEIGHT);
        let tx = cake_like_tx();
        assert!(!aux_flags(&tx, &block).round_fee);
    }

    #[test]
    fn round_payment_true_for_a_200_000_sat_output() {
        let (tx, block) = custom_block(
            &[(300_000, p2wpkh_spk(), EPOCH_TEST_HEIGHT - 10)],
            vec![TxOut {
                value: Amount::from_sat(200_000),
                script_pubkey: p2wpkh_spk(),
            }],
            EPOCH_TEST_HEIGHT,
        );
        assert!(aux_flags(&tx, &block).round_payment);
    }

    #[test]
    fn round_payment_false_when_no_output_is_a_multiple_of_100_000() {
        // The Cake-shaped fixture's outputs are 29_358 and 429_919 sat.
        let block = block_with(cake_like_tx(), EPOCH_TEST_HEIGHT);
        let tx = cake_like_tx();
        assert!(!aux_flags(&tx, &block).round_payment);
    }

    #[test]
    fn address_reuse_true_when_an_output_pays_back_to_an_input_script() {
        // block_with resolves every prevout to p2wpkh_spk(), which cake_like_tx's outputs pay.
        let block = block_with(cake_like_tx(), EPOCH_TEST_HEIGHT);
        let tx = cake_like_tx();
        assert!(aux_flags(&tx, &block).address_reuse);
    }

    #[test]
    fn address_reuse_false_when_outputs_pay_a_different_script() {
        let (tx, block) = custom_block(
            &[(100_000, p2wpkh_spk(), EPOCH_TEST_HEIGHT - 10)],
            vec![TxOut {
                value: Amount::from_sat(90_000),
                script_pubkey: other_spk(),
            }],
            EPOCH_TEST_HEIGHT,
        );
        assert!(!aux_flags(&tx, &block).address_reuse);
    }

    #[test]
    fn changeless_true_for_a_single_output() {
        let input = TxIn {
            previous_output: OutPoint::new(Txid::from_raw_hash(Hash::all_zeros()), 0),
            script_sig: ScriptBuf::new(),
            sequence: Sequence(0xffffffff),
            witness: Witness::from_slice(&[vec![0u8; 71], vec![0u8; 33]]),
        };
        let tx = Transaction {
            version: bitcoin::transaction::Version(2),
            lock_time: bitcoin::absolute::LockTime::ZERO,
            input: vec![input],
            output: vec![TxOut {
                value: Amount::from_sat(50_000),
                script_pubkey: p2wpkh_spk(),
            }],
        };
        let block = block_with(tx.clone(), EPOCH_TEST_HEIGHT);
        assert!(aux_flags(&tx, &block).changeless);
    }

    #[test]
    fn changeless_false_for_two_outputs() {
        // cake_like_tx has two outputs.
        let block = block_with(cake_like_tx(), EPOCH_TEST_HEIGHT);
        let tx = cake_like_tx();
        assert!(!aux_flags(&tx, &block).changeless);
    }

    #[test]
    fn same_block_parent_true_when_creation_height_equals_block_height() {
        let (tx, block) = custom_block(
            &[(100_000, p2wpkh_spk(), EPOCH_TEST_HEIGHT)],
            vec![TxOut {
                value: Amount::from_sat(90_000),
                script_pubkey: p2wpkh_spk(),
            }],
            EPOCH_TEST_HEIGHT,
        );
        assert!(aux_flags(&tx, &block).same_block_parent);
    }

    #[test]
    fn same_block_parent_false_when_the_spent_coin_is_older() {
        // block_with sets creation_height to height - 10.
        let block = block_with(cake_like_tx(), EPOCH_TEST_HEIGHT);
        let tx = cake_like_tx();
        assert!(!aux_flags(&tx, &block).same_block_parent);
    }

    #[test]
    fn cake_like_block_address_reuse_matches_direct_construction() {
        let block = cake_like_block(EPOCH_TEST_HEIGHT);
        let tx = &block.block.txdata[1];
        assert!(aux_flags(tx, &block).address_reuse);
    }

    #[test]
    fn round_fee_and_address_reuse_go_through_the_library() {
        use bitcoin::{Amount, OutPoint, ScriptBuf, Sequence, TxIn, TxOut, Witness};
        use lumen_primitives::traits::SourcedBlock;
        use lumen_primitives::traits::block_source::SpentOutput;
        use std::collections::HashMap;

        // Prevout script reused as an output script; fee = 6000 - 5000 = 1000 (round).
        let spk = ScriptBuf::from_hex("0014000102030405060708090a0b0c0d0e0f10111213").unwrap();
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
                value: Amount::from_sat(5000),
                script_pubkey: spk.clone(),
            }],
        };
        let mut prevouts = HashMap::new();
        prevouts.insert(
            tx.input[0].previous_output,
            SpentOutput {
                txout: TxOut {
                    value: Amount::from_sat(6000),
                    script_pubkey: spk,
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

        let aux = aux_flags(&tx, &block);
        assert!(aux.round_fee);
        assert!(aux.address_reuse);
    }
}
