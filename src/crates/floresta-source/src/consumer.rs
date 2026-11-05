use std::collections::HashMap;
use std::sync::mpsc::SyncSender;

use bitcoin::{Block, OutPoint};
use floresta_chain::{BlockConsumer, UtxoData};
use lumen_primitives::traits::SourcedBlock;
use lumen_primitives::traits::block_source::SpentOutput;

/// A full bounded channel blocks the node: intended backpressure that keeps memory flat
/// when analysis lags download.
pub(crate) struct SurveyConsumer {
    sender: SyncSender<SourcedBlock>,
    start_height: u32,
    stop_height: Option<u32>,
}

impl SurveyConsumer {
    pub(crate) fn new(
        sender: SyncSender<SourcedBlock>,
        start_height: u32,
        stop_height: Option<u32>,
    ) -> Self {
        Self {
            sender,
            start_height,
            stop_height,
        }
    }
}

impl BlockConsumer for SurveyConsumer {
    fn wants_spent_utxos(&self) -> bool {
        true
    }

    fn on_block(
        &self,
        block: &Block,
        height: u32,
        spent_utxos: Option<&HashMap<OutPoint, UtxoData>>,
    ) {
        if height < self.start_height {
            return;
        }
        if let Some(stop) = self.stop_height
            && height > stop
        {
            return;
        }

        let prevouts = spent_utxos
            .map(|map| {
                map.iter()
                    .map(|(outpoint, data)| {
                        (
                            *outpoint,
                            SpentOutput {
                                txout: data.txout.clone(),
                                creation_height: data.creation_height,
                            },
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();

        let sourced = SourcedBlock {
            height,
            block: block.clone(),
            prevouts,
        };

        // A closed receiver means the survey stopped; not fatal.
        let _ = self.sender.send(sourced);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitcoin::{Amount, OutPoint, ScriptBuf, TxOut};
    use std::sync::mpsc::sync_channel;

    fn genesis() -> bitcoin::Block {
        bitcoin::constants::genesis_block(bitcoin::Network::Bitcoin)
    }

    #[test]
    fn forwards_blocks_in_window_and_drops_those_below_start() {
        let (tx, rx) = sync_channel(4);
        let consumer = SurveyConsumer::new(tx, 100, Some(102));

        let mut prevouts = HashMap::new();
        prevouts.insert(
            OutPoint::null(),
            UtxoData {
                txout: TxOut {
                    value: Amount::from_sat(5),
                    script_pubkey: ScriptBuf::new(),
                },
                is_coinbase: false,
                creation_height: 99,
                creation_time: 0,
            },
        );

        consumer.on_block(&genesis(), 99, Some(&prevouts));
        consumer.on_block(&genesis(), 100, Some(&prevouts));

        let got = rx.try_recv().expect("one block forwarded");
        assert_eq!(got.height, 100);
        let spent = got
            .prevouts
            .get(&OutPoint::null())
            .expect("prevout carried through");
        assert_eq!(spent.txout.value, Amount::from_sat(5));
        assert_eq!(
            spent.creation_height, 99,
            "creation height must survive the bridge"
        );
        assert!(rx.try_recv().is_err(), "height 99 must not be forwarded");
    }

    #[test]
    fn enforces_height_window_boundaries() {
        let (tx, rx) = sync_channel(10);
        let consumer = SurveyConsumer::new(tx, 100, Some(102));

        consumer.on_block(&genesis(), 99, Some(&HashMap::new()));
        consumer.on_block(&genesis(), 100, Some(&HashMap::new()));
        consumer.on_block(&genesis(), 101, Some(&HashMap::new()));
        consumer.on_block(&genesis(), 102, Some(&HashMap::new()));
        consumer.on_block(&genesis(), 103, Some(&HashMap::new()));

        assert_eq!(rx.try_recv().unwrap().height, 100);
        assert_eq!(rx.try_recv().unwrap().height, 101);
        assert_eq!(rx.try_recv().unwrap().height, 102);
        assert!(
            rx.try_recv().is_err(),
            "no block above stop_height should be forwarded"
        );
    }

    #[test]
    fn wants_spent_utxos_is_true() {
        let (tx, _rx) = sync_channel(1);
        assert!(SurveyConsumer::new(tx, 0, None).wants_spent_utxos());
    }
}
