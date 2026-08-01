use std::collections::HashMap;

use bitcoin::{Block, OutPoint, TxOut};

/// An output being spent, with the height of the block that created it.
///
/// `creation_height` is what makes input age observable: coin age ordering, and
/// whether a transaction spends a very recently confirmed (or same-block) output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpentOutput {
    pub txout: TxOut,
    pub creation_height: u32,
}

/// One block together with the resolved previous output of every input it spends.
///
/// `prevouts` is keyed by outpoint and, for utreexo-sourced blocks, also contains
/// outputs created earlier *within the same block* (a block-internal spend resolves
/// there). Always look up by `txin.previous_output`; never treat the map as "the
/// set of coins this block spent".
#[derive(Debug, Clone)]
pub struct SourcedBlock {
    pub height: u32,
    pub block: Block,
    pub prevouts: HashMap<OutPoint, SpentOutput>,
}

impl SourcedBlock {
    /// The spent output with its creation height, if the source resolved it.
    pub fn spent(&self, outpoint: &OutPoint) -> Option<&SpentOutput> {
        self.prevouts.get(outpoint)
    }

    /// The output an input spends, if the source resolved it.
    pub fn prevout(&self, outpoint: &OutPoint) -> Option<&TxOut> {
        self.prevouts.get(outpoint).map(|s| &s.txout)
    }

    /// Confirmations the spent coin had when this block was mined.
    ///
    /// Returns `Some(0)` only when the coin was created in this same block (a
    /// same-block parent spend), which is a meaningful signal used to measure transaction
    /// fingerprints.
    ///
    /// # Precondition
    ///
    /// A conformant `BlockSource` implementation must never yield a prevout with
    /// `creation_height > height` — all coins spent in a block must have been created
    /// in or before that block. This method uses `saturating_sub` to remain safe even
    /// if a buggy implementation violates this precondition, but a debug build will
    /// assert the invariant to catch implementation bugs early.
    pub fn confirmations(&self, outpoint: &OutPoint) -> Option<u32> {
        self.prevouts.get(outpoint).map(|s| {
            debug_assert!(
                s.creation_height <= self.height,
                "spent coin created at height {} but spent in block at height {}",
                s.creation_height,
                self.height
            );
            self.height.saturating_sub(s.creation_height)
        })
    }
}

/// A forward-only stream of blocks with resolved prevouts, in ascending height order.
///
/// Implementors: `FlorestaSource` (utreexo leaf data over p2p) and, if the Floresta
/// throughput gate fails, `CoreDatadirSource` (blk*.dat + rev*.dat undo files).
pub trait BlockSource {
    type Error: std::error::Error + Send + Sync + 'static;

    /// The next block, or `None` when the requested window is exhausted.
    /// Blocking: may wait on network I/O.
    fn next_block(&mut self) -> Result<Option<SourcedBlock>, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitcoin::consensus::deserialize;
    use bitcoin::{Amount, ScriptBuf, TxOut};

    /// A source that yields a fixed list of blocks then None. Proves the trait is
    /// implementable by something other than Floresta (this is what keeps the
    /// Core-datadir fallback cheap).
    struct VecSource(std::vec::IntoIter<SourcedBlock>);

    #[derive(Debug)]
    struct Never;
    impl std::fmt::Display for Never {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "never")
        }
    }
    impl std::error::Error for Never {}

    impl BlockSource for VecSource {
        type Error = Never;
        fn next_block(&mut self) -> Result<Option<SourcedBlock>, Never> {
            Ok(self.0.next())
        }
    }

    fn dummy_block() -> bitcoin::Block {
        // Mainnet genesis block, the one block whose bytes are stable and well known.
        let raw = hex::decode(
            "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd\
             7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c01\
             01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff\
             4d04ffff001d0104455468652054696d65732030332f4a616e2f32303039204368616e63656c6c6f72\
             206f6e206272696e6b206f66207365636f6e64206261696c6f757420666f722062616e6b73ffffffff\
             0100f2052a01000000434104678afdb0fe5548271967f1a67130b7105cd6a828e03909a67962e0ea1f\
             61deb649f6bc3f4cef38c4f35504e51ec112de5c384df7ba0b8d578a4c702b6bf11d5fac00000000",
        )
        .unwrap();
        deserialize(&raw).unwrap()
    }

    #[test]
    fn yields_blocks_then_none() {
        let outpoint = OutPoint::null();
        let txout = TxOut {
            value: Amount::from_sat(1234),
            script_pubkey: ScriptBuf::new(),
        };
        let mut prevouts = HashMap::new();
        prevouts.insert(
            outpoint,
            SpentOutput {
                txout: txout.clone(),
                creation_height: 40,
            },
        );

        let sourced = SourcedBlock {
            height: 42,
            block: dummy_block(),
            prevouts,
        };
        let mut source = VecSource(vec![sourced].into_iter());

        let first = source.next_block().unwrap().expect("one block");
        assert_eq!(first.height, 42);
        assert_eq!(first.prevout(&outpoint), Some(&txout));
        assert_eq!(
            first.prevout(&bitcoin::OutPoint::new(
                bitcoin::Txid::from_raw_hash(bitcoin::hashes::Hash::all_zeros()),
                7
            )),
            None
        );

        assert!(source.next_block().unwrap().is_none());
    }

    #[test]
    fn confirmations_count_from_creation_height() {
        let outpoint = OutPoint::null();
        let mut prevouts = HashMap::new();
        prevouts.insert(
            outpoint,
            SpentOutput {
                txout: TxOut {
                    value: Amount::from_sat(1),
                    script_pubkey: ScriptBuf::new(),
                },
                creation_height: 40,
            },
        );
        let block = SourcedBlock {
            height: 42,
            block: dummy_block(),
            prevouts,
        };

        assert_eq!(block.confirmations(&outpoint), Some(2));

        // Created in this very block: a same-block parent spend, 0 confirmations.
        let mut same = block.clone();
        same.prevouts.get_mut(&outpoint).unwrap().creation_height = 42;
        assert_eq!(same.confirmations(&outpoint), Some(0));
    }

    #[test]
    fn confirmations_same_block_parent_spend_boundary() {
        // Explicitly test the boundary case: creation_height == height yields Some(0).
        // This is a meaningful signal — a same-block parent spend is a transaction
        // fingerprint axis, and this test ensures the zero is not confused with
        // a clamped result from a malformed block source.
        let outpoint = OutPoint::null();
        let mut prevouts = HashMap::new();
        prevouts.insert(
            outpoint,
            SpentOutput {
                txout: TxOut {
                    value: Amount::from_sat(1),
                    script_pubkey: ScriptBuf::new(),
                },
                creation_height: 100,
            },
        );
        let block = SourcedBlock {
            height: 100,
            block: dummy_block(),
            prevouts,
        };

        assert_eq!(block.confirmations(&outpoint), Some(0));
    }
}
