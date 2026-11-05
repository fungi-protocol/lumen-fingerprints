use bitcoin::Amount;

use crate::{OutputType, ScriptPubkeyHash, classify_script_pubkey};

pub trait TxConstituent {
    type Handle: AbstractTransaction;
    fn containing_tx(&self) -> Self::Handle;

    fn vout(&self) -> usize;
}

pub trait OutputCount: AbstractTransaction {
    fn output_count(&self) -> usize;
}

pub trait InputCount: AbstractTransaction {
    fn input_count(&self) -> usize;
}

// TODO: find a better name for this
pub trait EnumerateOutputValueInArbitraryOrder: AbstractTransaction {
    fn output_values(&self) -> impl Iterator<Item = Amount>;
}

pub trait EnumerateInputValueInArbitraryOrder: AbstractTransaction {
    fn input_values(&self) -> impl Iterator<Item = Amount>;
}

pub trait AbstractTxIn: HasSequence + HasWitness + HasScriptSig + HasPrevOutpoint {}

pub trait AbstractTxOut: HasScriptPubkey + HasValue {}

pub trait AbstractTransaction: HasNLockTime + HasVersion {
    fn inputs(&self) -> Box<dyn Iterator<Item = Box<dyn AbstractTxIn + '_>> + '_>;
    fn outputs(&self) -> Box<dyn Iterator<Item = Box<dyn AbstractTxOut + '_>> + '_>;
    fn input_len(&self) -> usize;
    fn output_len(&self) -> usize;
    fn output_at(&self, index: usize) -> Option<Box<dyn AbstractTxOut + '_>>;
    fn is_coinbase(&self) -> bool;
}

pub trait HasNLockTime {
    fn locktime(&self) -> u32;
}

pub trait HasVersion {
    fn version(&self) -> i32;
}

pub trait HasSequence {
    fn sequence(&self) -> u32;
}

pub trait HasWitness {
    fn witness_items(&self) -> Vec<Vec<u8>>;
}

pub trait HasScriptSig {
    fn script_sig_bytes(&self) -> Vec<u8>;
}

pub trait HasScriptPubkey {
    fn script_pubkey_bytes(&self) -> Vec<u8>;

    fn script_pubkey_hash(&self) -> ScriptPubkeyHash {
        use bitcoin::hashes::{Hash, hash160};
        hash160::Hash::hash(&self.script_pubkey_bytes()).to_byte_array()
    }

    fn output_type(&self) -> OutputType {
        classify_script_pubkey(&self.script_pubkey_bytes())
    }

    fn is_spendable(&self) -> bool {
        self.output_type().is_spendable()
    }
}

pub trait HasValue {
    fn value(&self) -> Amount;
}

/// Coinbase inputs carry the null outpoint (all-zero txid, vout = u32::MAX).
pub trait HasPrevOutpoint {
    /// Internal (wire) byte order.
    fn prev_outpoint_txid_bytes(&self) -> [u8; 32];
    fn prev_outpoint_vout(&self) -> u32;
}

impl AbstractTxIn for bitcoin::TxIn {}

impl HasSequence for bitcoin::TxIn {
    fn sequence(&self) -> u32 {
        self.sequence.0
    }
}

impl HasWitness for bitcoin::TxIn {
    fn witness_items(&self) -> Vec<Vec<u8>> {
        self.witness.iter().map(|item| item.to_vec()).collect()
    }
}

impl HasScriptSig for bitcoin::TxIn {
    fn script_sig_bytes(&self) -> Vec<u8> {
        self.script_sig.to_bytes()
    }
}

impl HasScriptPubkey for bitcoin::TxOut {
    fn script_pubkey_bytes(&self) -> Vec<u8> {
        self.script_pubkey.to_bytes()
    }
}

impl HasVersion for bitcoin::Transaction {
    fn version(&self) -> i32 {
        self.version.0
    }
}

impl HasValue for bitcoin::TxOut {
    fn value(&self) -> Amount {
        self.value
    }
}

impl HasPrevOutpoint for bitcoin::TxIn {
    fn prev_outpoint_txid_bytes(&self) -> [u8; 32] {
        let txid_ref: &[u8] = self.previous_output.txid.as_ref();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(txid_ref);
        bytes
    }

    fn prev_outpoint_vout(&self) -> u32 {
        self.previous_output.vout
    }
}
