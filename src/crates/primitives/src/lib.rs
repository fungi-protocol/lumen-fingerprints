pub mod output_type;
pub mod traits;

pub use output_type::{OutputType, classify_script_pubkey};
pub use traits::abstract_types::{
    AbstractTransaction, AbstractTxIn, AbstractTxOut, HasPrevOutpoint, HasScriptPubkey,
    HasScriptSig, HasSequence, HasValue, HasVersion, HasWitness,
};

pub type ScriptPubkeyHash = [u8; 20];
