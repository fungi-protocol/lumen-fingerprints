pub mod abstract_types;
pub mod block_source;

pub use abstract_types::{
    AbstractTransaction, AbstractTxIn, AbstractTxOut, HasNLockTime, HasPrevOutpoint,
    HasScriptPubkey, HasScriptSig, HasSequence, HasValue, HasVersion, HasWitness, InputCount,
};

pub use block_source::{BlockSource, SourcedBlock};
