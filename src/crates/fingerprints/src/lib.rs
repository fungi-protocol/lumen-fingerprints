pub mod classify;
pub mod fingerprints;
pub mod types;

pub use classify::classify_script_pubkey;
pub use fingerprints::input::{self, HasInputFingerprints};
pub use fingerprints::input_with_prevout;
pub use fingerprints::output::{self, HasOutputFingerprints};
pub use fingerprints::transaction;
pub use types::{
    InputSortingType, LocktimeOffsetType, NLockTimeType, NSequenceType, OutputStructureType,
    SighashType,
};

#[cfg(test)]
mod tests;
