use bitcoin::Script;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputType {
    P2pkh = 0,
    P2sh = 1,
    P2wpkh = 2,
    P2wsh = 3,
    P2tr = 4,
    OpReturn = 5,
    NonStandard = 6,
    // TODO: pay2anchor
    /// Last so existing `as_u32()` values hold.
    P2pk = 7,
}

impl OutputType {
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    pub fn is_spendable(self) -> bool {
        self != OutputType::OpReturn && self != OutputType::NonStandard
    }
}

pub fn classify_script_pubkey(spk: &[u8]) -> OutputType {
    let script = Script::from_bytes(spk);

    if script.is_op_return() {
        return OutputType::OpReturn;
    }
    // P2PK cannot overlap the other types, so its position is irrelevant.
    if script.is_p2pk() {
        return OutputType::P2pk;
    }
    if script.is_p2pkh() {
        return OutputType::P2pkh;
    }
    if script.is_p2sh() {
        return OutputType::P2sh;
    }
    if script.is_p2wpkh() {
        return OutputType::P2wpkh;
    }
    if script.is_p2wsh() {
        return OutputType::P2wsh;
    }
    if script.is_p2tr() {
        return OutputType::P2tr;
    }

    OutputType::NonStandard
}
