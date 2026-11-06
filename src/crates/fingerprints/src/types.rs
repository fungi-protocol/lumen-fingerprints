use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputSortingType {
    Single = 0,
    Ascending = 1,
    Descending = 2,
    Bip69 = 3,
    Historical = 4,
    Unknown = 5,
}

impl InputSortingType {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputStructureType {
    Single = 0,
    Double = 1,
    Multi = 2,
    Unknown = 3,
}

impl OutputStructureType {
    pub const ALL: [Self; 4] = [Self::Single, Self::Double, Self::Multi, Self::Unknown];

    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NSequenceType {
    // Serialized via nsequence_str, not the variant name.
    CakeGroupC = 0, // [0x01, MAX, ...]
    Lone0x01 = 1,   // any input 0x01 (not the Cake pattern)
    Rbf = 2,        // all 0xfffffffd
    Final = 3,      // all 0xfffffffe
    Max = 4,        // all 0xffffffff
    MixedOther = 5,
}

impl NSequenceType {
    pub const ALL: [Self; 6] = [
        Self::CakeGroupC,
        Self::Lone0x01,
        Self::Rbf,
        Self::Final,
        Self::Max,
        Self::MixedOther,
    ];

    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SighashType {
    TaprootDefault = 0,  // 64-byte schnorr, implicit SIGHASH_DEFAULT
    TaprootExplicit = 1, // 65-byte schnorr, explicit sighash byte
    All = 2,             // ECDSA 0x01
    None = 3,            // 0x02
    Single = 4,          // 0x03
    AcpAll = 5,          // 0x81
    AcpNone = 6,         // 0x82
    AcpSingle = 7,       // 0x83
    Other = 8,
    Mixed = 9, // >1 distinct type across inputs
    Na = 10,   // no signatures observed
}

impl SighashType {
    pub const ALL: [Self; 11] = [
        Self::TaprootDefault,
        Self::TaprootExplicit,
        Self::All,
        Self::None,
        Self::Single,
        Self::AcpAll,
        Self::AcpNone,
        Self::AcpSingle,
        Self::Other,
        Self::Mixed,
        Self::Na,
    ];

    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Core backdates ~10% of locktimes up to 100 blocks; since the creation-time tip is
/// unknown, exact tip vs backdated is only separable as a distribution (`LocktimeOffsetType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NLockTimeType {
    Zero = 0,
    /// `(height-100, height]`
    AntiFeeSnipe = 1,
    /// More than 100 blocks back.
    Backdated = 2,
    /// Above the containing block.
    Future = 3,
    /// >= 500_000_000: a unix timestamp, not a height.
    Timestamp = 4,
}

impl NLockTimeType {
    pub const ALL: [Self; 5] = [
        Self::Zero,
        Self::AntiFeeSnipe,
        Self::Backdated,
        Self::Future,
        Self::Timestamp,
    ];

    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Exact-tip wallets concentrate near zero (confirmation delay); backdating spreads
/// across the range. Hence the finest buckets near zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LocktimeOffsetType {
    /// Outside the `AntiFeeSnipe` window.
    NotApplicable = 0,
    Zero = 1,
    One = 2,
    Two = 3,
    Three = 4,
    /// 4-6.
    FourToSix = 5,
    /// 7-12.
    SevenToTwelve = 6,
    /// 13-25.
    ThirteenToTwentyFive = 7,
    /// 26-50.
    TwentySixToFifty = 8,
    /// 51-100.
    FiftyOneToHundred = 9,
}

impl LocktimeOffsetType {
    pub const ALL: [Self; 10] = [
        Self::NotApplicable,
        Self::Zero,
        Self::One,
        Self::Two,
        Self::Three,
        Self::FourToSix,
        Self::SevenToTwelve,
        Self::ThirteenToTwentyFive,
        Self::TwentySixToFifty,
        Self::FiftyOneToHundred,
    ];

    pub fn as_u32(self) -> u32 {
        self as u32
    }
}
