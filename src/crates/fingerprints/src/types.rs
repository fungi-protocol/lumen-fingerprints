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

// `Hash` is required here (beyond what the brief's derive list names) because
// `FingerprintVector` derives `Hash` and stores `output_structure: OutputStructureType`
// directly — without it the survey crate does not compile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputStructureType {
    Single = 0,
    Double = 1,
    Multi = 2,
    Unknown = 3,
}

impl OutputStructureType {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Multi-valued nSequence signature over a transaction's inputs. Mirrors decluster's
/// x_nsequence so its measured bits (e.g. the Cake `cake_group_c` bug) transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NSequenceType {
    // Emitted as the descriptive pattern string (see nsequence_str), not "CakeGroupC":
    // the survey shows the collected value, not a research-group codename.
    CakeGroupC = 0, // [0x01, MAX, ...]: first input 0x01, rest 0xffffffff
    Lone0x01 = 1,   // any input 0x01 (not the Cake pattern)
    Rbf = 2,        // all 0xfffffffd
    Final = 3,      // all 0xfffffffe
    Max = 4,        // all 0xffffffff
    MixedOther = 5,
}

impl NSequenceType {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Signature-hash signature over a transaction's inputs (first witness item per input).
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
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// nLockTime interpreted against the height of the block containing the transaction.
///
/// `AntiFeeSnipe` covers the whole `(height-100, height]` window: Bitcoin Core sets
/// nLockTime to the current tip height at creation time, then ~10% of the time
/// backdates by a further 0-100 blocks (Electrum and Liana reimplement the same rule).
///
/// This used to be split into `ExactTip` (`locktime == block_height`) and
/// `RecentBackdated` (`block_height - 100 < locktime < block_height`), to try to
/// separate those two behaviours. That split was reverted: it is not just hard to
/// observe, it is IMPOSSIBLE to observe per transaction. A wallet writes the tip height
/// at *creation* time, but the transaction confirms in a *later* block, so
/// `locktime == block_height` requires zero confirmation delay — which essentially
/// never happens. Measured over 85.2M mainnet transactions, `ExactTip` came out at
/// 0.000% and `RecentBackdated` absorbed the entire 4.592% anti-fee-sniping mass,
/// exactly as this single bucket does. Worse, the split cannot even be reconstructed
/// after the fact: given a confirmed locktime and height, the tip height at creation
/// time is simply unknown (somewhere in `[locktime, height)`). Exact-tip and
/// backdating are only separable *distributionally* — see `locktime_offset`, which
/// measures the observable `block_height - locktime` gap instead of trying to draw a
/// per-transaction line that cannot exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NLockTimeType {
    /// 0 — no anti-fee-sniping (Cake, Boltz).
    Zero = 0,
    /// `(height-100, height]`: Core-style anti-fee-sniping, exact tip and randomized
    /// backdating alike — the two are not distinguishable per transaction (see this
    /// type's doc comment). Reuses discriminant 1, the original single bucket's value
    /// from before the (reverted) `ExactTip`/`RecentBackdated` split.
    AntiFeeSnipe = 1,
    /// A height more than 100 blocks back.
    Backdated = 2,
    /// A height above the containing block.
    Future = 3,
    /// >= 500_000_000: a unix timestamp, not a height.
    Timestamp = 4,
}

impl NLockTimeType {
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// `block_height - locktime`, bucketed — only meaningful when nLockTime falls in the
/// `NLockTimeType::AntiFeeSnipe` window; for every other `NLockTimeType` value the
/// offset is not interpretable at all (`Zero`/`Backdated`/`Future` carry no tip-height
/// signal, `Timestamp` is not a height in the first place), so those map to
/// `NotApplicable` rather than a bucket.
///
/// This is the axis that actually carries the exact-tip-vs-backdating signal
/// `NLockTimeType` cannot: a wallet that always uses the exact tip produces a small
/// offset equal to its own confirmation delay (concentrated in the low buckets), while
/// Core's ~10% randomized-backdating tail adds a further 0-100 block offset on top,
/// spreading mass across the whole range. The signal is the *shape* of this
/// distribution, not any single transaction's bucket.
///
/// Boundary convention: nine closed-interval buckets over the offset, widening as the
/// offset grows (single-block resolution matters most near zero, where confirmation
/// delay lives; coarser buckets suffice further out): `0`, `1`, `2`, `3`, `4-6`,
/// `7-12`, `13-25`, `26-50`, `51-100`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LocktimeOffsetType {
    /// nLockTime is not in the anti-fee-sniping window, so no offset applies.
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
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}
