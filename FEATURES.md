# FEATURES.md — `lumen scan --features` schema

`lumen scan --features <path.csv[.zst]>` streams one numeric row per **classified**
transaction to a CSV, in the same pass as the epoch scan. This is the wide,
redundancy-tolerant feature matrix meant for loading into pandas/Polars/etc. for
clustering and ML — as opposed to the curated, orthogonal axis values used internally
for the sparsity joint vector (`axis_value` in `src/crates/core/src/vector.rs`, which
this file does not describe).

- A `.zst` suffix on the `--features` path selects zstd-compressed output (level 3);
  any other/missing suffix writes plain CSV. Decompressed content is byte-identical to
  the plain-CSV form.
- **Population**: exactly one row per transaction the scan successfully classified —
  the same set counted in the scan summary's `txs`. Transactions the scan could not
  classify ("defects") produce no row at all; they are not represented as nulls or
  zero-rows.
- **Column order is append-only**: the header is `COLUMN_NAMES.join(",")`
  (`src/crates/core/src/features.rs`). New feature versions may append new columns at
  the end but must never reorder, rename, or remove existing ones — a script written
  against column *names* keeps working across versions; a script relying on column
  *position* is safe for existing columns but must tolerate new trailing columns
  appearing.
- Every value in the CSV is numeric (`f64`); there are no string-valued feature
  columns other than the two identity columns.

There are 175 columns total: 2 identity + 173 feature values.

## Identity (columns 0–1)

| column | type | meaning |
|---|---|---|
| `txid` | string | hex transaction id |
| `block_height` | integer | height of the block the tx was confirmed in |

## Plain booleans (columns 2–11)

Each is `1.0` / `0.0`.

| column | meaning |
|---|---|
| `op_return` | tx has at least one `OP_RETURN` output |
| `round_feerate` | feerate falls on a "round" value (heuristic) |
| `low_s_yes` | ECDSA signature(s) use low-S encoding — collapsed from the underlying tri-state axis: `1.0` only when the axis is definitively `Yes`; both `No` and `Indeterminate` read as `0.0`. (Contrast with `low_r`/`uncompressed_pubkey` below, which keep the indeterminate case as its own column.) |
| `round_fee` | absolute fee is a "round" value (heuristic, computed per-row from the tx/block at scan time) |
| `round_payment` | a non-change output value is "round" (heuristic) |
| `uih1` | Unnecessary Input Heuristic 1 fires |
| `uih2` | Unnecessary Input Heuristic 2 fires |
| `address_reuse` | an input's previous output address reappears elsewhere in this tx's context |
| `same_block_parent` | at least one input's previous tx confirmed in the same block |
| `changeless` | heuristic finds no likely change output |

## Tri-state pairs (columns 12–15)

Encoded as two booleans per axis: `<axis>_yes` and `<axis>_indeterminate`. Both `0.0`
means the axis is definitively `No`.

| columns | axis |
|---|---|
| `low_r_yes`, `low_r_indeterminate` | low-R signature nonce encoding |
| `uncompressed_pubkey_yes`, `uncompressed_pubkey_indeterminate` | an uncompressed public key is used |

## One-hot groups (columns 16–73)

Exactly one column in each group is `1.0` per row; the rest are `0.0`.

**`version_*`** (16–19): `version_1`, `version_2`, `version_3`, `version_other`.
`nVersion` is an open (arbitrary `i32`) domain, so any value other than 1/2/3 sets
`version_other`.

**`nsequence_*`** (20–25), closed/exhaustive over `NSequenceType`:
`nsequence_cake_group_c`, `nsequence_lone_0x01`, `nsequence_rbf`, `nsequence_final`,
`nsequence_max`, `nsequence_mixed_other`.

**`nlocktime_*`** (26–30), closed/exhaustive over `NLockTimeType`:
`nlocktime_zero`, `nlocktime_anti_fee_snipe`, `nlocktime_backdated`,
`nlocktime_future`, `nlocktime_timestamp`.

**`input_order_*`** (31–36), closed/exhaustive over `OrderClass`:
`input_order_bip69`, `input_order_value_ascending`, `input_order_value_descending`,
`input_order_age_ascending`, `input_order_other`, `input_order_indeterminate`.

**`output_order_*`** (37–42) — the same six `OrderClass` values, applied to output
ordering instead: `output_order_bip69`, `output_order_value_ascending`,
`output_order_value_descending`, `output_order_age_ascending`, `output_order_other`,
`output_order_indeterminate`.

`output_order_age_ascending` is **always `0.0`**: outputs have no creation age (only
inputs do — age comes from a spent prevout's `creation_height`), so `output_order_class`
can never return `AgeAscending`. The column exists only for one-hot symmetry with the
input side (`input_order_*`), so the two 6-wide order vectors line up positionally. (Same
treatment as `input_count_op_return`, which is always `0`.)

**`input_subtype_*`** (43–52), closed/exhaustive over `InputSubtype`:
`input_subtype_p2sh_p2wpkh`, `input_subtype_p2sh_multisig`, `input_subtype_p2sh_other`,
`input_subtype_p2wsh_multisig`, `input_subtype_p2wsh_other`,
`input_subtype_taproot_key_path`, `input_subtype_taproot_script_path`,
`input_subtype_bare`, `input_subtype_mixed`, `input_subtype_indeterminate`.

**`sighash_*`** (53–63), closed/exhaustive over `SighashType`:
`sighash_taproot_default`, `sighash_taproot_explicit`, `sighash_all`, `sighash_none`,
`sighash_single`, `sighash_acp_all`, `sighash_acp_none`, `sighash_acp_single`,
`sighash_other`, `sighash_mixed`, `sighash_na`.

**`input_types_*`** (64–73) — a 10-way one-hot over `InputTypeClass`: the 8
`Uniform(OutputType)` variants (`input_types_uniform_p2pkh`,
`input_types_uniform_p2sh`, `input_types_uniform_p2wpkh`,
`input_types_uniform_p2wsh`, `input_types_uniform_p2tr`,
`input_types_uniform_op_return`, `input_types_uniform_nonstandard`,
`input_types_uniform_p2pk`, in `OutputType`'s own order), plus `input_types_mixed` and
`input_types_unknown` as their own columns when the tx's inputs are not all one
script type, or that type could not be determined.

## Ordinal index columns (columns 74–79)

Ascending integer index, larger = "larger bucket" on that axis. Stored as `f64` but
always integer-valued (except `feerate_bucket_index`'s sentinel, see below).

| column | bucket → index mapping |
|---|---|
| `ecdsa_sigs_index` | `None`=0, `One`=1, `Few`=2, `Many`=3 |
| `input_age_index` | `SameBlock`=0, `Within6`=1, `Within144`=2, `Older`=3, `Indeterminate`=4 |
| `output_structure_index` | `Single`=0, `Double`=1, `Multi`=2, `Unknown`=3 |
| `feerate_bucket_index` | rank of the feerate bucket string's parsed integer lower bound. Buckets look like `"0"`..`"9"`, `"10-19"`, `"20-29"`, …, `"100+"`, or `"Unknown"`. The value is the integer before the first `-`/`+` (so `"10-19"` → `10.0`, `"100+"` → `100.0`); `"Unknown"` and any unparseable bucket string both map to `f64::from(u16::MAX)` (`65535.0`) so they sort after every real bucket. This is a monotonic rank for feeding a model, not the feerate itself. |
| `locktime_offset_index` | `NotApplicable`=0, `Zero`=0, `One`=1, `Two`=2, `Three`=3, `FourToSix`=4, `SevenToTwelve`=5, `ThirteenToTwentyFive`=6, `TwentySixToFifty`=7, `FiftyOneToHundred`=8. Note `NotApplicable` and `Zero` share index 0 — use the next column to tell them apart. |
| `locktime_offset_not_applicable` | boolean flag, `1.0` iff the tx's locktime offset axis was `NotApplicable` (e.g. `nLockTime` isn't a height/time the offset heuristic applies to) rather than a genuine zero-block offset. |

## Heuristic-derived columns (columns 80–84)

Kept here for clustering; never part of the sparsity survey's joint key.

**`change_position_*`** (80–83), one-hot over `ChangePosition`: `change_position_first`,
`change_position_last`, `change_position_middle`, `change_position_indeterminate` —
where the heuristically-identified change output sits among the tx's outputs.

**`change_type_present`** (84): boolean, `1.0` iff the heuristic identified a change
output at all (regardless of its position).

## Group A — nSequence bit aggregates (columns 85–90)

These replace the old presence-only view of `nSequence` with a decomposition of the
32-bit field's actual bit layout: **bit 31** is the disable flag, **bit 22** is the
relative-locktime type flag, **bits 0–15** are the relative-locktime value, and **bits
16–21 and 23–30** are reserved (should be zero). Computed per-input across all of a
tx's inputs, then folded to booleans/a max over the tx.

| column | meaning |
|---|---|
| `bip125_rbf` | `1.0` iff any input's sequence is `< 0xFFFFFFFE` — BIP-125 opt-in replace-by-fee signaling. |
| `all_inputs_final` | `1.0` iff every input's sequence is exactly `0xFFFFFFFF` (and the tx has at least one input). |
| `bip68_active` | `1.0` iff the tx version is `>= 2` AND at least one input has bit 31 clear — i.e. BIP-68 relative timelock semantics are in effect for that input. Always `0.0` for version-1 txs, regardless of sequence values. |
| `bip68_type_time` | `1.0` iff any BIP-68-active input (bit 31 clear, version >= 2) has bit 22 set — the relative-locktime unit is 512-second time intervals rather than blocks. |
| `nsequence_reserved_bits_set` | `1.0` iff any bit-31-clear input has any of the should-be-zero reserved bits set (mask `0x7FBF0000` = bits 16–21 and 23–30). A nonstandard-construction fingerprint — well-behaved wallets never set these. |
| `bip68_relative_value_max` | the maximum of the low-16-bit value (bits 0–15) taken over all BIP-68-active inputs (bit 31 clear, version >= 2); `0.0` if there are no such inputs. Not itself a boolean. |

## Group B — nLockTime coupling (column 91)

| column | meaning |
|---|---|
| `nlocktime_dead` | `1.0` iff `nLockTime` is non-zero AND `all_inputs_final` is true. Consensus only enforces `nLockTime` when at least one input is non-final (sequence `!= 0xFFFFFFFF`); when every input is final, a non-zero `nLockTime` is silently ignored by validation. A "dead" locktime like this leaks an approximate creation height/time for zero actual benefit — a wallet fingerprint with no upside. |

## Group C — type counts and positional signal (columns 92–114)

Where the removed `input_has_*`/`output_has_*` presence booleans only said "does this
tx touch script type X at all", these columns carry the real per-type multiset and the
tx's positional layout, which is a stronger clustering/fingerprinting signal.

| column | meaning |
|---|---|
| `input_count` | number of inputs (tx size on the spending side). |
| `output_count` | number of outputs (tx size on the creation side). |

**`input_count_*`** (94–101) and **`output_count_*`** (102–109), 8 columns per side, one
per `OutputType` variant in `OutputType`'s own order: `input_count_p2pkh`,
`input_count_p2sh`, `input_count_p2wpkh`, `input_count_p2wsh`, `input_count_p2tr`,
`input_count_op_return`, `input_count_nonstandard`, `input_count_p2pk`, and the
matching `output_count_p2pkh`, `output_count_p2sh`, `output_count_p2wpkh`,
`output_count_p2wsh`, `output_count_p2tr`, `output_count_op_return`,
`output_count_nonstandard`, `output_count_p2pk` set. Each is an integer count
(`f64`-valued) of how many of that tx's inputs/outputs classify as that type — the real
per-type multiset, not a presence bit, so e.g. a tx spending three P2WPKH inputs has
`input_count_p2wpkh = 3.0`.

`input_count_op_return` is **always `0`**: an `OP_RETURN` output is provably
unspendable, so it can never appear as an input's previous-output type. The column is
kept anyway for input/output symmetry (so the 8-wide input and output count vectors
line up positionally).

| column | meaning |
|---|---|
| `first_output_type_index` | the `OutputType` ordinal (0–7, in `OutputType`'s own order — the same order as the `*_count_*` columns) of the tx's first output; `-1.0` if the tx has no outputs. |
| `last_output_type_index` | the `OutputType` ordinal of the tx's last output; `-1.0` if the tx has no outputs. |
| `first_output_matches_input_type` | `1.0` iff the tx's inputs are a single uniform type (all inputs classify to the same `OutputType`) AND that type equals the first output's type. `0.0` if the inputs are mixed/unknown-typed or there is no first output. |
| `last_output_matches_input_type` | same, but compared against the last output's type instead. |

`last_output_matches_input_type = true` together with `first_output_matches_input_type
= false` is the classic payment-then-change layout signal: the first output pays a
different (destination) type while the last output returns to the sender's own input
type — as opposed to the reverse or "both match"/"neither matches" cases.

| column | meaning |
|---|---|
| `outputs_type_grouped` | `1.0` iff same-typed outputs are contiguous — no type appears, is followed by a different type, and then reappears later. `0.0` means the outputs were interleaved by type, which is evidence they were *not* produced by a simple unshuffled payment+change construction (e.g. `[A, B, A]` is not grouped; `[A, A, B]` is). |

## Positional type encoding (sub-project 3e) (columns 115–137)

CSV-only, feature-matrix-only columns — none of these enter the sparsity joint vector
or `axis_value`. Every type index here uses the same encoding as `first_output_type_index`
above: `t as usize` (0–7, `OutputType`'s own order), `-1.0` when there is no type at that
position (no input/output there, or an empty tx).

**Input-side mirror** (115–117): the exact counterpart of `first_output_type_index` /
`last_output_type_index` / `outputs_type_grouped`, computed over the tx's ordered input
types (each input's resolved prevout scriptPubKey type) instead of its outputs.

| column | meaning |
|---|---|
| `first_input_type_index` | the `OutputType` ordinal of the tx's first input's prevout type; `-1.0` if the tx has no inputs. |
| `last_input_type_index` | the `OutputType` ordinal of the tx's last input's prevout type; `-1.0` if the tx has no inputs. |
| `inputs_type_grouped` | `1.0` iff same-typed inputs are contiguous — same contiguous-run rule as `outputs_type_grouped`, applied to the input side. |

**Capped per-position arrays** (118–137): each tx's input types and output types laid out
positionally, one column per position, up to a cap of **10 positions per side**. A
position beyond the tx's actual input/output count reads `-1.0`; a position at or beyond
the cap (input/output index ≥ 10) is simply not represented — the cap discards signal
past position 9, trading a small amount of information on unusually large transactions
for a fixed-width row.

| columns | meaning |
|---|---|
| `input_type_at_pos_0` … `input_type_at_pos_9` (118–127) | the `OutputType` ordinal of the input at that position (0-indexed), `-1.0` if the tx has no input at that position. |
| `output_type_at_pos_0` … `output_type_at_pos_9` (128–137) | the `OutputType` ordinal of the output at that position (0-indexed), `-1.0` if the tx has no output at that position. |

## Feerate (columns 138–145)

Raw fee/size primitives plus derived feerates in two unit conventions and two
block-relative ratios. All are `f64`-valued.

| column | meaning |
|---|---|
| `fee_sat` | absolute fee in satoshis (inputs − outputs), computed per-row from the tx and its resolved prevouts. A raw primitive: kept alongside `vsize`/`weight` so a consumer can recompute any feerate convention itself and infer which one the wallet targeted. |
| `vsize` | virtual size in vbytes (`weight` rounded up to the nearest vbyte, i.e. `ceil(weight/4)`). Raw primitive. |
| `weight` | transaction weight units (BIP-141). Raw primitive; `vsize` and `weight` differ by the rounding step, and exposing both lets a consumer see it. |
| `feerate_sat_per_vb` | `fee_sat / vsize` — satoshis per virtual byte, the convention wallet UIs and mempool policy quote. |
| `feerate_sat_per_kwu` | `fee_sat * 1000 / weight` — satoshis per 1000 weight units, the rust-bitcoin / Lightning `FeeRate` convention. Present next to `feerate_sat_per_vb` precisely so the two unit conventions can be distinguished: a wallet that rounds to a whole number in one convention but not the other reveals which one it computes in. |
| `fee_is_multiple_of_vsize` | `1.0` iff `fee_sat` is an exact integer multiple of `vsize` — i.e. the feerate is a whole number of sat/vB with no remainder. A wallet that always sets fees this way is fingerprintable. |
| `feerate_over_block_min` | `feerate_sat_per_vb` divided by the minimum feerate over the *containing* block's fee-paying transactions. Sentinel `-1.0` when that minimum is absent or non-positive (e.g. a block with no resolvable fee-paying tx). A value near `1.0` means this tx paid close to the cheapest feerate that still confirmed in its block. |
| `feerate_over_prev_block_min` | same ratio, but against the *preceding* block's minimum feerate. Sentinel `-1.0` for the first block of a scan (no preceding block) or when that minimum is absent/non-positive. The preceding block is the fee market that was actually observable when this tx was built and broadcast — a wallet setting its fee off "the last block's cheapest" leaves this ratio clustered near a constant. |

Note the containing-block minimum is computed once per block by the scan engine over
every tx whose fee and vsize resolve, then carried forward one block so the next block
sees it as its `prev` minimum.

**Caveat — rounding direction is not observable.** These columns expose the *magnitude*
of a tx's fee and feerate, and whether the fee is an exact multiple of vsize, but not the
*direction* a wallet rounded to reach it: an on-chain tx records only the final fee, not
whether the wallet rounded its target feerate up or down (or rounded the fee, or the
feerate, or neither) to arrive there. Two wallets that round oppositely can produce the
same row.

## Amounts (columns 146–151)

Digit-structure measures of the transaction's output values — how "round" or
"structured" the amounts look in various bases, a payment-vs-change and wallet-UI
fingerprint. All `f64`-valued; the base-n counts and spans are integer-valued.

| column | meaning |
|---|---|
| `max_output_value_sat` | the largest output value in satoshis (`0.0` if the tx has no outputs). The reference amount the following four digit measures are computed over. |
| `hamming_decimal` | count of non-zero **base-10** digits of `max_output_value_sat` (its decimal Hamming weight). `100000000` (1.0 BTC) → 1; `123450000` → 5. |
| `hamming_base2` | count of set **bits** of `max_output_value_sat` (its binary Hamming weight / population count). |
| `hamming_base3` | count of non-zero **base-3** digits of `max_output_value_sat`. A third base so a value that is "round" in one base but not others is distinguishable. |
| `decimal_sig_fig_span` | bounded-precision measure: the number of decimal positions spanned from the most-significant to the least-significant non-zero digit, inclusive (`0` for a zero value). This separates values with the same decimal Hamming weight but very different precision: `1.1` BTC = `110000000` sat has span **2** (the two significant digits are adjacent), whereas `1.00000001` BTC = `100000001` sat has span **9** (the two significant digits sit nine places apart) — yet both have `hamming_decimal` = 2. A small span means a coarsely-rounded amount; a large span means sub-satoshi-level precision that a human is unlikely to have typed. |
| `hamming_decimal_min_over_outputs` | the minimum `hamming_decimal` taken over **all** the tx's outputs, **excluding zero-value outputs** (so an `OP_RETURN` or provably-empty output does not force this to 0). Sentinel `-1.0` when the tx has no non-zero output. Where `hamming_decimal` looks only at the largest output, this finds the "roundest" output anywhere in the tx — often the payment or change amount a wallet chose. |

## Taproot (columns 152–153)

Script-path spend structure over the tx's taproot inputs. Both `f64`-valued and
integer-valued.

| column | meaning |
|---|---|
| `taproot_max_merkle_depth` | the maximum taproot script-path merkle depth over the tx's inputs, computed as `(control_block_len - 33) / 32` for each script-path input's control block. `0` when the tx has no script-path taproot input (all key-path spends, or no taproot inputs at all). Annex-aware: a witness whose last element begins with `0x50` is a taproot annex, so the control block is read from the second-to-last witness element instead of the last. A deeper merkle tree implies a more complex spending policy (more script leaves), a wallet/protocol fingerprint. |
| `taproot_script_path_input_count` | how many of the tx's inputs are taproot **script-path** spends (`0` if none). Distinguishes a single complex-script input from many. |

## Standardness (columns 154–160)

A subset of Bitcoin Core's `IsStandardTx` / `AreOutputsStandard` policy checks, evaluated
per transaction. `is_standard` is the headline column; the six `nonstd_*` columns are the
individual reasons, each `1.0` when that specific rule is violated. All are `1.0` / `0.0`.

| column | rule (`1.0` when violated) |
|---|---|
| `is_standard` | the **negation** of the OR of the six `nonstd_*` reasons: `1.0` iff none of them fired, i.e. the tx passes every standardness check modelled here. |
| `nonstd_version` | `nVersion` is not in `{1, 2, 3}` — Core relays only versions 1–3 (`TX_MAX_STANDARD_VERSION`); any other value is non-standard. |
| `nonstd_weight` | transaction weight `> 400000` weight units (`MAX_STANDARD_TX_WEIGHT`). Heavier txs are non-standard even though they remain consensus-valid and mineable. |
| `nonstd_output_type` | at least one output has a genuinely non-standard scriptPubKey — it does not classify as any recognised template (P2PKH/P2SH/P2WPKH/P2WSH/P2TR/P2PK/OP_RETURN) and is not a bare-multisig template. Bare multisig and OP_RETURN are handled by their own rules below and never set this flag. |
| `nonstd_dust` | at least one **non-OP_RETURN** output pays below its dust threshold. The threshold follows Core's `GetDustThreshold` at the default `dustRelayFee` (3000 sat/kvB): `3 * (9 + spk_len + spend_cost)` sats, where `spend_cost` is `67` for a witness-program output and `148` otherwise — the `3*(9+spk_len+(67 witness | 148))` formula. (P2WPKH → 294, P2PKH → 546.) The check runs on every non-OP_RETURN output, including bare-multisig outputs, since those can be dust too; OP_RETURN outputs (which are legitimately zero-value) are exempt. |
| `nonstd_multi_op_return` | the tx has **more than one** OP_RETURN output. Core's default policy permits at most one data-carrier output per transaction. |
| `nonstd_bare_multisig` | at least one output is a bare-multisig template `OP_m <pubkey>… OP_n OP_CHECKMULTISIG` with `n > 3`. Core relays bare multisig only up to 3-of-3; a larger `n` is non-standard. |

**What non-standard means on-chain.** Every transaction scored here was *confirmed*, so a
`nonstd_*` flag does not mean the tx was invalid — it means the tx **bypassed default relay
policy** to get mined: it reached a miner without traversing a default-configured node's
mempool, e.g. submitted direct-to-miner (out-of-band / accelerator) or relayed through a
node running more permissive policy. That bypass is itself a fingerprint.

**Caveat — policy of today.** Core's standardness rules are policy, not consensus, and they
evolve (dust relay fee changes, the OP_RETURN data-carrier limit, version ceilings, added
templates). These columns score every transaction against **the current rules as encoded
here**, regardless of when it was mined. A transaction that was perfectly standard under the
policy in force at its confirmation height can therefore be flagged non-standard by today's
rules (and vice-versa). Read the `nonstd_*` columns as "non-standard by current modelled
policy", not "was non-standard when broadcast".

**Not covered.** This is a subset of `IsStandardTx`. It does **not** model the scriptSig
push-only requirement (standard inputs must carry only data pushes, no opcodes) nor the
exact sigops-count limit, among other finer checks; a tx these columns call standard could
still be non-standard for a reason not modelled here.

## Protocol markers (columns 161–162)

Presence detection for two on-chain protocols that piggyback on ordinary Bitcoin
transaction fields (a taproot witness / an `OP_RETURN` output) rather than a distinct
transaction type. Both are `1.0` / `0.0`.

| column | meaning |
|---|---|
| `has_inscription_envelope` | `1.0` iff any input's taproot **script-path** witness reveals a tapscript containing the canonical Ordinals/BRC-20 inscription envelope `OP_FALSE OP_IF OP_PUSHBYTES_3 "ord"` — the 6-byte marker `00 63 03 6f 72 64` — as a contiguous byte sequence anywhere in that tapscript. The tapscript is read the same annex-aware way as `taproot_max_merkle_depth`: the witness element immediately before the control block (skipping a trailing annex that starts with `0x50`). Key-path spends and non-taproot inputs have no tapscript and never set this flag. |
| `has_runestone` | `1.0` iff any output is an `OP_RETURN` script whose second opcode is `OP_PUSHNUM_13` (`0x5d`) — the Runes protocol marker `6a 5d …`. This is the Runestone output-recognition rule only; it does not decode or validate the Runestone payload that follows. |

**Caveat — high-certainty positive, not exhaustive.** Both columns are pattern matches
on the canonical, well-formed encoding of each protocol and are reliable when they fire:
a `1.0` is strong evidence the protocol is present. They are **not** exhaustive detectors,
however — a fragmented, obfuscated, or otherwise non-canonical envelope/marker (e.g. an
inscription envelope split across an unusual script structure, or a Runestone-adjacent
encoding that does not use the exact `6a 5d` prefix) can escape detection and read as
`0.0`. Both columns are presence detection over the raw bytes, not content decoding: they
say a protocol marker is there, not what it contains (inscription content/type, Rune
name/amount, etc.).

**Not covered.** Older token/asset protocols that also piggyback on ordinary Bitcoin
outputs — Counterparty and Omni/Mastercoin, among others — have no column here. Detecting
them follows the same presence-detection pattern (a fixed marker/prefix check on an
`OP_RETURN` or similar output) and is a straightforward extension of this section, not
implemented in this sub-project.

## Feerate, appended (columns 163–164)

Appended after the protocol markers (append-only column order), the "by weight" analogues
of `fee_is_multiple_of_vsize` in the Feerate section above. `1.0` / `0.0`.

| column | meaning |
|---|---|
| `fee_is_multiple_of_kwu` | `1.0` iff `fee_sat * 1000` is an exact integer multiple of `weight` — i.e. the sat/kwu rate (`feerate_sat_per_kwu`) is a whole number with no remainder, the weight-unit parallel to `fee_is_multiple_of_vsize`. Present alongside the vsize version precisely so a wallet that rounds cleanly in one unit convention but not the other is distinguishable (see `feerate_sat_per_kwu`). The `* 1000` is computed in `u128` to avoid overflow on large fees. |
| `fee_is_multiple_of_weight` | `1.0` iff `fee_sat` is an exact integer multiple of `weight` — i.e. the sat/wu rate is a whole number with no remainder. Exact integer sat/wu; stricter than the sat/kwu flag (implies it): any tx whose fee is an exact multiple of `weight` is automatically an exact multiple of `weight / 1000`'s scaled form too, so `fee_is_multiple_of_weight` implies `fee_is_multiple_of_kwu` but not vice versa. A wallet targeting exact sat/wu sets both columns; a wallet only hitting a clean sat/kwu rate may set only the kwu column. Plain `u64` modulo (no `* 1000` scaling, so no overflow risk). |

## Multisig & output-value structure (sub-project 3f) (columns 165–169)

Feature-matrix only — none of these 5 columns change the sparsity joint vector
(`CORE_AXES`/`EXTENDED_AXES`/`axis_value` in `vector.rs` are untouched). Multisig config
is read from the **spending inputs**, not the outputs: for each input, the prevout script
itself (bare multisig), else the P2SH redeem script (the last scriptSig push), else the
P2WSH witness script (the last witness item) — mirroring the navigation
`input_subtype_class` already does. An input that spends a non-multisig template
contributes nothing.

For a P2TR input spent via script-path, the revealed tapscript is also decoded when it is
a canonical BIP-342 CHECKSIGADD threshold multisig (`multi_a`:
`<pk1> OP_CHECKSIG (<pk_i> OP_CHECKSIGADD)* <m> OP_NUMEQUAL[VERIFY]`, each key 32 bytes
x-only) — so `multisig_m`/`multisig_n` are now populated for taproot multisig inputs where
they were previously `0`/`0`. Thresholds above 16, which must be pushed as data rather than
an `OP_PUSHNUM`, are not decoded (`None`, same as an unrecognized shape).

| column | meaning |
|---|---|
| `multisig_m` | the `m` (signature threshold) of the **dominant** multisig input — the input config with the largest `n` across all of this tx's inputs. `0` when no input spends a multisig. |
| `multisig_n` | the `n` (total keys) of that same dominant multisig input's config. `0` when no input spends a multisig. |
| `multisig_mixed` | `1.0` iff the tx's multisig-spending inputs carry **two or more distinct** `(m, n)` configs — e.g. one input 2-of-3, another 3-of-5. `0.0` when there are zero or one distinct configs (including the "no multisig inputs at all" case). |
| `distinct_output_value_count` | the number of distinct output values (in satoshis) among this tx's outputs — low values suggest denomination (e.g. equal-value CoinJoin outputs); the max possible is `output_count`. |
| `max_equal_value_output_count` | the size of the largest group of outputs sharing the same value — the classic CoinJoin/denomination signal (e.g. `3` when 3-of-4 outputs share one value). |

## Two more fingerprints (sub-project 3g) (columns 170–171)

Feature-matrix only — these 2 columns do not change the sparsity joint vector
(`CORE_AXES`/`EXTENDED_AXES`/`axis_value` in `vector.rs` are untouched). Both are computed
by the `lumen-fingerprints-lib` classifiers directly, not by new logic in the core crate.

| column | meaning |
|---|---|
| `nlocktime_optin_without_use` | `1.0` iff the tx opts in to nLockTime enforcement — some input's nSequence is in the RBF/enforcement range (`< 0xFFFFFFFE`) — but sets `nLockTime = 0`, so the opt-in has no effect. The complement of `nlocktime_dead`: that column flags a *non-zero* locktime consensus ignores because all inputs are final; this one flags an *enforceable* locktime the tx declines to use. Computed by `lumen_fingerprints_lib::transaction::nlocktime_optin_without_use`. |
| `taproot_keyspend_non_default_sighash` | `1.0` iff any input is a taproot key-path spend (P2TR prevout, one witness item — or two if the second is an annex) whose signature is 65 bytes, i.e. carries an explicit sighash byte instead of the compact 64-byte default-sighash form. Finer-grained than the tx-level `sighash` axis (`sighash_taproot_explicit`/`sighash_taproot_default`), since it isolates key-path spends specifically. Computed per-input by `lumen_fingerprints_lib::input_with_prevout::taproot_keyspend_non_default_sighash` against each input's resolved prevout, then aggregated with `any` across the tx's inputs. |

## Sig-op cost and change-heuristic identity (columns 172–174)

Feature-matrix only — these 3 columns do not change the sparsity joint vector. They answer
two collaborator requests: use a signature measure broader than the ECDSA-only bucket, and
record *which* change heuristic fired (the detecting heuristic is itself a fingerprint).

| column | meaning |
|---|---|
| `sigop_count` | Total sig-op cost of the tx — legacy sig ops (×4), P2SH, and witness — from `bitcoin::Transaction::total_sigop_cost` over the resolved prevouts. Unlike `ecdsa_sigs_index`, it is not scoped to ECDSA, so Schnorr (taproot) and multisig sig ops are counted. |
| `change_detected_by_round_number` | `1.0` iff the value-based round-number change heuristic fires: exactly two outputs, exactly one a non-zero round multiple of 1000 sat (the deliberately entered payment) and the other not (the leftover change). Independent of the script-type heuristic that feeds `change_type_present`/`change_position`. |
| `change_heuristics_agree` | `1.0` iff the script-type and round-number heuristics both fired *and* identified the same output as change. With `change_type_present` and `change_detected_by_round_number` also present, the four combinations — neither, script-type only, round-number only, both-agree, both-disagree — are all recoverable. |

## Loading

```python
import pandas as pd
df = pd.read_csv("features.csv")          # plain CSV
df = pd.read_csv("features.csv.zst")      # pandas/pyarrow handle zstd transparently
```

The header row is exactly `COLUMN_NAMES.join(",")` from
`src/crates/core/src/features.rs` — if this document and that constant ever
disagree, the source is authoritative and this file has drifted.
