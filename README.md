# Lumen Fingerprints

Surveys how Bitcoin transactions are *built* — script types, ordering, nSequence, sighash,
feerate rounding, amounts, change — and reports how identifying each fingerprint is.

## Dashboard

```sh
nix run .#dashboard
```

Opens a local Explorer; click **Run a scan** or browse the field catalog.

## CLI

```sh
lumen scan   --datadir <dir> --out epochs.jsonl    # scan a block range via Floresta
lumen report --epochs epochs.jsonl --out-dir out/  # aggregate into report.json
```

A scan covers up to ~20k blocks — Floresta's assume-utreexo floor (height 939,969) to the
chain tip. `nix run .#scan` does scan → report → serve in one command.
