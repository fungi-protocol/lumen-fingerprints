use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use lumen_core::{
    EpochRow, ScanConfig, epoch_rows, load_templates, resume_point, scan, write_report,
};
use lumen_floresta_source::{FlorestaConfig, FlorestaSource, sync_tip};

const ASSUME_UTREEXO_HEIGHT: u32 = 939_969;

/// The templates shipped in the repo, used by `lumen report` when `--templates` is
/// omitted. Report-time matching (see `lumen_core::report::build_report`) means
/// a `wallets.toml` correction takes effect the next time `report` runs, so defaulting
/// to the shipped file — rather than requiring `--templates` on every invocation, or
/// worse, silently producing a report with no template series at all — is what makes
/// that "edit the TOML, see the fix" loop actually convenient day to day. An explicit
/// `--templates` still overrides it, e.g. to try a wallet developer's proposed PR
/// before merging it.
fn default_templates_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/crates/core/templates/wallets.toml")
}

fn usage() -> ! {
    eprintln!(
        "usage:
  lumen scan   --datadir <dir> --out <epochs.jsonl> [--features <path.csv[.zst]>]
                [--start-height N] [--stop-height N] [--epoch-size N] [--peer host:port]
  lumen report --epochs <epochs.jsonl[.zst]> --out-dir <dir> [--templates <wallets.toml>]
  lumen series --epochs <epochs.jsonl[.zst]> --out <series.csv> [--min-share F]
  lumen tip    --datadir <dir>                       # print the chain floor/tip as JSON

  --start-height defaults to the last complete epoch in --out, else {ASSUME_UTREEXO_HEIGHT}
  (Floresta's mainnet assume-utreexo checkpoint; blocks below it need the backfill path).
  `scan` takes no --templates: it no longer matches against wallet templates at all —
  matching happens entirely at report time (see `report`'s --templates below), so a scan
  is a one-time investment in the raw data, independent of any wallet list.
  report's --templates defaults to the shipped src/crates/core/templates/wallets.toml;
  matching is recomputed at report time, so a corrected template takes effect on the next
  `report` run with no rescan needed.
  `series` exports one row per (epoch, axis, value) as a compact, committable CSV —
  see `write_series_csv` in this file. --min-share (default 0.0, i.e. export
  everything) aggregates any value below the given share into a per-axis `(other)` row
  instead, to keep the file small enough to commit."
    );
    std::process::exit(2);
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() {
    // Surface Floresta's `tracing` logs (peer discovery, utreexo-peer connect/ban, sync)
    // to stderr so a scan that is quietly hunting for a utreexo peer is diagnosable rather
    // than looking hung. `RUST_LOG` overrides; unset defaults to node/peer info.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        tracing_subscriber::EnvFilter::new("floresta_wire=info,floresta_chain=info")
    });
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();

    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("scan") => cmd_scan(&args),
        Some("tip") => cmd_tip(&args),
        Some("report") => cmd_report(&args),
        Some("series") => cmd_series(&args),
        _ => usage(),
    }
}

fn cmd_scan(args: &[String]) {
    let datadir = arg(args, "--datadir").unwrap_or_else(|| usage());
    let out = PathBuf::from(arg(args, "--out").unwrap_or_else(|| usage()));
    let epoch_size: u32 = arg(args, "--epoch-size")
        .map(|s| s.parse().expect("--epoch-size must be a number"))
        .unwrap_or(144);
    let stop_height: Option<u32> =
        arg(args, "--stop-height").map(|s| s.parse().expect("--stop-height must be a number"));
    let features_out = arg(args, "--features").map(PathBuf::from);

    let resumed = resume_point(&out).expect("reading resume point");
    let start_height: u32 = arg(args, "--start-height")
        .map(|s| s.parse().expect("--start-height must be a number"))
        .or(resumed)
        .unwrap_or(ASSUME_UTREEXO_HEIGHT);

    if let Some(h) = resumed {
        println!("resuming from height {h} (last complete epoch on disk)");
    }
    println!("scanning from height {start_height}, epoch size {epoch_size}");

    let mut config = FlorestaConfig::new(PathBuf::from(datadir), start_height);
    config.stop_height = stop_height;
    config.fixed_peers = args
        .iter()
        .enumerate()
        .filter(|(_, a)| a.as_str() == "--peer")
        .filter_map(|(i, _)| args.get(i + 1).cloned())
        .collect();

    let mut source = FlorestaSource::start(config).expect("starting Floresta");

    let summary = scan(
        &mut source,
        ScanConfig {
            out_path: out,
            epoch_size,
            start_height,
            features_out,
        },
    )
    .expect("scan failed");

    println!(
        "done: {} epochs, {} blocks, {} txs, {} defects",
        summary.epochs, summary.blocks, summary.txs, summary.defects
    );
    if summary.coverage_gaps > 0 {
        eprintln!(
            "WARNING: {} epoch(s) covered fewer blocks than their height span — the \
             source skipped blocks silently. This run is NOT a faithful measurement of \
             its window. Re-scan from a fresh datadir in a single pass (no resume).",
            summary.coverage_gaps
        );
    }
}

fn cmd_tip(args: &[String]) {
    let datadir = arg(args, "--datadir").unwrap_or_else(|| usage());
    let fixed_peers: Vec<String> = args
        .iter()
        .enumerate()
        .filter(|(_, a)| a.as_str() == "--peer")
        .filter_map(|(i, _)| args.get(i + 1).cloned())
        .collect();
    eprintln!("syncing headers to find the live tip (needs peers; can take a few minutes)...");
    let info = sync_tip(Path::new(&datadir), fixed_peers).expect("syncing chain tip");
    println!(
        "{{\"tip\":{},\"floor\":{},\"in_ibd\":{}}}",
        info.tip, info.floor, info.in_ibd
    );
}

fn cmd_report(args: &[String]) {
    let epochs = PathBuf::from(arg(args, "--epochs").unwrap_or_else(|| usage()));
    let out_dir = PathBuf::from(arg(args, "--out-dir").unwrap_or_else(|| usage()));
    let templates_path = arg(args, "--templates")
        .map(PathBuf::from)
        .unwrap_or_else(default_templates_path);

    let templates = load_templates(&templates_path).expect("loading templates");
    println!(
        "matching against {} wallet templates from {}",
        templates.len(),
        templates_path.display()
    );

    write_report(&epochs, &out_dir, &templates).expect("writing report");
    println!("wrote {}", out_dir.join("report.json").display());
}

fn cmd_series(args: &[String]) {
    let epochs = PathBuf::from(arg(args, "--epochs").unwrap_or_else(|| usage()));
    let out = PathBuf::from(arg(args, "--out").unwrap_or_else(|| usage()));
    let min_share: f64 = arg(args, "--min-share")
        .map(|s| s.parse().expect("--min-share must be a number"))
        .unwrap_or(0.0);

    write_series_csv(&epochs, &out, min_share).expect("writing series");

    let bytes = std::fs::metadata(&out).expect("stat series output").len();
    println!("wrote {} ({bytes} bytes)", out.display());
}

/// Escape one CSV field per RFC 4180: wrap in quotes (doubling any embedded quote)
/// only when the field contains a character that would otherwise be ambiguous.
fn csv_escape(field: &str) -> std::borrow::Cow<'_, str> {
    if field.contains([',', '"', '\n', '\r']) {
        std::borrow::Cow::Owned(format!("\"{}\"", field.replace('"', "\"\"")))
    } else {
        std::borrow::Cow::Borrowed(field)
    }
}

/// Write one row per (epoch, axis, value) — see the `series` usage text above for the
/// column layout.
///
/// Streams `EpochRow`s straight off disk via `lumen_core::epoch_rows` instead of
/// reading the whole file into a `Vec<EpochRow>` first: this is a pure per-row
/// transformation (read one row, write its lines, drop it, move to the next), so
/// nothing is gained by materializing the dataset up front, and on the full-size
/// dataset (2.7 GB on disk, several GB once deserialized into `Vec<EpochRow>`) holding
/// it all was the entire memory cost of this command.
///
/// Rows are written in file order with no additional sort. The old implementation
/// collected into a `Vec` and called `sort_by_key(|r| r.start_height)` before writing,
/// but that sort was always a no-op in practice: `scan` (`src/crates/core/src/
/// engine.rs`) only ever appends epoch rows in increasing height order — each row is
/// written as `acc.flush()` produces it, immediately after the previous one, and
/// `resume_point` picks up exactly where the file left off — so the file is already in
/// ascending `start_height` order before this function ever sees it, and `Vec::sort_by
/// _key` is stable, so sorting an already-sorted sequence changes nothing. Streaming in
/// file order therefore produces byte-identical output to the old collect-then-sort
/// version. (Within a row, `axis_counts` is a `BTreeMap<String, BTreeMap<String,
/// u64>>`, so iterating it is already sorted by axis, then by value — the exact order
/// the spec asks for — independent of the across-row question above.)
///
/// `axis_counts` already includes every axis the scan computed — core, extended, and
/// heuristic alike (see `EpochAccumulator::ingest` in `src/crates/core/src/epoch.rs`)
/// — so no axis needs to be singled out here; iterating the whole map is complete by
/// construction.
///
/// `min_share` (0.0 by default, meaning "export everything") aggregates any value whose
/// share of that epoch's `txs` falls below it into a single per-axis `(other)` row, so
/// a long tail of rare values does not blow up the file size. A `(other)` row is only
/// emitted when there is a genuine remainder to report.
fn write_series_csv(epochs_path: &Path, out: &Path, min_share: f64) -> std::io::Result<()> {
    let mut w = BufWriter::new(std::fs::File::create(out)?);
    writeln!(w, "start_height,end_height,txs,axis,value,count,share")?;

    for row in epoch_rows(epochs_path).map_err(report_error_to_io)? {
        write_series_row(&mut w, &row.map_err(report_error_to_io)?, min_share)?;
    }

    w.flush()
}

/// Writes one `EpochRow`'s worth of series lines. Factored out of `write_series_csv` so
/// that function's body is just "stream a row in, write it, drop it" — the per-row
/// logic itself (the `min_share` bucketing in particular) is unchanged from before this
/// was made to stream.
fn write_series_row(w: &mut impl Write, row: &EpochRow, min_share: f64) -> std::io::Result<()> {
    for (axis, values) in &row.axis_counts {
        let mut entries: Vec<(&str, u64)> = values.iter().map(|(v, &c)| (v.as_str(), c)).collect();

        if min_share > 0.0 && row.txs > 0 {
            let mut kept = Vec::new();
            let mut other = 0u64;
            for (value, count) in entries {
                if (count as f64) / (row.txs as f64) >= min_share {
                    kept.push((value, count));
                } else {
                    other += count;
                }
            }
            if other > 0 {
                kept.push(("(other)", other));
            }
            kept.sort_by_key(|(value, _)| *value);
            entries = kept;
        }

        for (value, count) in entries {
            let share = if row.txs > 0 {
                count as f64 / row.txs as f64
            } else {
                0.0
            };
            writeln!(
                w,
                "{},{},{},{},{},{},{:.6}",
                row.start_height,
                row.end_height,
                row.txs,
                csv_escape(axis),
                csv_escape(value),
                count,
                share
            )?;
        }
    }
    Ok(())
}

/// Converts a `lumen_core::ReportError` (I/O or JSON-parse failure while
/// streaming epoch rows) into an `io::Error`, so `write_series_csv` can keep returning
/// `std::io::Result` — the same error type it already used for its own file I/O —
/// rather than introducing a second error type into this small binary.
fn report_error_to_io(e: lumen_core::ReportError) -> std::io::Error {
    std::io::Error::other(e.to_string())
}
