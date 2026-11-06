use std::io::{BufWriter, Write};
use std::net::ToSocketAddrs;
use std::path::{Path, PathBuf};

use lumen_core::{
    EpochRow, ScanConfig, epoch_rows, load_templates, resume_point, scan, write_report,
};
use lumen_floresta_source::{FlorestaConfig, FlorestaSource, sync_tip};

const ASSUME_UTREEXO_HEIGHT: u32 = 939_969;

fn default_templates_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/crates/core/templates/wallets.toml")
}

fn usage() -> ! {
    eprintln!(
        "usage:
  lumen scan   --datadir <dir> --out <epochs.jsonl> [--features <path.csv[.zst]>]
                [--vectors <path.csv[.zst]>] [--fresh] [--no-default-peers]
                [--start-height N] [--stop-height N] [--epoch-size N] [--peer host:port]...
  lumen report --epochs <epochs.jsonl[.zst]> --out-dir <dir> [--templates <wallets.toml>]
                # writes <dir>/report.json AND <dir>/wallet-axis-cross.csv (the ML cross)
  lumen series --epochs <epochs.jsonl[.zst]> --out <series.csv> [--min-share F]
  lumen tip    --datadir <dir>                       # print the chain floor/tip as JSON

  --start-height defaults to the last complete epoch in --out, else {ASSUME_UTREEXO_HEIGHT}
  (Floresta's mainnet assume-utreexo checkpoint; blocks below it need the backfill path).
  --peer may be repeated. With no --peer, scan dials default utreexo peers (the
  x1000.bitcoin.seed.dlsouza.lol seed, resolved fresh; LUMEN_SCAN_PEERS overrides,
  --no-default-peers disables). --fresh wipes the datadir and epochs output and scans from
  the floor in one pass, the only faithful way to measure a full window: utreexo cannot
  re-serve blocks a resumed datadir already synced past.
  Wallet templates are matched only by `report`, so a template change needs no rescan.
  report's --templates defaults to the shipped src/crates/core/templates/wallets.toml.
  `series` exports one row per (epoch, axis, value) as a compact CSV. --min-share
  (default 0.0, export everything) folds values below the given share into a per-axis
  `(other)` row."
    );
    std::process::exit(2);
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

/// DNS discovery is unreliable and a wiped datadir has no `anchors.json`. The two
/// hardcoded nodes serve utreexo (`0x1000`) and full blocks.
fn default_utreexo_peers() -> Vec<String> {
    if let Ok(s) = std::env::var("LUMEN_SCAN_PEERS") {
        let v: Vec<String> = s.split_whitespace().map(str::to_string).collect();
        if !v.is_empty() {
            return v;
        }
    }
    let mut peers: Vec<String> = ("x1000.bitcoin.seed.dlsouza.lol", 8333u16)
        .to_socket_addrs()
        .map(|it| it.map(|a| a.to_string()).collect())
        .unwrap_or_default();
    if peers.is_empty() {
        peers.push("189.44.63.101:8333".to_string());
        peers.push("1.228.21.110:8333".to_string());
    }
    peers
}

fn main() {
    // Floresta's logs show a scan hunting for a utreexo peer instead of looking hung.
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
    let vectors_out = arg(args, "--vectors").map(PathBuf::from);

    // A datadir synced past the floor silently skips blocks utreexo can't re-serve, so a
    // faithful full measurement wipes it and scans from the floor in one pass.
    if has_flag(args, "--fresh") {
        let _ = std::fs::remove_dir_all(Path::new(&datadir));
        let _ = std::fs::create_dir_all(Path::new(&datadir));
        let _ = std::fs::remove_file(&out);
        println!("--fresh: wiped datadir {datadir} and epochs output; scanning from the floor");
    }

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
    if config.fixed_peers.is_empty() && !has_flag(args, "--no-default-peers") {
        config.fixed_peers = default_utreexo_peers();
        println!(
            "no --peer given; dialing default utreexo peers: {}",
            config.fixed_peers.join(", ")
        );
    }

    let mut source = FlorestaSource::start(config).expect("starting Floresta");

    let summary = scan(
        &mut source,
        ScanConfig {
            out_path: out,
            epoch_size,
            start_height,
            features_out,
            vectors_out,
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
    let mut fixed_peers: Vec<String> = args
        .iter()
        .enumerate()
        .filter(|(_, a)| a.as_str() == "--peer")
        .filter_map(|(i, _)| args.get(i + 1).cloned())
        .collect();
    if fixed_peers.is_empty() && !has_flag(args, "--no-default-peers") {
        fixed_peers = default_utreexo_peers();
    }
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

/// RFC 4180: quote only when needed.
fn csv_escape(field: &str) -> std::borrow::Cow<'_, str> {
    if field.contains([',', '"', '\n', '\r']) {
        std::borrow::Cow::Owned(format!("\"{}\"", field.replace('"', "\"\"")))
    } else {
        std::borrow::Cow::Borrowed(field)
    }
}

/// File order is already ascending (`scan` only appends). Values below `min_share` of an
/// epoch's `txs` fold into a per-axis `(other)` row.
fn write_series_csv(epochs_path: &Path, out: &Path, min_share: f64) -> std::io::Result<()> {
    let mut w = BufWriter::new(std::fs::File::create(out)?);
    writeln!(w, "start_height,end_height,txs,axis,value,count,share")?;

    for row in epoch_rows(epochs_path).map_err(report_error_to_io)? {
        write_series_row(&mut w, &row.map_err(report_error_to_io)?, min_share)?;
    }

    w.flush()
}

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

fn report_error_to_io(e: lumen_core::ReportError) -> std::io::Error {
    std::io::Error::other(e.to_string())
}
