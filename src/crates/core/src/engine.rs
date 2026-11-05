use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use lumen_primitives::traits::BlockSource;

use crate::epoch::{EpochAccumulator, EpochRow};
use crate::features::COLUMN_NAMES;

#[derive(Debug, Clone)]
pub struct ScanConfig {
    pub out_path: PathBuf,
    pub epoch_size: u32,
    pub start_height: u32,
    pub features_out: Option<PathBuf>,
    pub vectors_out: Option<PathBuf>,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ScanSummary {
    pub epochs: u64,
    pub blocks: u64,
    pub txs: u64,
    pub defects: u64,
    /// Epochs missing blocks the source skipped silently: the window is not faithfully measured.
    pub coverage_gaps: u64,
}

#[derive(Debug)]
pub enum ScanError {
    Io(std::io::Error),
    Source(String),
    Serialize(serde_json::Error),
    /// A per-tx CSV can only truncate, so resuming would lose its prior data.
    ResumeWithPerTxSink,
}

impl std::fmt::Display for ScanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::Source(msg) => write!(f, "block source: {msg}"),
            Self::Serialize(e) => write!(f, "serializing epoch row: {e}"),
            Self::ResumeWithPerTxSink => write!(
                f,
                "cannot resume a scan with --features/--vectors: the per-tx CSV would be \
                 truncated, discarding prior data. Scan a single pass to a fresh --out, or omit \
                 --features/--vectors when resuming"
            ),
        }
    }
}

impl std::error::Error for ScanError {}

/// Per-tx CSV writer. Must be closed with `finish`: `Drop` on `BufWriter` and on the
/// zstd encoder ignores IO errors and can silently truncate the file.
enum FeatureSink {
    Plain(BufWriter<std::fs::File>),
    Zstd(zstd::stream::write::Encoder<'static, std::fs::File>),
}

impl std::io::Write for FeatureSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Self::Plain(w) => w.write(buf),
            Self::Zstd(w) => w.write(buf),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Self::Plain(w) => w.flush(),
            Self::Zstd(w) => w.flush(),
        }
    }
}

impl FeatureSink {
    fn finish(self) -> std::io::Result<()> {
        match self {
            Self::Plain(mut w) => w.flush(),
            Self::Zstd(w) => {
                w.finish()?;
                Ok(())
            }
        }
    }
}

/// One past the last epoch on disk, so a crash loses at most one epoch of work.
pub fn resume_point(out_path: &Path) -> Result<Option<u32>, ScanError> {
    if !out_path.exists() {
        return Ok(None);
    }
    let file = std::fs::File::open(out_path).map_err(ScanError::Io)?;
    let mut last: Option<EpochRow> = None;
    for line in BufReader::new(file).lines() {
        let line = line.map_err(ScanError::Io)?;
        if line.trim().is_empty() {
            continue;
        }
        last = Some(serde_json::from_str(&line).map_err(ScanError::Serialize)?);
    }
    Ok(last.map(|row| row.end_height + 1))
}

/// Memory stays flat: only one epoch of counters is held.
pub fn scan<S: BlockSource>(source: &mut S, config: ScanConfig) -> Result<ScanSummary, ScanError> {
    let resuming = std::fs::metadata(&config.out_path)
        .map(|m| m.len() > 0)
        .unwrap_or(false);
    if resuming && (config.features_out.is_some() || config.vectors_out.is_some()) {
        return Err(ScanError::ResumeWithPerTxSink);
    }

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&config.out_path)
        .map_err(ScanError::Io)?;
    let mut out = BufWriter::new(file);

    let mut acc = EpochAccumulator::new(config.start_height, config.epoch_size);
    let mut summary = ScanSummary::default();
    let mut pending = false;
    let mut epoch_blocks: u32 = 0;
    let mut epoch_start = config.start_height;

    let mut features: Option<FeatureSink> = match &config.features_out {
        None => None,
        Some(path) => Some(open_sink(path, &COLUMN_NAMES.join(","))?),
    };

    let mut vectors: Option<FeatureSink> = match &config.vectors_out {
        None => None,
        Some(path) => Some(open_sink(
            path,
            &crate::FingerprintVector::vector_csv_header(),
        )?),
    };

    loop {
        let block = match source.next_block() {
            Ok(Some(block)) => block,
            Ok(None) => break,
            Err(e) => return Err(ScanError::Source(e.to_string())),
        };

        let height = block.height;
        if features.is_some() || vectors.is_some() {
            let mut sink_err: Option<std::io::Error> = None;
            acc.ingest_with(&block, &mut |tx, _height, vector, _flags, row| {
                if sink_err.is_some() {
                    return;
                }
                if let Some(w) = features.as_mut()
                    && let Err(e) = write_feature_row(w, row)
                {
                    sink_err = Some(e);
                    return;
                }
                if let Some(w) = vectors.as_mut() {
                    let txid = tx.compute_txid().to_string();
                    if let Err(e) = writeln!(w, "{}", vector.to_vector_csv_line(&txid)) {
                        sink_err = Some(e);
                    }
                }
            });
            if let Some(e) = sink_err {
                return Err(ScanError::Io(e));
            }
        } else {
            acc.ingest(&block);
        }
        summary.blocks += 1;
        epoch_blocks += 1;
        pending = true;
        drop(block); // keeps memory flat: only counters survive

        if acc.boundary_reached(height) {
            if epoch_blocks != height - epoch_start + 1 {
                summary.coverage_gaps += 1;
            }
            let row = acc.flush(height + 1);
            write_row(&mut out, &row, &mut summary)?;
            pending = false;
            epoch_blocks = 0;
            epoch_start = height + 1;
        }
    }

    if pending {
        let row = acc.flush(0);
        write_row(&mut out, &row, &mut summary)?;
    }

    out.flush().map_err(ScanError::Io)?;

    if let Some(sink) = features {
        sink.finish().map_err(ScanError::Io)?;
    }
    if let Some(sink) = vectors {
        sink.finish().map_err(ScanError::Io)?;
    }

    Ok(summary)
}

fn write_row(
    out: &mut BufWriter<std::fs::File>,
    row: &EpochRow,
    summary: &mut ScanSummary,
) -> Result<(), ScanError> {
    let line = serde_json::to_string(row).map_err(ScanError::Serialize)?;
    writeln!(out, "{line}").map_err(ScanError::Io)?;
    // Visible to `resume_point` at once. No fsync: survives a crash, not power loss.
    out.flush().map_err(ScanError::Io)?;
    summary.epochs += 1;
    summary.txs += row.txs;
    summary.defects += row.defects;
    Ok(())
}

/// Truncates `path` and writes `header`. A `.zst` suffix selects zstd.
fn open_sink(path: &Path, header: &str) -> Result<FeatureSink, ScanError> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(ScanError::Io)?;
    let mut w = if path.extension().is_some_and(|e| e == "zst") {
        FeatureSink::Zstd(zstd::stream::write::Encoder::new(file, 3).map_err(ScanError::Io)?)
    } else {
        FeatureSink::Plain(BufWriter::new(file))
    };
    writeln!(w, "{header}").map_err(ScanError::Io)?;
    Ok(w)
}

fn write_feature_row(
    w: &mut FeatureSink,
    row: &crate::features::FeatureRow,
) -> std::io::Result<()> {
    use std::fmt::Write as _;
    let mut line = String::with_capacity(row.values.len() * 4 + row.txid.len() + 12);
    line.push_str(&row.txid);
    let _ = write!(line, ",{}", row.height);
    for v in &row.values {
        let _ = write!(line, ",{v}");
    }
    writeln!(w, "{line}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::tests_support::cake_like_block;
    use lumen_primitives::traits::{BlockSource, SourcedBlock};

    #[derive(Debug)]
    struct Never;
    impl std::fmt::Display for Never {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "never")
        }
    }
    impl std::error::Error for Never {}

    struct VecSource(std::vec::IntoIter<SourcedBlock>);
    impl BlockSource for VecSource {
        type Error = Never;
        fn next_block(&mut self) -> Result<Option<SourcedBlock>, Never> {
            Ok(self.0.next())
        }
    }

    fn blocks(from: u32, count: u32) -> VecSource {
        VecSource(
            (from..from + count)
                .map(cake_like_block)
                .collect::<Vec<_>>()
                .into_iter(),
        )
    }

    fn blocks_at(heights: &[u32]) -> VecSource {
        VecSource(
            heights
                .iter()
                .map(|h| cake_like_block(*h))
                .collect::<Vec<_>>()
                .into_iter(),
        )
    }

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        if name.contains('.') {
            p.push(format!("survey-test-{name}"));
        } else {
            p.push(format!("survey-test-{name}.jsonl"));
        }
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn writes_one_row_per_epoch() {
        let out = tmp_path("epochs");
        let summary = scan(
            &mut blocks(1000, 20),
            ScanConfig {
                out_path: out.clone(),
                epoch_size: 5,
                start_height: 1000,
                features_out: None,
                vectors_out: None,
            },
        )
        .unwrap();

        assert_eq!(summary.blocks, 20);
        assert_eq!(summary.epochs, 4, "20 blocks / 5 per epoch");

        let text = std::fs::read_to_string(&out).unwrap();
        let rows: Vec<EpochRow> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].start_height, 1000);
        assert_eq!(rows[0].end_height, 1004);
        assert_eq!(rows[1].start_height, 1005);
        assert_eq!(rows.iter().map(|r| r.txs).sum::<u64>(), 20);
    }

    #[test]
    fn a_contiguous_scan_reports_no_coverage_gaps() {
        let out = tmp_path("no-gap");
        let summary = scan(
            &mut blocks(1000, 10),
            ScanConfig {
                out_path: out,
                epoch_size: 5,
                start_height: 1000,
                features_out: None,
                vectors_out: None,
            },
        )
        .unwrap();
        assert_eq!(summary.coverage_gaps, 0);
    }

    #[test]
    fn resuming_with_a_pertx_sink_errors_instead_of_truncating() {
        let out = tmp_path("resume-epochs");
        std::fs::write(
            &out,
            "{\"start_height\":1000,\"end_height\":1004,\"txs\":5}\n",
        )
        .unwrap();
        let feat = tmp_path("resume-feat.csv");
        std::fs::write(&feat, "header\nrow_from_before\n").unwrap();

        let err = scan(
            &mut blocks(1005, 5),
            ScanConfig {
                out_path: out.clone(),
                epoch_size: 5,
                start_height: 1005,
                features_out: Some(feat.clone()),
                vectors_out: None,
            },
        )
        .unwrap_err();

        assert!(matches!(err, ScanError::ResumeWithPerTxSink));
        assert_eq!(
            std::fs::read_to_string(&feat).unwrap(),
            "header\nrow_from_before\n"
        );
    }

    #[test]
    fn resuming_with_a_vectors_sink_errors_instead_of_truncating() {
        let out = tmp_path("resume-vec-epochs");
        std::fs::write(
            &out,
            "{\"start_height\":1000,\"end_height\":1004,\"txs\":5}\n",
        )
        .unwrap();
        let vec_path = tmp_path("resume-vec.csv");
        std::fs::write(&vec_path, "header\nrow_from_before\n").unwrap();

        let err = scan(
            &mut blocks(1005, 5),
            ScanConfig {
                out_path: out.clone(),
                epoch_size: 5,
                start_height: 1005,
                features_out: None,
                vectors_out: Some(vec_path.clone()),
            },
        )
        .unwrap_err();

        assert!(matches!(err, ScanError::ResumeWithPerTxSink));
        assert_eq!(
            std::fs::read_to_string(&vec_path).unwrap(),
            "header\nrow_from_before\n"
        );
    }

    #[test]
    fn an_epoch_missing_blocks_is_flagged_as_a_coverage_gap() {
        let out = tmp_path("gap");
        let summary = scan(
            &mut blocks_at(&[1000, 1001, 1003, 1004, 1005, 1006, 1007, 1008, 1009]),
            ScanConfig {
                out_path: out.clone(),
                epoch_size: 5,
                start_height: 1000,
                features_out: None,
                vectors_out: None,
            },
        )
        .unwrap();

        assert_eq!(
            summary.coverage_gaps, 1,
            "the first epoch is missing block 1002"
        );
        assert_eq!(summary.blocks, 9);

        let text = std::fs::read_to_string(&out).unwrap();
        let rows: Vec<EpochRow> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        // The row itself looks complete; only the block count reveals the gap.
        assert_eq!(rows[0].start_height, 1000);
        assert_eq!(rows[0].end_height, 1004);
        assert_eq!(rows[0].defects, 0);
        assert_eq!(rows[1].start_height, 1005);
        assert_eq!(rows[1].end_height, 1009);
    }

    #[test]
    fn flushes_a_trailing_partial_epoch() {
        let out = tmp_path("partial");
        let summary = scan(
            &mut blocks(1000, 7),
            ScanConfig {
                out_path: out.clone(),
                epoch_size: 5,
                start_height: 1000,
                features_out: None,
                vectors_out: None,
            },
        )
        .unwrap();

        assert_eq!(summary.epochs, 2, "one full epoch plus a 2-block remainder");
        let rows = std::fs::read_to_string(&out).unwrap();
        assert_eq!(rows.lines().count(), 2);
    }

    #[test]
    fn resume_point_is_after_the_last_complete_epoch() {
        let out = tmp_path("resume");
        scan(
            &mut blocks(1000, 10),
            ScanConfig {
                out_path: out.clone(),
                epoch_size: 5,
                start_height: 1000,
                features_out: None,
                vectors_out: None,
            },
        )
        .unwrap();

        assert_eq!(resume_point(&out).unwrap(), Some(1010));
        assert_eq!(
            resume_point(std::path::Path::new("/nonexistent")).unwrap(),
            None
        );
    }

    #[test]
    fn resume_after_a_crash_mid_epoch_covers_every_block_once() {
        // 7 blocks with epoch_size 5 leave a trailing partial epoch, as a crash would.
        let out = tmp_path("resume-mid-epoch");
        let first = scan(
            &mut blocks(1000, 7),
            ScanConfig {
                out_path: out.clone(),
                epoch_size: 5,
                start_height: 1000,
                features_out: None,
                vectors_out: None,
            },
        )
        .unwrap();
        assert_eq!(first.blocks, 7);

        let resume_at = resume_point(&out).unwrap();
        assert_eq!(
            resume_at,
            Some(1007),
            "next unprocessed height must be exactly one past the last block actually \
             ingested (1006), not padded out to the next epoch boundary"
        );
        let resume_at = resume_at.unwrap();

        let second = scan(
            &mut blocks(resume_at, 8),
            ScanConfig {
                out_path: out.clone(),
                epoch_size: 5,
                start_height: resume_at,
                features_out: None,
                vectors_out: None,
            },
        )
        .unwrap();
        assert_eq!(second.blocks, 8);

        let text = std::fs::read_to_string(&out).unwrap();
        let rows: Vec<EpochRow> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(rows.len(), 4, "one full + one partial epoch from each run");

        for pair in rows.windows(2) {
            assert_eq!(
                pair[1].start_height,
                pair[0].end_height + 1,
                "epoch rows must be contiguous across the resume boundary: {:?} -> {:?}",
                pair[0],
                pair[1]
            );
        }

        assert_eq!((rows[0].start_height, rows[0].end_height), (1000, 1004));
        assert_eq!((rows[1].start_height, rows[1].end_height), (1005, 1006));
        assert_eq!((rows[2].start_height, rows[2].end_height), (1007, 1011));
        assert_eq!((rows[3].start_height, rows[3].end_height), (1012, 1014));

        // One classifiable tx per block.
        let total_blocks = first.blocks + second.blocks;
        assert_eq!(total_blocks, 15);
        assert_eq!(rows.iter().map(|r| r.txs).sum::<u64>(), total_blocks);
    }

    #[test]
    fn scanning_appends_rather_than_truncating() {
        let out = tmp_path("append");
        scan(
            &mut blocks(1000, 5),
            ScanConfig {
                out_path: out.clone(),
                epoch_size: 5,
                start_height: 1000,
                features_out: None,
                vectors_out: None,
            },
        )
        .unwrap();
        scan(
            &mut blocks(1005, 5),
            ScanConfig {
                out_path: out.clone(),
                epoch_size: 5,
                start_height: 1005,
                features_out: None,
                vectors_out: None,
            },
        )
        .unwrap();

        let text = std::fs::read_to_string(&out).unwrap();
        assert_eq!(
            text.lines().count(),
            2,
            "second run must not clobber the first"
        );
    }

    #[test]
    fn features_sink_writes_one_row_per_tx_with_header() {
        let out = tmp_path("feat-epochs");
        let feat = tmp_path("feat.csv");
        let summary = scan(
            &mut blocks(1000, 5),
            ScanConfig {
                out_path: out,
                epoch_size: 5,
                start_height: 1000,
                features_out: Some(feat.clone()),
                vectors_out: None,
            },
        )
        .unwrap();

        let text = std::fs::read_to_string(&feat).unwrap();
        let mut lines = text.lines();
        let header = lines.next().unwrap();
        assert_eq!(header, crate::COLUMN_NAMES.join(","));
        let rows: Vec<&str> = lines.collect();
        assert_eq!(rows.len() as u64, summary.txs);
        for row in &rows {
            for field in row.split(',').skip(2) {
                field.parse::<f64>().expect("feature field is numeric");
            }
        }

        // `address_reuse` is the only aux flag the fixture sets; checking it by name
        // catches aux values landing in the wrong column.
        let address_reuse_col = crate::COLUMN_NAMES
            .iter()
            .position(|c| *c == "address_reuse")
            .unwrap();
        for row in &rows {
            let fields: Vec<&str> = row.split(',').collect();
            assert_eq!(
                fields[address_reuse_col].parse::<f64>().unwrap(),
                1.0,
                "address_reuse must land in its own named column, not a neighboring one"
            );
        }
    }

    #[test]
    fn vectors_sink_writes_one_row_per_tx_with_header() {
        let out = tmp_path("vec-epochs");
        let vec_path = tmp_path("vec.csv");
        let summary = scan(
            &mut blocks(1000, 5),
            ScanConfig {
                out_path: out,
                epoch_size: 5,
                start_height: 1000,
                features_out: None,
                vectors_out: Some(vec_path.clone()),
            },
        )
        .unwrap();

        let text = std::fs::read_to_string(&vec_path).unwrap();
        let mut lines = text.lines();
        let header = lines.next().unwrap();
        assert_eq!(header, crate::FingerprintVector::vector_csv_header());
        let rows: Vec<&str> = lines.collect();
        assert_eq!(rows.len() as u64, summary.txs);
        for row in &rows {
            let fields: Vec<&str> = row.split(',').collect();
            assert_eq!(
                fields.len(),
                1 + crate::vector::VECTOR_AXES.len(),
                "row must carry txid + one value per VECTOR_AXES entry"
            );
            assert_eq!(fields[0].len(), 64, "first field must be the txid");
        }
    }

    #[test]
    fn features_and_vectors_sinks_can_both_be_populated_in_one_pass() {
        let out = tmp_path("both-epochs");
        let feat = tmp_path("both-feat.csv");
        let vec_path = tmp_path("both-vec.csv");
        let summary = scan(
            &mut blocks(1000, 5),
            ScanConfig {
                out_path: out,
                epoch_size: 5,
                start_height: 1000,
                features_out: Some(feat.clone()),
                vectors_out: Some(vec_path.clone()),
            },
        )
        .unwrap();

        let feat_rows = std::fs::read_to_string(&feat).unwrap().lines().count();
        let vec_rows = std::fs::read_to_string(&vec_path).unwrap().lines().count();
        assert_eq!(feat_rows as u64, summary.txs + 1);
        assert_eq!(vec_rows as u64, summary.txs + 1);
    }

    #[test]
    fn features_sink_populates_block_relative_feerate() {
        let out = tmp_path("feat-blockmin-epochs");
        let feat = tmp_path("feat-blockmin.csv");
        let summary = scan(
            &mut blocks(1000, 2),
            ScanConfig {
                out_path: out,
                epoch_size: 1,
                start_height: 1000,
                features_out: Some(feat.clone()),
                vectors_out: None,
            },
        )
        .unwrap();

        let text = std::fs::read_to_string(&feat).unwrap();
        let header: Vec<&str> = text.lines().next().unwrap().split(',').collect();
        let idx = |n: &str| header.iter().position(|c| *c == n).unwrap();
        let rows: Vec<&str> = text.lines().skip(1).collect();
        assert_eq!(rows.len() as u64, summary.txs);

        for r in &rows {
            let f: Vec<&str> = r.split(',').collect();
            f[idx("feerate_over_block_min")].parse::<f64>().unwrap();
            f[idx("feerate_over_prev_block_min")]
                .parse::<f64>()
                .unwrap();
        }

        // Column 1 is `block_height`; -1.0 is the no-previous-block sentinel.
        let height = |r: &str| r.split(',').nth(1).unwrap().parse::<u32>().unwrap();
        let prev_min = |r: &str| {
            r.split(',')
                .nth(idx("feerate_over_prev_block_min"))
                .unwrap()
                .parse::<f64>()
                .unwrap()
        };
        let block1 = rows.iter().find(|r| height(r) == 1000).unwrap();
        let block2 = rows.iter().find(|r| height(r) == 1001).unwrap();
        assert_eq!(prev_min(block1), -1.0, "block 1 has no preceding block");
        assert!(
            prev_min(block2) >= 0.0,
            "block 2's prev-min ratio references block 1's positive minimum"
        );
    }

    #[test]
    fn no_features_path_leaves_behaviour_unchanged() {
        let out = tmp_path("no-feat");
        let summary = scan(
            &mut blocks(1000, 5),
            ScanConfig {
                out_path: out,
                epoch_size: 5,
                start_height: 1000,
                features_out: None,
                vectors_out: None,
            },
        )
        .unwrap();
        assert!(summary.txs > 0);
    }

    #[test]
    fn features_sink_zstd_round_trips() {
        let out = tmp_path("feat-z-epochs");
        let plain = tmp_path("feat-plain.csv");
        let zstd = tmp_path("feat-z.csv.zst");
        let cfg = |p: std::path::PathBuf| ScanConfig {
            out_path: out.clone(),
            epoch_size: 5,
            start_height: 1000,
            features_out: Some(p),
            vectors_out: None,
        };
        scan(&mut blocks(1000, 5), cfg(plain.clone())).unwrap();
        let _ = std::fs::remove_file(&out);
        scan(&mut blocks(1000, 5), cfg(zstd.clone())).unwrap();

        let plain_text = std::fs::read_to_string(&plain).unwrap();
        let zbytes = std::fs::read(&zstd).unwrap();
        let ztext = String::from_utf8(zstd::decode_all(&zbytes[..]).unwrap()).unwrap();
        assert_eq!(plain_text, ztext);
    }

    #[test]
    fn scan_to_epochs_to_report_reflects_the_scanned_transactions() {
        use crate::report::{build_report, read_epoch_rows, write_report};
        use crate::vector::tests_support::{block_with_many, cake_like_tx, legacy_p2pkh_tx};

        let start = 500_000u32;

        let mut fixture: Vec<SourcedBlock> = (start..start + 8).map(cake_like_block).collect();

        // A 9th block with three shapes gives `nsequence` a known 9/1/1 split.
        let cake_extra = cake_like_tx();
        let mut pubkey = vec![0xab; 33];
        pubkey[0] = 0x02;
        let mut legacy_max = legacy_p2pkh_tx(pubkey.clone());
        legacy_max.input[0].previous_output.vout = 100;
        let mut legacy_rbf = legacy_p2pkh_tx(pubkey);
        legacy_rbf.input[0].previous_output.vout = 200;
        legacy_rbf.input[0].sequence = bitcoin::Sequence(0xffff_fffd);
        let mixed_height = start + 8;
        fixture.push(block_with_many(
            vec![cake_extra, legacy_max, legacy_rbf],
            mixed_height,
        ));

        let mut source = VecSource(fixture.into_iter());

        let mut out = std::env::temp_dir();
        out.push(format!("wfs-pipeline-epochs-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&out);
        let mut report_dir = std::env::temp_dir();
        report_dir.push(format!("wfs-pipeline-report-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&report_dir);

        let epoch_size = 4;
        let summary = scan(
            &mut source,
            ScanConfig {
                out_path: out.clone(),
                epoch_size,
                start_height: start,
                features_out: None,
                vectors_out: None,
            },
        )
        .unwrap();

        assert_eq!(summary.blocks, 9);
        assert_eq!(
            summary.epochs, 3,
            "8 blocks / 4 per epoch, plus a trailing 1-block partial epoch"
        );
        assert_eq!(
            summary.txs, 11,
            "8 single-tx cake blocks + 3 txs in the mixed block"
        );
        assert_eq!(
            summary.coverage_gaps, 0,
            "the fixture is contiguous, height by height"
        );

        write_report(&out, &report_dir, &[]).unwrap();
        let report_json_path = report_dir.join("report.json");
        assert!(
            report_json_path.exists(),
            "write_report must produce report.json"
        );
        let report_json = std::fs::read_to_string(&report_json_path).unwrap();
        assert!(!report_json.trim().is_empty());

        // `Report` has no `Deserialize`, so typed checks rebuild it from the same file.
        let rows = read_epoch_rows(&out).unwrap();
        let report = build_report(&rows, &[]);

        assert_eq!(
            report.totals.txs, summary.txs,
            "the whole scan -> epochs -> report pipeline must conserve the tx count"
        );
        assert_eq!(report.window.start_height, start);
        assert_eq!(report.window.end_height, mixed_height);

        assert!(!report.axis_summaries.is_empty());
        let nsequence = report
            .axis_summaries
            .get("nsequence")
            .expect("nsequence is a CORE axis and must be present");
        let total_share: f64 = nsequence.values.iter().map(|v| v.share).sum();
        assert!(
            (total_share - 1.0).abs() < 1e-9,
            "per-value shares on an axis must sum to ~1.0, got {total_share}"
        );
        let total_count: u64 = nsequence.values.iter().map(|v| v.count).sum();
        assert_eq!(
            total_count, summary.txs,
            "every classified tx must land in exactly one nsequence value"
        );
        assert_eq!(
            nsequence
                .values
                .iter()
                .find(|v| v.value == "first 0x01 rest 0xffffffff")
                .map(|v| v.count),
            Some(9),
            "9 of the 11 fixture txs are Cake-shaped"
        );
        assert_eq!(
            nsequence
                .values
                .iter()
                .find(|v| v.value == "Max")
                .map(|v| v.count),
            Some(1)
        );
        assert_eq!(
            nsequence
                .values
                .iter()
                .find(|v| v.value == "Rbf")
                .map(|v| v.count),
            Some(1)
        );

        let nseq_cond = report
            .conditional_anonymity
            .get("nsequence")
            .expect("nsequence is a CORE axis and must have a conditional_anonymity entry");
        assert!(!nseq_cond.is_empty());
        let cond_total: u64 = nseq_cond
            .values()
            .map(|c| c.buckets.values().sum::<u64>())
            .sum();
        assert_eq!(
            cond_total, summary.txs,
            "conditional_anonymity's bucket counts must account for every classified tx exactly once"
        );
        for cond in nseq_cond.values() {
            assert!((0.0..=1.0).contains(&cond.share_lt10));
            assert!((0.0..=1.0).contains(&cond.share_lt100));
            assert!(
                cond.share_lt10 <= cond.share_lt100,
                "a set below 10 is also below 100"
            );
        }

        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_dir_all(&report_dir);
    }
}
