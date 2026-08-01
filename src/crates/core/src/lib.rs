// `warn`, not `deny`: these should surface in normal builds so new code doesn't
// silently regress, but should never fail CI outright over a pre-existing gap in an
// unrelated corner of the crate. `missing_docs` is deliberately left out — this crate
// has plenty of existing `pub` items (e.g. most `ScanConfig`/`Report` fields) with no
// doc comment, and turning it on here would produce a wall of unrelated warnings
// rather than surfacing anything about this change.
#![warn(missing_debug_implementations, rust_2018_idioms)]

pub mod aux;
pub mod change;
pub mod engine;
pub mod epoch;
mod features;
pub mod report;
pub mod template;
pub mod vector;

pub use aux::{AuxFlags, aux_flags};
pub use engine::{ScanConfig, ScanError, ScanSummary, resume_point, scan};
pub use epoch::{EpochAccumulator, EpochRow};
pub use features::{COLUMN_NAMES, FeatureRow, TxShape, feature_row, tx_shape};
pub use report::{Report, ReportError, build_report, epoch_rows, read_epoch_rows, write_report};
pub use template::{Confidence, TemplateError, WalletTemplate, load_templates, parse_templates};
pub use vector::{FingerprintVector, InputTypeClass, OrderClass, Tri, classify_tx};
