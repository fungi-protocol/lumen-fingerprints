// `warn`, not `deny`: surface regressions without failing CI on pre-existing gaps.
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
