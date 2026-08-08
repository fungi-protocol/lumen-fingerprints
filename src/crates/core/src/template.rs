use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::vector::{CORE_AXES, EXTENDED_AXES, FingerprintVector, HEURISTIC_AXES};

/// Every axis a template may constrain. Anything else in the TOML is an error, so a
/// typo in a wallet's entry fails loudly instead of silently matching everything.
/// Derived from the vector's own axis sets, so adding an axis to `FingerprintVector`
/// automatically makes it constrainable in a template — they cannot drift apart.
///
/// Includes `HEURISTIC_AXES`: `change_position`/`change_type` are excluded from both
/// joint-vector keys (see that constant's doc comment), but they are still real,
/// still-computed axes reachable through `axis_value`, so a template is still allowed
/// to constrain on them — only the sparsity-measuring joint key excludes them.
fn is_known_axis(axis: &str) -> bool {
    CORE_AXES.contains(&axis) || EXTENDED_AXES.contains(&axis) || HEURISTIC_AXES.contains(&axis)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confidence {
    /// Decoded from a real on-chain transaction.
    ChainProven,
    /// Read from the wallet's source, not yet observed on-chain.
    CodePredicted,
}

#[derive(Debug)]
pub enum TemplateError {
    Io(std::io::Error),
    Parse(String),
    UnknownAxis(String),
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "reading templates: {e}"),
            Self::Parse(msg) => write!(f, "parsing templates: {msg}"),
            Self::UnknownAxis(axis) => write!(f, "unknown axis in template: {axis}"),
        }
    }
}

impl std::error::Error for TemplateError {}

/// A wallet's expected fingerprint values over one version interval (an "era").
/// Axes absent from `axes` are wildcards. Multiple entries may share a `name`;
/// each is an independent template. Version strings are opaque labels — nothing
/// in code orders or compares them; interval semantics are `[from, until)`.
#[derive(Debug, Clone)]
pub struct WalletTemplate {
    pub name: String,
    pub confidence: Confidence,
    /// Which codebase the version bounds refer to (e.g. "cake_wallet",
    /// "bitcoin_base"). Required whenever either bound is present.
    pub software: Option<String>,
    /// First version (inclusive) producing this era's fingerprint.
    pub from_version: Option<String>,
    /// First version (exclusive) that no longer produces it.
    pub until_version: Option<String>,
    pub axes: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct RawFile {
    wallet: Vec<BTreeMap<String, String>>,
}

pub fn load_templates(path: &Path) -> Result<Vec<WalletTemplate>, TemplateError> {
    let text = std::fs::read_to_string(path).map_err(TemplateError::Io)?;
    parse_templates(&text)
}

pub fn parse_templates(text: &str) -> Result<Vec<WalletTemplate>, TemplateError> {
    let raw: RawFile = toml::from_str(text).map_err(|e| TemplateError::Parse(e.to_string()))?;

    let templates: Vec<WalletTemplate> = raw
        .wallet
        .into_iter()
        .map(|mut entry| {
            let name = entry
                .remove("name")
                .ok_or_else(|| TemplateError::Parse("wallet entry without a name".into()))?;
            let confidence = match entry.remove("confidence").as_deref() {
                Some("chain-proven") => Confidence::ChainProven,
                Some("code-predicted") | None => Confidence::CodePredicted,
                Some(other) => {
                    return Err(TemplateError::Parse(format!("unknown confidence: {other}")));
                }
            };
            let software = entry.remove("software");
            let from_version = entry.remove("from_version");
            let until_version = entry.remove("until_version");
            if software.is_none() && (from_version.is_some() || until_version.is_some()) {
                return Err(TemplateError::Parse(format!(
                    "wallet {name}: from_version/until_version require software"
                )));
            }
            for axis in entry.keys() {
                if !is_known_axis(axis) {
                    return Err(TemplateError::UnknownAxis(axis.clone()));
                }
            }
            Ok(WalletTemplate {
                name,
                confidence,
                software,
                from_version,
                until_version,
                axes: entry,
            })
        })
        .collect::<Result<_, _>>()?;

    let mut seen = std::collections::BTreeSet::new();
    for t in &templates {
        if !seen.insert(t.era_label()) {
            return Err(TemplateError::Parse(format!(
                "duplicate era: {}",
                t.era_label()
            )));
        }
    }
    Ok(templates)
}

impl WalletTemplate {
    /// Display / series key for this era. Unversioned templates keep their bare
    /// `name`, so every existing `template_series` consumer is unaffected until a
    /// wallet actually grows a second era.
    pub fn era_label(&self) -> String {
        let Some(software) = &self.software else {
            return self.name.clone();
        };
        match (&self.from_version, &self.until_version) {
            (Some(from), Some(until)) => format!("{} ({software} {from}–{until})", self.name),
            (Some(from), None) => format!("{} ({software} ≥{from})", self.name),
            (None, Some(until)) => format!("{} ({software} <{until})", self.name),
            (None, None) => format!("{} ({software})", self.name),
        }
    }

    /// The one place the no-signal / consistent-with semantics live. Both public entry
    /// points (`matches` and `matches_axes`) delegate here rather than each
    /// re-implementing the rule, so the two can never quietly diverge — this codebase
    /// has already been bitten twice by exactly that class of duplication bug.
    ///
    /// `lookup` maps an axis name to its observed value, however that value happens to
    /// be stored (a `FingerprintVector`'s typed fields, or a parsed `axis=value` map).
    /// An axis whose observed value is `Indeterminate` / `Unknown` / `Na` carries no
    /// signal and therefore cannot contradict anything — this is Ishaana's
    /// "consistent-with" semantics, and it is why anonymity sets measured here are
    /// upper bounds on distinguishability.
    ///
    /// `lookup` returns `Cow` rather than `String` so that neither caller needs to
    /// allocate just to answer a lookup: `matches` forwards `FingerprintVector::axis_value`'s
    /// own `Cow` straight through, and `matches_axes` borrows out of its `BTreeMap<String,
    /// String>` instead of cloning.
    fn matches_by<'a, F: Fn(&str) -> Option<Cow<'a, str>>>(&self, lookup: F) -> bool {
        self.axes.iter().all(|(axis, expected)| {
            let Some(actual) = lookup(axis) else {
                return true;
            };
            // No signal on this axis: cannot contradict.
            if actual.as_ref() == "Indeterminate"
                || actual.as_ref() == "Unknown"
                || actual.as_ref() == "Na"
            {
                return true;
            }
            actual.as_ref() == expected.as_str()
        })
    }

    /// True when no constrained axis contradicts the observed vector. Used at scan
    /// time, where a `FingerprintVector` is in hand.
    pub fn matches(&self, vector: &FingerprintVector) -> bool {
        self.matches_by(|axis| vector.axis_value(axis))
    }

    /// True when no constrained axis contradicts an axis map parsed back out of a
    /// `vectors_extended` key (see `vector::parse_key`). Used at report time, where
    /// only the joint-vector histogram survives on disk — no `FingerprintVector` exists
    /// to call `matches` on.
    pub fn matches_axes(&self, axes: &BTreeMap<String, String>) -> bool {
        self.matches_by(|axis| axes.get(axis).map(|s| Cow::Borrowed(s.as_str())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::classify_tx;
    use crate::vector::tests_support::{EPOCH_TEST_HEIGHT, cake_like_block};

    fn cake_vector() -> FingerprintVector {
        let block = cake_like_block(EPOCH_TEST_HEIGHT);
        let tx = block.block.txdata.last().unwrap().clone();
        classify_tx(&tx, &block).unwrap()
    }

    #[test]
    fn loads_the_shipped_templates() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("templates/wallets.toml");
        let templates = load_templates(&path).expect("templates parse");
        assert!(templates.iter().any(|t| t.name == "Cake Wallet"));
        assert!(templates.iter().any(|t| t.name == "Bitcoin Core"));

        let cake = templates.iter().find(|t| t.name == "Cake Wallet").unwrap();
        assert_eq!(cake.confidence, Confidence::ChainProven);
    }

    #[test]
    fn cake_template_matches_the_cake_shape_and_core_does_not() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("templates/wallets.toml");
        let templates = load_templates(&path).unwrap();
        let v = cake_vector();

        let cake = templates.iter().find(|t| t.name == "Cake Wallet").unwrap();
        let core = templates.iter().find(|t| t.name == "Bitcoin Core").unwrap();

        assert!(
            cake.matches(&v),
            "the Cake template must match the Cake shape"
        );
        assert!(
            !core.matches(&v),
            "Core diverges on nsequence and nlocktime"
        );
    }

    #[test]
    fn omitted_axis_is_a_wildcard() {
        let template = WalletTemplate {
            name: "anything".into(),
            confidence: Confidence::CodePredicted,
            software: None,
            from_version: None,
            until_version: None,
            axes: BTreeMap::new(),
        };
        assert!(template.matches(&cake_vector()));
    }

    #[test]
    fn indeterminate_axis_matches_any_expectation() {
        let mut axes = BTreeMap::new();
        // The Cake fixture is pure-ECDSA, so uncompressed_pubkey resolves to No, not
        // Indeterminate. input_order on a 1-input tx is what we want to exercise here.
        axes.insert("input_order".to_string(), "Other".to_string());
        let template = WalletTemplate {
            name: "t".into(),
            confidence: Confidence::CodePredicted,
            software: None,
            from_version: None,
            until_version: None,
            axes,
        };

        let block = cake_like_block(EPOCH_TEST_HEIGHT);
        let mut tx = block.block.txdata.last().unwrap().clone();
        tx.input.truncate(1);
        let v = classify_tx(&tx, &block).unwrap();

        assert_eq!(v.input_order, crate::vector::OrderClass::Indeterminate);
        assert!(
            template.matches(&v),
            "an indeterminate axis carries no signal, so it cannot contradict a template"
        );
    }

    #[test]
    fn unknown_axis_name_is_an_error() {
        let toml = r#"
            [[wallet]]
            name = "bogus"
            confidence = "code-predicted"
            not_an_axis = "x"
        "#;
        let err = parse_templates(toml).unwrap_err();
        assert!(
            format!("{err}").contains("not_an_axis"),
            "error names the bad axis"
        );
    }

    #[test]
    fn no_signal_feerate_bucket_cannot_contradict_template() {
        // Build a template that constrains feerate_bucket to a specific value.
        // A transaction with no-signal feerate (cannot compute the value) should
        // still match, because no-signal axes cannot contradict templates.
        let mut axes = BTreeMap::new();
        axes.insert("feerate_bucket".to_string(), "5".to_string());
        let template = WalletTemplate {
            name: "test-feerate".into(),
            confidence: Confidence::CodePredicted,
            software: None,
            from_version: None,
            until_version: None,
            axes,
        };

        // Construct a vector with feerate_bucket set to "Unknown" (the no-signal sentinel).
        // This simulates a transaction where the feerate cannot be computed.
        let vector = FingerprintVector {
            version: 2,
            nsequence: lumen_fingerprints_lib::NSequenceType::MixedOther,
            nlocktime: lumen_fingerprints_lib::NLockTimeType::Zero,
            locktime_offset: lumen_fingerprints_lib::LocktimeOffsetType::NotApplicable,
            input_order: crate::vector::OrderClass::Other,
            output_order: crate::vector::OrderClass::Other,
            output_structure: lumen_fingerprints_lib::OutputStructureType::Multi,
            input_types: crate::vector::InputTypeClass::Unknown,
            output_types: vec![],
            low_r: crate::vector::Tri::No,
            sighash: lumen_fingerprints_lib::SighashType::Mixed,
            uncompressed_pubkey: crate::vector::Tri::No,
            op_return: false,
            input_subtype: crate::vector::InputSubtype::Mixed,
            low_s: crate::vector::Tri::No,
            ecdsa_sigs: crate::vector::EcdsaSigCount::None,
            input_age: crate::vector::AgeClass::Older,
            feerate_bucket: "Unknown".to_string(),
            round_feerate: false,
            change_position: crate::vector::ChangePosition::Indeterminate,
            change_type: None,
        };

        assert!(
            template.matches(&vector),
            "a no-signal feerate_bucket (Unknown) cannot contradict a template \
             even if the template constrains feerate_bucket to a specific value"
        );
    }

    #[test]
    fn matches_axes_agrees_with_matches_on_the_same_vector() {
        // The whole point of `matches_by` is that `matches` and `matches_axes` cannot
        // diverge. Exercise every shipped template against the same vector through
        // both entry points and require identical verdicts.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("templates/wallets.toml");
        let templates = load_templates(&path).unwrap();
        let v = cake_vector();

        let all_axes: Vec<&str> = CORE_AXES
            .iter()
            .chain(EXTENDED_AXES.iter())
            .chain(HEURISTIC_AXES.iter())
            .copied()
            .collect();
        let axes = crate::vector::parse_key(&v.key_for(&all_axes));

        for template in &templates {
            assert_eq!(
                template.matches(&v),
                template.matches_axes(&axes),
                "template {} disagrees between matches() and matches_axes()",
                template.name
            );
        }
    }

    #[test]
    fn matches_axes_treats_indeterminate_as_no_signal() {
        let mut axes = BTreeMap::new();
        axes.insert("feerate_bucket".to_string(), "5".to_string());
        let template = WalletTemplate {
            name: "t".into(),
            confidence: Confidence::CodePredicted,
            software: None,
            from_version: None,
            until_version: None,
            axes,
        };

        let mut observed = BTreeMap::new();
        observed.insert("feerate_bucket".to_string(), "Unknown".to_string());
        assert!(
            template.matches_axes(&observed),
            "a no-signal observed value cannot contradict a template via matches_axes either"
        );
    }

    #[test]
    fn matches_axes_treats_a_missing_axis_as_a_wildcard() {
        let mut axes = BTreeMap::new();
        axes.insert("feerate_bucket".to_string(), "5".to_string());
        let template = WalletTemplate {
            name: "t".into(),
            confidence: Confidence::CodePredicted,
            software: None,
            from_version: None,
            until_version: None,
            axes,
        };

        let observed: BTreeMap<String, String> = BTreeMap::new();
        assert!(template.matches_axes(&observed));
    }

    #[test]
    fn matches_axes_rejects_a_genuine_contradiction() {
        let mut axes = BTreeMap::new();
        axes.insert("nsequence".to_string(), "CakeGroupC".to_string());
        let template = WalletTemplate {
            name: "t".into(),
            confidence: Confidence::CodePredicted,
            software: None,
            from_version: None,
            until_version: None,
            axes,
        };

        let mut observed = BTreeMap::new();
        observed.insert("nsequence".to_string(), "Rbf".to_string());
        assert!(!template.matches_axes(&observed));
    }

    #[test]
    fn parses_a_versioned_entry() {
        let toml = r#"
            [[wallet]]
            name = "Cake Wallet"
            software = "cake_wallet"
            from_version = "4.28.0"
            confidence = "code-predicted"
            nsequence = "CakeGroupC"
        "#;
        let templates = parse_templates(toml).unwrap();
        assert_eq!(templates.len(), 1);
        let t = &templates[0];
        assert_eq!(t.software.as_deref(), Some("cake_wallet"));
        assert_eq!(t.from_version.as_deref(), Some("4.28.0"));
        assert_eq!(t.until_version, None);
        assert_eq!(
            t.axes.get("nsequence").map(String::as_str),
            Some("CakeGroupC")
        );
    }

    #[test]
    fn version_bound_without_software_is_an_error() {
        let toml = r#"
            [[wallet]]
            name = "bogus"
            from_version = "1.0.0"
        "#;
        let err = parse_templates(toml).unwrap_err();
        assert!(
            format!("{err}").contains("require software"),
            "error must say version bounds require software"
        );
    }

    #[test]
    fn era_label_covers_all_range_shapes() {
        let mut t = WalletTemplate {
            name: "W".into(),
            confidence: Confidence::CodePredicted,
            software: None,
            from_version: None,
            until_version: None,
            axes: BTreeMap::new(),
        };
        assert_eq!(t.era_label(), "W");
        t.software = Some("lib".into());
        assert_eq!(t.era_label(), "W (lib)");
        t.from_version = Some("2.0".into());
        assert_eq!(t.era_label(), "W (lib ≥2.0)");
        t.until_version = Some("3.0".into());
        assert_eq!(t.era_label(), "W (lib 2.0–3.0)");
        t.from_version = None;
        assert_eq!(t.era_label(), "W (lib <3.0)");
    }

    #[test]
    fn two_eras_of_one_wallet_both_parse_with_distinct_labels() {
        let toml = r#"
            [[wallet]]
            name = "Cake Wallet"
            software = "cake_wallet"
            until_version = "4.28.0"
            input_order = "Bip69"

            [[wallet]]
            name = "Cake Wallet"
            software = "cake_wallet"
            from_version = "4.28.0"
            input_order = "Other"
        "#;
        let templates = parse_templates(toml).unwrap();
        assert_eq!(templates.len(), 2);
        assert_ne!(templates[0].era_label(), templates[1].era_label());
    }

    #[test]
    fn duplicate_era_labels_are_an_error() {
        let toml = r#"
            [[wallet]]
            name = "W"
            nsequence = "Rbf"

            [[wallet]]
            name = "W"
            nsequence = "Max"
        "#;
        let err = parse_templates(toml).unwrap_err();
        assert!(
            format!("{err}").contains("duplicate era"),
            "two entries with identical era labels would collide as template_series keys"
        );
    }
}
