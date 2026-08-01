use lumen_primitives::{HasScriptPubkey, OutputType};

/// Classify an output by its scriptPubKey.
pub fn output_type(output: &(impl HasScriptPubkey + ?Sized)) -> OutputType {
    output.output_type()
}

/// Bundled trait for output-level fingerprints.
pub trait HasOutputFingerprints: HasScriptPubkey {
    // TODO: add output level wallet fingerprints beyond scriptPubKey classification.
    fn output_type(&self) -> OutputType {
        HasScriptPubkey::output_type(self)
    }
}

impl<T: HasScriptPubkey> HasOutputFingerprints for T {}
