use lumen_primitives::{HasScriptPubkey, OutputType};

pub fn output_type(output: &(impl HasScriptPubkey + ?Sized)) -> OutputType {
    output.output_type()
}

pub trait HasOutputFingerprints: HasScriptPubkey {
    // TODO: add output level wallet fingerprints beyond scriptPubKey classification.
    fn output_type(&self) -> OutputType {
        HasScriptPubkey::output_type(self)
    }
}

impl<T: HasScriptPubkey> HasOutputFingerprints for T {}
