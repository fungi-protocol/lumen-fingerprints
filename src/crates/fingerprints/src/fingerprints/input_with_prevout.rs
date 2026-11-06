use bitcoin::Script;
use bitcoin::script::Instruction;
use lumen_primitives::{
    HasScriptPubkey, HasWitness, OutputType, traits::abstract_types::HasScriptSig,
};

pub fn input_type(prevout: &impl HasScriptPubkey) -> OutputType {
    prevout.output_type()
}

/// P2PK embeds the key in the prevout; P2PKH pushes it in the scriptSig.
pub fn has_uncompressed_pubkey(
    input: &(impl HasWitness + HasScriptSig + ?Sized),
    prevout: &(impl HasScriptPubkey + ?Sized),
) -> bool {
    let spk_bytes = prevout.script_pubkey_bytes();
    let script = Script::from_bytes(&spk_bytes);

    if let Some(pk) = script.p2pk_public_key()
        && !pk.compressed
    {
        return true;
    }

    if script.is_p2pkh() {
        let sig_bytes = input.script_sig_bytes();
        let sig_script = Script::from_bytes(&sig_bytes);
        return sig_script
            .instructions()
            .filter_map(|instr| match instr {
                Ok(Instruction::PushBytes(bytes)) => Some(bytes.as_bytes().to_vec()),
                _ => None,
            })
            .any(|data| data.len() == 65 && data[0] == 0x04);
    }

    false
}

/// Key-spend witness is `[sig]` or `[sig, annex]` (annex starts 0x50); a 65-byte sig
/// carries an explicit sighash byte.
pub fn taproot_keyspend_non_default_sighash(
    input: &(impl HasWitness + ?Sized),
    prevout: &(impl HasScriptPubkey + ?Sized),
) -> bool {
    let spk_bytes = prevout.script_pubkey_bytes();
    let script = Script::from_bytes(&spk_bytes);

    if !script.is_p2tr() {
        return false;
    }

    let items = input.witness_items();

    let sig = match items.len() {
        1 => &items[0],
        2 if items[1].first() == Some(&0x50) => &items[0],
        _ => return false,
    };

    sig.len() == 65
}

pub trait HasInputWithPrevoutFingerprints: HasWitness + HasScriptSig + HasScriptPubkey {
    fn has_uncompressed_pubkey(&self, prevout: &impl HasScriptPubkey) -> bool {
        has_uncompressed_pubkey(self, prevout)
    }
    fn taproot_keyspend_non_default_sighash(&self, prevout: &impl HasScriptPubkey) -> bool {
        taproot_keyspend_non_default_sighash(self, prevout)
    }
}
