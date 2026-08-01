use bitcoin::Script;
use bitcoin::ecdsa::Signature as EcdsaSignature;
use bitcoin::script::Instruction;
use lumen_primitives::OutputType;

/// Classify a scriptPubKey by type from raw bytes.
pub fn classify_script_pubkey(spk: &[u8]) -> OutputType {
    lumen_primitives::classify_script_pubkey(spk)
}

/// Extract ECDSA signature byte vectors from scriptSig bytes.
pub fn extract_signatures_from_script_sig(script_sig_bytes: &[u8]) -> Vec<Vec<u8>> {
    let script = Script::from_bytes(script_sig_bytes);
    script
        .instructions()
        .filter_map(|instr| match instr {
            Ok(Instruction::PushBytes(bytes)) => Some(bytes.as_bytes().to_vec()),
            _ => None,
        })
        .filter(|data| data.len() >= 9 && data[0] == 0x30)
        .collect()
}

/// Extract ECDSA signature byte vectors from witness items.
pub fn extract_signatures_from_witness(witness_items: &[Vec<u8>]) -> Vec<Vec<u8>> {
    witness_items
        .iter()
        .filter(|data| EcdsaSignature::from_slice(data).is_ok())
        .cloned()
        .collect()
}

/// Check if an ECDSA signature has low-R (R value's high bit is unset).
pub fn has_low_r_signature(sig: &EcdsaSignature) -> bool {
    sig.signature.serialize_compact()[0] < 0x80
}

// TODO the above needs tests.
