use std::mem;

use crate::error::{LiteLlmError, LiteLlmResult};
use crate::types::{ExpertKey, TierId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractReport {
    pub tier_id_width_bits: usize,
    pub tier_id_compatible: bool,
    pub expert_key_codec_compatible: bool,
    pub hash_compatible: bool,
}

impl ContractReport {
    pub fn is_compatible(&self) -> bool {
        self.tier_id_compatible && self.expert_key_codec_compatible && self.hash_compatible
    }
}

pub fn verify_shared_contracts() -> LiteLlmResult<ContractReport> {
    let runtime_bits = mem::size_of::<lite_llm_runtime::TierId>() * 8;
    let storage_bits = mem::size_of::<lite_llm_storage::TierId>() * 8;
    let training_bits = mem::size_of::<lite_llm_training::TierId>() * 8;
    let inference_bits = mem::size_of::<lite_llm_inference::TierId>() * 8;
    let security_bits = mem::size_of::<lite_llm_security::TierId>() * 8;

    let tier_id_compatible = runtime_bits == 16
        && storage_bits == 16
        && training_bits == 16
        && inference_bits == 16
        && security_bits == 16;

    let canonical_key = ExpertKey::new(TierId(7), 3, 11);
    let canonical_encoded = canonical_key.encode();

    let storage_key: lite_llm_storage::ExpertKey = canonical_key.into();
    let training_key: lite_llm_training::ExpertKey = canonical_key.into();
    let inference_key: lite_llm_inference::ExpertKey = canonical_key.into();

    let storage_roundtrip = lite_llm_storage::ExpertKey::parse(&storage_key.encode())
        .map(ExpertKey::from)
        .map(|key| key.encode());
    let training_roundtrip = lite_llm_training::ExpertKey::parse(&training_key.encode())
        .map(ExpertKey::from)
        .map(|key| key.encode());
    let inference_roundtrip = lite_llm_inference::ExpertKey::parse(&inference_key.encode())
        .map(ExpertKey::from)
        .map(|key| key.encode());

    let expert_key_codec_compatible = storage_roundtrip.as_deref()
        == Some(canonical_encoded.as_str())
        && training_roundtrip.as_deref() == Some(canonical_encoded.as_str())
        && inference_roundtrip.as_deref() == Some(canonical_encoded.as_str());

    let payload = b"lite-llm-contract-drift-check";
    let storage_hash = lite_llm_storage::fnv64_hex(payload);
    let training_hash = lite_llm_training::fnv64_hex(payload);
    let inference_hash = lite_llm_inference::fnv64_hex(payload);
    let security_hash = lite_llm_security::fnv64_hex(payload);
    let hash_compatible = storage_hash == training_hash
        && training_hash == inference_hash
        && inference_hash == security_hash;

    let report = ContractReport {
        tier_id_width_bits: runtime_bits,
        tier_id_compatible,
        expert_key_codec_compatible,
        hash_compatible,
    };

    if !report.is_compatible() {
        if !tier_id_compatible {
            return Err(LiteLlmError::ContractDrift(
                "TierId width mismatch across crates",
            ));
        }
        if !expert_key_codec_compatible {
            return Err(LiteLlmError::ContractDrift(
                "ExpertKey encode/parse mismatch across crates",
            ));
        }
        return Err(LiteLlmError::ContractDrift(
            "fnv64 hash implementation mismatch across crates",
        ));
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::verify_shared_contracts;

    #[test]
    fn contracts_are_compatible() {
        let report = verify_shared_contracts().expect("contracts should match");
        assert!(report.is_compatible());
    }
}
