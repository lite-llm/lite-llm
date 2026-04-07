#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TierId(pub u16);

impl TierId {
    pub const fn new(value: u16) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u16 {
        self.0
    }
}

impl From<u16> for TierId {
    fn from(value: u16) -> Self {
        Self(value)
    }
}

impl From<TierId> for u16 {
    fn from(value: TierId) -> Self {
        value.0
    }
}

impl From<lite_llm_runtime::TierId> for TierId {
    fn from(value: lite_llm_runtime::TierId) -> Self {
        Self(value.0)
    }
}

impl From<TierId> for lite_llm_runtime::TierId {
    fn from(value: TierId) -> Self {
        lite_llm_runtime::TierId::new(value.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpertKey {
    pub tier: TierId,
    pub group: u32,
    pub expert: u32,
}

impl ExpertKey {
    pub const fn new(tier: TierId, group: u32, expert: u32) -> Self {
        Self {
            tier,
            group,
            expert,
        }
    }

    pub fn encode(self) -> String {
        format!("{}:{}:{}", self.tier.0, self.group, self.expert)
    }

    pub fn parse(value: &str) -> Option<Self> {
        let parts: Vec<&str> = value.split(':').collect();
        if parts.len() != 3 {
            return None;
        }

        Some(Self {
            tier: TierId(parts[0].parse::<u16>().ok()?),
            group: parts[1].parse::<u32>().ok()?,
            expert: parts[2].parse::<u32>().ok()?,
        })
    }
}

impl From<lite_llm_runtime::ExpertKey> for ExpertKey {
    fn from(value: lite_llm_runtime::ExpertKey) -> Self {
        Self {
            tier: value.tier.into(),
            group: value.group,
            expert: value.expert,
        }
    }
}

impl From<ExpertKey> for lite_llm_runtime::ExpertKey {
    fn from(value: ExpertKey) -> Self {
        Self {
            tier: value.tier.into(),
            group: value.group,
            expert: value.expert,
        }
    }
}

impl From<lite_llm_storage::ExpertKey> for ExpertKey {
    fn from(value: lite_llm_storage::ExpertKey) -> Self {
        Self {
            tier: TierId(value.tier),
            group: value.group,
            expert: value.expert,
        }
    }
}

impl From<ExpertKey> for lite_llm_storage::ExpertKey {
    fn from(value: ExpertKey) -> Self {
        Self {
            tier: value.tier.0,
            group: value.group,
            expert: value.expert,
        }
    }
}

impl From<lite_llm_training::ExpertKey> for ExpertKey {
    fn from(value: lite_llm_training::ExpertKey) -> Self {
        Self {
            tier: TierId(value.tier),
            group: value.group,
            expert: value.expert,
        }
    }
}

impl From<ExpertKey> for lite_llm_training::ExpertKey {
    fn from(value: ExpertKey) -> Self {
        Self {
            tier: value.tier.0,
            group: value.group,
            expert: value.expert,
        }
    }
}

impl From<lite_llm_inference::ExpertKey> for ExpertKey {
    fn from(value: lite_llm_inference::ExpertKey) -> Self {
        Self {
            tier: TierId(value.tier),
            group: value.group,
            expert: value.expert,
        }
    }
}

impl From<ExpertKey> for lite_llm_inference::ExpertKey {
    fn from(value: ExpertKey) -> Self {
        Self {
            tier: value.tier.0,
            group: value.group,
            expert: value.expert,
        }
    }
}

impl From<lite_llm_distributed::ExpertAddress> for ExpertKey {
    fn from(value: lite_llm_distributed::ExpertAddress) -> Self {
        Self {
            tier: TierId(value.tier),
            group: value.group,
            expert: value.expert,
        }
    }
}

impl From<ExpertKey> for lite_llm_distributed::ExpertAddress {
    fn from(value: ExpertKey) -> Self {
        Self {
            tier: value.tier.0,
            group: value.group,
            expert: value.expert,
        }
    }
}

pub fn runtime_tierset(tiers: &[TierId], cumulative: bool) -> lite_llm_runtime::TierSet {
    let values = tiers
        .iter()
        .copied()
        .map(lite_llm_runtime::TierId::from)
        .collect::<Vec<lite_llm_runtime::TierId>>();
    lite_llm_runtime::TierSet::new(values, cumulative)
}

pub fn inference_tierset(tiers: &[TierId], cumulative: bool) -> lite_llm_inference::TierSet {
    lite_llm_inference::TierSet::new(tiers.iter().map(|tier| tier.0).collect(), cumulative)
}

#[cfg(test)]
mod tests {
    use super::{ExpertKey, TierId};

    #[test]
    fn expert_key_roundtrip_is_stable() {
        let key = ExpertKey::new(TierId(10), 7, 3);
        assert_eq!(ExpertKey::parse(&key.encode()), Some(key));
    }

    #[test]
    fn cross_crate_expert_key_conversions_are_lossless() {
        let canonical = ExpertKey::new(TierId(2), 1, 9);

        let runtime: lite_llm_runtime::ExpertKey = canonical.into();
        let runtime_back: ExpertKey = runtime.into();
        assert_eq!(runtime_back, canonical);

        let storage: lite_llm_storage::ExpertKey = canonical.into();
        let storage_back: ExpertKey = storage.into();
        assert_eq!(storage_back, canonical);

        let training: lite_llm_training::ExpertKey = canonical.into();
        let training_back: ExpertKey = training.into();
        assert_eq!(training_back, canonical);

        let inference: lite_llm_inference::ExpertKey = canonical.into();
        let inference_back: ExpertKey = inference.into();
        assert_eq!(inference_back, canonical);
    }
}
