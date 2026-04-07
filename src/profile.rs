#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartupProfile {
    Development,
    Deterministic,
    Throughput,
    Recovery,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StartupTuning {
    pub router_quantization_scale: f32,
    pub cache_capacity_bytes: u64,
    pub selector_base_latency_ms: f32,
    pub enable_pipeline_compression: bool,
    pub strict_security: bool,
    pub heartbeat_timeout_steps: u64,
}

impl StartupProfile {
    pub fn tuning(self) -> StartupTuning {
        match self {
            Self::Development => StartupTuning {
                router_quantization_scale: 1_000.0,
                cache_capacity_bytes: 8 * 1024 * 1024,
                selector_base_latency_ms: 1.0,
                enable_pipeline_compression: true,
                strict_security: false,
                heartbeat_timeout_steps: 30,
            },
            Self::Deterministic => StartupTuning {
                router_quantization_scale: 1_000_000.0,
                cache_capacity_bytes: 64 * 1024 * 1024,
                selector_base_latency_ms: 2.0,
                enable_pipeline_compression: false,
                strict_security: true,
                heartbeat_timeout_steps: 20,
            },
            Self::Throughput => StartupTuning {
                router_quantization_scale: 100_000.0,
                cache_capacity_bytes: 128 * 1024 * 1024,
                selector_base_latency_ms: 3.0,
                enable_pipeline_compression: true,
                strict_security: true,
                heartbeat_timeout_steps: 40,
            },
            Self::Recovery => StartupTuning {
                router_quantization_scale: 1_000_000.0,
                cache_capacity_bytes: 32 * 1024 * 1024,
                selector_base_latency_ms: 2.5,
                enable_pipeline_compression: false,
                strict_security: true,
                heartbeat_timeout_steps: 10,
            },
        }
    }
}

impl Default for StartupProfile {
    fn default() -> Self {
        Self::Deterministic
    }
}
