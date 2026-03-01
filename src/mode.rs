use std::path::PathBuf;

use crate::profile::StartupProfile;
use crate::types::TierId;

#[derive(Debug, Clone)]
pub struct BootstrapConfig {
    pub profile: StartupProfile,
    pub runtime: lite_llm_runtime::RuntimeOptions,
    pub manifest_text: String,
    pub initial_active_tiers: Option<Vec<TierId>>,
    pub distributed: lite_llm_distributed::ParallelismConfig,
    pub snapshot_root: PathBuf,
    pub training_checkpoint_root: PathBuf,
    pub node_id: String,
    pub signer_id: String,
    pub signing_secret: String,
}

#[derive(Debug, Clone)]
pub struct TrainingEntrypoint {
    pub model_identifier: lite_llm_training::ModelIdentifier,
    pub checkpoint_id: String,
    pub world_size: u32,
    pub optimizer_name: String,
    pub curriculum_plan: lite_llm_training::TierExpansionPlan,
    pub accumulation: lite_llm_training::AccumulationConfig,
}

#[derive(Debug, Clone)]
pub struct TrainingHandle {
    pub scheduler: lite_llm_training::CurriculumScheduler,
    pub accumulation: lite_llm_training::AccumulationState,
    pub checkpoint_repository: lite_llm_training::DistributedCheckpointRepository,
    pub replay: lite_llm_training::ReplayContext,
}

#[derive(Debug, Clone)]
pub struct InferenceEntrypoint {
    pub fixed_tiers: lite_llm_inference::FixedModeTierSets,
    pub tier_profiles: Vec<lite_llm_inference::TierProfile>,
    pub selection_request: lite_llm_inference::TierSetSelectionRequest,
    pub kv_cache: lite_llm_inference::KvCacheConfig,
}

#[derive(Debug, Clone)]
pub struct InferenceHandle {
    pub selection: lite_llm_inference::TierSetSelectionResult,
    pub selector: lite_llm_inference::TierSetSelector,
    pub pipeline: lite_llm_inference::DeterministicInferencePipeline,
    pub streaming: lite_llm_inference::StreamingRuntime,
}

#[derive(Debug, Clone)]
pub struct ReplayEntrypoint {
    pub expected: lite_llm_training::ReplayContext,
    pub observed: lite_llm_training::ReplayContext,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayHandle {
    pub replay_hash: String,
    pub event_count: usize,
}

#[derive(Debug, Clone)]
pub struct RecoveryEntrypoint {
    pub manifest_text: String,
    pub failure_event: lite_llm_distributed::FailureEvent,
    pub resume_active: bool,
    pub snapshot_id: Option<String>,
    pub selected_tiers: Option<Vec<TierId>>,
}

#[derive(Debug, Clone)]
pub struct RecoveryHandle {
    pub action: lite_llm_distributed::RecoveryAction,
    pub runtime_status: lite_llm_runtime::RuntimeStatus,
    pub restored_snapshot: Option<lite_llm_storage::RestoredSnapshot>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveMode {
    Training,
    Inference,
    Replay,
    Recovery,
}
