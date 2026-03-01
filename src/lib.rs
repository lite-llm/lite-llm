pub mod contracts;
pub mod error;
pub mod mode;
pub mod orchestrator;
pub mod profile;
pub mod spec_gate;
pub mod types;

pub use contracts::{verify_shared_contracts, ContractReport};
pub use error::{LiteLlmError, LiteLlmResult};
pub use mode::{
    ActiveMode, BootstrapConfig, InferenceEntrypoint, InferenceHandle, RecoveryEntrypoint,
    RecoveryHandle, ReplayEntrypoint, ReplayHandle, TrainingEntrypoint, TrainingHandle,
};
pub use orchestrator::{DistributedStack, LiteLlm, SecurityStack, StorageStack};
pub use profile::{StartupProfile, StartupTuning};
pub use spec_gate::{
    spec_compliance_matrix, SpecComplianceEntry, REQUIRED_SPEC_END, REQUIRED_SPEC_START,
};
pub use types::{inference_tierset, runtime_tierset, ExpertKey, TierId};

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::{
        BootstrapConfig, InferenceEntrypoint, LiteLlm, RecoveryEntrypoint, ReplayEntrypoint,
        StartupProfile, TierId, TrainingEntrypoint,
    };

    fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nanos}"))
    }

    fn sample_runtime_options(training_mode: bool) -> lite_llm_runtime::RuntimeOptions {
        lite_llm_runtime::RuntimeOptions {
            routing_seed: lite_llm_runtime::RoutingSeed::new(42),
            available_tiers: vec![
                lite_llm_runtime::TierConfig {
                    id: lite_llm_runtime::TierId::new(1),
                    groups: 2,
                    experts_per_group: 2,
                    placement: lite_llm_runtime::Placement::Hot,
                },
                lite_llm_runtime::TierConfig {
                    id: lite_llm_runtime::TierId::new(2),
                    groups: 2,
                    experts_per_group: 2,
                    placement: lite_llm_runtime::Placement::Warm,
                },
            ],
            expected_manifest_version: 1,
            training_mode,
        }
    }

    fn sample_manifest() -> String {
        "version=1\n\
         tiers=1,2\n\
         cumulative=false\n\
         base_checksum=abc123\n\
         router_checksum=def456\n\
         optimizer_checksum=xyz999\n\
         shard=base|aa11|1024\n\
         shard=exp|bb22|2048\n"
            .to_owned()
    }

    fn bootstrap_config(training_mode: bool) -> BootstrapConfig {
        BootstrapConfig {
            profile: StartupProfile::Deterministic,
            runtime: sample_runtime_options(training_mode),
            manifest_text: sample_manifest(),
            initial_active_tiers: Some(vec![TierId(1), TierId(2)]),
            distributed: lite_llm_distributed::ParallelismConfig {
                data_parallel: 1,
                tensor_parallel: 1,
                pipeline_parallel: 1,
                expert_parallel: 1,
            },
            snapshot_root: unique_temp_dir("lite-llm-snapshots"),
            training_checkpoint_root: unique_temp_dir("lite-llm-training-checkpoints"),
            node_id: "node-1".to_owned(),
            signer_id: "signer-1".to_owned(),
            signing_secret: "secret-1".to_owned(),
        }
    }

    #[test]
    fn bootstrap_wires_all_stacks_and_contracts() {
        let llm = LiteLlm::bootstrap(bootstrap_config(false)).expect("bootstrap should succeed");
        assert!(llm.contracts().is_compatible());
        assert_eq!(llm.distributed().parallelism.world_size(), 1);
        assert!(llm.storage().snapshots.root().exists());
    }

    #[test]
    fn inference_entrypoint_selects_tiers_and_sets_mode() {
        let mut llm =
            LiteLlm::bootstrap(bootstrap_config(false)).expect("bootstrap should succeed");

        let handle = llm
            .start_inference(InferenceEntrypoint {
                fixed_tiers: lite_llm_inference::FixedModeTierSets {
                    fast: lite_llm_inference::TierSet::new(vec![1], false),
                    balanced: lite_llm_inference::TierSet::new(vec![1, 2], false),
                    deep: lite_llm_inference::TierSet::new(vec![1, 2], false),
                    max: lite_llm_inference::TierSet::new(vec![1, 2], false),
                },
                tier_profiles: vec![
                    lite_llm_inference::TierProfile {
                        id: 1,
                        label: "hot".to_owned(),
                        capacity_value: 100,
                        latency_cost_ms: 1.0,
                        monetary_cost_units: 0.1,
                        available: true,
                    },
                    lite_llm_inference::TierProfile {
                        id: 2,
                        label: "warm".to_owned(),
                        capacity_value: 200,
                        latency_cost_ms: 2.0,
                        monetary_cost_units: 0.3,
                        available: true,
                    },
                ],
                selection_request: lite_llm_inference::TierSetSelectionRequest {
                    mode: lite_llm_inference::SelectionMode::Balanced,
                    explicit_tiers: None,
                    include_tiers: vec![],
                    exclude_tiers: vec![],
                    budget: lite_llm_inference::BudgetSpec::default(),
                },
                kv_cache: lite_llm_inference::KvCacheConfig {
                    hot_token_limit: 4,
                    warm_token_limit: 4,
                    total_token_limit: 16,
                    sliding_window_tokens: None,
                },
            })
            .expect("inference should start");

        assert!(!handle.selection.selected.tiers.is_empty());
        assert_eq!(llm.active_mode(), Some(crate::ActiveMode::Inference));
    }

    #[test]
    fn training_entrypoint_initializes_scheduler_and_replay() {
        let mut llm = LiteLlm::bootstrap(bootstrap_config(true)).expect("bootstrap should succeed");

        let handle = llm
            .start_training(TrainingEntrypoint {
                model_identifier: lite_llm_training::ModelIdentifier::parse("lite-llm-v1.0.0")
                    .expect("model id should parse"),
                checkpoint_id: "ckpt-1".to_owned(),
                world_size: 1,
                optimizer_name: "adamw".to_owned(),
                curriculum_plan: lite_llm_training::TierExpansionPlan {
                    new_tier: 2,
                    window: lite_llm_training::ExpansionWindow {
                        start_step: 0,
                        preparation_steps: 1,
                        isolation_steps: 1,
                        integration_steps: 1,
                        joint_training_steps: 1,
                    },
                    integration_schedule: lite_llm_training::IntegrationSchedule::Linear,
                    deterministic_seed: 7,
                },
                accumulation: lite_llm_training::AccumulationConfig {
                    micro_batch_size: 2,
                    accumulation_steps: 2,
                    data_parallel_ranks: 1,
                    scale_learning_rate: false,
                },
            })
            .expect("training should start");

        assert_eq!(handle.replay.world_size, 1);
        assert_eq!(llm.active_mode(), Some(crate::ActiveMode::Training));
    }

    #[test]
    fn replay_and_recovery_entrypoints_are_available() {
        let mut replay_llm =
            LiteLlm::bootstrap(bootstrap_config(false)).expect("bootstrap should succeed");

        let mut expected = lite_llm_training::ReplayContext::new("ckpt-1", "lite-llm-v1.0.0", 1);
        expected
            .push_event(lite_llm_training::ReplayEvent {
                sequence: 0,
                update_step: 0,
                micro_batch_index: 0,
                kind: lite_llm_training::ReplayEventKind::MicroBatch {
                    data_shard: "train-0".to_owned(),
                    seed: 1,
                },
            })
            .expect("event append should succeed");

        let replay = replay_llm
            .start_replay(ReplayEntrypoint {
                expected: expected.clone(),
                observed: expected,
            })
            .expect("replay should start");
        assert_eq!(replay.event_count, 1);
        assert_eq!(replay_llm.active_mode(), Some(crate::ActiveMode::Replay));

        let mut recovery_llm =
            LiteLlm::bootstrap(bootstrap_config(false)).expect("bootstrap should succeed");
        let recovery = recovery_llm
            .start_recovery(RecoveryEntrypoint {
                manifest_text: sample_manifest(),
                failure_event: lite_llm_distributed::FailureEvent {
                    step: 10,
                    class: lite_llm_distributed::FailureClass::Recoverable,
                    domain: lite_llm_distributed::FailureDomain::Network,
                    description: "temporary partition".to_owned(),
                },
                resume_active: false,
                snapshot_id: None,
                selected_tiers: None,
            })
            .expect("recovery should start");

        assert_eq!(
            recovery.runtime_status.state,
            lite_llm_runtime::RuntimeState::Warm
        );
        assert_eq!(
            recovery_llm.active_mode(),
            Some(crate::ActiveMode::Recovery)
        );
    }
}
