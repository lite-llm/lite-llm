#[cfg(test)]
use std::collections::{BTreeMap, BTreeSet};
#[cfg(test)]
use std::path::Path;
#[cfg(test)]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(test)]
use lite_llm_distributed::CollectiveOps;
#[cfg(test)]
use lite_llm_inference::InferencePipeline;
#[cfg(test)]
use lite_llm_runtime::Router;
#[cfg(test)]
use lite_llm_security::AuditSink;

#[derive(Debug, Clone, Copy)]
pub struct SpecComplianceEntry {
    pub spec_id: u16,
    pub title: &'static str,
    pub module_path: &'static str,
    pub test_refs: &'static [&'static str],
}

pub const REQUIRED_SPEC_START: u16 = 1;
pub const REQUIRED_SPEC_END: u16 = 60;

const RUNTIME_PROCESS_TESTS: &[&str] = &[
    "lite-llm-runtime::process::tests::full_boot_load_activate_flow",
    "lite-llm-runtime::process::tests::recovery_restores_to_active",
];
const RUNTIME_ROUTING_TESTS: &[&str] = &[
    "lite-llm-runtime::routing::tests::stable_top_k_uses_seeded_tie_break",
    "lite-llm-runtime::routing::tests::deterministic_router_replays_identically",
];
const RUNTIME_CONFIG_TESTS: &[&str] = &[
    "lite-llm-runtime::config::tests::tierset_resolve_handles_cumulative_mode",
    "lite-llm-runtime::config::tests::routing_seed_is_deterministic",
];
const RUNTIME_COMPUTE_TESTS: &[&str] = &[
    "lite-llm-runtime::compute::tests::enforce_bound_rejects_overflow_routes",
    "lite-llm-runtime::routing::tests::deterministic_router_respects_compute_bound",
];
const RUNTIME_ERROR_TESTS: &[&str] = &[
    "lite-llm-runtime::process::tests::parse_manifest_requires_boot",
    "lite-llm-runtime::config::tests::routing_config_requires_positive_k",
];
const RUNTIME_STATE_MACHINE_TESTS: &[&str] = &[
    "lite-llm-runtime::state_machine::tests::transition_rules_match_spec",
    "lite-llm-runtime::state_machine::tests::replay_is_deterministic_for_state_transitions",
];

const DISTRIBUTED_PARALLEL_TESTS: &[&str] = &[
    "lite-llm-distributed::parallelism::tests::coordinate_roundtrip_is_lossless",
    "lite-llm-distributed::parallelism::tests::expert_owner_is_deterministic",
];
const DISTRIBUTED_COLLECTIVE_TESTS: &[&str] = &[
    "lite-llm-distributed::collectives::tests::all_reduce_uses_fixed_global_order",
    "lite-llm-distributed::collectives::tests::all_to_all_is_stable",
];
const DISTRIBUTED_CONSENSUS_TESTS: &[&str] =
    &["lite-llm-distributed::consensus::tests::selection_is_deterministic_with_same_seed"];
const DISTRIBUTED_TRANSPORT_TESTS: &[&str] = &[
    "lite-llm-distributed::transport::tests::tagged_send_receive_roundtrip",
    "lite-llm-distributed::transport::tests::send_requires_monotonic_tag_order",
];
const DISTRIBUTED_FAULT_TESTS: &[&str] = &[
    "lite-llm-distributed::fault_tolerance::tests::transient_failure_retries_then_checkpoint_reload",
    "lite-llm-distributed::fault_tolerance::tests::heartbeat_timeout_detection_is_deterministic",
];

const STORAGE_PLACEMENT_TESTS: &[&str] = &[
    "lite-llm-storage::placement::tests::adaptive_policy_promotes_hot_expert",
    "lite-llm-storage::placement::tests::static_policy_keeps_hint_tier",
];
const STORAGE_CACHE_TESTS: &[&str] = &[
    "lite-llm-storage::cache::tests::deterministic_eviction_keeps_recent_hot_entries",
    "lite-llm-storage::cache::tests::lazy_loader_fetches_in_deterministic_key_order",
];
const STORAGE_CHECKPOINT_TESTS: &[&str] = &[
    "lite-llm-storage::checkpoint::tests::manifest_roundtrip_is_lossless",
    "lite-llm-storage::checkpoint::tests::shard_verification_detects_corruption",
];
const STORAGE_SNAPSHOT_TESTS: &[&str] = &[
    "lite-llm-storage::snapshot::tests::restore_roundtrip_is_consistent",
    "lite-llm-storage::snapshot::tests::restore_detects_corrupted_shard",
];

const TRAINING_CURRICULUM_TESTS: &[&str] = &[
    "lite-llm-training::curriculum::tests::scheduler_phase_progression_matches_spec",
    "lite-llm-training::curriculum::tests::scheduler_replay_is_stable",
];
const TRAINING_LOAD_BALANCE_TESTS: &[&str] =
    &["lite-llm-training::load_balancing::tests::hierarchical_loss_computes_all_levels"];
const TRAINING_STARVATION_TESTS: &[&str] =
    &["lite-llm-training::starvation::tests::starvation_analysis_detects_zero_assignment_experts"];
const TRAINING_OPTIMIZER_TESTS: &[&str] = &[
    "lite-llm-training::optimizer::tests::sgd_momentum_update_is_deterministic",
    "lite-llm-training::optimizer::tests::adam_state_shard_roundtrip_restores_state",
];
const TRAINING_PRECISION_TESTS: &[&str] =
    &["lite-llm-training::precision::tests::precision_cast_is_deterministic"];
const TRAINING_ACCUMULATION_TESTS: &[&str] = &[
    "lite-llm-training::accumulation::tests::mean_gradients_and_deterministic_update_are_stable",
    "lite-llm-training::accumulation::tests::micro_batch_seed_is_deterministic",
];
const TRAINING_CHECKPOINT_TESTS: &[&str] = &[
    "lite-llm-training::checkpoint::tests::distributed_checkpoint_roundtrip_restores_payloads",
    "lite-llm-training::checkpoint::tests::restore_detects_shard_corruption",
];
const TRAINING_REPLAY_TESTS: &[&str] = &[
    "lite-llm-training::replay::tests::replay_roundtrip_is_lossless",
    "lite-llm-training::replay::tests::replay_verifier_detects_mismatch",
];
const TRAINING_VERSIONING_TESTS: &[&str] = &[
    "lite-llm-training::versioning::tests::compatibility_reports_forward_minor_with_ignored_tiers",
    "lite-llm-training::versioning::tests::compatibility_rejects_major_mismatch",
];

const INFERENCE_TIERSET_TESTS: &[&str] = &[
    "lite-llm-inference::tierset_selection::tests::fixed_fast_mode_selects_hot",
    "lite-llm-inference::tierset_selection::tests::budget_solver_maximizes_capacity_under_budget",
];
const INFERENCE_PIPELINE_TESTS: &[&str] = &[
    "lite-llm-inference::pipeline::tests::pipeline_output_is_deterministic",
    "lite-llm-inference::pipeline::tests::stable_tie_breaking_is_seeded",
];
const INFERENCE_PREFETCH_TESTS: &[&str] =
    &["lite-llm-inference::prefetch::tests::plan_is_deterministic"];
const INFERENCE_KV_TESTS: &[&str] =
    &["lite-llm-inference::kv_cache::tests::deterministic_tier_assignment_after_rebalance"];
const INFERENCE_STREAMING_TESTS: &[&str] = &[
    "lite-llm-inference::streaming::tests::token_generation_is_deterministic",
    "lite-llm-inference::streaming::tests::replay_prefix_is_idempotent",
];
const INFERENCE_COST_TESTS: &[&str] =
    &["lite-llm-inference::cost_adaptive::tests::selection_is_deterministic"];
const INFERENCE_TELEMETRY_TESTS: &[&str] =
    &["lite-llm-inference::telemetry::tests::sampled_recording_is_deterministic"];
const INFERENCE_TENANT_TESTS: &[&str] = &[
    "lite-llm-inference::tenant::tests::tier_authorization_enforces_isolation",
    "lite-llm-inference::tenant::tests::tenant_usage_state_is_isolated",
];

const SECURITY_MEMORY_TESTS: &[&str] =
    &["lite-llm-security::memory_safety::tests::fully_compliant_profile_is_accepted"];
const SECURITY_INTEGRITY_TESTS: &[&str] = &[
    "lite-llm-security::integrity::tests::load_succeeds_when_hash_and_signature_match",
    "lite-llm-security::integrity::tests::load_fails_on_signature_mismatch",
];
const SECURITY_ENCRYPTION_TESTS: &[&str] = &[
    "lite-llm-security::encryption::tests::encryption_roundtrip_succeeds",
    "lite-llm-security::encryption::tests::tampered_ciphertext_is_rejected",
];
const SECURITY_ZEROIZATION_TESTS: &[&str] =
    &["lite-llm-security::zeroization::tests::session_zeroization_records_events"];
const SECURITY_ACCESS_TESTS: &[&str] = &[
    "lite-llm-security::access_control::tests::allows_authorized_tier_access",
    "lite-llm-security::access_control::tests::denies_when_action_role_missing",
];
const SECURITY_AUDIT_TESTS: &[&str] = &[
    "lite-llm-security::audit::tests::append_and_verify_chain_succeeds",
    "lite-llm-security::audit::tests::tamper_is_detected",
];
const SECURITY_KEY_TESTS: &[&str] = &[
    "lite-llm-security::key_management::tests::retrieval_requires_authorization",
    "lite-llm-security::key_management::tests::rotation_creates_new_version",
];
const SECURITY_SANDBOX_TESTS: &[&str] = &[
    "lite-llm-security::sandbox::tests::capability_tokens_enforce_expiry_and_scope",
    "lite-llm-security::sandbox::tests::resource_limits_are_enforced",
];
const SECURITY_COMPLIANCE_TESTS: &[&str] = &[
    "lite-llm-security::compliance::tests::artifact_generation_is_deterministic",
    "lite-llm-security::compliance::tests::hipaa_report_is_ready_when_controls_enabled",
];
const SECURITY_HARDENING_TESTS: &[&str] = &[
    "lite-llm-security::hardening::tests::checklist_report_tracks_coverage",
    "lite-llm-security::hardening::tests::incident_plan_has_all_phases",
];

pub const SPEC_COMPLIANCE: &[SpecComplianceEntry] = &[
    SpecComplianceEntry {
        spec_id: 1,
        title: "Runtime Architecture Overview",
        module_path: "../lite-llm-runtime/src/lib.rs",
        test_refs: RUNTIME_PROCESS_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 2,
        title: "Process Model & Execution Lifecycle",
        module_path: "../lite-llm-runtime/src/process.rs",
        test_refs: RUNTIME_PROCESS_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 3,
        title: "Deterministic Routing Engine",
        module_path: "../lite-llm-runtime/src/routing.rs",
        test_refs: RUNTIME_ROUTING_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 4,
        title: "Tiered Parameter Architecture (TPA)",
        module_path: "../lite-llm-runtime/src/config.rs",
        test_refs: RUNTIME_CONFIG_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 5,
        title: "Hierarchical Sparse Expert Routing (HSER)",
        module_path: "../lite-llm-runtime/src/routing.rs",
        test_refs: RUNTIME_ROUTING_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 6,
        title: "Active Compute Bounding Model",
        module_path: "../lite-llm-runtime/src/compute.rs",
        test_refs: RUNTIME_COMPUTE_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 7,
        title: "Runtime Memory Model",
        module_path: "../lite-llm-runtime/src/process.rs",
        test_refs: RUNTIME_PROCESS_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 8,
        title: "Error Model & Failure Domains",
        module_path: "../lite-llm-runtime/src/error.rs",
        test_refs: RUNTIME_ERROR_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 9,
        title: "Concurrency & Threading Model",
        module_path: "../lite-llm-runtime/src/process.rs",
        test_refs: RUNTIME_STATE_MACHINE_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 10,
        title: "Runtime State Machine",
        module_path: "../lite-llm-runtime/src/state_machine.rs",
        test_refs: RUNTIME_STATE_MACHINE_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 11,
        title: "Data Parallel Specification",
        module_path: "../lite-llm-distributed/src/parallelism.rs",
        test_refs: DISTRIBUTED_PARALLEL_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 12,
        title: "Tensor Parallel Specification",
        module_path: "../lite-llm-distributed/src/parallelism.rs",
        test_refs: DISTRIBUTED_PARALLEL_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 13,
        title: "Pipeline Parallel Specification",
        module_path: "../lite-llm-distributed/src/parallelism.rs",
        test_refs: DISTRIBUTED_PARALLEL_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 14,
        title: "Expert Parallel Specification",
        module_path: "../lite-llm-distributed/src/parallelism.rs",
        test_refs: DISTRIBUTED_PARALLEL_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 15,
        title: "All-to-All Communication Protocol",
        module_path: "../lite-llm-distributed/src/collectives.rs",
        test_refs: DISTRIBUTED_COLLECTIVE_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 16,
        title: "Routing Consensus Protocol",
        module_path: "../lite-llm-distributed/src/consensus.rs",
        test_refs: DISTRIBUTED_CONSENSUS_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 17,
        title: "Cross-Node Synchronization Guarantees",
        module_path: "../lite-llm-distributed/src/collectives.rs",
        test_refs: DISTRIBUTED_COLLECTIVE_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 18,
        title: "Deterministic Collective Operations",
        module_path: "../lite-llm-distributed/src/collectives.rs",
        test_refs: DISTRIBUTED_COLLECTIVE_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 19,
        title: "Network Transport Abstraction",
        module_path: "../lite-llm-distributed/src/transport.rs",
        test_refs: DISTRIBUTED_TRANSPORT_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 20,
        title: "Fault-Tolerant Distributed Execution",
        module_path: "../lite-llm-distributed/src/fault_tolerance.rs",
        test_refs: DISTRIBUTED_FAULT_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 21,
        title: "Tier Placement Policy",
        module_path: "../lite-llm-storage/src/placement.rs",
        test_refs: STORAGE_PLACEMENT_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 22,
        title: "Hot Cache Management",
        module_path: "../lite-llm-storage/src/cache.rs",
        test_refs: STORAGE_CACHE_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 23,
        title: "Warm Tier Staging Protocol",
        module_path: "../lite-llm-storage/src/placement.rs",
        test_refs: STORAGE_PLACEMENT_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 24,
        title: "Cold Tier Streaming & Prefetch",
        module_path: "../lite-llm-storage/src/cache.rs",
        test_refs: STORAGE_CACHE_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 25,
        title: "Archive Tier Retrieval Model",
        module_path: "../lite-llm-storage/src/snapshot.rs",
        test_refs: STORAGE_SNAPSHOT_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 26,
        title: "Lazy Expert Loading Contract",
        module_path: "../lite-llm-storage/src/cache.rs",
        test_refs: STORAGE_CACHE_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 27,
        title: "Tier Eviction Strategy",
        module_path: "../lite-llm-storage/src/cache.rs",
        test_refs: STORAGE_CACHE_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 28,
        title: "Parameter Sharding Format",
        module_path: "../lite-llm-storage/src/checkpoint.rs",
        test_refs: STORAGE_CHECKPOINT_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 29,
        title: "Checkpoint Manifest Specification",
        module_path: "../lite-llm-storage/src/checkpoint.rs",
        test_refs: STORAGE_CHECKPOINT_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 30,
        title: "Snapshot & Restore Semantics",
        module_path: "../lite-llm-storage/src/snapshot.rs",
        test_refs: STORAGE_SNAPSHOT_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 31,
        title: "Curriculum Tier Expansion Protocol",
        module_path: "../lite-llm-training/src/curriculum.rs",
        test_refs: TRAINING_CURRICULUM_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 32,
        title: "Load Balancing Loss Formalization",
        module_path: "../lite-llm-training/src/load_balancing.rs",
        test_refs: TRAINING_LOAD_BALANCE_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 33,
        title: "Expert Starvation Guarantees",
        module_path: "../lite-llm-training/src/starvation.rs",
        test_refs: TRAINING_STARVATION_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 34,
        title: "Optimizer Abstraction Interface",
        module_path: "../lite-llm-training/src/optimizer.rs",
        test_refs: TRAINING_OPTIMIZER_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 35,
        title: "Mixed Precision Policy",
        module_path: "../lite-llm-training/src/precision.rs",
        test_refs: TRAINING_PRECISION_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 36,
        title: "Gradient Accumulation Model",
        module_path: "../lite-llm-training/src/accumulation.rs",
        test_refs: TRAINING_ACCUMULATION_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 37,
        title: "Sharded Optimizer State Format",
        module_path: "../lite-llm-training/src/checkpoint.rs",
        test_refs: TRAINING_OPTIMIZER_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 38,
        title: "Distributed Checkpointing",
        module_path: "../lite-llm-training/src/checkpoint.rs",
        test_refs: TRAINING_CHECKPOINT_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 39,
        title: "Deterministic Training Replay Engine",
        module_path: "../lite-llm-training/src/replay.rs",
        test_refs: TRAINING_REPLAY_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 40,
        title: "Model Evolution & Versioning",
        module_path: "../lite-llm-training/src/versioning.rs",
        test_refs: TRAINING_VERSIONING_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 41,
        title: "TierSet Selection Engine",
        module_path: "../lite-llm-inference/src/tierset_selection.rs",
        test_refs: INFERENCE_TIERSET_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 42,
        title: "Latency Budget Solver",
        module_path: "../lite-llm-inference/src/tierset_selection.rs",
        test_refs: INFERENCE_TIERSET_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 43,
        title: "Token Routing Execution Pipeline",
        module_path: "../lite-llm-inference/src/pipeline.rs",
        test_refs: INFERENCE_PIPELINE_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 44,
        title: "Expert Packing & Dispatch",
        module_path: "../lite-llm-inference/src/pipeline.rs",
        test_refs: INFERENCE_PIPELINE_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 45,
        title: "Dynamic Tier Prefetching",
        module_path: "../lite-llm-inference/src/prefetch.rs",
        test_refs: INFERENCE_PREFETCH_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 46,
        title: "KV-Cache Architecture",
        module_path: "../lite-llm-inference/src/kv_cache.rs",
        test_refs: INFERENCE_KV_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 47,
        title: "Streaming Inference Protocol",
        module_path: "../lite-llm-inference/src/streaming.rs",
        test_refs: INFERENCE_STREAMING_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 48,
        title: "Cost-Adaptive Routing",
        module_path: "../lite-llm-inference/src/cost_adaptive.rs",
        test_refs: INFERENCE_COST_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 49,
        title: "Inference Telemetry Model",
        module_path: "../lite-llm-inference/src/telemetry.rs",
        test_refs: INFERENCE_TELEMETRY_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 50,
        title: "Multi-Tenant Isolation Model",
        module_path: "../lite-llm-inference/src/tenant.rs",
        test_refs: INFERENCE_TENANT_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 51,
        title: "Memory Safety Guarantees (Rust Mapping)",
        module_path: "../lite-llm-security/src/memory_safety.rs",
        test_refs: SECURITY_MEMORY_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 52,
        title: "Secure Model Loading & Integrity Verification",
        module_path: "../lite-llm-security/src/integrity.rs",
        test_refs: SECURITY_INTEGRITY_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 53,
        title: "Tier Encryption at Rest",
        module_path: "../lite-llm-security/src/encryption.rs",
        test_refs: SECURITY_ENCRYPTION_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 54,
        title: "In-Memory Zeroization Policy",
        module_path: "../lite-llm-security/src/zeroization.rs",
        test_refs: SECURITY_ZEROIZATION_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 55,
        title: "Access Control & Tier Authorization",
        module_path: "../lite-llm-security/src/access_control.rs",
        test_refs: SECURITY_ACCESS_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 56,
        title: "Deterministic Audit Logging",
        module_path: "../lite-llm-security/src/audit.rs",
        test_refs: SECURITY_AUDIT_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 57,
        title: "Secure Distributed Key Management",
        module_path: "../lite-llm-security/src/key_management.rs",
        test_refs: SECURITY_KEY_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 58,
        title: "Runtime Sandboxing & Capability Isolation",
        module_path: "../lite-llm-security/src/sandbox.rs",
        test_refs: SECURITY_SANDBOX_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 59,
        title: "Compliance & Regulatory Readiness",
        module_path: "../lite-llm-security/src/compliance.rs",
        test_refs: SECURITY_COMPLIANCE_TESTS,
    },
    SpecComplianceEntry {
        spec_id: 60,
        title: "Threat Model & Security Hardening Guide",
        module_path: "../lite-llm-security/src/hardening.rs",
        test_refs: SECURITY_HARDENING_TESTS,
    },
];

pub fn spec_compliance_matrix() -> &'static [SpecComplianceEntry] {
    SPEC_COMPLIANCE
}

#[cfg(test)]
fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{nanos}"))
}

#[cfg(test)]
fn runtime_manifest_text() -> &'static str {
    "version=1\n\
     tiers=1,2\n\
     cumulative=false\n\
     base_checksum=abc123\n\
     router_checksum=def456\n\
     optimizer_checksum=xyz999\n\
     shard=base|aa11|1024\n\
     shard=exp|bb22|2048\n"
}

#[cfg(test)]
fn sample_storage_manifest() -> (
    lite_llm_storage::CheckpointManifest,
    BTreeMap<String, Vec<u8>>,
) {
    let dense_bytes = b"dense-shard".to_vec();
    let expert_bytes = b"expert-shard".to_vec();

    let manifest = lite_llm_storage::CheckpointManifest {
        model_id: "lite-llm-base".to_owned(),
        epoch: 1,
        step: 100,
        tiers: vec![
            lite_llm_storage::TierManifestEntry {
                tier_id: 1,
                name: "hot".to_owned(),
                size_budget_bytes: 1024 * 1024,
                placement_policy: lite_llm_storage::PlacementPolicyKind::Prioritized,
                priority_score: 1,
            },
            lite_llm_storage::TierManifestEntry {
                tier_id: 2,
                name: "warm".to_owned(),
                size_budget_bytes: 2 * 1024 * 1024,
                placement_policy: lite_llm_storage::PlacementPolicyKind::Lru,
                priority_score: 2,
            },
        ],
        shards: vec![
            lite_llm_storage::ShardDescriptor {
                path: "dense/l0/q.bin".to_owned(),
                shard_type: lite_llm_storage::ShardType::Dense,
                tensor_name: Some("attention_q".to_owned()),
                expert_key: None,
                shape: vec![2, 5],
                dtype: "bf16".to_owned(),
                tier_hint: lite_llm_storage::StorageTier::Hot,
                checksum_hex: lite_llm_storage::fnv64_hex(&dense_bytes),
                hash_algorithm: "fnv64".to_owned(),
                version: 1,
                bytes: dense_bytes.len() as u64,
            },
            lite_llm_storage::ShardDescriptor {
                path: "experts/t2/g0/e1.bin".to_owned(),
                shard_type: lite_llm_storage::ShardType::Expert,
                tensor_name: None,
                expert_key: Some(lite_llm_storage::ExpertKey::new(2, 0, 1)),
                shape: vec![2, 6],
                dtype: "bf16".to_owned(),
                tier_hint: lite_llm_storage::StorageTier::Warm,
                checksum_hex: lite_llm_storage::fnv64_hex(&expert_bytes),
                hash_algorithm: "fnv64".to_owned(),
                version: 2,
                bytes: expert_bytes.len() as u64,
            },
        ],
        optim_state: Some(lite_llm_storage::OptimStateRef {
            path: "optim/state.bin".to_owned(),
            checksum_hex: lite_llm_storage::fnv64_hex(b"optim-state"),
            hash_algorithm: "fnv64".to_owned(),
            version: 1,
        }),
        router_state: Some(lite_llm_storage::RouterStateRef {
            path: "router/state.bin".to_owned(),
            checksum_hex: lite_llm_storage::fnv64_hex(b"router-state"),
            hash_algorithm: "fnv64".to_owned(),
            version: 1,
            base_seed: 42,
            layer_seeds: vec![lite_llm_storage::RouterSeedRef { layer: 0, seed: 99 }],
        }),
        metadata_version: 1,
    };

    let mut shards = BTreeMap::new();
    shards.insert("dense/l0/q.bin".to_owned(), dense_bytes);
    shards.insert("experts/t2/g0/e1.bin".to_owned(), expert_bytes);

    (manifest, shards)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_spec_matrix_maps_001_to_060_only() {
        let matrix = spec_compliance_matrix();
        assert_eq!(
            matrix.len(),
            usize::from(REQUIRED_SPEC_END - REQUIRED_SPEC_START + 1)
        );

        let mut ids = matrix
            .iter()
            .map(|entry| entry.spec_id)
            .collect::<Vec<u16>>();
        ids.sort_unstable();

        let expected = (REQUIRED_SPEC_START..=REQUIRED_SPEC_END).collect::<Vec<u16>>();
        assert_eq!(ids, expected);

        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        for entry in matrix {
            assert!(!entry.title.trim().is_empty());
            assert!(!entry.test_refs.is_empty());

            let module = Path::new(manifest_dir).join(entry.module_path);
            assert!(
                module.exists(),
                "missing mapped module for SPEC-{:03}: {}",
                entry.spec_id,
                module.display()
            );
        }

        assert!(matrix
            .iter()
            .all(|entry| entry.spec_id <= REQUIRED_SPEC_END));
    }

    #[test]
    fn determinism_gate_same_seed_input_topology_produces_identical_outputs_and_logs() {
        let tier_catalog = vec![
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
        ];

        let router_a = lite_llm_runtime::DeterministicRouter::new(
            lite_llm_runtime::RoutingSeed::new(99),
            tier_catalog.clone(),
        );
        let router_b = lite_llm_runtime::DeterministicRouter::new(
            lite_llm_runtime::RoutingSeed::new(99),
            tier_catalog,
        );

        let tiers = lite_llm_runtime::TierSet::new(
            vec![
                lite_llm_runtime::TierId::new(1),
                lite_llm_runtime::TierId::new(2),
            ],
            false,
        );
        let cfg = lite_llm_runtime::RoutingConfig {
            k_tier: 1,
            k_group: 2,
            k_expert: 2,
        };
        let token = vec![0.1, -0.3, 0.8, 0.2, -0.5];

        let routes_a = router_a
            .route(&token, 3, 17, &tiers, cfg)
            .expect("routing should succeed");
        let routes_b = router_b
            .route(&token, 3, 17, &tiers, cfg)
            .expect("routing should succeed");
        assert_eq!(routes_a, routes_b);

        let topology = lite_llm_distributed::ParallelismConfig {
            data_parallel: 1,
            tensor_parallel: 1,
            pipeline_parallel: 1,
            expert_parallel: 2,
        };
        topology.validate().expect("topology should be valid");

        let collectives_a =
            lite_llm_distributed::DeterministicCollectives::new(topology.world_size())
                .expect("collectives should initialize");
        let collectives_b =
            lite_llm_distributed::DeterministicCollectives::new(topology.world_size())
                .expect("collectives should initialize");

        let payloads = vec![
            vec![b"0->0".to_vec(), b"0->1".to_vec()],
            vec![b"1->0".to_vec(), b"1->1".to_vec()],
        ];
        let all_to_all_a = collectives_a
            .all_to_all(&payloads)
            .expect("all_to_all should succeed");
        let all_to_all_b = collectives_b
            .all_to_all(&payloads)
            .expect("all_to_all should succeed");
        assert_eq!(all_to_all_a, all_to_all_b);

        let pipeline = lite_llm_inference::DeterministicInferencePipeline::default();
        let input = lite_llm_inference::PipelineInput {
            seed: 77,
            rank: 0,
            world_size: 2,
            top_k: 2,
            tokens: vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]],
            expert_scores: vec![
                vec![
                    lite_llm_inference::ExpertScore {
                        expert: lite_llm_inference::ExpertKey::new(1, 0, 0),
                        destination_rank: 0,
                        score: 0.9,
                    },
                    lite_llm_inference::ExpertScore {
                        expert: lite_llm_inference::ExpertKey::new(2, 0, 0),
                        destination_rank: 1,
                        score: 0.7,
                    },
                ],
                vec![
                    lite_llm_inference::ExpertScore {
                        expert: lite_llm_inference::ExpertKey::new(1, 1, 0),
                        destination_rank: 0,
                        score: 0.8,
                    },
                    lite_llm_inference::ExpertScore {
                        expert: lite_llm_inference::ExpertKey::new(2, 1, 0),
                        destination_rank: 1,
                        score: 0.6,
                    },
                ],
            ],
        };
        let pipe_a = pipeline.run(&input).expect("pipeline should run");
        let pipe_b = pipeline.run(&input).expect("pipeline should run");
        assert_eq!(pipe_a, pipe_b);

        let mut log_a = lite_llm_security::DeterministicAuditLog::new("n1", "signer", "secret");
        let mut log_b = lite_llm_security::DeterministicAuditLog::new("n1", "signer", "secret");

        let events = vec![
            lite_llm_security::AuditEvent {
                sequence: 0,
                timestamp_ms: 1,
                category: lite_llm_security::AuditCategory::Routing,
                actor: "runtime".to_owned(),
                action: "route".to_owned(),
                payload: "token=0".to_owned(),
            },
            lite_llm_security::AuditEvent {
                sequence: 1,
                timestamp_ms: 2,
                category: lite_llm_security::AuditCategory::Cache,
                actor: "runtime".to_owned(),
                action: "cache-hit".to_owned(),
                payload: "expert=1:0:0".to_owned(),
            },
        ];

        for event in events.clone() {
            log_a.append(event).expect("append should succeed");
        }
        for event in events {
            log_b.append(event).expect("append should succeed");
        }

        assert_eq!(log_a.records(), log_b.records());
        assert_eq!(log_a.root_hash(), log_b.root_hash());
        assert!(log_a.verify_chain().is_ok());
        assert!(log_b.verify_chain().is_ok());
    }

    #[test]
    fn security_gate_validates_manifest_authz_and_audit_chain() {
        let mut store = lite_llm_security::InMemoryArtifactStore::default();
        store.insert("tier1/shard.bin", b"weights-tier1".to_vec());
        store.insert("tier2/shard.bin", b"weights-tier2".to_vec());

        let shards = vec![
            lite_llm_security::ManifestShard {
                path: "tier1/shard.bin".to_owned(),
                tier: 1,
                digest: lite_llm_security::ArtifactDigest::from_payload(b"weights-tier1", "sha256")
                    .expect("digest should build"),
                bytes: 13,
            },
            lite_llm_security::ManifestShard {
                path: "tier2/shard.bin".to_owned(),
                tier: 2,
                digest: lite_llm_security::ArtifactDigest::from_payload(b"weights-tier2", "sha256")
                    .expect("digest should build"),
                bytes: 13,
            },
        ];

        let mut manifest = lite_llm_security::SecureModelManifest {
            model_id: "lite-llm".to_owned(),
            version_major: 1,
            version_minor: 0,
            version_patch: 0,
            tiers: vec![1, 2],
            shards,
            manifest_hash_hex: String::new(),
            signature: lite_llm_security::SignatureEnvelope {
                signer_id: "publisher".to_owned(),
                key_id: "pub-1".to_owned(),
                signature_hex: String::new(),
            },
        };

        manifest.manifest_hash_hex = manifest.recompute_hash();
        let pub_material = "publisher-public-material";
        manifest.signature.signature_hex = lite_llm_security::SignatureVerifier::sign_for_testing(
            pub_material,
            &manifest.signature.signer_id,
            &manifest.manifest_hash_hex,
        );

        let mut signature_verifier = lite_llm_security::SignatureVerifier::default();
        signature_verifier.register_key("pub-1", pub_material);

        let loader = lite_llm_security::SecureModelLoader {
            verifier: lite_llm_security::DeterministicDigestVerifier,
            signature_verifier,
            supported_major_version: 1,
            expected_tiers: BTreeSet::from([1, 2]),
        };

        let loaded = loader
            .load(&manifest, &store)
            .expect("secure load should succeed");
        assert_eq!(loaded.model_id, "lite-llm");

        let mut bad_hash = manifest.clone();
        bad_hash.manifest_hash_hex = "deadbeefdeadbeef".to_owned();
        assert!(loader.load(&bad_hash, &store).is_err());

        let mut bad_signature = manifest.clone();
        bad_signature.signature.signature_hex = "00".to_owned();
        assert!(loader.load(&bad_signature, &store).is_err());

        let mut access = lite_llm_security::AccessController::default();
        access.set_action_roles(
            lite_llm_security::Action::RunInference,
            BTreeSet::from(["inference".to_owned(), "admin".to_owned()]),
        );
        access.set_tier_policy(lite_llm_security::TierPolicy {
            tier: 1,
            allowed_roles: BTreeSet::from(["inference".to_owned(), "admin".to_owned()]),
            allowed_tenants: BTreeSet::from(["tenant-a".to_owned()]),
            downgrade_tier: None,
        });

        let allow = access
            .authorize(
                &lite_llm_security::Principal {
                    id: "user-a".to_owned(),
                    tenant_id: "tenant-a".to_owned(),
                    roles: BTreeSet::from(["inference".to_owned()]),
                    scopes: BTreeSet::new(),
                },
                lite_llm_security::Action::RunInference,
                Some(1),
            )
            .expect("authorization should run");
        assert!(matches!(
            allow,
            lite_llm_security::AuthorizationDecision::Allow
        ));

        let deny = access
            .authorize(
                &lite_llm_security::Principal {
                    id: "user-b".to_owned(),
                    tenant_id: "tenant-b".to_owned(),
                    roles: BTreeSet::from(["viewer".to_owned()]),
                    scopes: BTreeSet::new(),
                },
                lite_llm_security::Action::RunInference,
                Some(1),
            )
            .expect("authorization should run");
        assert!(matches!(
            deny,
            lite_llm_security::AuthorizationDecision::Deny { .. }
        ));

        let mut audit = lite_llm_security::DeterministicAuditLog::new("n1", "signer", "secret");
        audit
            .append(lite_llm_security::AuditEvent {
                sequence: 0,
                timestamp_ms: 10,
                category: lite_llm_security::AuditCategory::Security,
                actor: "runtime".to_owned(),
                action: "load".to_owned(),
                payload: "ok".to_owned(),
            })
            .expect("append should succeed");
        audit
            .append(lite_llm_security::AuditEvent {
                sequence: 1,
                timestamp_ms: 11,
                category: lite_llm_security::AuditCategory::AccessControl,
                actor: "runtime".to_owned(),
                action: "authorize".to_owned(),
                payload: "allow".to_owned(),
            })
            .expect("append should succeed");

        assert!(audit.verify_chain().is_ok());
        assert!(audit
            .append(lite_llm_security::AuditEvent {
                sequence: 3,
                timestamp_ms: 12,
                category: lite_llm_security::AuditCategory::Error,
                actor: "runtime".to_owned(),
                action: "bad-sequence".to_owned(),
                payload: "reject".to_owned(),
            })
            .is_err());
    }

    #[test]
    fn recovery_gate_snapshot_restore_returns_identical_resumed_state() {
        let snapshot_root = unique_temp_dir("lite-llm-spec-gate-snapshots");
        let repository = lite_llm_storage::SnapshotRepository::new(&snapshot_root)
            .expect("repo should initialize");
        let verifier = lite_llm_storage::Fnv64HashVerifier;

        let (manifest, shard_bytes) = sample_storage_manifest();
        repository
            .commit_snapshot("gate-snap-1", &manifest, &shard_bytes, &verifier)
            .expect("snapshot commit should succeed");

        let restored = repository
            .restore_snapshot("gate-snap-1", None, &verifier)
            .expect("snapshot restore should succeed");

        assert_eq!(restored.manifest, manifest);
        assert_eq!(restored.shard_bytes, shard_bytes);

        let mut runtime =
            lite_llm_runtime::RuntimeLifecycle::new(lite_llm_runtime::RuntimeOptions {
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
                training_mode: false,
            })
            .expect("runtime options should be valid");

        runtime.boot().expect("boot should succeed");
        runtime
            .parse_manifest(runtime_manifest_text())
            .expect("manifest parse should succeed");
        runtime
            .load_base_parameters()
            .expect("base load should succeed");
        runtime
            .register_experts()
            .expect("expert registration should succeed");
        runtime
            .load_router_parameters()
            .expect("router load should succeed");
        runtime
            .load_optimizer_state()
            .expect("optimizer stage should be no-op for inference mode");
        runtime
            .complete_model_load()
            .expect("model load completion should succeed");

        let active = lite_llm_runtime::TierSet::new(
            vec![
                lite_llm_runtime::TierId::new(1),
                lite_llm_runtime::TierId::new(2),
            ],
            false,
        );
        runtime
            .activate_tiers(active)
            .expect("tier activation should succeed");
        runtime.start_serving().expect("serving should start");

        let status_before = runtime.status();
        let seed_before = runtime.routing_seed();

        runtime
            .begin_recovery()
            .expect("recovery transition should succeed");
        runtime
            .restore_after_crash(runtime_manifest_text(), true)
            .expect("restore after crash should succeed");

        let status_after = runtime.status();
        assert_eq!(seed_before, runtime.routing_seed());
        assert_eq!(status_before.state, status_after.state);
        assert_eq!(
            status_before.model_load_stage,
            status_after.model_load_stage
        );
        assert_eq!(status_before.active_tiers, status_after.active_tiers);

        let _ = std::fs::remove_dir_all(snapshot_root);
    }
}
