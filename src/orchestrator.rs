use std::collections::BTreeSet;
use std::path::PathBuf;

use lite_llm_security::AuditSink;

use crate::contracts::{verify_shared_contracts, ContractReport};
use crate::error::{LiteLlmError, LiteLlmResult};
use crate::mode::{
    ActiveMode, BootstrapConfig, InferenceEntrypoint, InferenceHandle, RecoveryEntrypoint,
    RecoveryHandle, ReplayEntrypoint, ReplayHandle, TrainingEntrypoint, TrainingHandle,
};
use crate::profile::{StartupProfile, StartupTuning};

#[derive(Debug, Clone)]
pub struct DistributedStack {
    pub parallelism: lite_llm_distributed::ParallelismConfig,
    pub collectives: lite_llm_distributed::DeterministicCollectives,
    pub consensus: lite_llm_distributed::RoutingConsensus,
    pub transport: lite_llm_distributed::InMemoryTaggedTransport,
    pub recovery: lite_llm_distributed::RecoveryCoordinator,
}

#[derive(Debug, Clone)]
pub struct StorageStack {
    pub snapshots: lite_llm_storage::SnapshotRepository,
    pub shard_verifier: lite_llm_storage::Fnv64HashVerifier,
    pub hot_cache: lite_llm_storage::HotExpertCache,
    pub training_checkpoint_root: PathBuf,
}

#[derive(Debug, Clone)]
pub struct SecurityStack {
    pub access: lite_llm_security::AccessController,
    pub audit: lite_llm_security::DeterministicAuditLog,
    pub keys: lite_llm_security::KeyManager,
    pub sandbox: lite_llm_security::SandboxRuntime,
    pub compliance: lite_llm_security::ComplianceEngine,
    pub hardening: lite_llm_security::HardeningChecklist,
    pub memory_profile: lite_llm_security::MemorySafetyProfile,
}

#[derive(Debug, Clone)]
pub struct LiteLlm {
    profile: StartupProfile,
    tuning: StartupTuning,
    contracts: ContractReport,
    runtime: lite_llm_runtime::RuntimeLifecycle,
    router: lite_llm_runtime::DeterministicRouter,
    distributed: DistributedStack,
    storage: StorageStack,
    security: SecurityStack,
    active_mode: Option<ActiveMode>,
}

impl LiteLlm {
    pub fn bootstrap(config: BootstrapConfig) -> LiteLlmResult<Self> {
        if config.manifest_text.trim().is_empty() {
            return Err(LiteLlmError::InvalidMode("manifest_text must not be empty"));
        }
        if config.node_id.trim().is_empty()
            || config.signer_id.trim().is_empty()
            || config.signing_secret.trim().is_empty()
        {
            return Err(LiteLlmError::InvalidProfile(
                "node_id, signer_id and signing_secret are required",
            ));
        }

        config.distributed.validate()?;
        let contracts = verify_shared_contracts()?;
        let tuning = config.profile.tuning();

        let routing_seed = config.runtime.routing_seed;
        let available_tiers = config.runtime.available_tiers.clone();

        let mut runtime = lite_llm_runtime::RuntimeLifecycle::new(config.runtime)?;
        runtime.boot()?;
        runtime.parse_manifest(&config.manifest_text)?;
        runtime.load_base_parameters()?;
        runtime.register_experts()?;
        runtime.load_router_parameters()?;
        runtime.load_optimizer_state()?;
        runtime.complete_model_load()?;

        let runtime_active = if let Some(initial) = &config.initial_active_tiers {
            lite_llm_runtime::TierSet::new(
                initial
                    .iter()
                    .map(|tier| lite_llm_runtime::TierId::new(tier.0))
                    .collect(),
                false,
            )
        } else {
            runtime.status().active_tiers
        };
        runtime.activate_tiers(runtime_active)?;

        let router =
            lite_llm_runtime::DeterministicRouter::new(routing_seed, available_tiers.clone())
                .with_quantization_scale(tuning.router_quantization_scale);

        let world_size = config.distributed.world_size();
        let consensus_seed = low64(routing_seed.base) ^ (world_size as u64);
        let distributed = DistributedStack {
            parallelism: config.distributed,
            collectives: lite_llm_distributed::DeterministicCollectives::new(world_size)?,
            consensus: lite_llm_distributed::RoutingConsensus::new(
                lite_llm_distributed::ConsensusConfig::new(consensus_seed),
            ),
            transport: lite_llm_distributed::InMemoryTaggedTransport::new(world_size)?,
            recovery: lite_llm_distributed::RecoveryCoordinator::new(
                lite_llm_distributed::RecoveryPolicy {
                    checkpoint_interval_steps: 200,
                    max_retries: 3,
                    base_backoff_millis: 50,
                    heartbeat_timeout_steps: tuning.heartbeat_timeout_steps,
                },
            ),
        };

        let storage = StorageStack {
            snapshots: lite_llm_storage::SnapshotRepository::new(config.snapshot_root)?,
            shard_verifier: lite_llm_storage::Fnv64HashVerifier,
            hot_cache: lite_llm_storage::HotExpertCache::new(
                tuning.cache_capacity_bytes,
                consensus_seed,
            )?,
            training_checkpoint_root: config.training_checkpoint_root,
        };

        let security = build_security_stack(
            &available_tiers,
            config.node_id,
            config.signer_id,
            config.signing_secret,
            tuning,
            consensus_seed,
        )?;

        Ok(Self {
            profile: config.profile,
            tuning,
            contracts,
            runtime,
            router,
            distributed,
            storage,
            security,
            active_mode: None,
        })
    }

    pub fn profile(&self) -> StartupProfile {
        self.profile
    }

    pub fn tuning(&self) -> StartupTuning {
        self.tuning
    }

    pub fn contracts(&self) -> &ContractReport {
        &self.contracts
    }

    pub fn runtime(&self) -> &lite_llm_runtime::RuntimeLifecycle {
        &self.runtime
    }

    pub fn router(&self) -> &lite_llm_runtime::DeterministicRouter {
        &self.router
    }

    pub fn distributed(&self) -> &DistributedStack {
        &self.distributed
    }

    pub fn storage(&self) -> &StorageStack {
        &self.storage
    }

    pub fn security(&self) -> &SecurityStack {
        &self.security
    }

    pub fn active_mode(&self) -> Option<ActiveMode> {
        self.active_mode
    }

    pub fn start_training(&mut self, config: TrainingEntrypoint) -> LiteLlmResult<TrainingHandle> {
        if config.checkpoint_id.trim().is_empty() {
            return Err(LiteLlmError::InvalidMode("checkpoint_id must not be empty"));
        }
        if config.world_size as usize != self.distributed.parallelism.world_size() {
            return Err(LiteLlmError::InvalidMode(
                "training world_size must match distributed world size",
            ));
        }

        ensure_runtime_serving(&mut self.runtime)?;

        let scheduler = lite_llm_training::CurriculumScheduler::new(config.curriculum_plan, 0);
        let accumulation = lite_llm_training::AccumulationState::new(config.accumulation)?;
        let checkpoint_repository = lite_llm_training::DistributedCheckpointRepository::new(
            &self.storage.training_checkpoint_root,
        )?;
        let replay = lite_llm_training::ReplayContext::new(
            config.checkpoint_id,
            config.model_identifier.as_string(),
            config.world_size,
        );

        let audit_sequence = self.security.audit.records().len() as u64;
        self.security.audit.append(lite_llm_security::AuditEvent {
            sequence: audit_sequence,
            timestamp_ms: audit_sequence,
            category: lite_llm_security::AuditCategory::ModelLoad,
            actor: "lite-llm".to_owned(),
            action: format!("start_training:{}", config.optimizer_name),
            payload: "training entrypoint initialized".to_owned(),
        })?;

        self.active_mode = Some(ActiveMode::Training);

        Ok(TrainingHandle {
            scheduler,
            accumulation,
            checkpoint_repository,
            replay,
        })
    }

    pub fn start_inference(
        &mut self,
        config: InferenceEntrypoint,
    ) -> LiteLlmResult<InferenceHandle> {
        let selector = lite_llm_inference::TierSetSelector::new(
            self.tuning.selector_base_latency_ms,
            config.fixed_tiers,
            config.tier_profiles,
        )?;

        let selection = selector.select(&config.selection_request)?;

        let runtime_tiers = lite_llm_runtime::TierSet::new(
            selection
                .selected
                .tiers
                .iter()
                .map(|tier| lite_llm_runtime::TierId::new(*tier))
                .collect(),
            selection.selected.cumulative,
        );
        self.runtime.activate_tiers(runtime_tiers)?;
        ensure_runtime_serving(&mut self.runtime)?;

        let pipeline = lite_llm_inference::DeterministicInferencePipeline {
            enable_compression: self.tuning.enable_pipeline_compression,
        };
        let streaming = lite_llm_inference::StreamingRuntime::new(config.kv_cache)?;

        let audit_sequence = self.security.audit.records().len() as u64;
        self.security.audit.append(lite_llm_security::AuditEvent {
            sequence: audit_sequence,
            timestamp_ms: audit_sequence,
            category: lite_llm_security::AuditCategory::Routing,
            actor: "lite-llm".to_owned(),
            action: "start_inference".to_owned(),
            payload: format!("tiers={:?}", selection.selected.tiers),
        })?;

        self.active_mode = Some(ActiveMode::Inference);

        Ok(InferenceHandle {
            selection,
            selector,
            pipeline,
            streaming,
        })
    }

    pub fn start_replay(&mut self, config: ReplayEntrypoint) -> LiteLlmResult<ReplayHandle> {
        config.expected.verify_against(&config.observed)?;
        let replay_hash = config.expected.replay_hash()?;

        let audit_sequence = self.security.audit.records().len() as u64;
        self.security.audit.append(lite_llm_security::AuditEvent {
            sequence: audit_sequence,
            timestamp_ms: audit_sequence,
            category: lite_llm_security::AuditCategory::Security,
            actor: "lite-llm".to_owned(),
            action: "start_replay".to_owned(),
            payload: replay_hash.clone(),
        })?;

        self.active_mode = Some(ActiveMode::Replay);

        Ok(ReplayHandle {
            event_count: config.expected.events.len(),
            replay_hash,
        })
    }

    pub fn start_recovery(&mut self, config: RecoveryEntrypoint) -> LiteLlmResult<RecoveryHandle> {
        self.runtime.begin_recovery()?;
        let action = self
            .distributed
            .recovery
            .handle_failure(&config.failure_event)?;
        self.runtime
            .restore_after_crash(&config.manifest_text, config.resume_active)?;

        let restored_snapshot = if let Some(snapshot_id) = config.snapshot_id {
            let selected_raw = config
                .selected_tiers
                .map(|tiers| tiers.into_iter().map(|tier| tier.0).collect::<Vec<u16>>());
            Some(self.storage.snapshots.restore_snapshot(
                &snapshot_id,
                selected_raw.as_deref(),
                &self.storage.shard_verifier,
            )?)
        } else {
            None
        };

        let runtime_status = self.runtime.status();
        let audit_sequence = self.security.audit.records().len() as u64;
        self.security.audit.append(lite_llm_security::AuditEvent {
            sequence: audit_sequence,
            timestamp_ms: audit_sequence,
            category: lite_llm_security::AuditCategory::Error,
            actor: "lite-llm".to_owned(),
            action: "start_recovery".to_owned(),
            payload: format!("action={:?}", action),
        })?;

        self.active_mode = Some(ActiveMode::Recovery);

        Ok(RecoveryHandle {
            action,
            runtime_status,
            restored_snapshot,
        })
    }
}

fn ensure_runtime_serving(runtime: &mut lite_llm_runtime::RuntimeLifecycle) -> LiteLlmResult<()> {
    if runtime.status().state != lite_llm_runtime::RuntimeState::Active {
        runtime.start_serving()?;
    }
    Ok(())
}

fn low64(value: u128) -> u64 {
    (value & u64::MAX as u128) as u64
}

fn build_security_stack(
    tiers: &[lite_llm_runtime::TierConfig],
    node_id: String,
    signer_id: String,
    signing_secret: String,
    tuning: StartupTuning,
    seed: u64,
) -> LiteLlmResult<SecurityStack> {
    let mut access = lite_llm_security::AccessController::default();
    access.set_action_roles(
        lite_llm_security::Action::RunInference,
        BTreeSet::from(["inference".to_owned(), "admin".to_owned()]),
    );
    access.set_action_roles(
        lite_llm_security::Action::LoadModel,
        BTreeSet::from(["trainer".to_owned(), "admin".to_owned()]),
    );

    let fallback_tier = tiers.iter().map(|tier| tier.id.0).min().unwrap_or(0);
    for tier in tiers {
        let downgrade = if tier.id.0 == fallback_tier {
            None
        } else {
            Some(fallback_tier)
        };

        access.set_tier_policy(lite_llm_security::TierPolicy {
            tier: tier.id.0,
            allowed_roles: BTreeSet::from([
                "inference".to_owned(),
                "trainer".to_owned(),
                "admin".to_owned(),
            ]),
            allowed_tenants: BTreeSet::from(["default".to_owned()]),
            downgrade_tier: downgrade,
        });
    }

    let audit = lite_llm_security::DeterministicAuditLog::new(node_id, signer_id, signing_secret);

    let mut keys = lite_llm_security::KeyManager::new(lite_llm_security::KeyRotationPolicy {
        rotate_every_days: 30,
        overlap_days: 7,
    })?;
    keys.add_access_policy(lite_llm_security::KeyAccessPolicy {
        identity: "runtime-loader".to_owned(),
        allowed_key_ids: BTreeSet::from(["model-key".to_owned()]),
        allowed_kinds: BTreeSet::from([
            lite_llm_security::KeyKind::Encryption,
            lite_llm_security::KeyKind::Signature,
        ]),
    });
    let _ = keys.generate_key(
        "model-key",
        lite_llm_security::KeyKind::Encryption,
        1,
        0,
        seed,
    )?;

    let mut sandbox = lite_llm_security::SandboxRuntime::default();
    sandbox.configure(lite_llm_security::SandboxConfig {
        allowed_syscalls: BTreeSet::from([
            "read".to_owned(),
            "write".to_owned(),
            "mmap".to_owned(),
            "futex".to_owned(),
        ]),
        max_memory_bytes: if tuning.strict_security {
            256 * 1024 * 1024
        } else {
            1024 * 1024 * 1024
        },
        max_cpu_millis: if tuning.strict_security {
            30_000
        } else {
            120_000
        },
    })?;

    let compliance =
        lite_llm_security::ComplianceEngine::new(lite_llm_security::ComplianceProfile {
            data_minimization: true,
            deletion_requests_supported: true,
            encryption_at_rest: true,
            access_control: true,
            audit_logging: true,
            incident_response_plan: true,
            zeroization: true,
            telemetry_pii_redaction: tuning.strict_security,
        });

    let memory_profile = lite_llm_security::MemorySafetyProfile {
        require_miri: tuning.strict_security,
        require_fuzzing: tuning.strict_security,
        unsafe_policy: if tuning.strict_security {
            lite_llm_security::UnsafeBlockPolicy::AllowWithReview
        } else {
            lite_llm_security::UnsafeBlockPolicy::AllowWithAudit
        },
        max_unsafe_blocks: if tuning.strict_security { 8 } else { 64 },
        require_ffi_layout_validation: tuning.strict_security,
        require_ffi_lifetime_validation: tuning.strict_security,
    };

    Ok(SecurityStack {
        access,
        audit,
        keys,
        sandbox,
        compliance,
        hardening: lite_llm_security::HardeningChecklist::default_items(),
        memory_profile,
    })
}
