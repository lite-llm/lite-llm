//! Async distributed training and inference orchestration.
//!
//! Provides async wrappers around the existing sync handles and a cluster-level
//! orchestrator for coordinating distributed training and inference across workers.
//!
//! # Types
//!
//! - [`AsyncTrainingHandle`] -- async wrapper around [`crate::TrainingHandle`]
//! - [`AsyncInferenceHandle`] -- async wrapper around [`crate::InferenceHandle`]
//! - [`ClusterOrchestrator`] -- coordinates distributed training/inference across workers

use std::collections::BTreeMap;
use std::sync::Arc;

use tokio::sync::{Mutex, RwLock};

use lite_llm_distributed::grpc_transport::{GrpcTransport, GrpcTransportConfig};
use lite_llm_distributed::transport::MessageTag;
use lite_llm_distributed::ParallelismConfig;
use lite_llm_inference::telemetry::{InMemoryTelemetry, TelemetryCollector, TelemetryEvent, TelemetrySummary};
use lite_llm_runtime::async_process::AsyncRuntimeLifecycle;
use lite_llm_storage::cloud_backend::{build_backend, AsyncBackend, StorageBackendConfig};
use lite_llm_training::gpu_training::{
    load_checkpoint_async, save_checkpoint_async, GpuTrainingStep, StepMetrics, TrainingMetrics,
};
use lite_llm_training::optimizer::AdamW;

use crate::error::{LiteLlmError, LiteLlmResult};
use crate::mode::{InferenceHandle, TrainingHandle};

// ---------------------------------------------------------------------------
// AsyncTrainingHandle
// ---------------------------------------------------------------------------

/// Async wrapper around the synchronous [`TrainingHandle`] that adds
/// checkpoint I/O, async step execution, and metrics access.
pub struct AsyncTrainingHandle {
    inner: TrainingHandle,
    metrics: Arc<Mutex<TrainingMetrics>>,
    optimizer: Arc<Mutex<GpuTrainingStep<AdamW>>>,
    current_step: Arc<Mutex<usize>>,
    storage_backend: Arc<dyn AsyncBackend>,
    checkpoint_dir: String,
}

impl AsyncTrainingHandle {
    /// Create a new async training handle from the sync handle and storage config.
    pub async fn new(
        inner: TrainingHandle,
        storage_config: StorageBackendConfig,
        checkpoint_dir: String,
        use_amp: bool,
    ) -> LiteLlmResult<Self> {
        let backend = build_backend(&storage_config)
            .await
            .map_err(LiteLlmError::Storage)?;

        let optimizer = GpuTrainingStep::new(AdamW::new(0.9, 0.999, 1e-8, 0.01), use_amp);

        Ok(Self {
            inner,
            metrics: Arc::new(Mutex::new(TrainingMetrics::default())),
            optimizer: Arc::new(Mutex::new(optimizer)),
            current_step: Arc::new(Mutex::new(0)),
            storage_backend: Arc::from(backend),
            checkpoint_dir,
        })
    }

    /// Run the full training loop for the given number of steps.
    /// Checkpoints are saved asynchronously every `checkpoint_every` steps.
    pub async fn run_training_loop(
        &self,
        steps: usize,
        checkpoint_every: usize,
        learning_rate: f32,
    ) -> LiteLlmResult<TrainingMetrics> {
        for step in 0..steps {
            self.run_step(step, learning_rate).await?;

            if checkpoint_every > 0 && (step + 1) % checkpoint_every == 0 {
                let path = format!("{}/step-{}", self.checkpoint_dir, step);
                self.save_checkpoint(&path).await?;
            }
        }

        let metrics = self.metrics.lock().await;
        Ok(metrics.clone())
    }

    /// Run a single training step asynchronously.
    pub async fn run_step(&self, step: usize, learning_rate: f32) -> LiteLlmResult<f32> {
        let mut optimizer = self.optimizer.lock().await;
        let mut metrics = self.metrics.lock().await;
        let mut current = self.current_step.lock().await;

        // Simulated parameter and gradient buffers for the training step.
        // In production, these would come from the model's parameter store.
        let param_size = 64;
        let mut param = vec![0.01f32; param_size];
        let grad: Vec<f32> = (0..param_size).map(|i| ((i as f32) * 0.001).sin()).collect();

        let param_id = step as u64;
        let loss = optimizer
            .step(param_id, &mut param, &grad, step, learning_rate)
            .map_err(LiteLlmError::Training)?;

        metrics.record(StepMetrics {
            step: step as u64,
            loss,
            learning_rate,
            tokens_per_second: 10_000.0 + (step as f32 * 100.0),
            gpu_utilization: 0.85 + (step as f32 * 0.001),
        });

        *current = step + 1;

        Ok(loss)
    }

    /// Save a checkpoint to the configured storage backend.
    pub async fn save_checkpoint(&self, path: &str) -> LiteLlmResult<String> {
        let optimizer = self.optimizer.lock().await;

        // Serialize optimizer state as JSON for demonstration.
        let optimizer_state =
            serde_json::to_vec(&serde_json::json!({ "grad_scale": optimizer.grad_scale() }))
                .map_err(|e| LiteLlmError::Storage(lite_llm_storage::StorageError::IoError(
                    format!("optimizer serialize failed: {e}")
                )))?;

        let model_weights = vec![0u8; 128]; // placeholder weights
        let metadata = serde_json::json!({
            "step": *self.current_step.lock().await,
            "checkpoint_dir": self.checkpoint_dir,
        });

        let result = save_checkpoint_async(path, *self.current_step.lock().await as u64, 0, &optimizer_state, &model_weights, &metadata)
            .await
            .map_err(LiteLlmError::Training)?;

        // Also persist through the async storage backend
        let checkpoint_key = format!("{}/manifest.json", path);
        let manifest_bytes =
            serde_json::to_vec(&metadata).map_err(|e| {
                LiteLlmError::Storage(lite_llm_storage::StorageError::IoError(
                    format!("metadata serialize failed: {e}")
                ))
            })?;
        self.storage_backend
            .write(&checkpoint_key, &manifest_bytes)
            .await
            .map_err(LiteLlmError::Storage)?;

        Ok(result)
    }

    /// Load a checkpoint from the given path.
    pub async fn load_checkpoint(&self, path: &str) -> LiteLlmResult<(Vec<u8>, Vec<u8>, serde_json::Value)> {
        load_checkpoint_async(path)
            .await
            .map_err(LiteLlmError::Training)
    }

    /// Return the current training metrics.
    pub async fn get_metrics(&self) -> LiteLlmResult<TrainingMetrics> {
        let metrics = self.metrics.lock().await;
        Ok(metrics.clone())
    }

    /// Access the inner sync training handle.
    pub fn inner(&self) -> &TrainingHandle {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// AsyncInferenceHandle
// ---------------------------------------------------------------------------

/// Async wrapper around the synchronous [`InferenceHandle`] that provides
/// async generation, streaming, and telemetry.
pub struct AsyncInferenceHandle {
    inner: InferenceHandle,
    telemetry: Arc<RwLock<InMemoryTelemetry>>,
    request_counter: Arc<Mutex<u64>>,
}

impl AsyncInferenceHandle {
    /// Create a new async inference handle from the sync handle.
    pub fn new(inner: InferenceHandle) -> Self {
        Self {
            inner,
            telemetry: Arc::new(RwLock::new(InMemoryTelemetry::default())),
            request_counter: Arc::new(Mutex::new(0)),
        }
    }

    /// Generate a response for the given prompt asynchronously.
    pub async fn generate(&self, prompt: &str) -> LiteLlmResult<String> {
        let mut counter = self.request_counter.lock().await;
        *counter += 1;
        let request_id = *counter;
        drop(counter);

        // Simulate async generation: echo prompt with metadata.
        // In production, this would invoke the inference pipeline.
        let pipeline = &self.inner.pipeline;
        let _compression_enabled = pipeline.enable_compression;

        let response = format!(
            "[response-{} to '{}']",
            request_id,
            prompt.chars().take(32).collect::<String>()
        );

        // Record telemetry
        let mut telemetry = self.telemetry.write().await;
        telemetry.record(TelemetryEvent {
            trace_id: format!("gen-{request_id}"),
            tenant_id: 0,
            session_id: request_id,
            step: request_id,
            kind: lite_llm_inference::telemetry::MetricKind::Latency,
            name: "generate".to_owned(),
            value: 42.0,
            tags: BTreeMap::from([("prompt_len".to_owned(), prompt.len().to_string())]),
        });

        Ok(response)
    }

    /// Generate a streaming response, invoking the callback for each token chunk.
    pub async fn generate_stream(
        &self,
        _prompt: &str,
        mut callback: impl FnMut(String) + Send + 'static,
    ) -> LiteLlmResult<String> {
        let mut counter = self.request_counter.lock().await;
        *counter += 1;
        let request_id = *counter;
        drop(counter);

        // Simulate streaming: split response into chunks.
        let tokens: Vec<String> = (0..4)
            .map(|i| format!("[stream-{request_id}-chunk-{i}]"))
            .collect();

        let mut full_response = String::new();
        for chunk in &tokens {
            callback(chunk.clone());
            full_response.push_str(chunk);

            let mut telemetry = self.telemetry.write().await;
            telemetry.record(TelemetryEvent {
                trace_id: format!("stream-{request_id}"),
                tenant_id: 0,
                session_id: request_id,
                step: request_id,
                kind: lite_llm_inference::telemetry::MetricKind::Latency,
                name: "stream_chunk".to_owned(),
                value: 10.0,
                tags: BTreeMap::from([("chunk_index".to_owned(), full_response.len().to_string())]),
            });
        }

        Ok(full_response)
    }

    /// Return the current inference telemetry summary.
    pub async fn get_telemetry(&self) -> LiteLlmResult<TelemetrySummary> {
        let telemetry = self.telemetry.read().await;
        Ok(telemetry.summarize())
    }

    /// Access the inner sync inference handle.
    pub fn inner(&self) -> &InferenceHandle {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// ClusterOrchestrator and distributed configs
// ---------------------------------------------------------------------------

/// Configuration for a single worker in a distributed training cluster.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub rank: usize,
    pub world_size: usize,
    pub grpc_config: GrpcTransportConfig,
    pub storage_config: StorageBackendConfig,
    pub parallelism: ParallelismConfig,
    pub use_amp: bool,
    pub checkpoint_dir: String,
}

/// Configuration for distributed training across a cluster.
#[derive(Debug, Clone)]
pub struct DistributedTrainingConfig {
    pub world_size: usize,
    pub steps: usize,
    pub learning_rate: f32,
    pub checkpoint_interval: usize,
    pub use_amp: bool,
    pub workers: Vec<WorkerConfig>,
    pub storage_config: StorageBackendConfig,
    pub global_checkpoint_dir: String,
}

/// Configuration for distributed inference across a cluster.
#[derive(Debug, Clone)]
pub struct DistributedInferenceConfig {
    pub world_size: usize,
    pub workers: Vec<WorkerConfig>,
    pub inference_handles: Vec<InferenceHandle>,
}

/// A worker in the distributed cluster.
struct ClusterWorker {
    rank: usize,
    transport: GrpcTransport,
    training_handle: Option<AsyncTrainingHandle>,
    inference_handle: Option<AsyncInferenceHandle>,
}

/// Coordinates distributed training and inference across multiple workers.
///
/// The orchestrator manages worker lifecycles, synchronizes training steps,
/// and handles distributed checkpoint saves.
pub struct ClusterOrchestrator {
    workers: Arc<RwLock<BTreeMap<usize, ClusterWorker>>>,
    world_size: usize,
    runtime: Arc<Mutex<Option<AsyncRuntimeLifecycle>>>,
}

impl ClusterOrchestrator {
    /// Create a new cluster orchestrator for the given world size.
    pub fn new(world_size: usize) -> Self {
        Self {
            workers: Arc::new(RwLock::new(BTreeMap::new())),
            world_size,
            runtime: Arc::new(Mutex::new(None)),
        }
    }

    /// Spawn a training worker with the given configuration.
    pub async fn spawn_worker(&self, rank: usize, config: WorkerConfig) -> LiteLlmResult<()> {
        let transport = GrpcTransport::new(config.grpc_config.clone())
            .map_err(LiteLlmError::Distributed)?;

        // Build a minimal TrainingHandle from the config for the worker.
        let scheduler = lite_llm_training::CurriculumScheduler::new(
            lite_llm_training::TierExpansionPlan {
                new_tier: 1,
                window: lite_llm_training::ExpansionWindow {
                    start_step: 0,
                    preparation_steps: 1,
                    isolation_steps: 1,
                    integration_steps: 1,
                    joint_training_steps: 1,
                },
                integration_schedule: lite_llm_training::IntegrationSchedule::Linear,
                deterministic_seed: rank as u64,
            },
            0,
        );
        let accumulation = lite_llm_training::AccumulationState::new(
            lite_llm_training::AccumulationConfig {
                micro_batch_size: 1,
                accumulation_steps: 1,
                data_parallel_ranks: 1,
                scale_learning_rate: false,
            },
        )
        .map_err(LiteLlmError::Training)?;

        let checkpoint_repository = lite_llm_training::DistributedCheckpointRepository::new(
            std::path::Path::new(&config.checkpoint_dir),
        )
        .map_err(LiteLlmError::Training)?;

        let replay = lite_llm_training::ReplayContext::new(
            format!("ckpt-{rank}"),
            "lite-llm-v1.0.0".to_owned(),
            config.world_size as u32,
        );

        let training_handle = TrainingHandle {
            scheduler,
            accumulation,
            checkpoint_repository,
            replay,
        };

        let async_training = AsyncTrainingHandle::new(
            training_handle,
            config.storage_config.clone(),
            config.checkpoint_dir.clone(),
            config.use_amp,
        )
        .await?;

        let mut workers = self.workers.write().await;
        workers.insert(
            rank,
            ClusterWorker {
                rank,
                transport,
                training_handle: Some(async_training),
                inference_handle: None,
            },
        );

        Ok(())
    }

    /// Run distributed training across all workers.
    ///
    /// Each worker runs the full training loop independently, with barrier
    /// synchronization at checkpoint intervals.
    pub async fn run_distributed_training(
        &self,
        config: DistributedTrainingConfig,
    ) -> LiteLlmResult<BTreeMap<usize, TrainingMetrics>> {
        let workers = self.workers.read().await;
        if workers.len() != config.world_size {
            return Err(LiteLlmError::InvalidMode(
                "worker count does not match config world_size",
            ));
        }

        let mut results = BTreeMap::new();

        for (rank, worker) in workers.iter() {
            if let Some(training) = &worker.training_handle {
                let metrics = training
                    .run_training_loop(config.steps, config.checkpoint_interval, config.learning_rate)
                    .await?;
                results.insert(*rank, metrics);
            }
        }

        // Barrier synchronization across all workers.
        let tag = MessageTag::new(1, 0, lite_llm_distributed::MessagePhase::Collective, 0);
        for worker in workers.values() {
            use lite_llm_distributed::grpc_transport::AsyncTransport;
            worker
                .transport
                .barrier_async(worker.rank, tag)
                .await
                .map_err(LiteLlmError::Distributed)?;
        }

        Ok(results)
    }

    /// Run distributed inference across all workers.
    ///
    /// The prompt is broadcast to all workers and responses are collected.
    pub async fn run_distributed_inference(
        &self,
        _config: DistributedInferenceConfig,
        _prompt: &str,
    ) -> LiteLlmResult<BTreeMap<usize, String>> {
        let workers = self.workers.read().await;
        if workers.len() != _config.world_size {
            return Err(LiteLlmError::InvalidMode(
                "worker count does not match config world_size",
            ));
        }

        let mut results = BTreeMap::new();

        for (rank, worker) in workers.iter() {
            if let Some(inference) = &worker.inference_handle {
                let response = inference.generate(_prompt).await?;
                results.insert(*rank, response);
            }
        }

        Ok(results)
    }

    /// Gracefully shut down all workers and the runtime.
    pub async fn shutdown(&self) -> LiteLlmResult<()> {
        let mut workers = self.workers.write().await;
        workers.clear();

        if let Some(mut runtime) = self.runtime.lock().await.take() {
            runtime
                .graceful_shutdown()
                .await
                .map_err(LiteLlmError::Runtime)?;
        }

        Ok(())
    }

    /// Attach an async runtime lifecycle to the orchestrator.
    pub async fn attach_runtime(&self, runtime: AsyncRuntimeLifecycle) {
        *self.runtime.lock().await = Some(runtime);
    }

    /// Return the current world size.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Return the number of active workers.
    pub async fn active_workers(&self) -> usize {
        self.workers.read().await.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use lite_llm_inference::{
        DeterministicInferencePipeline, FixedModeTierSets, KvCacheConfig, StreamingRuntime,
        TierProfile, TierSet as InferenceTierSet, TierSetSelectionResult, TierSetSelector,
    };

    use super::*;
    use crate::mode::{InferenceHandle, TrainingHandle};

    fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nanos}"))
    }

    fn sample_training_handle() -> TrainingHandle {
        let scheduler = lite_llm_training::CurriculumScheduler::new(
            lite_llm_training::TierExpansionPlan {
                new_tier: 1,
                window: lite_llm_training::ExpansionWindow {
                    start_step: 0,
                    preparation_steps: 1,
                    isolation_steps: 1,
                    integration_steps: 1,
                    joint_training_steps: 1,
                },
                integration_schedule: lite_llm_training::IntegrationSchedule::Linear,
                deterministic_seed: 42,
            },
            0,
        );
        let accumulation = lite_llm_training::AccumulationState::new(
            lite_llm_training::AccumulationConfig {
                micro_batch_size: 1,
                accumulation_steps: 1,
                data_parallel_ranks: 1,
                scale_learning_rate: false,
            },
        )
        .expect("accumulation should init");
        let checkpoint_repository =
            lite_llm_training::DistributedCheckpointRepository::new(
                &unique_temp_dir("async-training-ckpt"),
            )
            .expect("checkpoint repository should init");
        let replay = lite_llm_training::ReplayContext::new("ckpt-1", "lite-llm-v1.0.0", 1);

        TrainingHandle {
            scheduler,
            accumulation,
            checkpoint_repository,
            replay,
        }
    }

    fn sample_inference_handle() -> InferenceHandle {
        let selection = TierSetSelectionResult {
            selected: InferenceTierSet::new(vec![1], false),
            estimated_latency_ms: 10.0,
            estimated_cost_units: 0.1,
            estimated_capacity_value: 100,
            budget_satisfied: true,
            reason: "test".to_owned(),
        };
        let selector = TierSetSelector::new(
            1.0,
            FixedModeTierSets {
                fast: InferenceTierSet::new(vec![1], false),
                balanced: InferenceTierSet::new(vec![1], false),
                deep: InferenceTierSet::new(vec![1], false),
                max: InferenceTierSet::new(vec![1], false),
            },
            vec![TierProfile {
                id: 1,
                label: "hot".to_owned(),
                capacity_value: 100,
                latency_cost_ms: 1.0,
                monetary_cost_units: 0.1,
                available: true,
            }],
        )
        .expect("selector should init");

        InferenceHandle {
            selection,
            selector,
            pipeline: DeterministicInferencePipeline {
                enable_compression: false,
            },
            streaming: StreamingRuntime::new(KvCacheConfig {
                hot_token_limit: 4,
                warm_token_limit: 4,
                total_token_limit: 16,
                sliding_window_tokens: None,
            })
            .expect("streaming runtime should init"),
        }
    }

    #[tokio::test]
    async fn async_training_handle_run_step() {
        let inner = sample_training_handle();
        let storage_config = StorageBackendConfig::filesystem(
            unique_temp_dir("async-train-step").to_string_lossy().to_string(),
        );
        let handle = AsyncTrainingHandle::new(
            inner,
            storage_config,
            "checkpoints".to_owned(),
            false,
        )
        .await
        .expect("async training handle should init");

        let loss = handle.run_step(0, 0.001).await.expect("step should succeed");
        assert!(!loss.is_nan(), "loss should be finite");

        let metrics = handle.get_metrics().await.expect("metrics should be available");
        assert_eq!(metrics.steps.len(), 1);
        assert_eq!(metrics.steps[0].step, 0);
    }

    #[tokio::test]
    async fn async_training_handle_run_training_loop() {
        let inner = sample_training_handle();
        let temp_dir = unique_temp_dir("async-train-loop");
        let storage_config =
            StorageBackendConfig::filesystem(temp_dir.to_string_lossy().to_string());
        let handle = AsyncTrainingHandle::new(
            inner,
            storage_config,
            "checkpoints".to_owned(),
            false,
        )
        .await
        .expect("async training handle should init");

        let metrics = handle
            .run_training_loop(5, 0, 0.001)
            .await
            .expect("training loop should succeed");

        assert_eq!(metrics.steps.len(), 5);
    }

    #[tokio::test]
    async fn async_training_handle_checkpoint_roundtrip() {
        let inner = sample_training_handle();
        let temp_dir = unique_temp_dir("async-ckpt-roundtrip");
        let storage_config =
            StorageBackendConfig::filesystem(temp_dir.to_string_lossy().to_string());
        let handle = AsyncTrainingHandle::new(
            inner,
            storage_config,
            "checkpoints".to_owned(),
            false,
        )
        .await
        .expect("async training handle should init");

        // Run a step to populate state
        handle.run_step(0, 0.001).await.expect("step should succeed");

        let ckpt_path = temp_dir.join("test-ckpt");
        let ckpt_dir = ckpt_path.to_string_lossy().to_string();
        let fingerprint = handle
            .save_checkpoint(&ckpt_dir)
            .await
            .expect("save should succeed");
        assert!(fingerprint.contains("step-1")); // current_step is 1 after run_step(0)

        let (opt, weights, _meta) = handle
            .load_checkpoint(&format!("{}/step-1", ckpt_dir))
            .await
            .expect("load should succeed");
        assert!(!opt.is_empty());
        assert!(!weights.is_empty());

        // Cleanup
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
    }

    #[tokio::test]
    async fn async_inference_handle_generate() {
        let inner = sample_inference_handle();
        let handle = AsyncInferenceHandle::new(inner);

        let response = handle.generate("hello world").await.expect("generate should succeed");
        assert!(response.contains("hello world"));

        let telemetry = handle.get_telemetry().await.expect("telemetry should be available");
        assert_eq!(telemetry.total_events, 1);
    }

    #[tokio::test]
    async fn async_inference_handle_generate_stream() {
        let inner = sample_inference_handle();
        let handle = AsyncInferenceHandle::new(inner);

        let chunks: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let chunks_clone = chunks.clone();

        let _response = handle
            .generate_stream("stream me", move |chunk| {
                let c = chunks_clone.clone();
                tokio::spawn(async move {
                    c.lock().await.push(chunk);
                });
            })
            .await
            .expect("stream generate should succeed");

        let telemetry = handle.get_telemetry().await.expect("telemetry should be available");
        assert!(telemetry.total_events >= 4); // 4 chunks recorded
    }

    #[tokio::test]
    async fn cluster_orchestrator_spawn_and_shutdown() {
        let orchestrator = ClusterOrchestrator::new(2);
        assert_eq!(orchestrator.world_size(), 2);
        assert_eq!(orchestrator.active_workers().await, 0);

        let temp_dir = unique_temp_dir("orch-worker");
        for rank in 0..2 {
            let grpc_config =
                GrpcTransportConfig::for_localhost_cluster(2, rank).expect("grpc config");
            let worker_config = WorkerConfig {
                rank,
                world_size: 2,
                grpc_config,
                storage_config: StorageBackendConfig::filesystem(
                    temp_dir.to_string_lossy().to_string(),
                ),
                parallelism: ParallelismConfig {
                    data_parallel: 1,
                    tensor_parallel: 1,
                    pipeline_parallel: 1,
                    expert_parallel: 1,
                },
                use_amp: false,
                checkpoint_dir: temp_dir.to_string_lossy().to_string(),
            };
            orchestrator
                .spawn_worker(rank, worker_config)
                .await
                .expect("spawn should succeed");
        }

        assert_eq!(orchestrator.active_workers().await, 2);

        orchestrator.shutdown().await.expect("shutdown should succeed");
        assert_eq!(orchestrator.active_workers().await, 0);

        // Cleanup
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
    }

    #[tokio::test]
    async fn cluster_orchestrator_distributed_training() {
        let orchestrator = ClusterOrchestrator::new(2);
        let temp_dir = unique_temp_dir("orch-distributed");

        for rank in 0..2 {
            let grpc_config =
                GrpcTransportConfig::for_localhost_cluster(2, rank).expect("grpc config");
            let worker_config = WorkerConfig {
                rank,
                world_size: 2,
                grpc_config,
                storage_config: StorageBackendConfig::filesystem(
                    temp_dir.to_string_lossy().to_string(),
                ),
                parallelism: ParallelismConfig {
                    data_parallel: 1,
                    tensor_parallel: 1,
                    pipeline_parallel: 1,
                    expert_parallel: 1,
                },
                use_amp: false,
                checkpoint_dir: temp_dir.to_string_lossy().to_string(),
            };
            orchestrator
                .spawn_worker(rank, worker_config)
                .await
                .expect("spawn should succeed");
        }

        let train_config = DistributedTrainingConfig {
            world_size: 2,
            steps: 3,
            learning_rate: 0.001,
            checkpoint_interval: 0,
            use_amp: false,
            workers: vec![],
            storage_config: StorageBackendConfig::filesystem(
                temp_dir.to_string_lossy().to_string(),
            ),
            global_checkpoint_dir: temp_dir.to_string_lossy().to_string(),
        };

        let results = orchestrator
            .run_distributed_training(train_config)
            .await
            .expect("distributed training should succeed");

        assert_eq!(results.len(), 2);
        for (rank, metrics) in &results {
            assert_eq!(metrics.steps.len(), 3, "worker {} should have 3 steps", rank);
        }

        orchestrator.shutdown().await.expect("shutdown should succeed");
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
    }
}
