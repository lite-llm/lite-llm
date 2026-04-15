# lite-llm

Orchestrator integration crate for Lite LLM — the top-level API that wires all component crates together.

## Overview
Serves as the integration layer that bootstraps and coordinates all lite-llm component crates (runtime, distributed, storage, training, inference, security) into a cohesive system with mode-specific entrypoints and release compliance gates.

This crate provides: shared compatibility types with cross-crate conversions, shared-contract drift checks for tier/codec/hash parity, bootstrap orchestration for runtime + distributed + storage + security stacks, startup profiles for deterministic/throughput/recovery tuning, mode APIs for training/inference/replay/recovery, async distributed training and inference orchestration with cluster coordination, and SPEC-001..060 release gate tests for compliance validation.

## Features

### Feature Flag: `default` (all features enabled)
No optional feature flags. All component crates are always included.

## Dependencies
| Crate | Version | Purpose |
|-------|---------|---------|
| lite-llm-runtime | 0.1.0 (path) | Core runtime lifecycle and routing |
| lite-llm-distributed | 0.1.0 (path) | Distributed execution and gRPC transport |
| lite-llm-storage | 0.1.0 (path) | Tiered storage and cloud backends |
| lite-llm-training | 0.1.0 (path) | Training runtime and GPU ops |
| lite-llm-inference | 0.1.0 (path) | Inference runtime and GPU backend |
| lite-llm-security | 0.1.0 (path) | Security controls and encryption |
| clap | 4.0 | CLI argument parsing |
| async-trait | 0.1 | Async trait support |
| serde / serde_json | 1.0 | Serialization |
| tokio | 1 | Async runtime |

## Key Modules
- `orchestrator` — `LiteLlm` bootstrap with runtime, distributed, storage, security stacks
- `mode` — mode entrypoints: training, inference, replay, recovery with handles
- `async_training` — async training/inference handles, cluster orchestrator
- `profile` — startup profiles (Deterministic, Throughput, Recovery)
- `contracts` — shared-contract drift checks (TierId width, codec/hash parity)
- `spec_gate` — SPEC-001..060 release compliance gate and compliance matrix
- `types` — top-level type re-exports and TierSet factory functions
- `error` — unified error model across all crates

## Public API
### Core Types
- `LiteLlm` — top-level orchestr with distributed, storage, security stack access
- `BootstrapConfig` — full bootstrap configuration with runtime, distributed, manifest, paths
- `DistributedStack` / `StorageStack` / `SecurityStack` — component stack references
- `ActiveMode` — active mode enumeration (Training, Inference, Replay, Recovery)
- `TrainingEntrypoint` / `TrainingHandle` — training mode entry and handle
- `InferenceEntrypoint` / `InferenceHandle` — inference mode entry and handle
- `ReplayEntrypoint` / `ReplayHandle` — replay mode entry and handle
- `RecoveryEntrypoint` / `RecoveryHandle` — recovery mode entry and handle
- `AsyncTrainingHandle` — async wrapper with checkpoint I/O and metrics
- `AsyncInferenceHandle` — async wrapper with streaming and telemetry
- `ClusterOrchestrator` — coordinates distributed training/inference across workers
- `DistributedTrainingConfig` / `DistributedInferenceConfig` — distributed configs
- `WorkerConfig` — per-worker configuration with gRPC, storage, AMP settings
- `StartupProfile` / `StartupTuning` — startup profile tuning parameters
- `ContractReport` — shared-contract drift check results
- `SpecComplianceEntry` — per-spec compliance entry
- `TierId` / `ExpertKey` — top-level shared types

### Core Functions
- `LiteLlm::bootstrap()` — full bootstrap of all stacks
- `spec_compliance_matrix()` — return compliance matrix for all specs
- `verify_shared_contracts()` — cross-crate contract drift checks
- `inference_tierset()` / `runtime_tierset()` — TierSet factory functions

## Quick Start
```rust
use lite_llm::{
    LiteLlm, BootstrapConfig, StartupProfile,
    TrainingEntrypoint, InferenceEntrypoint,
};
use lite_llm_runtime::{RuntimeOptions, RoutingSeed, TierConfig, TierId, Placement};
use lite_llm_distributed::ParallelismConfig;

// Configure bootstrap
let config = BootstrapConfig {
    profile: StartupProfile::Deterministic,
    runtime: RuntimeOptions {
        routing_seed: RoutingSeed::new(42),
        available_tiers: vec![
            TierConfig { id: TierId(1), groups: 4, experts_per_group: 4, placement: Placement::Hot },
        ],
        expected_manifest_version: 1,
        training_mode: false,
    },
    manifest_text: "version=1\ntiers=1\ncumulative=false\n\
                    base_checksum=abc\nrouter_checksum=def\nshard=base|aa|1024\n",
    initial_active_tiers: Some(vec![TierId(1)]),
    distributed: ParallelismConfig { data_parallel: 1, tensor_parallel: 1, pipeline_parallel: 1, expert_parallel: 1 },
    snapshot_root: "/tmp/snapshots".into(),
    training_checkpoint_root: "/tmp/checkpoints".into(),
    node_id: "node-1".into(),
    signer_id: "signer-1".into(),
    signing_secret: "secret-1".into(),
};

// Bootstrap the system
let mut llm = LiteLlm::bootstrap(config)?;

// Check contracts
assert!(llm.contracts().is_compatible());

// Start inference mode
let handle = llm.start_inference(InferenceEntrypoint { ... })?;
```

## Building
```bash
cargo build --release
```

## Usage
```bash
# Generate text
cargo run -- generate --prompt "Hello world" --max-length 100 --temperature 0.7

# Train model
cargo run -- train --epochs 10 --batch-size 4 --learning-rate 0.01

# Show info
cargo run -- info
```

## Running Tests
```bash
cargo fmt
cargo test
```

## Architecture
This crate is the integration layer for the entire lite-llm platform. It bootstraps all component crates in the correct order: runtime → distributed → storage → security → training/inference. The async training module (`async_training`) provides distributed training orchestration with `ClusterOrchestrator` managing worker lifecycles, barrier synchronization, and distributed checkpoint persistence. Release gate tests in `spec_gate.rs` validate SPEC-001..060 compliance before every release.

## License
DOSL-IIE-1.0
