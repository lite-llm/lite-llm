# lite-llm

Orchestrated integration crate for Lite LLM.

## Purpose
`lite-llm` wires the independently versioned crates into one startup and operations API:

- runtime
- distributed
- storage
- training
- inference
- security

## Key Capabilities
- Bootstrap orchestration via `LiteLlm::bootstrap(...)`.
- Shared contract compatibility checks across crates.
- Startup profiles (`Development`, `Deterministic`, `Throughput`, `Recovery`).
- Mode-specific entrypoints:
  - `start_training(...)`
  - `start_inference(...)`
  - `start_replay(...)`
  - `start_recovery(...)`
- SPEC-compliance release gate for `SPEC-001`..`SPEC-060`.

## Public Modules
- `src/orchestrator.rs`: integration state and mode entrypoints
- `src/mode.rs`: bootstrap/mode config and handle types
- `src/contracts.rs`: shared contract drift checks
- `src/types.rs`: canonical cross-crate type conversions
- `src/spec_gate.rs`: release gate matrix and required compliance tests
- `src/profile.rs`: startup tuning profiles
- `src/error.rs`: unified error surface

## Build and Test
```bash
cargo fmt
cargo test
```

## Documentation
- Comprehensive docs: `../lite-llm-docs/README.md`
- Compliance gate details: `../lite-llm-docs/operations/release-gate.md`

## Changelog
See `CHANGELOG.md`.

## License
See `LICENSE`.
