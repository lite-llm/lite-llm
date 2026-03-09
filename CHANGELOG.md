# Changelog

All notable changes to `lite-llm` are documented in this file.

## [0.1.0] - 2026-03-01
### Added
- Orchestrated integration API wiring runtime, distributed, storage, training, inference, and security crates.
- Startup profiles with mode-aware tuning (`Development`, `Deterministic`, `Throughput`, `Recovery`).
- Mode-specific entrypoints for training, inference, replay, and recovery.
- Shared type compatibility adapters and contract drift checks.
- Release gate implementation for `SPEC-001` to `SPEC-060`.
- Determinism, security, and recovery gate tests.

### Notes
- `SPEC-061` to `SPEC-075` are intentionally excluded from required release-gate coverage.
