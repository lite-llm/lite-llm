use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_bootstrap_config_validation(c: &mut Criterion) {
    use lite_llm::mode::BootstrapConfig;

    let config = BootstrapConfig {
        profile: lite_llm::profile::StartupProfile::default(),
        runtime: lite_llm_runtime::RuntimeOptions::default(),
        manifest_text: "test".into(),
        initial_active_tiers: None,
        distributed: lite_llm_distributed::ParallelismConfig::default(),
        snapshot_root: std::path::PathBuf::from("."),
        training_checkpoint_root: std::path::PathBuf::from("."),
        node_id: "test-node".into(),
        signer_id: "test-signer".into(),
        signing_secret: "test-secret-123".into(),
    };

    c.bench_function("bootstrap_config_validate", |b| {
        b.iter(|| config.validate())
    });
}

fn bench_bootstrap_config_from_env(c: &mut Criterion) {
    c.bench_function("bootstrap_config_from_env", |b| {
        b.iter(|| {
            let config = BootstrapConfig::from_env();
            black_box(config)
        })
    });
}

criterion_group!(
    benches,
    bench_bootstrap_config_validation,
    bench_bootstrap_config_from_env
);
criterion_main!(benches);
