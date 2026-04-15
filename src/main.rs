use clap::{Parser, Subcommand, ValueEnum};
use lite_llm_inference::{InferenceConfig, InferenceEngine};
use lite_llm_storage::StorageBackendConfig;
use lite_llm_training::TrainerConfig;

/// Configuration for the model source (filesystem, S3, or MinIO).
#[derive(Debug, Clone)]
struct ModelSourceConfig {
    /// Source type: "filesystem", "s3", or "minio".
    source: String,
    /// S3 bucket name (used for s3/minio sources).
    bucket: Option<String>,
    /// S3 key prefix / filesystem path prefix.
    prefix: Option<String>,
    /// Custom S3 endpoint URL (for MinIO or S3-compatible services).
    endpoint: Option<String>,
}

impl ModelSourceConfig {
    /// Convert to a `StorageBackendConfig` for the storage layer.
    fn into_backend_config(self) -> Option<StorageBackendConfig> {
        match self.source.as_str() {
            "filesystem" => Some(StorageBackendConfig::filesystem(
                self.prefix.unwrap_or_else(|| "models".to_string()),
            )),
            "s3" | "minio" => {
                let bucket = self.bucket?;
                let prefix = self.prefix.unwrap_or_default();
                if self.source == "minio" {
                    let endpoint = self.endpoint.unwrap_or_default();
                    Some(StorageBackendConfig::minio(endpoint, bucket, prefix))
                } else {
                    Some(StorageBackendConfig::s3(bucket, prefix))
                }
            }
            _ => None,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "lite-llm")]
#[command(about = "Lite LLM - Lightweight Language Model", long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,

    #[arg(short, long, default_value_t = false)]
    verbose: bool,

    // -- Model source flags (S3 / MinIO / filesystem) --

    /// Model source type: filesystem, s3, or minio.
    #[arg(long, default_value = "filesystem", value_name = "SOURCE")]
    model_source: String,

    /// S3 bucket name for model storage (required when --model-source is s3 or minio).
    #[arg(long, value_name = "BUCKET")]
    model_bucket: Option<String>,

    /// S3 key prefix or filesystem path prefix for model files.
    #[arg(long, value_name = "PREFIX")]
    model_prefix: Option<String>,

    /// Custom S3 endpoint URL (e.g., http://localhost:9000 for MinIO).
    #[arg(long, value_name = "URL")]
    model_endpoint: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Generate {
        #[arg(short, long, default_value = "Hello, how are you?")]
        prompt: String,

        #[arg(long, value_enum, default_value = "small")]
        model: ModelSize,

        #[arg(long, default_value_t = 50)]
        max_length: usize,

        #[arg(short, long, default_value_t = 1.0)]
        temperature: f32,

        #[arg(long, default_value_t = 50)]
        top_k: usize,

        #[arg(long, default_value_t = 0.9)]
        top_p: f32,

        #[arg(long)]
        seed: Option<u64>,
    },
    Train {
        #[arg(long, default_value_t = 10)]
        epochs: usize,

        #[arg(long, default_value_t = 4)]
        batch_size: usize,

        #[arg(long, default_value_t = 0.01)]
        learning_rate: f32,

        #[arg(long, default_value_t = 32)]
        seq_length: usize,

        #[arg(long)]
        data: Option<String>,
    },
    Info,
}

#[derive(Debug, Clone, ValueEnum)]
enum ModelSize {
    Small,
    Medium,
    Large,
}

impl ModelSize {
    fn to_string(&self) -> String {
        match self {
            ModelSize::Small => "small",
            ModelSize::Medium => "medium",
            ModelSize::Large => "large",
        }
        .to_string()
    }
}

fn run_generate(
    prompt: &str,
    model: &ModelSize,
    max_length: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: Option<u64>,
    verbose: bool,
) {
    if verbose {
        println!("Lite LLM Inference");
        println!("==================");
        println!("Model: {}", model.to_string());
        println!("Prompt: {}", prompt);
        println!("Max length: {}", max_length);
        println!("Temperature: {}", temperature);
        println!("Top-k: {}", top_k);
        println!("Top-p: {}", top_p);
        if let Some(s) = seed {
            println!("Seed: {}", s);
        }
        println!();
    }

    let config = InferenceConfig {
        model_size: model.to_string(),
        max_length,
        temperature,
        top_k,
        top_p,
        seed,
    };

    print!("Generating...");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    let engine = InferenceEngine::new(config);

    match engine.generate(prompt) {
        Ok(result) => {
            println!("\r           \r");
            println!("Response:");
            println!("{}", result);
        }
        Err(e) => {
            eprintln!("Error: {:?}", e);
            std::process::exit(1);
        }
    }
}

fn run_train(
    epochs: usize,
    batch_size: usize,
    learning_rate: f32,
    seq_length: usize,
    data: Option<String>,
    verbose: bool,
) {
    if verbose {
        println!("Lite LLM Training");
        println!("=================");
        println!("Epochs: {}", epochs);
        println!("Batch size: {}", batch_size);
        println!("Learning rate: {}", learning_rate);
        println!("Seq length: {}", seq_length);
        if let Some(d) = &data {
            println!("Data: {}", d);
        }
        println!();
    }

    let config = TrainerConfig {
        learning_rate,
        batch_size,
        epochs,
        seq_length,
    };

    let trainer = lite_llm_training::Trainer::new(config);

    let sample_texts = vec![
        "hello world".to_string(),
        "the quick brown fox".to_string(),
        "machine learning is fun".to_string(),
    ];

    println!("Training on {} sample texts...", sample_texts.len());
    trainer.train(&sample_texts);
    println!("Training complete!");
}

fn run_info(verbose: bool) {
    println!("Lite LLM - Lightweight Language Model");
    println!("======================================");
    println!();
    println!("A minimal transformer-based language model implementation.");
    println!();
    println!("Commands:");
    println!("  generate   - Generate text from a prompt");
    println!("  train      - Train the model on text data");
    println!("  info       - Show model information");
    println!();
    println!("Examples:");
    println!("  lite-llm generate --prompt 'Hello world'");
    println!("  lite-llm train --epochs 10 --batch-size 4");
    println!("  lite-llm info");

    if verbose {
        println!();
        println!("Verbose mode enabled");
    }
}

fn main() {
    let args = Args::parse();

    // Build model source configuration from CLI flags.
    let model_source_config = ModelSourceConfig {
        source: args.model_source.clone(),
        bucket: args.model_bucket.clone(),
        prefix: args.model_prefix.clone(),
        endpoint: args.model_endpoint.clone(),
    };

    if args.verbose {
        eprintln!("Model source: {}", model_source_config.source);
        if let Some(ref bucket) = model_source_config.bucket {
            eprintln!("Model bucket: {}", bucket);
        }
        if let Some(ref prefix) = model_source_config.prefix {
            eprintln!("Model prefix: {}", prefix);
        }
        if let Some(ref endpoint) = model_source_config.endpoint {
            eprintln!("Model endpoint: {}", endpoint);
        }
    }

    let backend_config = model_source_config.clone().into_backend_config();
    if args.verbose {
        if let Some(ref cfg) = backend_config {
            eprintln!("Storage backend type: {}", cfg.backend_type);
        }
    }

    match args.command {
        Some(Commands::Generate {
            prompt,
            model,
            max_length,
            temperature,
            top_k,
            top_p,
            seed,
        }) => {
            run_generate(
                &prompt,
                &model,
                max_length,
                temperature,
                top_k,
                top_p,
                seed,
                args.verbose,
            );
        }
        Some(Commands::Train {
            epochs,
            batch_size,
            learning_rate,
            seq_length,
            data,
        }) => {
            run_train(
                epochs,
                batch_size,
                learning_rate,
                seq_length,
                data,
                args.verbose,
            );
        }
        Some(Commands::Info) => {
            run_info(args.verbose);
        }
        None => {
            println!("Lite LLM - Lightweight Language Model");
            println!("======================================");
            println!();
            println!("Usage: lite-llm <COMMAND>");
            println!();
            println!("Commands:");
            println!("  generate   - Generate text from a prompt");
            println!("  train      - Train the model on text data");
            println!("  info       - Show model information");
            println!();
            println!("Model Source Options:");
            println!("  --model-source filesystem  (default, local disk)");
            println!("  --model-source s3          (AWS S3 or compatible)");
            println!("  --model-source minio       (MinIO with custom endpoint)");
            println!();
            println!("S3/MinIO Options:");
            println!("  --model-bucket BUCKET      S3 bucket name");
            println!("  --model-prefix PREFIX      S3 key prefix or path");
            println!("  --model-endpoint URL       Custom endpoint (for MinIO)");
            println!();
            println!("Examples:");
            println!("  lite-llm generate --prompt 'Hello world'");
            println!("  lite-llm generate --model-source s3 --model-bucket my-models --model-prefix checkpoints/");
            println!("  lite-llm generate --model-source minio --model-bucket models --model-prefix v1/ --model-endpoint http://localhost:9000");
            println!("  lite-llm train --epochs 10 --batch-size 4");
            println!("  lite-llm info");
        }
    }
}
