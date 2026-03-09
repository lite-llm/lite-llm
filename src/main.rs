use clap::{Parser, Subcommand, ValueEnum};
use lite_llm_inference::{InferenceConfig, InferenceEngine};
use lite_llm_training::TrainerConfig;

#[derive(Parser, Debug)]
#[command(name = "lite-llm")]
#[command(about = "Lite LLM - Lightweight Language Model", long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,

    #[arg(short, long, default_value_t = false)]
    verbose: bool,
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
            println!("Run 'lite-llm --help' for more information.");
        }
    }
}
