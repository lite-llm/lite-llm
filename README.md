# lite-llm

A lightweight language model implementation in Rust.

## Features

- Character-level tokenizer (~90 vocab)
- Transformer-based language model
- Multiple sampling strategies (greedy, temperature, top-k, top-p)
- Training utilities
- Model checkpoint save/load
- CLI interface

## Building

```bash
cargo build --release
```

## Usage

### Generate Text

```bash
# Basic generation
cargo run -- generate --prompt "Hello world"

# With custom parameters
cargo run -- generate --prompt "Hello" --max-length 100 --temperature 0.7 --top-k 40 --top-p 0.9

# With seed for reproducibility
cargo run -- generate --prompt "Hello" --seed 42
```

### Train Model

```bash
cargo run -- train --epochs 10 --batch-size 4 --learning-rate 0.01
```

### Show Info

```bash
cargo run -- info
```

## Architecture

- **Tokenizer**: Character-level with ~90 vocab size
- **Model**: Transformer with embedding, multi-head attention, feed-forward networks
- **Sampling**: Supports greedy, temperature, top-k, and top-p sampling

## License
MIT
