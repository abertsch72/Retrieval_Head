# Measuring Attention Sinks in Transformer Models

This tool analyzes the attention patterns in transformer models to quantify the "attention sink" phenomenon - the tendency of attention heads to allocate a significant portion of their attention to the first few tokens of a sequence.

## Overview

Attention sinks are an important aspect of transformer model behavior, especially for long-context models. This script:

1. Loads a specified transformer model
2. Generates or uses sample input documents of varying lengths (from 100 to 50,000 tokens)
3. Measures the percentage of attention mass directed to the first N tokens (default: 20)
4. Analyzes which layers and heads exhibit the strongest sink behavior
5. Outputs detailed statistics and visualizations

## Usage

```bash
python measure_attention_sinks.py --model_name MODEL_PATH [OPTIONS]
```

### Required Arguments

- `--model_name`: Path or name of the model to analyze (e.g., "meta-llama/Llama-2-7b-hf")

### Optional Arguments

- `--num_docs`: Number of documents to analyze (default: 10)
- `--min_length`: Minimum document length in tokens (default: 100)
- `--max_length`: Maximum document length in tokens (default: 50000)
- `--sink_window`: Size of the attention sink window in tokens (default: 20)
- `--output_file`: Path to save results (default: MODEL_NAME_sinks.json)

## Example

```bash
python measure_attention_sinks.py --model_name "meta-llama/Llama-2-7b-hf" --num_docs 5 --sink_window 10
```

## Output

The script generates a JSON file with detailed results, including:

- Overall average sink percentage
- Per-layer sink percentages
- Per-head sink percentages
- Per-layer-head sink percentages
- Top sink layers and heads
- Metadata about the analysis

Example output summary:

```
ATTENTION SINK ANALYSIS FOR meta-llama/Llama-2-7b-hf
==================================================
Average attention to first 20 tokens: 42.18%

Top 5 sink layers:
  Layer 15: 68.45%
  Layer 22: 65.32%
  Layer 8: 61.77%
  Layer 29: 58.91%
  Layer 3: 52.14%

Top 5 layer-head combinations:
  Layer 15, Head 12: 89.76%
  Layer 22, Head 7: 87.32%
  Layer 8, Head 31: 85.19%
  Layer 29, Head 2: 83.45%
  Layer 3, Head 18: 79.88%
==================================================
```

## Supported Models

The script supports various model architectures including:
- LLaMA
- Qwen (2, 2.5, 3)
- OLMo2
- Mixtral
- Mistral
- Phi3

## Requirements

- PyTorch
- Transformers
- NumPy
- tqdm

## Notes

- For very large models, you may need significant GPU memory
- Processing long sequences (e.g., 50,000 tokens) can be time-consuming
- The script uses the custom implementations from the faiss_attn module for model loading