#!/usr/bin/env python3
"""
Measure attention sinks in transformer models.

This script loads a specified model and measures the percentage of attention
mass that is concentrated in the first 20 tokens across different input lengths.
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import random
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any
from transformers import AutoTokenizer, AutoConfig

# Add faiss_attn to path
sys.path.append("./faiss_attn/")

# Import model classes from faiss_attn
from source.modeling_llama import LlamaForCausalLM
from source.modeling_qwen2 import Qwen2ForCausalLM
from source.modeling_qwen2_5 import Qwen2ForCausalLM as Qwen25ForCausalLM
from source.modeling_qwen3 import Qwen3ForCausalLM
from source.modeling_olmo2 import Olmo2ForCausalLM
from source.modeling_mixtral import MixtralForCausalLM
from source.modeling_mistral import MistralForCausalLM
from source.modeling_phi3 import Phi3ForCausalLM

# Set up logging
import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_name: str) -> Tuple[torch.nn.Module, Any]:
    """
    Load the model and tokenizer based on the model name.
    
    Args:
        model_name: Path or name of the model to load
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model and tokenizer from {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # Get model configuration
    config = AutoConfig.from_pretrained(model_name)
    
    # Determine model type and load appropriate model
    model_version = model_name.split("/")[-1]
    
    if "Qwen3" in model_version:
        model = Qwen3ForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map='auto', 
            use_flash_attention_2="flash_attention_2"
        ).eval()
    elif "Qwen2.5" in model_version or "Qwen2-5" in model_version:
        model = Qwen25ForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map='auto',
            use_flash_attention_2="flash_attention_2"
        ).eval()
    elif "Qwen2" in model_version:
        model = Qwen2ForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map='auto',
            use_flash_attention_2="flash_attention_2"
        ).eval()
    elif "OLMo2" in model_version:
        model = Olmo2ForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map='auto',
            use_flash_attention_2="flash_attention_2"
        ).eval()
    elif "Mixtral" in model_version:
        model = MixtralForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map='auto',
            use_flash_attention_2="flash_attention_2", trust_remote_code=True,
        ).eval()
    elif "Mistral" in model_version:
        model = MistralForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map='auto',
            use_flash_attention_2="flash_attention_2", trust_remote_code=True,
        ).eval()
    elif "Phi3" in model_version:
        model = Phi3ForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map='auto',
            use_flash_attention_2="flash_attention_2", trust_remote_code=True,
        ).eval()
    else:
        # Default to LLaMA for other models
        model = LlamaForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map='auto',
            use_flash_attention_2="flash_attention_2"
        ).eval()
    
    return model, tokenizer

def generate_input_documents(tokenizer, num_docs=10, min_length=100, max_length=50000) -> List[Dict]:
    """
    Generate input documents with varying lengths.
    
    Args:
        tokenizer: The tokenizer to use for encoding
        num_docs: Number of documents to generate
        min_length: Minimum document length in tokens
        max_length: Maximum document length in tokens
        
    Returns:
        List of dictionaries with 'text' and 'input_ids' keys
    """
    logger.info(f"Generating {num_docs} input documents with lengths from {min_length} to {max_length} tokens")
    
    # Use logarithmic spacing for lengths to better sample the range
    lengths = np.logspace(np.log10(min_length), np.log10(max_length), num=num_docs).astype(int)
    
    # Sample text from Wikipedia or other sources
    # For this implementation, we'll generate random text
    documents = []
    
    # Try to load sample text from a file if available
    sample_texts = []
    try:
        with open("sample_texts.txt", "r") as f:
            sample_texts = f.read().split("\n\n")
    except FileNotFoundError:
        # Use some default text samples if file not found
        sample_texts = [
            "The quick brown fox jumps over the lazy dog. " * 1000,
            "In computer science, artificial intelligence (AI) is intelligence demonstrated by machines, " * 1000,
            "Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', " * 1000,
            "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence " * 1000,
            "Transformer models have revolutionized natural language processing since their introduction in 2017. " * 1000,
        ]
    
    for i, length in enumerate(lengths):
        # Select a random text sample or generate random text
        if sample_texts:
            text = random.choice(sample_texts)
            # Repeat the text to reach desired length if needed
            text = (text + " ") * (length // len(tokenizer.encode(text)) + 1)
        else:
            # Generate random text if no samples available
            words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", 
                    "artificial", "intelligence", "machine", "learning", "transformer", 
                    "attention", "neural", "network", "deep", "language", "model"]
            text = " ".join(random.choices(words, k=length * 2))  # Approximate word count
        
        # Encode and truncate to desired length
        input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")[0]
        if len(input_ids) > length:
            input_ids = input_ids[:length]
        
        # Decode back to text for reference
        text = tokenizer.decode(input_ids)
        
        documents.append({
            "text": text,
            "input_ids": input_ids,
            "length": len(input_ids)
        })
    
    return documents

def measure_attention_sinks(model, documents, sink_window_size=20):
    """
    Measure the percentage of attention mass in the first sink_window_size tokens.
    
    Args:
        model: The model to analyze
        documents: List of documents with 'input_ids' key
        sink_window_size: Number of tokens to consider as the attention sink window
        
    Returns:
        Dictionary with results per layer, head, and document length
    """
    logger.info(f"Measuring attention sinks (first {sink_window_size} tokens)")
    
    results = {
        "per_layer_head": {},
        "per_layer": {},
        "per_head": {},
        "overall": {},
        "metadata": {
            "sink_window_size": sink_window_size,
            "document_lengths": [doc["length"] for doc in documents]
        }
    }
    
    # Get model configuration
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    
    # Initialize accumulators
    layer_head_sinks = np.zeros((num_layers, num_heads, len(documents)))
    
    # Process each document
    for doc_idx, doc in enumerate(tqdm(documents, desc="Processing documents")):
        input_ids = doc["input_ids"].unsqueeze(0).to(model.device)
        
        # Forward pass with attention outputs
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True, attn_mode="torch")
        
        # Extract attention weights
        attentions = outputs.attentions  # Tuple of tensors (batch_size, num_heads, seq_len, seq_len)
        
        # Calculate sink percentages for each layer and head
        for layer_idx, layer_attention in enumerate(attentions):
            # layer_attention shape: (batch_size, num_heads, seq_len, seq_len)
            for head_idx in range(num_heads):
                # Get attention weights for this head
                # Shape: (seq_len, seq_len)
                head_attention = layer_attention[0, head_idx]
                
                # For each token, measure how much attention it pays to the sink window
                seq_len = head_attention.size(0)
                
                # Calculate the total attention mass directed to the sink window
                # We look at the last row (last token) and sum attention to first sink_window_size tokens
                sink_window = min(sink_window_size, seq_len)
                
                # Sum over all query positions (rows) to get total attention to each key position
                total_attention_per_key = head_attention.sum(dim=0)
                
                # Calculate percentage of attention to sink window
                sink_attention = total_attention_per_key[:sink_window].sum().item()
                total_attention = total_attention_per_key.sum().item()
                sink_percentage = (sink_attention / total_attention) * 100
                
                # Store result
                layer_head_sinks[layer_idx, head_idx, doc_idx] = sink_percentage
    
    # Calculate averages across documents
    layer_head_avg_sinks = np.mean(layer_head_sinks, axis=2)
    
    # Store detailed results
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            key = f"layer_{layer_idx}_head_{head_idx}"
            results["per_layer_head"][key] = float(layer_head_avg_sinks[layer_idx, head_idx])
    
    # Calculate per-layer averages
    layer_avg_sinks = np.mean(layer_head_avg_sinks, axis=1)
    for layer_idx in range(num_layers):
        results["per_layer"][f"layer_{layer_idx}"] = float(layer_avg_sinks[layer_idx])
    
    # Calculate per-head averages (across layers)
    head_avg_sinks = np.mean(layer_head_avg_sinks, axis=0)
    for head_idx in range(num_heads):
        results["per_head"][f"head_{head_idx}"] = float(head_avg_sinks[head_idx])
    
    # Calculate overall average
    overall_avg_sink = float(np.mean(layer_head_avg_sinks))
    results["overall"]["average_sink_percentage"] = overall_avg_sink
    
    # Calculate top sink layers and heads
    top_layers_indices = np.argsort(layer_avg_sinks)[::-1][:5]
    top_heads_indices = np.argsort(head_avg_sinks)[::-1][:5]
    
    results["overall"]["top_sink_layers"] = [int(idx) for idx in top_layers_indices]
    results["overall"]["top_sink_heads"] = [int(idx) for idx in top_heads_indices]
    
    # Find the layer-head combinations with highest sink percentages
    flat_indices = np.argsort(layer_head_avg_sinks.flatten())[::-1][:10]
    top_layer_head_combinations = []
    for idx in flat_indices:
        layer_idx = idx // num_heads
        head_idx = idx % num_heads
        sink_percentage = float(layer_head_avg_sinks[layer_idx, head_idx])
        top_layer_head_combinations.append({
            "layer": int(layer_idx),
            "head": int(head_idx),
            "sink_percentage": sink_percentage
        })
    
    results["overall"]["top_layer_head_combinations"] = top_layer_head_combinations
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Measure attention sinks in transformer models")
    parser.add_argument("--model_name", type=str, required=True, help="Path or name of the model to analyze")
    parser.add_argument("--num_docs", type=int, default=10, help="Number of documents to analyze")
    parser.add_argument("--min_length", type=int, default=100, help="Minimum document length in tokens")
    parser.add_argument("--max_length", type=int, default=50000, help="Maximum document length in tokens")
    parser.add_argument("--sink_window", type=int, default=20, help="Size of the attention sink window (in tokens)")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save results (default: model_name_sinks.json)")
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if args.output_file is None:
        model_short_name = args.model_name.split("/")[-1]
        args.output_file = f"{model_short_name}_sinks.json"
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    
    # Generate input documents
    documents = generate_input_documents(
        tokenizer, 
        num_docs=args.num_docs, 
        min_length=args.min_length, 
        max_length=args.max_length
    )
    
    # Measure attention sinks
    results = measure_attention_sinks(model, documents, sink_window_size=args.sink_window)
    
    # Add metadata
    results["metadata"]["model_name"] = args.model_name
    results["metadata"]["num_documents"] = args.num_docs
    results["metadata"]["min_length"] = args.min_length
    results["metadata"]["max_length"] = args.max_length
    
    # Save results
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output_file}")
    
    # Print summary
    print("\n" + "="*50)
    print(f"ATTENTION SINK ANALYSIS FOR {args.model_name}")
    print("="*50)
    print(f"Average attention to first {args.sink_window} tokens: {results['overall']['average_sink_percentage']:.2f}%")
    print("\nTop 5 sink layers:")
    for layer_idx in results["overall"]["top_sink_layers"]:
        print(f"  Layer {layer_idx}: {results['per_layer'][f'layer_{layer_idx}']:.2f}%")
    
    print("\nTop 5 layer-head combinations:")
    for combo in results["overall"]["top_layer_head_combinations"][:5]:
        print(f"  Layer {combo['layer']}, Head {combo['head']}: {combo['sink_percentage']:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()