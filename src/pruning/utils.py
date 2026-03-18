"""Shared pruning utilities: layer iteration, sparsity verification, calibration, model saving."""
import os
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.utils


def is_prunable_linear(name: str, module) -> bool:
    """Determine if a module should be pruned.

    Prune: All nn.Linear in attention and MLP layers.
    Skip: Embedding layers, LM head, layer norms.

    Architecture-specific layer names:
    - Mistral/Gemma: model.layers[i].self_attn.{q,k,v,o}_proj
                     model.layers[i].mlp.{gate,up,down}_proj
    - Phi-3.5: model.layers[i].self_attn.{qkv,o}_proj (FUSED QKV)
               model.layers[i].mlp.{gate_up,down}_proj (FUSED gate_up)
    """
    if not isinstance(module, nn.Linear):
        return False
    skip_patterns = ["embed", "lm_head", "norm"]
    return not any(pat in name.lower() for pat in skip_patterns)


def iter_prunable_layers(model):
    """Yield (name, module) for all prunable linear layers."""
    for name, module in model.named_modules():
        if is_prunable_linear(name, module):
            yield name, module


def verify_sparsity(model) -> dict:
    """Count zeros vs total weights in all prunable layers."""
    total_params = 0
    zero_params = 0
    per_layer = {}
    for name, module in iter_prunable_layers(model):
        w = module.weight
        n = w.size
        z = mx.sum(w == 0).item()
        total_params += n
        zero_params += z
        per_layer[name] = {"total": n, "zeros": z, "sparsity": z / n}
    return {
        "global_sparsity": zero_params / total_params if total_params > 0 else 0,
        "total_params": total_params,
        "zero_params": zero_params,
        "per_layer": per_layer,
    }


def get_calibration_data(dataset_name, subset, num_samples, seed, tokenizer, seq_length=2048):
    """Load calibration data from C4, tokenize, return list of token ID arrays."""
    from datasets import load_dataset
    import numpy as np

    np.random.seed(seed)
    ds = load_dataset(dataset_name, subset, split="train", streaming=True)
    calibration_data = []
    for example in ds:
        if len(calibration_data) >= num_samples:
            break
        tokens = tokenizer.encode(example["text"])
        if len(tokens) >= seq_length:
            calibration_data.append(mx.array(tokens[:seq_length]))
    return calibration_data[:num_samples]


def save_pruned_model(model, tokenizer, output_dir):
    """Save pruned model weights and tokenizer to output directory."""
    from mlx_lm.utils import save_model
    save_model(output_dir, model)
    tokenizer.save_pretrained(output_dir)
    print(f"  Saved pruned model to {output_dir}")
