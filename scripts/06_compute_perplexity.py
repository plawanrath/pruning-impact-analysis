"""Compute perplexity on Tulu-3 for all 39 model configurations."""
import gc
import os
import sys

import mlx.core as mx
import mlx_lm
import numpy as np
import pandas as pd
import yaml
from mlx_lm.perplexity import load_data, eval_ppl
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def build_config_list(cfg):
    """Build list of 39 model configs grouped by model family."""
    models_dir = cfg["pruning"]["models_dir"]
    configs = []

    for model_cfg in cfg["models"]:
        name = model_cfg["name"]
        # Dense baseline
        configs.append({
            "model_name": name,
            "prune_method": "dense",
            "sparsity": 0,
            "model_path": os.path.join(models_dir, f"{name}-bf16"),
        })
        # Pruned variants
        for method in cfg["pruning"]["methods"]:
            for sparsity in cfg["pruning"]["sparsity_levels"]:
                if sparsity == 0:
                    continue
                configs.append({
                    "model_name": name,
                    "prune_method": method,
                    "sparsity": sparsity,
                    "model_path": os.path.join(models_dir, f"{name}-{method}-s{sparsity}"),
                })

    return configs


def load_completed(csv_path):
    """Load set of completed (model, prune_method, sparsity) tuples."""
    completed = set()
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            completed.add((row["model"], row["prune_method"], int(row["sparsity"])))
    return completed


def compute_perplexity(config_path="config.yaml"):
    cfg = yaml.safe_load(open(config_path))
    ppl_cfg = cfg["perplexity"]
    data_path = ppl_cfg["data_path"]
    num_samples = ppl_cfg["num_samples"]
    seq_length = ppl_cfg["sequence_length"]
    batch_size = ppl_cfg["batch_size"]
    ppl_seed = ppl_cfg["seed"]

    output_dir = "results/perplexity"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "perplexity_results.csv")

    configs = build_config_list(cfg)
    completed = load_completed(csv_path)

    # Group configs by model family for tokenizer/data reuse
    families = {}
    for c in configs:
        families.setdefault(c["model_name"], []).append(c)

    for family_name, family_configs in families.items():
        # Find the dense model path to load tokenizer for data prep
        dense_config = [c for c in family_configs if c["prune_method"] == "dense"][0]
        dense_path = dense_config["model_path"]

        if not os.path.exists(dense_path):
            print(f"Skipping family {family_name}: dense model not found at {dense_path}")
            continue

        # Check if all configs in this family are already done
        remaining = [c for c in family_configs
                     if (c["model_name"], c["prune_method"], c["sparsity"]) not in completed]
        if not remaining:
            print(f"Skipping family {family_name}: all {len(family_configs)} configs complete")
            continue

        # Load tokenizer from dense model for data preparation
        print(f"\nFamily: {family_name} ({len(remaining)} configs remaining)")
        _, tokenizer = mlx_lm.load(dense_path)

        # Load evaluation data once per family
        np.random.seed(ppl_seed)
        data = load_data(tokenizer, data_path, num_samples, seq_length)
        print(f"  Eval data shape: {data.shape}")

        # Free the dense model we loaded just for the tokenizer
        gc.collect()

        for config in tqdm(remaining, desc=f"  {family_name}"):
            key = (config["model_name"], config["prune_method"], config["sparsity"])
            if key in completed:
                continue

            model_path = config["model_path"]
            if not os.path.exists(model_path):
                print(f"\n  Skipping {config['prune_method']}-s{config['sparsity']}: not found at {model_path}")
                continue

            model, _ = mlx_lm.load(model_path)
            ppl, ppl_se = eval_ppl(model, data, batch_size=batch_size)

            print(f"\n  {config['prune_method']}-s{config['sparsity']}: "
                  f"PPL={ppl:.2f} ± {ppl_se:.4f}")

            # Append to CSV
            row = pd.DataFrame([{
                "model": config["model_name"],
                "prune_method": config["prune_method"],
                "sparsity": config["sparsity"],
                "perplexity": round(ppl, 4),
                "perplexity_se": round(ppl_se, 6),
            }])
            row.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path),
                       index=False)
            completed.add(key)

            del model
            gc.collect()
            mx.metal.clear_cache()

    print(f"\nPerplexity results saved to {csv_path}")


if __name__ == "__main__":
    compute_perplexity()
