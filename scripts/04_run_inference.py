"""Run BBQ inference across all 39 model configurations (3 dense + 36 pruned)."""
import gc
import json
import os
import sys
import time

import mlx.core as mx
import mlx_lm
import yaml
from mlx_lm.sample_utils import make_sampler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.prompt_templates import format_prompt
from src.response_parser import parse_response


def build_config_list(cfg):
    """Build list of 39 model configs: 3 dense baselines + 36 pruned."""
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
            "output_tag": f"{name}-dense-s0",
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
                    "output_tag": f"{name}-{method}-s{sparsity}",
                })

    return configs


def load_completed(output_path):
    """Load set of (item_id, seed) pairs already completed."""
    completed = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    completed.add((rec["item_id"], rec["seed"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def load_dataset(dataset_path):
    """Load BBQ items from JSONL."""
    items = []
    with open(dataset_path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def run_inference(config_path="config.yaml"):
    cfg = yaml.safe_load(open(config_path))
    dataset_path = cfg["dataset"]["output_path"]
    raw_dir = cfg["output"]["raw_results_dir"]
    seeds = cfg["inference"]["seeds"]
    temperature = cfg["inference"]["temperature"]
    max_tokens = cfg["inference"]["max_tokens"]

    os.makedirs(raw_dir, exist_ok=True)

    items = load_dataset(dataset_path)
    if not items:
        print(f"ERROR: No items found in {dataset_path}. Run 02_prepare_dataset.py first.")
        return

    print(f"Loaded {len(items)} BBQ items, {len(seeds)} seeds → {len(items) * len(seeds)} inferences per config")

    configs = build_config_list(cfg)
    sampler = make_sampler(temp=temperature)

    for config in tqdm(configs, desc="Model configs"):
        model_path = config["model_path"]
        output_path = os.path.join(raw_dir, f"{config['output_tag']}.jsonl")

        if not os.path.exists(model_path):
            print(f"\nSkipping {config['output_tag']}: model not found at {model_path}")
            continue

        # Check resume state
        completed = load_completed(output_path)
        total_expected = len(items) * len(seeds)
        if len(completed) >= total_expected:
            print(f"\nSkipping {config['output_tag']}: all {total_expected} inferences complete")
            continue

        print(f"\nLoading {config['output_tag']} from {model_path}...")
        model, tokenizer = mlx_lm.load(model_path)

        remaining = []
        for item in items:
            for seed in seeds:
                if (item["item_id"], seed) not in completed:
                    remaining.append((item, seed))

        print(f"  {len(completed)} done, {len(remaining)} remaining")

        with open(output_path, "a") as f_out:
            for item, seed in tqdm(remaining, desc=f"  {config['output_tag']}", leave=False):
                prompt = format_prompt(item, config["model_name"], tokenizer)

                # Seed immediately before generation for reproducibility
                mx.random.seed(seed)
                t0 = time.time()
                raw_response = mlx_lm.generate(
                    model, tokenizer, prompt,
                    max_tokens=max_tokens, sampler=sampler,
                )
                elapsed = time.time() - t0

                parsed = parse_response(raw_response)

                record = {
                    "model": config["model_name"],
                    "prune_method": config["prune_method"],
                    "sparsity": config["sparsity"],
                    "item_id": item["item_id"],
                    "category": item["category"],
                    "seed": seed,
                    "raw_response": raw_response,
                    "parsed_answer": parsed,
                    "stereotype_target_index": item["stereotype_target_index"],
                    "anti_stereotype_target_index": item["anti_stereotype_target_index"],
                    "unknown_index": item["unknown_index"],
                    "elapsed_s": round(elapsed, 4),
                }
                f_out.write(json.dumps(record) + "\n")
                f_out.flush()

        # Free memory between configs
        del model, tokenizer
        gc.collect()
        mx.metal.clear_cache()
        print(f"  Completed {config['output_tag']}")

    print("\nAll inference complete.")


if __name__ == "__main__":
    run_inference()
