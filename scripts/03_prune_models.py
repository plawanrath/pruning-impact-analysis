"""Apply pruning methods to all models at all sparsity levels, save pruned MLX models."""
import os
import sys
import json
import time
import yaml
import mlx_lm

# Ensure project root is on sys.path for absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pruning.random_pruning import prune as random_prune
from src.pruning.magnitude import prune as magnitude_prune
from src.pruning.wanda import prune as wanda_prune
from src.pruning.utils import verify_sparsity, save_pruned_model, get_calibration_data


def prune_all(config_path="config.yaml"):
    cfg = yaml.safe_load(open(config_path))
    models_dir = cfg["pruning"]["models_dir"]
    methods = cfg["pruning"]["methods"]
    sparsity_levels = [s for s in cfg["pruning"]["sparsity_levels"] if s > 0]

    cal_cfg = cfg["pruning"]

    for model_cfg in cfg["models"]:
        model_name = model_cfg["name"]
        dense_path = os.path.join(models_dir, f"{model_name}-bf16")

        if not os.path.exists(dense_path):
            print(f"ERROR: Dense model not found at {dense_path}. Run 01_download_models.py first.")
            continue

        for method in methods:
            # Load calibration data once per model if needed for Wanda
            calibration_data = None

            for sparsity in sparsity_levels:
                output_dir = os.path.join(models_dir, f"{model_name}-{method}-s{sparsity}")

                if os.path.exists(output_dir):
                    print(f"Skipping {output_dir} (already exists)")
                    continue

                print(f"\n{'='*60}")
                print(f"Pruning: {model_name} | method={method} | sparsity={sparsity}%")
                print(f"{'='*60}")

                # Load fresh dense model for each config
                t0 = time.time()
                print(f"  Loading dense model from {dense_path}...")
                model, tokenizer = mlx_lm.load(dense_path)
                t_load = time.time() - t0
                print(f"  Loaded in {t_load:.1f}s")

                # Apply pruning
                sparsity_ratio = sparsity / 100.0
                t0 = time.time()

                if method == "random":
                    model = random_prune(model, sparsity_ratio, seed=42)
                elif method == "magnitude":
                    model = magnitude_prune(model, sparsity_ratio)
                elif method == "wanda":
                    if calibration_data is None:
                        print(f"  Loading calibration data ({cal_cfg['calibration_samples']} samples from C4)...")
                        calibration_data = get_calibration_data(
                            cal_cfg["calibration_dataset"],
                            cal_cfg["calibration_subset"],
                            cal_cfg["calibration_samples"],
                            cal_cfg["calibration_seed"],
                            tokenizer,
                        )
                        print(f"  Calibration data ready: {len(calibration_data)} samples")
                    model = wanda_prune(model, tokenizer, sparsity_ratio, calibration_data)
                else:
                    print(f"  Unknown method: {method}, skipping")
                    continue

                t_prune = time.time() - t0
                print(f"  Pruning done in {t_prune:.1f}s")

                # Verify sparsity
                sp = verify_sparsity(model)
                actual = sp["global_sparsity"] * 100
                print(f"  Target sparsity: {sparsity}% | Actual: {actual:.2f}%")
                print(f"  Total prunable params: {sp['total_params']:,} | Zeros: {sp['zero_params']:,}")

                # Save
                # Copy config.json from dense model
                dense_config_path = os.path.join(dense_path, "config.json")
                if os.path.exists(dense_config_path):
                    os.makedirs(output_dir, exist_ok=True)
                    with open(dense_config_path) as f:
                        model_config = json.load(f)
                    with open(os.path.join(output_dir, "config.json"), "w") as f:
                        json.dump(model_config, f, indent=2)

                save_pruned_model(model, tokenizer, output_dir)

                # Save pruning metadata
                meta = {
                    "model": model_name,
                    "method": method,
                    "target_sparsity": sparsity,
                    "actual_sparsity": round(actual, 4),
                    "total_prunable_params": sp["total_params"],
                    "zero_params": sp["zero_params"],
                    "prune_time_s": round(t_prune, 2),
                }
                with open(os.path.join(output_dir, "pruning_meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)

                # Free memory
                del model
                print(f"  Done: {output_dir}")


if __name__ == "__main__":
    prune_all()
