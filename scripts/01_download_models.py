"""Download BF16 models from HuggingFace and convert to MLX format."""
import os
import subprocess
import yaml


def download_models(config_path="config.yaml"):
    cfg = yaml.safe_load(open(config_path))
    models_dir = cfg["pruning"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    for model_cfg in cfg["models"]:
        out = os.path.join(models_dir, f"{model_cfg['name']}-bf16")
        if os.path.exists(out):
            print(f"Skipping {out} (already exists)")
            continue
        print(f"Downloading {model_cfg['hf_path']} -> {out}")
        cmd = [
            "python", "-m", "mlx_lm.convert",
            "--hf-path", model_cfg["hf_path"],
            "--mlx-path", out,
        ]
        subprocess.run(cmd, check=True)
        print(f"Done: {out}")


if __name__ == "__main__":
    download_models()
