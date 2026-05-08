# Pruning Impact Analysis — Bias Amplification in Pruned LLMs

Code and pruned-model artifacts for the AIIoT 2026 paper on weight-pruning–induced
bias amplification in instruction-tuned LLMs targeted at edge deployment.

## Pruned model collection

All 36 pruned checkpoints used in the paper are published on Hugging Face:

**[Weight Pruning Amplifies Bias (AIIoT 2026)](https://huggingface.co/collections/plawanrath/weight-pruning-amplifies-bias-aiiot-2026-69fd6be2a5cbdc6e78a88ee7)**

The collection contains a 3 × 3 × 4 grid:

| Base model | Pruning methods | Target sparsities |
|---|---|---|
| `google/gemma-2-9b-it` | random, magnitude, wanda | 10%, 30%, 50%, 70% |
| `mistralai/Mistral-7B-Instruct-v0.3` | random, magnitude, wanda | 10%, 30%, 50%, 70% |
| `microsoft/Phi-3.5-mini-instruct` | random, magnitude, wanda | 10%, 30%, 50%, 70% |

Each pruned repo follows the naming pattern
`plawanrath/<base-model>-<method>-s<sparsity>-pia`.

> Research artifacts only — not intended for production deployment.

## Repository layout

- `config.yaml` — models, pruning grid, dataset, inference, perplexity settings.
- `scripts/` — numbered pipeline (download → prepare data → prune → infer → parse → perplexity → analyze).
- `src/` — pruning implementations, prompt templates, response parser, metrics, stats.
- `data/` — processed BBQ ambiguous subset.
- `results/` — raw outputs, aggregated tables, figures.
- `paper.tex` — manuscript source.

## Reproducing the experiment

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python scripts/01_download_models.py
python scripts/02_prepare_dataset.py
python scripts/03_prune_models.py
python scripts/04_run_inference.py
python scripts/05_parse_responses.py
python scripts/06_compute_perplexity.py
python scripts/07_analyze.py
```

## Recreating the HF collection

`scripts/create_hf_collection.py` is idempotent — running it again will reuse the
existing collection (matched by title) and skip items already attached.

```bash
huggingface-cli login   # token needs model + collection write
python scripts/create_hf_collection.py
```
