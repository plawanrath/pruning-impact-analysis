"""Per-layer unstructured magnitude pruning — naive baseline."""
import mlx.core as mx
from src.pruning.utils import iter_prunable_layers


def prune(model, sparsity_ratio: float):
    for name, module in iter_prunable_layers(model):
        W = module.weight
        abs_W = mx.abs(W)
        flat = abs_W.reshape(-1)
        k = int(flat.size * sparsity_ratio)
        if k == 0:
            continue
        # Find the k-th smallest value as threshold
        sorted_flat = mx.sort(flat)
        threshold = sorted_flat[k - 1]
        mask = (abs_W > threshold).astype(W.dtype)
        module.weight = W * mask
    mx.eval(model.parameters())
    return model
