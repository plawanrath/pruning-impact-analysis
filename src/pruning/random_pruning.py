"""Random unstructured pruning — control condition.
Establishes: what happens when you remove capacity with zero intelligence."""
import mlx.core as mx
from src.pruning.utils import iter_prunable_layers


def prune(model, sparsity_ratio: float, seed: int = 42):
    mx.random.seed(seed)
    for name, module in iter_prunable_layers(model):
        W = module.weight
        mask = mx.random.uniform(shape=W.shape) >= sparsity_ratio
        module.weight = W * mask.astype(W.dtype)
    mx.eval(model.parameters())
    return model
