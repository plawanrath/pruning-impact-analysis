"""Wanda: Pruning by Weights AND Activations (Sun et al., ICLR 2024).
Importance = |W_ij| * ||X_j||_2.  Per-row pruning.

MLX implementation: no hooks available. Must manually iterate through
transformer layers, passing hidden states and capturing inputs to each
linear sublayer.

Architecture note for activation capture:
- model.model.embed_tokens -> model.model.layers[0..N] -> model.model.norm -> model.lm_head
- Each layer: self_attn (q/k/v/o projections) + mlp (gate/up/down projections)
- Input to q/k/v proj = hidden_states (after attention layer norm)
- Input to gate/up proj = hidden_states (after MLP layer norm, post-attention residual)
- Process one layer at a time to minimize memory
"""
import mlx.core as mx
from src.pruning.utils import iter_prunable_layers


def prune(model, tokenizer, sparsity_ratio: float, calibration_data: list):
    # Step 1: Collect input activation norms per linear layer
    activation_norms = _collect_activation_norms(model, calibration_data)

    # Step 2: Apply Wanda criterion per prunable layer
    for name, module in iter_prunable_layers(model):
        if name not in activation_norms:
            print(f"  Warning: no activation norms for {name}, skipping Wanda")
            continue
        W = module.weight  # [out_features, in_features]
        scaler = activation_norms[name]  # [in_features]

        importance = mx.abs(W) * mx.sqrt(scaler)[None, :]

        n_prune = int(W.shape[1] * sparsity_ratio)
        if n_prune == 0:
            continue

        # Per-row: zero the n_prune least important weights using rank-based mask
        sorted_indices = mx.argsort(importance, axis=1)
        # Build rank matrix: rank[i, sorted_indices[i,k]] = k
        ranks = mx.zeros(W.shape, dtype=mx.int32)
        row_idx = mx.broadcast_to(mx.arange(W.shape[0])[:, None], sorted_indices.shape)
        col_positions = mx.broadcast_to(mx.arange(W.shape[1])[None, :], sorted_indices.shape)
        ranks = ranks.at[row_idx, sorted_indices].add(col_positions)
        # Mask: keep positions with rank >= n_prune (higher rank = more important)
        mask = (ranks >= n_prune).astype(W.dtype)
        module.weight = W * mask

    mx.eval(model.parameters())
    return model


def _collect_activation_norms(model, calibration_data):
    """Run calibration data through the model layer-by-layer,
    accumulating sum(X^2) per input dimension for each linear layer.

    Returns dict mapping layer name -> mx.array of shape [in_features]
    containing the sum of squared activations across all calibration tokens.
    """
    activation_norms = {}
    backbone = model.model

    for cal_tokens in calibration_data:
        # Embed
        hidden_states = backbone.embed_tokens(cal_tokens[None, :])  # [1, seq_len, hidden]

        for layer_idx, layer in enumerate(backbone.layers):
            prefix = f"model.layers.{layer_idx}"
            hidden_states, norms = _process_layer(
                layer, hidden_states, prefix
            )
            # Accumulate norms
            for name, norm_vec in norms.items():
                if name in activation_norms:
                    activation_norms[name] = activation_norms[name] + norm_vec
                else:
                    activation_norms[name] = norm_vec

        mx.eval(hidden_states)

    return activation_norms


def _process_layer(layer, hidden_states, prefix):
    """Process a single transformer layer, capturing activation norms for each linear sublayer.

    Handles both standard (Gemma/Mistral) and fused (Phi-3.5) architectures.
    Returns (output_hidden_states, norms_dict).
    """
    norms = {}

    # --- Self-attention block ---
    residual = hidden_states
    # Input layer norm (attention)
    attn_input = _apply_input_norm(layer, hidden_states, block="attn")

    # Capture norms for attention projections
    _accumulate_norm(norms, f"{prefix}.self_attn", layer.self_attn, attn_input)

    # Run full attention forward to get correct output
    attn_output = layer.self_attn(attn_input, mask=_create_causal_mask(attn_input))
    # Some architectures return a tuple
    if isinstance(attn_output, tuple):
        attn_output = attn_output[0]

    hidden_states = residual + attn_output

    # --- MLP block ---
    residual = hidden_states
    mlp_input = _apply_input_norm(layer, hidden_states, block="mlp")

    # Capture norms for MLP projections
    _accumulate_norm(norms, f"{prefix}.mlp", layer.mlp, mlp_input)

    # Run full MLP forward
    mlp_output = layer.mlp(mlp_input)
    hidden_states = residual + mlp_output

    mx.eval(hidden_states)
    return hidden_states, norms


def _apply_input_norm(layer, hidden_states, block="attn"):
    """Apply the appropriate input layer norm before attn or mlp."""
    if block == "attn":
        # Try common attribute names for attention input norm
        for attr in ["input_layernorm", "self_attn_layer_norm", "ln_1", "pre_attention_layernorm"]:
            if hasattr(layer, attr):
                return getattr(layer, attr)(hidden_states)
        # Fallback: return as-is
        return hidden_states
    else:
        # MLP input norm
        for attr in ["post_attention_layernorm", "mlp_layer_norm", "ln_2", "post_feedforward_layernorm"]:
            if hasattr(layer, attr):
                return getattr(layer, attr)(hidden_states)
        return hidden_states


def _accumulate_norm(norms, prefix, submodule, input_tensor):
    """Accumulate sum(X^2) for each linear layer within a submodule (attn or mlp).

    input_tensor: [batch, seq_len, hidden_dim] — the input to this submodule block.
    For attention: this is the input to q/k/v projections (NOT o_proj, which takes attn output).
    For MLP: this is the input to gate/up projections (NOT down_proj, which takes gate*up output).

    We only capture norms for layers whose input IS this tensor. o_proj and down_proj have
    different inputs that we don't capture here — they'll be skipped by Wanda (fallback to
    magnitude-like pruning via the warning in the main prune function).
    """
    # Flatten to [total_tokens, hidden_dim]
    flat_input = input_tensor.reshape(-1, input_tensor.shape[-1])
    # sum(x^2) per dimension
    input_sq_sum = mx.sum(flat_input * flat_input, axis=0)

    # Known projection names that receive the block input directly
    # Attention: q_proj, k_proj, v_proj, qkv_proj (fused)
    # MLP: gate_proj, up_proj, gate_up_proj (fused)
    # NOT: o_proj (receives attn output), down_proj (receives intermediate MLP output)
    input_proj_names = [
        "q_proj", "k_proj", "v_proj", "qkv_proj",  # attention
        "gate_proj", "up_proj", "gate_up_proj",       # MLP
    ]

    for attr_name in input_proj_names:
        attr = getattr(submodule, attr_name, None)
        if attr is None or not hasattr(attr, "weight"):
            continue
        full_name = f"{prefix}.{attr_name}"
        w_in = attr.weight.shape[1]
        if w_in == input_sq_sum.shape[0]:
            norms[full_name] = input_sq_sum


def _create_causal_mask(x):
    """Create a causal attention mask for the given input tensor."""
    seq_len = x.shape[1]
    mask = mx.triu(mx.full((seq_len, seq_len), float("-inf")), k=1)
    return mask
