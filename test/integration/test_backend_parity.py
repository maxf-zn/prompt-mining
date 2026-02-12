"""
Test that SAELens and HuggingFace backends return the same activations.
"""

import os
os.environ["HF_HOME"] = "/data/hf"
os.environ["TRANSFORMERS_CACHE"] = "/data/hf"

import numpy as np
import torch

from prompt_mining.model.model_wrapper import ModelConfig, ModelWrapper


def test_backend_parity():
    """Compare activations between SAELens and HuggingFace backends."""

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    test_prompt = "The capital of France is"
    test_layers = [19, 27]  # Test raw activations at multiple layers
    sae_layer = 19  # Only layer 19 has SAE available
    hook_point = "hook_resid_post"

    sae_configs = [
        {"sae_release": "goodfire-llama-3.1-8b-instruct", "sae_id": "layer_19"},
    ]

    print("=" * 60)
    print("Testing Backend Parity: SAELens vs HuggingFace")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Prompt: '{test_prompt}'")
    print(f"Layers: {test_layers}")
    print(f"Hook point: {hook_point}")
    print()

    # Load SAELens backend
    print("Loading SAELens backend...")
    saelens_config = ModelConfig(
        model_name=model_name,
        backend="saelens",
        dtype="float32",  # Use float32 to avoid precision differences
        sae_configs=sae_configs,
    )
    saelens_wrapper = ModelWrapper(saelens_config)
    saelens_wrapper.load()
    print()

    # Load HuggingFace backend
    print("Loading HuggingFace backend...")
    hf_config = ModelConfig(
        model_name=model_name,
        backend="huggingface",
        dtype="float32",  # Use float32 to avoid precision differences
        device_map={"": "cuda:0"},
        sae_configs=sae_configs,
    )
    hf_wrapper = ModelWrapper(hf_config)
    hf_wrapper.load()
    print()

    # Get activations from both backends
    print("Running forward pass with SAELens...")
    saelens_logits, saelens_cache = saelens_wrapper.run_with_cache(
        test_prompt,
        layers=test_layers,
        hook_points=[hook_point]
    )

    print("Running forward pass with HuggingFace...")
    hf_logits, hf_cache = hf_wrapper.run_with_cache(
        test_prompt,
        layers=test_layers,
        hook_points=[hook_point]
    )

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Compare tokenization
    saelens_tokens = saelens_wrapper.tokenize(test_prompt)["input_ids"]
    hf_tokens = hf_wrapper.tokenize(test_prompt)["input_ids"]

    print(f"\nTokenization:")
    print(f"  SAELens tokens shape: {saelens_tokens.shape}")
    print(f"  HuggingFace tokens shape: {hf_tokens.shape}")
    print(f"  Tokens match: {torch.equal(saelens_tokens.cpu(), hf_tokens.cpu())}")
    if not torch.equal(saelens_tokens.cpu(), hf_tokens.cpu()):
        print(f"  SAELens tokens: {saelens_tokens.tolist()}")
        print(f"  HuggingFace tokens: {hf_tokens.tolist()}")

    # Compare logits
    print(f"\nLogits:")
    print(f"  SAELens shape: {saelens_logits.shape}")
    print(f"  HuggingFace shape: {hf_logits.shape}")

    # Convert to same dtype for comparison
    sl = saelens_logits.float().cpu()
    hl = hf_logits.float().cpu()

    logits_diff = (sl - hl).abs()
    print(f"  Max absolute diff: {logits_diff.max().item():.6f}")
    print(f"  Mean absolute diff: {logits_diff.mean().item():.6f}")

    # Check top predictions match
    saelens_top = sl[0, -1].argmax().item()
    hf_top = hl[0, -1].argmax().item()
    print(f"  Top prediction (last pos) match: {saelens_top == hf_top}")
    print(f"    SAELens: {saelens_top} ({saelens_wrapper.tokenizer.decode([saelens_top])})")
    print(f"    HuggingFace: {hf_top} ({hf_wrapper.tokenizer.decode([hf_top])})")

    # Compare activations for each layer
    print(f"\nActivations (hook_resid_post):")
    all_close = True
    for layer in test_layers:
        key = f"blocks.{layer}.{hook_point}"

        if key not in saelens_cache:
            print(f"  Layer {layer}: Missing from SAELens cache!")
            continue
        if key not in hf_cache:
            print(f"  Layer {layer}: Missing from HuggingFace cache!")
            continue

        saelens_acts = saelens_cache[key].float().cpu()
        hf_acts = hf_cache[key].float().cpu()

        print(f"\n  Layer {layer}:")
        print(f"    SAELens shape: {saelens_acts.shape}")
        print(f"    HuggingFace shape: {hf_acts.shape}")

        if saelens_acts.shape != hf_acts.shape:
            print(f"    ⚠ Shape mismatch!")
            all_close = False
            continue

        diff = (saelens_acts - hf_acts).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Relative error (more meaningful for large values)
        rel_diff = diff / (saelens_acts.abs() + 1e-8)
        max_rel_diff = rel_diff.max().item()
        mean_rel_diff = rel_diff.mean().item()

        print(f"    Max absolute diff: {max_diff:.6f}")
        print(f"    Mean absolute diff: {mean_diff:.6f}")
        print(f"    Max relative diff: {max_rel_diff:.6f}")
        print(f"    Mean relative diff: {mean_rel_diff:.6f}")

        # Distribution of differences
        pct_99 = torch.quantile(diff.float(), 0.99).item()
        pct_999 = torch.quantile(diff.float(), 0.999).item()
        print(f"    99th percentile diff: {pct_99:.6f}")
        print(f"    99.9th percentile diff: {pct_999:.6f}")

        # Check correlation (should be very high if same structure)
        corr = torch.corrcoef(torch.stack([saelens_acts.flatten(), hf_acts.flatten()]))[0, 1].item()
        print(f"    Correlation: {corr:.8f}")

        # Check if close (using relative tolerance for numerical differences)
        is_close = mean_rel_diff < 0.01 and corr > 0.9999
        print(f"    Functionally equivalent: {'✓' if is_close else '✗'}")

        if not is_close:
            all_close = False

    # Compare SAE encodings (only for layer with SAE)
    print(f"\nSAE Encodings (layer {sae_layer}):")
    key = f"blocks.{sae_layer}.{hook_point}"
    if key in saelens_cache and key in hf_cache:
        saelens_acts = saelens_cache[key]
        hf_acts = hf_cache[key]

        saelens_encoded = saelens_wrapper.encode_layer_activations(saelens_acts, sae_layer)
        hf_encoded = hf_wrapper.encode_layer_activations(hf_acts, sae_layer)

        print(f"  SAELens encoded shape: {saelens_encoded.shape}")
        print(f"  HuggingFace encoded shape: {hf_encoded.shape}")

        se = saelens_encoded.float().cpu()
        he = hf_encoded.float().cpu()

        enc_diff = (se - he).abs()
        print(f"  Max absolute diff: {enc_diff.max().item():.6f}")
        print(f"  Mean absolute diff: {enc_diff.mean().item():.6f}")

        # Check non-zero features match
        saelens_nonzero = (se > 0).sum().item()
        hf_nonzero = (he > 0).sum().item()
        print(f"  Non-zero features: SAELens={saelens_nonzero}, HF={hf_nonzero}")

        # Correlation of SAE encodings
        se_flat = se.flatten()
        he_flat = he.flatten()
        # Only consider positions where at least one is non-zero
        mask = (se_flat > 0) | (he_flat > 0)
        if mask.sum() > 1:
            se_nz = se_flat[mask]
            he_nz = he_flat[mask]
            sae_corr = torch.corrcoef(torch.stack([se_nz, he_nz]))[0, 1].item()
            print(f"  SAE correlation (non-zero only): {sae_corr:.8f}")

        # Check which features are the same
        saelens_active = set((se > 0).nonzero()[:, -1].tolist())
        hf_active = set((he > 0).nonzero()[:, -1].tolist())
        overlap = len(saelens_active & hf_active)
        print(f"  Feature overlap: {overlap}/{len(saelens_active | hf_active)} ({100*overlap/max(len(saelens_active | hf_active), 1):.1f}%)")

    print()
    print("=" * 60)
    if all_close:
        print("✓ All activations match within tolerance!")
    else:
        print("✗ Some activations differ - investigate further")
    print("=" * 60)

    # Cleanup
    del saelens_wrapper, hf_wrapper
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test_backend_parity()
