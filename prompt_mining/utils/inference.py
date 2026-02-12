"""
Inference utilities for prompt processing.
"""
import torch
from typing import List, Tuple


def get_topk(logits: torch.Tensor, tokenizer, k: int = 5) -> List[Tuple[str, float]]:
    """
    Extract top-k tokens from logits.

    Args:
        logits: Model logits (typically shape: [batch, seq_len, vocab_size])
        tokenizer: HuggingFace tokenizer
        k: Number of top tokens to return

    Returns:
        List of (token_string, probability) tuples

    Example:
        >>> logits = model(input_ids)
        >>> top_tokens = get_topk(logits, tokenizer, k=10)
        >>> for token, prob in top_tokens:
        ...     print(f"{token}: {prob:.3f}")
    """
    # Get probabilities for last token position
    probs = torch.softmax(logits.squeeze()[-1], dim=-1)
    topk = torch.topk(probs, k)

    results = []
    for i in range(k):
        token_id = topk.indices[i].item()
        token_str = tokenizer.decode([token_id])
        prob = topk.values[i].item()
        results.append((token_str, prob))

    return results


def get_topk_logits(
    logits: torch.Tensor,
    k: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract top-k token IDs and probabilities from logits.

    Args:
        logits: Model logits (shape: [batch, seq_len, vocab_size])
        k: Number of top tokens

    Returns:
        Tuple of (token_ids, probabilities)
        - token_ids: shape (k,) - integers
        - probabilities: shape (k,) - floats

    Example:
        >>> logits = model(input_ids)
        >>> token_ids, probs = get_topk_logits(logits, k=10)
        >>> # Store in CompactGraph
    """
    # Get probabilities for last token position
    probs = torch.softmax(logits.squeeze()[-1], dim=-1)
    topk = torch.topk(probs, k)

    # Keep token IDs as integers, probabilities as floats
    return topk.indices.cpu().numpy(), topk.values.cpu().float().numpy()


def resolve_positions(
    position_specs: str | List[str],
    seq_len: int
) -> List[int]:
    """
    Resolve position specifications to actual indices.

    Args:
        position_specs: Either "all" for all positions, ["all"], or a list of position specifications
                       Specifications can be: "first", "last", or integer strings
        seq_len: Sequence length

    Returns:
        List of position indices

    Examples:
        >>> resolve_positions("all", seq_len=10)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> resolve_positions(["all"], seq_len=10)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> resolve_positions(["first", "last"], seq_len=45)
        [0, 44]
        >>> resolve_positions(["0", "10", "last"], seq_len=20)
        [0, 10, 19]
    """
    # Handle "all" as string
    if position_specs == "all":
        return list(range(seq_len))
    
    # Handle ["all"] as list
    if isinstance(position_specs, list) and len(position_specs) == 1 and position_specs[0] == "all":
        return list(range(seq_len))
    
    positions = []
    for spec in position_specs:
        if spec == "first":
            positions.append(0)
        elif spec == "last":
            positions.append(seq_len - 1)
        else:
            try:
                pos = int(spec)
                if pos < 0:
                    # Negative indexing
                    positions.append(seq_len + pos)
                else:
                    positions.append(pos)
            except ValueError:
                raise ValueError(f"Invalid position spec: {spec}")

    # Remove duplicates and sort
    positions = sorted(set(positions))

    # Validate
    for pos in positions:
        if pos < 0 or pos >= seq_len:
            raise ValueError(f"Position {pos} out of range [0, {seq_len})")

    return positions


def resolve_layers(
    layer_specs: str | List[int | str],
    num_layers: int
) -> List[int]:
    """
    Resolve layer specifications to actual layer indices.

    Args:
        layer_specs: Either "all" for all layers, ["all"], or a list of layer indices
        num_layers: Total number of layers in the model

    Returns:
        List of layer indices (positive integers only)

    Examples:
        >>> resolve_layers("all", num_layers=32)
        [0, 1, 2, ..., 31]
        >>> resolve_layers(["all"], num_layers=32)
        [0, 1, 2, ..., 31]
        >>> resolve_layers([0, 5, 10], num_layers=32)
        [0, 5, 10]
    """
    # Handle "all" as string
    if layer_specs == "all":
        return list(range(num_layers))
    
    # Handle ["all"] as list
    if isinstance(layer_specs, list) and len(layer_specs) == 1 and layer_specs[0] == "all":
        return list(range(num_layers))
    
    # Validate layer indices
    layers = []
    for layer in layer_specs:
        if not isinstance(layer, int):
            raise ValueError(f"Layer index must be an integer, got {type(layer)}")
        if layer < 0:
            raise ValueError(f"Negative layer indices not supported, got {layer}")
        if layer >= num_layers:
            raise ValueError(f"Layer {layer} out of range [0, {num_layers})")
        layers.append(layer)
    
    # Remove duplicates and sort
    return sorted(set(layers))
