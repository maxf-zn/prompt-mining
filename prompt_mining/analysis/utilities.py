"""
Analysis utilities for activation matrices.

These functions operate on activation matrices and compute directions,
separability metrics, and dimensionality reduction for exploratory analysis.

For classifier evaluation functions, use prompt_mining.classifiers instead.
"""
import torch
import numpy as np
from typing import List, Dict, Any, Union
from sklearn.decomposition import PCA


def compute_direction(
    cluster_a: Union[List[torch.Tensor], torch.Tensor],
    cluster_b: Union[List[torch.Tensor], torch.Tensor],
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute direction vector from cluster A to cluster B.

    This is the basic operation for computing steering vectors.

    Args:
        cluster_a: List of activation tensors or stacked tensor (n_samples, d_model)
        cluster_b: List of activation tensors or stacked tensor (n_samples, d_model)
        normalize: Whether to normalize the direction vector

    Returns:
        Direction vector (d_model,)

    Example:
        >>> success_acts = [...]  # List of activation tensors
        >>> failure_acts = [...]
        >>> direction = compute_direction(success_acts, failure_acts)
        >>> # Use direction for steering or analysis
    """
    # Convert to tensors if needed
    if isinstance(cluster_a, list):
        cluster_a = torch.stack(cluster_a)
    if isinstance(cluster_b, list):
        cluster_b = torch.stack(cluster_b)

    # Compute means
    mean_a = cluster_a.mean(dim=0)
    mean_b = cluster_b.mean(dim=0)

    # Direction from A to B
    direction = mean_b - mean_a

    if normalize:
        direction = direction / (direction.norm() + 1e-8)

    return direction


def compute_separability(
    cluster_a: Union[List[torch.Tensor], torch.Tensor],
    cluster_b: Union[List[torch.Tensor], torch.Tensor]
) -> Dict[str, float]:
    """
    Compute separability metrics between two clusters.

    Args:
        cluster_a: List of activation tensors or stacked tensor
        cluster_b: List of activation tensors or stacked tensor

    Returns:
        Dictionary with separability metrics:
        - l2_distance: L2 distance between means
        - cosine_similarity: Cosine similarity between means
        - normalized_distance: L2 distance normalized by magnitude
        - within_cluster_variance: Average within-cluster variance
        - between_cluster_variance: Variance of the direction

    Example:
        >>> metrics = compute_separability(success_acts, failure_acts)
        >>> print(f"L2 distance: {metrics['l2_distance']:.3f}")
        >>> print(f"Cosine sim: {metrics['cosine_similarity']:.3f}")
    """
    # Convert to tensors if needed
    if isinstance(cluster_a, list):
        cluster_a = torch.stack(cluster_a)
    if isinstance(cluster_b, list):
        cluster_b = torch.stack(cluster_b)

    # Compute means
    mean_a = cluster_a.mean(dim=0)
    mean_b = cluster_b.mean(dim=0)

    # L2 distance between means
    l2_distance = (mean_a - mean_b).norm().item()

    # Cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(
        mean_a.unsqueeze(0),
        mean_b.unsqueeze(0),
        dim=1
    ).item()

    # Normalized distance
    normalized_dist = l2_distance / (mean_a.norm() + mean_b.norm() + 1e-8).item()

    # Within-cluster variance
    var_a = cluster_a.var(dim=0).mean().item()
    var_b = cluster_b.var(dim=0).mean().item()
    within_cluster_var = (var_a + var_b) / 2

    # Between-cluster variance (variance along direction)
    direction = mean_b - mean_a
    direction_norm = direction / (direction.norm() + 1e-8)

    proj_a = (cluster_a @ direction_norm).var().item()
    proj_b = (cluster_b @ direction_norm).var().item()
    between_cluster_var = ((proj_a + proj_b) / 2)

    return {
        'l2_distance': l2_distance,
        'cosine_similarity': cosine_sim,
        'normalized_distance': normalized_dist,
        'within_cluster_variance': within_cluster_var,
        'between_cluster_variance': between_cluster_var,
        'global_separability_ratio': l2_distance / np.sqrt(within_cluster_var),
        'directional_separability': l2_distance / np.sqrt(between_cluster_var),
    }


def fit_pca(
    activations: Union[List[torch.Tensor], torch.Tensor, np.ndarray],
    n_components: int = 50
) -> PCA:
    """
    Fit PCA on activation matrix.

    Args:
        activations: Activation matrix (n_samples, d_model) or list of tensors
        n_components: Number of PCA components

    Returns:
        Fitted PCA object

    Example:
        >>> acts = np.stack([...])  # (n_samples, d_model)
        >>> pca = fit_pca(acts, n_components=50)
        >>> reduced = pca.transform(acts)
    """
    # Convert to numpy
    X = _prepare_tensors(activations)

    # Fit PCA
    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X)

    return pca


def transform_pca(
    activations: Union[List[torch.Tensor], torch.Tensor, np.ndarray],
    pca: PCA
) -> np.ndarray:
    """
    Transform activations using fitted PCA.

    Args:
        activations: Activation matrix or list of tensors
        pca: Fitted PCA object

    Returns:
        Reduced activation matrix (n_samples, n_components)
    """
    X = _prepare_tensors(activations)
    return pca.transform(X)


def compute_projection(
    activations: Union[torch.Tensor, np.ndarray],
    direction: Union[torch.Tensor, np.ndarray]
) -> np.ndarray:
    """
    Project activations onto direction vector.

    Args:
        activations: Activation matrix (n_samples, d_model)
        direction: Direction vector (d_model,)

    Returns:
        Projection values (n_samples,)

    Example:
        >>> direction = compute_direction(cluster_a, cluster_b)
        >>> projections = compute_projection(all_activations, direction)
        >>> # Plot histogram of projections
    """
    # Convert to numpy
    if isinstance(activations, torch.Tensor):
        activations = activations.cpu().numpy()
    if isinstance(direction, torch.Tensor):
        direction = direction.cpu().numpy()

    # Normalize direction
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    # Project
    projections = activations @ direction

    return projections


def _prepare_tensors(
    data: Union[List[torch.Tensor], torch.Tensor, np.ndarray]
) -> np.ndarray:
    """
    Convert various formats to numpy array for sklearn.

    Args:
        data: List of tensors, stacked tensor, or numpy array

    Returns:
        Numpy array (n_samples, d_model)
    """
    if isinstance(data, list):
        # Stack list of tensors
        if isinstance(data[0], torch.Tensor):
            data = torch.stack(data).cpu().numpy()
        else:
            data = np.stack(data)
    elif isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    return data


def compute_cluster_stats(
    activations: Union[torch.Tensor, np.ndarray]
) -> Dict[str, Any]:
    """
    Compute statistics for a cluster of activations.

    Args:
        activations: Activation matrix (n_samples, d_model)

    Returns:
        Dictionary with statistics
    """
    if isinstance(activations, torch.Tensor):
        activations = activations.cpu().numpy()

    return {
        'mean': activations.mean(axis=0),
        'std': activations.std(axis=0),
        'mean_norm': np.linalg.norm(activations.mean(axis=0)),
        'mean_std_norm': np.linalg.norm(activations.std(axis=0)),
        'n_samples': activations.shape[0],
        'd_model': activations.shape[1],
    }
