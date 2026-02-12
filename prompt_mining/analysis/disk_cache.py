"""
Disk cache for activation matrices.

Provides fast loading of pre-computed activation matrices by caching them
to disk on first load. Subsequent loads read from cache (~100x faster).
"""
import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from scipy import sparse

from prompt_mining.analysis.data_structures import ActivationMatrix


class ActivationDiskCache:
    """
    Disk cache for ActivationMatrix objects.

    Stores matrices in numpy .npz format for fast loading.
    Sparse matrices use scipy's sparse format.

    Cache files are stored in: {storage_root}/cache/activations/

    Example:
        >>> cache = ActivationDiskCache(Path("/data/my_registry"))
        >>>
        >>> # Check if cached
        >>> path = cache.get_path("raw", 31, "-5", is_sparse=False)
        >>> if cached := cache.load(path, is_sparse=False):
        ...     print("Loaded from cache!")
        >>> else:
        ...     # Load from source, then save
        ...     matrix = load_from_source(...)
        ...     cache.save(path, matrix, is_sparse=False)
    """

    def __init__(self, storage_root: Path):
        """
        Initialize disk cache.

        Args:
            storage_root: Root directory of the data registry
        """
        self.storage_root = Path(storage_root)
        self._cache_dir: Optional[Path] = None

    @property
    def cache_dir(self) -> Path:
        """Get or create cache directory."""
        if self._cache_dir is None:
            self._cache_dir = self.storage_root / "cache" / "activations"
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        return self._cache_dir

    def get_path(
        self,
        space: str,
        layer: int,
        position: Union[str, List[int]],
        is_sparse: bool
    ) -> Path:
        """
        Generate cache file path from parameters.

        Args:
            space: Activation space ('raw', 'sae', 'plt')
            layer: Layer index
            position: Position string or list of ints
            is_sparse: Whether matrix is sparse

        Returns:
            Path to cache file (.npz)
        """
        if isinstance(position, list):
            pos_str = ','.join(str(p) for p in sorted(position))
        else:
            pos_str = str(position)

        sparse_suffix = "_sparse" if is_sparse else ""
        filename = f"{space}_layer{layer}_pos{pos_str}{sparse_suffix}.npz"
        return self.cache_dir / filename

    def exists(self, cache_path: Path) -> bool:
        """Check if cache file exists."""
        return cache_path.exists()

    def load(
        self,
        cache_path: Path,
        is_sparse: bool
    ) -> Optional[ActivationMatrix]:
        """
        Load activation matrix from disk cache.

        Args:
            cache_path: Path to cache file
            is_sparse: Whether to load as sparse matrix

        Returns:
            ActivationMatrix if cache exists and is valid, None otherwise
        """
        if not cache_path.exists():
            return None

        try:
            if is_sparse:
                # Sparse: matrix in .npz, metadata in .meta.npz
                matrix = sparse.load_npz(cache_path)
                meta_path = Path(str(cache_path) + '.meta.npz')
                if not meta_path.exists():
                    return None
                meta = np.load(meta_path, allow_pickle=True)
            else:
                # Dense: everything in one .npz
                data = np.load(cache_path, allow_pickle=True)
                matrix = data['matrix']
                meta = data

            return ActivationMatrix(
                matrix=matrix,
                prompt_ids=meta['prompt_ids'].tolist(),
                run_ids=meta['run_ids'].tolist(),
                metadata=json.loads(str(meta['metadata']))
            )
        except Exception as e:
            # Cache corrupted or incompatible, ignore it
            print(f"Warning: Could not load cache {cache_path}: {e}")
            return None

    def save(
        self,
        cache_path: Path,
        activation_matrix: ActivationMatrix,
        is_sparse: bool
    ) -> bool:
        """
        Save activation matrix to disk cache.

        Args:
            cache_path: Path to cache file
            activation_matrix: Matrix to save
            is_sparse: Whether matrix is sparse

        Returns:
            True if save succeeded, False otherwise
        """
        try:
            if is_sparse:
                # Sparse: save matrix separately from metadata
                sparse.save_npz(cache_path, activation_matrix.matrix)
                np.savez(
                    str(cache_path) + '.meta',
                    run_ids=np.array(activation_matrix.run_ids),
                    prompt_ids=np.array(activation_matrix.prompt_ids),
                    metadata=json.dumps(activation_matrix.metadata)
                )
            else:
                # Dense: save everything in one file, convert to float32
                np.savez(
                    cache_path,
                    matrix=activation_matrix.matrix.astype(np.float32),
                    run_ids=np.array(activation_matrix.run_ids),
                    prompt_ids=np.array(activation_matrix.prompt_ids),
                    metadata=json.dumps(activation_matrix.metadata)
                )
            return True
        except Exception as e:
            print(f"Warning: Could not save cache {cache_path}: {e}")
            return False

    def clear(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache files.

        Args:
            pattern: Optional glob pattern to match (e.g., "raw_layer31_*")
                    If None, clears all cache files.

        Returns:
            Number of files deleted
        """
        if pattern is None:
            pattern = "*.npz"

        count = 0
        for path in self.cache_dir.glob(pattern):
            path.unlink()
            count += 1
            # Also remove .meta.npz for sparse caches
            meta_path = Path(str(path) + '.meta.npz')
            if meta_path.exists():
                meta_path.unlink()
                count += 1

        return count

    def list_cached(self) -> List[str]:
        """
        List all cached activation configurations.

        Returns:
            List of cache file names (without .npz extension)
        """
        files = []
        for path in self.cache_dir.glob("*.npz"):
            name = path.stem
            # Skip .meta files
            if not name.endswith('.meta'):
                files.append(name)
        return sorted(files)
