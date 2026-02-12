"""
Base classes and data structures for analysis pipeline.

This module provides:
- AnalysisData: Unified data accessor for loading feature matrices and activation matrices
- Analyzer: Abstract base class for all analyzers
- FeatureMatrix: Container for Prompt×Feature matrices
- ActivationMatrix: Container for Prompt×d_transcoder matrices
"""
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import numpy as np
import polars as pl
import zarr
from pathlib import Path
from scipy.sparse import csr_matrix
from tqdm import tqdm
from prompt_mining.registry.sqlite_registry import SQLiteRegistry
from prompt_mining.storage.base import StorageBackend
from prompt_mining.core.run_data import RunData
from prompt_mining.analysis.data_structures import ActivationMatrix, FeatureMatrix
from prompt_mining.analysis.disk_cache import ActivationDiskCache


def _load_raw_activation_worker(args: Tuple[str, int, List[int], str]) -> Tuple[str, Optional[np.ndarray]]:
    """
    Worker function for multiprocessing raw activation loading.

    Must be module-level for pickling. Each worker opens zarr independently,
    bypassing zarr's internal async serialization that blocks threading.

    Args:
        args: Tuple of (run_id, layer, pos_indices, storage_root_path)
              pos_indices is list of pre-resolved position indices (not 'last'/'first')

    Returns:
        Tuple of (run_id, activation_matrix or None)
        activation_matrix has shape (n_positions, d_model)
        Returns None if zarr file doesn't exist or positions not found
    """
    run_id, layer, pos_indices, storage_root = args

    run_date = run_id.split('_')[0]
    zarr_path = f"{storage_root}/runs/{run_date}/{run_id}/acts.zarr"

    try:
        acts = zarr.open(str(zarr_path), mode='r')
        array_path = f"raw/layer{layer}"
        if array_path not in acts:
            return run_id, None

        # Get saved positions and map to row indices
        raw_positions = np.array(acts.attrs.get('raw_positions', []))
        if len(raw_positions) == 0:
            return run_id, None

        pos_indices_arr = np.array(pos_indices)

        # Vectorized lookup: raw_positions is sorted, use searchsorted directly
        row_indices = np.searchsorted(raw_positions, pos_indices_arr)

        # Clamp to valid range and verify all positions were found
        row_indices = np.clip(row_indices, 0, len(raw_positions) - 1)
        if not np.all(raw_positions[row_indices] == pos_indices_arr):
            return run_id, None

        arr = acts[array_path]
        if np.max(row_indices) >= arr.shape[0]:
            return run_id, None

        # Extract all positions at once using fancy indexing
        # Result shape: (n_positions, d_model)
        return run_id, arr[row_indices, :]
    except Exception:
        return run_id, None


class AnalysisData:
    """
    Lazy-loading data accessor for analyzers.

    Provides unified interface for:
    - Loading runs from registry with filters
    - Building Prompt×Feature matrices from feature_incidence tables
    - Building Prompt×d_transcoder matrices from Zarr activations

    All data is loaded lazily and cached.

    Example:
        >>> registry = SQLiteRegistry("registry.sqlite")
        >>> storage = LocalStorage("file:///path/to/storage")
        >>>
        >>> # Load successful attacks
        >>> success_data = AnalysisData(
        ...     registry=registry,
        ...     storage=storage,
        ...     filters={'dataset_id': 'injecagent', 'labels': {'success': True}}
        ... )
        >>>
        >>> # Build feature matrix
        >>> feature_matrix = success_data.get_feature_matrix(top_k=200, format='binary')
        >>> print(f"Shape: {feature_matrix.matrix.shape}")  # (n_prompts, 200)
        >>>
        >>> # Build activation matrix
        >>> act_matrix = success_data.get_activation_matrix(layer=10, position='last')
        >>> print(f"Shape: {act_matrix.matrix.shape}")  # (n_prompts, 16384)
    """

    def __init__(
        self,
        registry: SQLiteRegistry,
        storage: StorageBackend,
        filters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize data accessor.

        Args:
            registry: SQLite registry for querying runs
            storage: Storage backend for reading artifacts
            filters: Filters for selecting runs (dataset_id, status, labels, etc.)
        """
        self.registry = registry
        self.storage = storage
        self.filters = filters or {}

        # Caches
        self._runs: Optional[List[Dict[str, Any]]] = None
        self._run_data_cache: Dict[str, RunData] = {}
        self._feature_matrix_cache: Dict[str, FeatureMatrix] = {}
        self._activation_matrix_cache: Dict[str, ActivationMatrix] = {}

        # Disk cache for activation matrices (lazy initialized)
        self._disk_cache: Optional["ActivationDiskCache"] = None

    @property
    def disk_cache(self) -> "ActivationDiskCache":
        """Get or create disk cache for activation matrices."""
        if self._disk_cache is None:
            from prompt_mining.analysis.disk_cache import ActivationDiskCache
            self._disk_cache = ActivationDiskCache(Path(self.storage.root))
        return self._disk_cache

    @property
    def runs(self) -> List[Dict[str, Any]]:
        """Get filtered runs (cached)."""
        if self._runs is None:
            self._runs = self.registry.get_runs(**self.filters)
        return self._runs

    def get_run_data(self, run_id: str) -> RunData:
        """Get RunData accessor for a specific run (cached)."""
        if run_id not in self._run_data_cache:
            self._run_data_cache[run_id] = RunData(run_id, self.storage, self.registry)
        return self._run_data_cache[run_id]

    def get_feature_matrix(
        self,
        top_k: int = 200,
        format: str = 'binary',
        max_ubiquity: float = 1.0,
        cache_key: Optional[str] = None
    ) -> FeatureMatrix:
        """
        Build Prompt×Feature matrix from feature_incidence tables.

        This method:
        1. Loads all feature_incidence/*.parquet files matching filters
        2. Counts feature frequency across prompts
        3. Filters features appearing in > max_ubiquity * n_prompts
        4. Selects top-K most frequent remaining features
        5. Builds matrix in specified format (binary/influence/activation)

        Args:
            top_k: Number of top features by frequency
            format: Matrix format:
                - 'binary': 0/1 (feature present/absent)
                - 'influence': Max influence value across positions
                - 'activation': Max activation value across positions
            max_ubiquity: Exclude features appearing in more than this fraction of prompts
                         (e.g., 0.9 = exclude features in >90% of prompts)
            cache_key: Optional key for caching (default: f"{format}_{top_k}_{max_ubiquity}")

        Returns:
            FeatureMatrix with:
                - matrix: (n_prompts, top_k) bfloat16
                - prompt_ids: List of prompt IDs
                - feature_coords: List of (layer, feature_idx) tuples
                - metadata: Dict with format, top_k, filters, etc.

        Size: ~4 MB for 10K prompts × 200 features (bfloat16)

        Example:
            >>> fm = analysis_data.get_feature_matrix(top_k=200, format='influence')
            >>> print(fm.matrix.shape)  # (n_prompts, 200)
            >>> print(fm.feature_coords[:5])  # [(10, 5421), (20, 8234), ...]
        """
        # Cache key
        if cache_key is None:
            cache_key = f"{format}_{top_k}_{max_ubiquity}"

        if cache_key in self._feature_matrix_cache:
            return self._feature_matrix_cache[cache_key]

        # Get runs
        runs = self.runs
        if not runs:
            raise ValueError(f"No runs found matching filters: {self.filters}")

        # Load all feature_incidence tables
        feature_tables = []
        for run in runs:
            table_path = self.storage.get_full_path(f"tables/feature_incidence/run={run['run_id']}.parquet")
            if self.storage.exists(table_path):
                df = pl.read_parquet(table_path)
                feature_tables.append(df)

        if not feature_tables:
            raise ValueError("No feature_incidence tables found for selected runs")

        # Concatenate all tables
        all_features = pl.concat(feature_tables)

        # Count feature frequency (unique prompts per feature)
        # NOTE: Polars 1.x uses `group_by` instead of the deprecated `groupby`.
        n_prompts = len(runs)
        max_count = int(n_prompts * max_ubiquity)
        
        feature_freq = (
            all_features
            .group_by(['layer', 'feature_idx'])
            .agg([
                pl.col('prompt_id').n_unique().alias('frequency'),
                pl.col('influence').max().alias('max_influence'),
                pl.col('activation_value').max().alias('max_activation')
            ])
            .filter(pl.col('frequency') <= max_count)
            .sort('frequency', descending=True)
            .head(top_k)
        )

        # Get top-K feature coordinates
        feature_coords = [
            (row['layer'], row['feature_idx'])
            for row in feature_freq.iter_rows(named=True)
        ]

        # Build matrix
        n_features = len(feature_coords)
        matrix = np.zeros((n_prompts, n_features), dtype=np.float16)

        prompt_id_to_idx = {run['prompt_id']: i for i, run in enumerate(runs)}
        feature_to_idx = {coord: i for i, coord in enumerate(feature_coords)}

        # Fill matrix based on format
        for row in all_features.iter_rows(named=True):
            prompt_id = row['prompt_id']
            feature_coord = (row['layer'], row['feature_idx'])

            if prompt_id in prompt_id_to_idx and feature_coord in feature_to_idx:
                i = prompt_id_to_idx[prompt_id]
                j = feature_to_idx[feature_coord]

                if format == 'binary':
                    matrix[i, j] = 1.0
                elif format == 'influence':
                    matrix[i, j] = max(matrix[i, j], row['influence'])
                elif format == 'activation':
                    matrix[i, j] = max(matrix[i, j], row['activation_value'])
                else:
                    raise ValueError(f"Unknown format: {format}")

        # Create FeatureMatrix
        feature_matrix = FeatureMatrix(
            matrix=matrix,
            prompt_ids=[run['prompt_id'] for run in runs],
            feature_coords=feature_coords,
            metadata={
                'format': format,
                'top_k': top_k,
                'max_ubiquity': max_ubiquity,
                'filters': self.filters,
                'n_runs': n_prompts,
                'total_features_seen': len(all_features),
            },
            run_ids=[run['run_id'] for run in runs]
        )

        # Cache and return
        self._feature_matrix_cache[cache_key] = feature_matrix
        return feature_matrix

    def _batch_load_plt_activations(
        self,
        runs: List[Dict[str, Any]],
        layer: int,
        position: str,
        d_sae: int,
        return_sparse: bool = True
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """
        Batch load PLT activations from parquet files.

        Builds explicit file paths from known run_ids and uses polars parallel
        reading internally. No glob or hive partitioning needed.

        Args:
            runs: List of run dictionaries with run_id keys
            layer: Layer index
            position: Position string ('last', 'first', or index like '0', '-1', '-5')
            d_sae: SAE dimension
            return_sparse: If True, return scipy.sparse.csr_matrix; if False, return dense

        Returns:
            Tuple of (matrix, valid_runs) where matrix is either dense or sparse
        """
        from pathlib import Path

        # Build explicit paths from known run_ids
        run_id_to_run = {run['run_id']: run for run in runs}
        all_paths = [
            (f"{self.storage.root}/tables/plt_activations/run={rid}/layer_{layer}.parquet", rid)
            for rid in run_id_to_run
        ]

        # Filter to existing files (batch Path.exists is fast)
        existing = [(p, rid) for p, rid in all_paths if Path(p).exists()]

        if not existing:
            raise ValueError(f"No PLT activation files found for layer {layer}")

        paths, valid_run_ids = zip(*existing)
        paths = list(paths)

        # Read all parquet files at once (polars parallelizes internally)
        print(f"Loading PLT activations (layer {layer}) from {len(paths)} files...")
        df = pl.read_parquet(paths)

        if df.height == 0:
            raise ValueError(f"No PLT activations found for layer {layer}")

        # Compute max position per run (needed for relative position filtering)
        max_pos_per_run = df.group_by('run_id').agg(
            pl.col('position').max().alias('max_position')
        )
        df = df.join(max_pos_per_run, on='run_id')

        # Handle position filtering
        if position == 'last':
            df = df.filter(pl.col('position') == pl.col('max_position'))
        elif position == 'first':
            df = df.filter(pl.col('position') == 0)
        else:
            pos_int = int(position)
            if pos_int < 0:
                # Relative to end: -1 = max_position, -5 = max_position - 4
                target_offset = -pos_int - 1
                df = df.filter(
                    pl.col('position') == (pl.col('max_position') - target_offset)
                )
            else:
                df = df.filter(pl.col('position') == pos_int)

        df = df.drop('max_position')

        if df.height == 0:
            raise ValueError(f"No activations found for position={position}")

        # Build run_id -> row_idx mapping
        unique_run_ids = df['run_id'].unique().sort()
        n_runs = unique_run_ids.len()

        mapping_df = pl.DataFrame({
            'run_id': unique_run_ids,
            'run_idx': pl.arange(0, n_runs, eager=True)
        })
        df = df.join(mapping_df, on='run_id')

        # Build valid_runs list in the same order as mapping
        valid_runs = [run_id_to_run[rid] for rid in unique_run_ids.to_list()]

        # Extract arrays for sparse matrix construction
        row_indices = df['run_idx'].to_numpy()
        col_indices = df['feature_idx'].to_numpy()
        values = df['activation_value'].to_numpy()

        # Filter out-of-bounds feature indices
        valid_mask = (col_indices >= 0) & (col_indices < d_sae)
        row_indices = row_indices[valid_mask]
        col_indices = col_indices[valid_mask]
        values = values[valid_mask]

        # Build sparse matrix
        matrix = csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(n_runs, d_sae),
            dtype=np.float32
        )

        print(f"Loaded {n_runs} runs, {matrix.nnz} non-zero activations")

        if return_sparse:
            return matrix, valid_runs
        else:
            return matrix.toarray(), valid_runs


    def _batch_load_raw_activations(
        self,
        runs: List[Dict[str, Any]],
        layer: int,
        position: Union[str, List[int]],
        n_workers: int = 8
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Union[str, List[int]]]:
        """
        Batch load raw activations using process parallelism.

        Uses ProcessPoolExecutor because zarr's internal async machinery
        blocks threading. Each process has its own event loop, achieving
        4-5x speedup over sequential/threaded loading.

        Args:
            runs: List of run dictionaries
            layer: Layer index
            position: Position string ('last', 'first', or index) or list of integer indices
            n_workers: Number of parallel processes (default 16)

        Returns:
            Tuple of (matrix, valid_runs, position) where:
                - matrix: (n_runs, d_model) if single position string,
                          (n_runs, n_positions, d_model) if list of ints
                - valid_runs: List of run dicts
                - position: Original position arg
        """
        from prompt_mining.utils.inference import resolve_positions

        # Get storage root for workers (they can't access self.storage)
        storage_root = str(self.storage.root)

        # Build args list: (run_id, layer, pos_indices, storage_root)
        args_list = []
        run_map = {}  # run_id -> run dict

        # Check if position is a list of ints (already resolved) or a string (needs resolution)
        is_int_list = isinstance(position, list)

        for run in runs:
            try:
                metadata = self.storage.read_metadata(run['run_id'])
                seq_len = metadata.get('input_length', 0)
                if seq_len == 0:
                    continue

                if is_int_list:
                    # Resolve negative indices relative to seq_len, then sort
                    pos_indices = sorted(p if p >= 0 else seq_len + p for p in position)
                else:
                    # Resolve string position to index
                    pos_indices = resolve_positions([position], seq_len)
                    if not pos_indices:
                        continue

                args_list.append((run['run_id'], layer, pos_indices, storage_root))
                run_map[run['run_id']] = run
            except Exception:
                continue

        if not args_list:
            raise ValueError(f"No valid runs found for position resolution")

        # Load in parallel using processes (bypasses zarr's async serialization)
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Use map with chunksize for efficiency
            chunksize = max(1, len(args_list) // (n_workers * 4))
            for result in tqdm(
                executor.map(_load_raw_activation_worker, args_list, chunksize=chunksize),
                total=len(args_list),
                desc=f"Loading raw activations (layer {layer})",
                miniters=1000
            ):
                results.append(result)

        # Collect valid results
        # Each activation is now (n_positions, d_model)
        valid_runs = []
        activations = []
        for run_id, act_matrix in results:
            if act_matrix is not None:
                valid_runs.append(run_map[run_id])
                activations.append(act_matrix)

        if not activations:
            raise ValueError(f"No valid raw activations found for layer={layer}, position={position}")

        # Sort by run_id for reproducibility (map preserves order, but let's be explicit)
        sorted_indices = sorted(range(len(valid_runs)), key=lambda i: valid_runs[i]['run_id'])
        valid_runs = [valid_runs[i] for i in sorted_indices]
        activations = [activations[i] for i in sorted_indices]

        # Stack: (n_runs, n_positions, d_model)
        matrix = np.stack(activations, axis=0).astype(np.float16)

        # Squeeze position dimension if effectively a single position (backward compat)
        # - position is a str (e.g. 'last') -> always a single resolved position
        # - position is a singleton list (e.g. [-5]) -> user intent is also a single position
        if (
            (isinstance(position, str) or (isinstance(position, list) and len(position) == 1))
            and matrix.ndim == 3
            and matrix.shape[1] == 1
        ):
            matrix = matrix[:, 0, :]  # (n_runs, d_model)

        return matrix, valid_runs, position

    def get_activation_matrix(
        self,
        layer: int,
        position: Union[str, List[int]] = 'last',
        space: str = 'plt',
        return_sparse: bool = True,
        cache_key: Optional[str] = None,
        force_reload: bool = False
    ) -> ActivationMatrix:
        """
        Build Prompt×d_sae matrix from sparse parquet or Zarr arrays.

        Works with both circuit_tracer transcoders and SAELens SAEs.

        This method:
        1. For PLT space: Uses batch loading for 2-5x speedup
        2. For raw space: Loads from Zarr arrays (supports multiple positions)
        3. Returns either dense or sparse matrix
        4. Caches results to disk for fast subsequent loads

        Args:
            layer: Which layer (0-27 for 28-layer model)
            position: Token position - either a string ('last', 'first', or index like '0', '-1')
                      or a list of integer indices (supports negative indexing).
                      Multiple positions (list) only supported for raw space.
            space: 'raw' (d_model), 'sae' or 'plt' (d_sae/d_transcoder, sparse)
            return_sparse: If True, return scipy.sparse.csr_matrix (sae/plt only); if False, return dense
            cache_key: Optional key for caching
            force_reload: If True, bypass disk cache and reload from source files

        Returns:
            ActivationMatrix with:
                - matrix: Shape depends on position arg:
                    - Single position (str): (n_prompts, d_model)
                    - Multiple positions (List[int], raw only): (n_prompts, n_positions, d_model)
                      (Special case: singleton list like [-5] returns (n_prompts, d_model).)
                - prompt_ids: List of prompt IDs
                - metadata: Dict with layer, position, space, is_sparse, filters, etc.

        Size: Dense ~320 MB, Sparse ~40 MB for 10K prompts × 16K features (98% sparsity)

        Example:
            >>> # Single position (backward compatible)
            >>> am = analysis_data.get_activation_matrix(layer=10, position='last', space='raw')
            >>> print(am.matrix.shape)  # (n_prompts, d_model)
            >>>
            >>> # Multiple positions (raw space only)
            >>> am = analysis_data.get_activation_matrix(layer=10, position=[-3, -2, -1], space='raw')
            >>> print(am.matrix.shape)  # (n_prompts, 3, d_model)
            >>>
            >>> # Force reload from source (bypass cache)
            >>> am = analysis_data.get_activation_matrix(layer=10, space='raw', force_reload=True)
        """
        # Build cache key from position
        if isinstance(position, list):
            positions_key = ','.join(str(p) for p in sorted(position))
        else:
            positions_key = position

        # Cache key
        if cache_key is None:
            sparse_str = 'sparse' if return_sparse else 'dense'
            cache_key = f"{space}_layer{layer}_{positions_key}_{sparse_str}"

        # Check in-memory cache first
        if cache_key in self._activation_matrix_cache and not force_reload:
            return self._activation_matrix_cache[cache_key]

        # Check disk cache (unless force_reload)
        disk_cache_path = self.disk_cache.get_path(space, layer, position, return_sparse)
        if not force_reload:
            cached = self.disk_cache.load(disk_cache_path, return_sparse)
            if cached is not None:
                print(f"Loaded from cache: {disk_cache_path.name}")
                self._activation_matrix_cache[cache_key] = cached
                return cached

        # Get runs
        runs = self.runs
        if not runs:
            raise ValueError(f"No runs found matching filters: {self.filters}")

        # For SAE/PLT space, use batch loading (OPTIMIZATION)
        # Accept both 'sae' and 'plt' as valid sparse activation spaces
        if space in ('plt', 'sae'):
            # Multiple positions not supported for PLT
            if isinstance(position, list):
                raise NotImplementedError("Multiple positions not supported for SAE/PLT space")

            # Get d_sae from first run's metadata (check both d_sae and d_transcoder)
            if not return_sparse:
                raise NotImplementedError("Dense loading is not supported for SAE/PLT space")
            d_sae = None
            for run in runs:
                try:
                    metadata = self.storage.read_metadata(run['run_id'])
                    d_sae = metadata.get('d_sae') or metadata.get('d_transcoder')
                    if d_sae:
                        break
                except Exception:
                    continue

            if d_sae is None:
                raise ValueError("Could not determine d_sae/d_transcoder from any run metadata")

            # Batch load all runs at once (single position)
            matrix, valid_runs = self._batch_load_plt_activations(
                runs=runs,
                layer=layer,
                position=position,
                d_sae=d_sae,
                return_sparse=return_sparse
            )
            loaded_position = position

        elif space == 'raw':
            # For raw space, use parallel batch loading (supports multiple positions)
            matrix, valid_runs, loaded_position = self._batch_load_raw_activations(
                runs=runs,
                layer=layer,
                position=position,
                n_workers=16
            )
        else:
            raise ValueError(f"Unknown space: {space}. Use 'sae', 'plt', or 'raw'")

        # Create ActivationMatrix
        activation_matrix = ActivationMatrix(
            matrix=matrix,
            prompt_ids=[run['prompt_id'] for run in valid_runs],
            metadata={
                'layer': layer,
                'position': loaded_position,
                'space': space,
                'is_sparse': return_sparse,
                'filters': self.filters,
                'n_runs': len(valid_runs),
                'd_model': matrix.shape[-1],  # Last dim is always d_model
            },
            run_ids=[run['run_id'] for run in valid_runs]
        )

        # Save to disk cache
        self.disk_cache.save(disk_cache_path, activation_matrix, return_sparse)
        print(f"Saved to cache: {disk_cache_path.name}")

        # Cache in memory and return
        self._activation_matrix_cache[cache_key] = activation_matrix
        return activation_matrix


class Analyzer(ABC):
    """
    Abstract base class for all analyzers.

    All analyzers operate on AnalysisData and produce results as dictionaries.

    Example:
        >>> class MyAnalyzer(Analyzer):
        ...     def run(self, data: AnalysisData) -> Dict[str, Any]:
        ...         feature_matrix = data.get_feature_matrix(top_k=200)
        ...         # Analyze...
        ...         return {'result': ...}
    """

    @abstractmethod
    def run(self, data: AnalysisData, **kwargs) -> Dict[str, Any]:
        """
        Run analysis on data.

        Args:
            data: AnalysisData accessor
            **kwargs: Additional analyzer-specific parameters

        Returns:
            Dictionary with analysis results
        """
        pass

    def save_results(self, results: Dict[str, Any], output_dir: Path):
        """
        Save analysis results to disk.

        Default implementation saves as JSON, but subclasses can override
        to save plots, matrices, etc.

        Args:
            results: Results dictionary from run()
            output_dir: Directory to save results
        """
        import json
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON-serializable results
        json_path = output_dir / f"{self.__class__.__name__}_results.json"
        with open(json_path, 'w') as f:
            # Filter out non-serializable objects (numpy arrays, etc.)
            serializable = {
                k: v for k, v in results.items()
                if isinstance(v, (str, int, float, bool, list, dict, type(None)))
            }
            json.dump(serializable, f, indent=2)