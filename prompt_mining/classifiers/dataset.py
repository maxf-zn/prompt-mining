"""
ClassificationDataset - Unified data loading for classification tasks.

Wraps AnalysisData to provide X, y, datasets for classifier training,
plus convenient LODO (Leave-One-Dataset-Out) split iteration.
"""
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Iterator, Optional, Dict, Any, Literal, Union, List, Sequence
from dataclasses import dataclass


@dataclass
class ClassificationData:
    """Container for classification data.

    Attributes:
        X: Feature matrix. Shape depends on position arg:
           - Single position (str): (n_samples, n_features)
           - Multiple positions (List[int]): (n_samples, n_positions, n_features)
        y: Labels (n_samples,)
        datasets: Dataset IDs (n_samples,)
        run_ids: Run IDs for traceability (n_samples,)
        metadata: Additional info (layer, position, space, etc.)
    """
    X: np.ndarray
    y: np.ndarray
    datasets: np.ndarray
    run_ids: np.ndarray
    metadata: Dict[str, Any]


class ClassificationDataset:
    """
    Unified data loader for classification tasks.

    Wraps AnalysisData and registry to provide:
    - Feature matrices (X) from activations
    - Labels (y) from run metadata
    - Dataset IDs for LODO evaluation
    - Convenient LODO split iteration

    Example:
        >>> from prompt_mining.classifiers import ClassificationDataset
        >>>
        >>> # From data directory
        >>> dataset = ClassificationDataset.from_path("/path/to/ingestion/output")
        >>> data = dataset.load(layer=23, space='raw')
        >>> print(f"Shape: {data.X.shape}, Labels: {data.y.mean():.1%} malicious")
        >>>
        >>> # LODO iteration
        >>> for test_ds, train_mask, test_mask in dataset.lodo_splits(data):
        ...     X_train, y_train = data.X[train_mask], data.y[train_mask]
        ...     X_test, y_test = data.X[test_mask], data.y[test_mask]
        ...     # Train and evaluate classifier
    """

    def __init__(
        self,
        registry,
        storage,
        filters: Optional[Dict[str, Any]] = None,
        label_key: str = 'malicious',
    ):
        """
        Initialize ClassificationDataset.

        Args:
            registry: SQLiteRegistry instance
            storage: LocalStorage instance
            filters: Filters for AnalysisData (default: {'status': 'completed'})
            label_key: Key in prompt_labels to use as binary label (default: 'malicious')
        """
        from prompt_mining.analysis import AnalysisData

        self.registry = registry
        self.storage = storage
        self.label_key = label_key
        self.filters = filters or {'status': 'completed'}

        self._analysis_data = AnalysisData(
            registry=registry,
            storage=storage,
            filters=self.filters,
        )

        # Cache run metadata for label extraction
        self._run_lookup: Optional[Dict[str, Dict]] = None

    @classmethod
    def from_path(
        cls,
        data_dir: Union[str, Path],
        label_key: str = 'malicious',
        filters: Optional[Dict[str, Any]] = None,
    ) -> "ClassificationDataset":
        """
        Create ClassificationDataset from a data directory path.

        Args:
            data_dir: Path to directory containing registry.sqlite and storage
            label_key: Key in prompt_labels for binary label
            filters: Optional filters for AnalysisData

        Returns:
            ClassificationDataset instance

        Example:
            >>> dataset = ClassificationDataset.from_path("/path/to/ingestion/output")
        """
        from prompt_mining.registry import SQLiteRegistry
        from prompt_mining.storage import LocalStorage

        data_path = Path(data_dir)
        registry = SQLiteRegistry(data_path / "registry.sqlite")
        storage = LocalStorage(f"file://{data_path}")

        return cls(
            registry=registry,
            storage=storage,
            filters=filters,
            label_key=label_key,
        )

    def _get_run_lookup(self) -> Dict[str, Dict]:
        """Get or create run metadata lookup."""
        if self._run_lookup is None:
            all_runs = self.registry.get_runs(**self.filters)
            self._run_lookup = {r['run_id']: r for r in all_runs}
        return self._run_lookup

    def load(
        self,
        layer: int = 23,
        position: Union[str, List[int]] = 'last',
        space: Literal['raw', 'plt'] = 'raw',
        return_sparse: bool = False,
        force_reload: bool = False,
    ) -> ClassificationData:
        """
        Load classification data from activations.

        Args:
            layer: Layer to extract activations from (default: 23)
            position: Token position - either a string ('last', 'first', or index like '0', '-1')
                      or a list of integer indices (supports negative indexing).
                      Multiple positions (list) only supported for raw space.
                      - Single position (str): X shape is (n_samples, n_features)
                      - Multiple positions (List[int]): X shape is (n_samples, n_positions, n_features)
            space: Activation space ('raw' for model activations, 'plt' for SAE)
            return_sparse: If True and space='plt', return sparse matrix
            force_reload: If True, bypass disk cache and reload from source files

        Returns:
            ClassificationData with X, y, datasets, run_ids, metadata

        Example:
            >>> # Single position (default)
            >>> data = dataset.load(layer=23, space='raw')
            >>> print(f"X: {data.X.shape}")  # (n_samples, d_model)
            >>>
            >>> # Multiple positions
            >>> data = dataset.load(layer=23, position=[-3, -2, -1], space='raw')
            >>> print(f"X: {data.X.shape}")  # (n_samples, 3, d_model)
            >>>
            >>> # Force reload (bypass cache)
            >>> data = dataset.load(layer=23, space='raw', force_reload=True)
        """
        # Load activation matrix
        activation_matrix = self._analysis_data.get_activation_matrix(
            layer=layer,
            position=position,
            space=space,
            return_sparse=return_sparse,
            force_reload=force_reload,
        )

        X = activation_matrix.matrix
        run_ids = np.array(activation_matrix.run_ids)

        # Extract labels and dataset IDs from run metadata
        run_lookup = self._get_run_lookup()

        datasets = np.array([
            run_lookup[rid]['dataset_id']
            for rid in run_ids
        ])

        y = np.array([
            int(run_lookup[rid]['prompt_labels'][self.label_key])
            for rid in run_ids
        ])

        # Get position from activation matrix metadata
        loaded_position = activation_matrix.metadata.get('position', position)
        n_positions = len(loaded_position) if isinstance(loaded_position, list) else 1

        return ClassificationData(
            X=X,
            y=y,
            datasets=datasets,
            run_ids=run_ids,
            metadata={
                'layer': layer,
                'position': loaded_position,
                'space': space,
                'label_key': self.label_key,
                'n_samples': len(y),
                'n_features': X.shape[-1],  # Last dim is always d_model
                'n_positions': n_positions,
                'n_datasets': len(np.unique(datasets)),
                'class_balance': float(y.mean()),
            }
        )

    def get_run_ids(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Get run IDs matching filters (no data loading).

        Args:
            filters: Optional registry filters (defaults to self.filters)

        Returns:
            List of run_ids

        Example:
            >>> run_ids = dataset.get_run_ids()
            >>> len(run_ids)  # Number of matching runs
        """
        effective_filters = filters or self.filters or {}
        runs = self.registry.get_runs(**effective_filters)
        return [r['run_id'] for r in runs]

    def load_labels(
        self,
        label_keys: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load labels-only metadata per run (no activations).

        This works for:
        - Full ingestion directories (registry.sqlite + artifacts)
        - Labels-only directories (registry.sqlite only)

        Args:
            label_keys: If provided, only include these keys from prompt_labels.
                       If None, include the full prompt_labels dict.
            filters: Optional registry filters (defaults to self.filters, e.g. {'status':'completed'})

        Returns:
            List of dicts, one per run, with keys:
              - run_id, prompt_id, dataset_id, prompt_labels
        """
        effective_filters = filters or self.filters or {}
        runs = self.registry.get_runs(**effective_filters)
        out: List[Dict[str, Any]] = []
        for r in runs:
            labels = r.get("prompt_labels") or {}
            if label_keys is not None:
                labels = {k: labels.get(k) for k in label_keys}
            out.append(
                {
                    "run_id": r.get("run_id"),
                    "prompt_id": r.get("prompt_id"),
                    "dataset_id": r.get("dataset_id"),
                    "prompt_labels": labels,
                }
            )
        return out

    def load_features(
        self,
        top_k: int = 200,
        format: Literal['binary', 'activation', 'influence'] = 'binary',
        max_ubiquity: float = 1.0,
    ) -> ClassificationData:
        """
        Load classification data from feature incidence tables.

        Args:
            top_k: Number of top features to include
            format: Feature matrix format
            max_ubiquity: Exclude features appearing in more than this fraction

        Returns:
            ClassificationData with sparse/dense feature matrix

        Example:
            >>> data = dataset.load_features(top_k=500, format='binary')
        """
        feature_matrix = self._analysis_data.get_feature_matrix(
            top_k=top_k,
            format=format,
            max_ubiquity=max_ubiquity,
        )

        X = feature_matrix.matrix
        run_ids = np.array(feature_matrix.run_ids)

        run_lookup = self._get_run_lookup()

        datasets = np.array([
            run_lookup[rid]['dataset_id']
            for rid in run_ids
        ])

        y = np.array([
            int(run_lookup[rid]['prompt_labels'][self.label_key])
            for rid in run_ids
        ])

        return ClassificationData(
            X=X,
            y=y,
            datasets=datasets,
            run_ids=run_ids,
            metadata={
                'mode': 'features',
                'top_k': top_k,
                'format': format,
                'max_ubiquity': max_ubiquity,
                'label_key': self.label_key,
                'n_samples': len(y),
                'n_features': X.shape[1],
                'n_datasets': len(np.unique(datasets)),
                'class_balance': float(y.mean()),
                'feature_coords': feature_matrix.feature_coords,
            }
        )

    def lodo_splits(
        self,
        data: ClassificationData,
        merge_datasets: Optional[Dict[str, str]] = None,
    ) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
        """
        Iterate over Leave-One-Dataset-Out splits.

        Args:
            data: ClassificationData from load() or load_features()
            merge_datasets: Optional dict mapping dataset names to merge targets
                           e.g., {'gandalf_summarization': 'mosscap'}

        Yields:
            (test_dataset_name, train_mask, test_mask) for each fold

        Example:
            >>> data = dataset.load(layer=23)
            >>> for test_ds, train_mask, test_mask in dataset.lodo_splits(data):
            ...     print(f"Testing on {test_ds}: {test_mask.sum()} samples")
            ...     X_train, y_train = data.X[train_mask], data.y[train_mask]
            ...     X_test, y_test = data.X[test_mask], data.y[test_mask]
        """
        datasets = data.datasets.copy()

        # Merge small datasets if specified
        if merge_datasets:
            for src, dst in merge_datasets.items():
                datasets[datasets == src] = dst

        unique_datasets = np.unique(datasets)

        for test_ds in unique_datasets:
            test_mask = datasets == test_ds
            train_mask = ~test_mask
            yield test_ds, train_mask, test_mask

    def get_dataset_counts(self, data: ClassificationData) -> Dict[str, int]:
        """
        Get sample counts per dataset.

        Args:
            data: ClassificationData instance

        Returns:
            Dict mapping dataset name to sample count
        """
        unique, counts = np.unique(data.datasets, return_counts=True)
        return dict(zip(unique, counts))

    def summary(self, data: ClassificationData) -> str:
        """
        Get human-readable summary of the dataset.

        Args:
            data: ClassificationData instance

        Returns:
            Formatted string summary
        """
        lines = [
            "ClassificationDataset Summary",
            "=" * 40,
            f"Samples: {data.metadata['n_samples']}",
            f"Features: {data.metadata['n_features']}",
            f"Datasets: {data.metadata['n_datasets']}",
            f"Class balance: {data.metadata['class_balance']:.1%} positive",
            "",
            "Per-dataset breakdown:",
        ]

        counts = self.get_dataset_counts(data)
        for ds, count in sorted(counts.items(), key=lambda x: -x[1]):
            ds_mask = data.datasets == ds
            ds_positive = data.y[ds_mask].mean()
            lines.append(f"  {ds}: {count} samples ({ds_positive:.1%} positive)")

        return "\n".join(lines)

    def get_prompts(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Get prompt texts for samples (no activation loading required).

        Reads prompt text from manifest.json files in the run storage.

        Args:
            filters: Optional registry filters (defaults to self.filters)

        Returns:
            List of prompt texts (in same order as get_run_ids())

        Example:
            >>> # Get all prompts (no activation loading)
            >>> prompts = dataset.get_prompts()
            >>>
            >>> # Get prompts with custom filters
            >>> prompts = dataset.get_prompts(filters={'dataset_id': 'tensortrust'})
        """
        run_ids = self.get_run_ids(filters=filters)
        return [self._get_prompt_for_run(rid) for rid in run_ids]

    def _get_prompt_for_run(self, run_id: str) -> str:
        """
        Load prompt text for a single run from manifest.json.

        Args:
            run_id: Run ID (e.g., '2025-12-04_48f54681')

        Returns:
            Prompt text string, or empty string if not found
        """
        # Use storage's _get_run_dir to find the run directory
        run_dir = Path(self.storage._get_run_dir(run_id))
        manifest_path = run_dir / "manifest.json"

        if not manifest_path.exists():
            return ""

        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            return manifest.get('input_text', '')
        except (json.JSONDecodeError, IOError):
            return ""

    def get_prompt_by_run_id(self, run_id: str) -> str:
        """
        Get prompt text for a specific run ID.

        Args:
            run_id: Run ID (e.g., '2025-12-04_48f54681')

        Returns:
            Prompt text string

        Example:
            >>> prompt = dataset.get_prompt_by_run_id('2025-12-04_48f54681')
            >>> print(prompt[:200])
        """
        return self._get_prompt_for_run(run_id)
