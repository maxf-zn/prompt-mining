"""
StorageBackend: Abstract base class for storage implementations.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import zarr
import numcodecs


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.

    Storage backends handle all file I/O operations for the platform.
    They are URI-configured (not factory-based):
    - LocalStorage: file:///path/to/storage
    - S3Storage: s3://bucket/prefix

    The backend provides:
    - CompactGraph storage (.pt format)
    - Zarr activation arrays (one group per run)
    - JSON metadata
    - Feature incidence tables (Parquet)
    - Registry database

    Storage layout:
        {storage_root}/
          registry.sqlite
          runs/YYYY-MM-DD/{run_id}/
            compact_graph.pt
            acts.zarr/
            topk_logits.json
            generation.json
            manifest.json
            run.log
          tables/
            feature_incidence/run={run_id}.parquet
          pipeline.log
    """

    def __init__(self, uri: str):
        """
        Initialize storage backend.

        Args:
            uri: Storage URI (e.g., "file:///path" or "s3://bucket/prefix")
        """
        self.uri = uri

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if path exists."""
        pass

    @abstractmethod
    def read_bytes(self, path: str) -> bytes:
        """Read file as bytes."""
        pass

    @abstractmethod
    def write_bytes(self, path: str, data: bytes):
        """Write bytes to file."""
        pass

    @abstractmethod
    def makedirs(self, path: str, exist_ok: bool = True):
        """Create directory."""
        pass

    @abstractmethod
    def get_full_path(self, relative_path: str) -> str:
        """Get full path from relative path."""
        pass

    # High-level operations (implemented in terms of primitives)

    def write_compact_graph(self, run_id: str, compact_graph):
        """
        Write CompactGraph to storage.

        Args:
            run_id: Unique run identifier
            compact_graph: CompactGraph instance
        """
        from ..core import CompactGraph

        # Create run directory
        run_dir = self._get_run_dir(run_id)
        self.makedirs(run_dir, exist_ok=True)

        # Write to temporary file then move (atomic-ish)
        graph_path = f"{run_dir}/compact_graph.pt"
        compact_graph.to_pt(graph_path)

    def read_compact_graph(self, run_id: str):
        """
        Read CompactGraph from storage.

        Args:
            run_id: Unique run identifier

        Returns:
            CompactGraph instance
        """
        from ..core import CompactGraph

        run_dir = self._get_run_dir(run_id)
        graph_path = f"{run_dir}/compact_graph.pt"

        if not self.exists(graph_path):
            raise FileNotFoundError(f"CompactGraph not found: {graph_path}")

        return CompactGraph.from_pt(graph_path)

    def write_activations(self, run_id: str, activations: Dict[str, np.ndarray],
                         metadata: Optional[Dict[str, Any]] = None):
        """
        Write Zarr activation arrays.

        Args:
            run_id: Unique run identifier
            activations: Dict mapping layer name to activation array
                        e.g., {'raw/layer10': (n_pos, d_model), ...}
            metadata: Optional metadata to store in .zattrs
        """
        import zarr

        run_dir = self._get_run_dir(run_id)
        self.makedirs(run_dir, exist_ok=True)

        zarr_path = f"{run_dir}/acts.zarr"
        root = zarr.open(zarr_path, mode='w')

        # Store metadata
        if metadata:
            root.attrs.update(metadata)

        # Write each layer's activations
        for layer_name, array in activations.items():
            root.create_array(
                layer_name,
                data=array,
                chunks=(array.shape[0], min(1024, array.shape[1])),  # Chunk along feature dim
                compressors=[zarr.codecs.BloscCodec(cname='zstd', clevel=3)]
            )

    def read_activations(self, run_id: str) -> zarr.Group:
        """
        Read Zarr activations.

        Args:
            run_id: Unique run identifier

        Returns:
            Zarr group handle
        """
        import zarr

        run_dir = self._get_run_dir(run_id)
        zarr_path = f"{run_dir}/acts.zarr"

        if not self.exists(zarr_path):
            raise FileNotFoundError(f"Activations not found: {zarr_path}")

        return zarr.open(zarr_path, mode='r')

    def get_activations_path(self, run_id: str) -> Path:
        """Get path to Zarr activations directory."""
        run_dir = self._get_run_dir(run_id)
        return Path(f"{run_dir}/acts.zarr")

    def write_metadata(self, run_id: str, metadata: Dict[str, Any]):
        """
        Write run metadata as JSON.

        Args:
            run_id: Unique run identifier
            metadata: Metadata dictionary
        """
        import json

        run_dir = self._get_run_dir(run_id)
        self.makedirs(run_dir, exist_ok=True)

        manifest_path = f"{run_dir}/manifest.json"
        json_str = json.dumps(metadata, indent=2)
        self.write_bytes(manifest_path, json_str.encode('utf-8'))

    def read_metadata(self, run_id: str) -> Dict[str, Any]:
        """
        Read run metadata from JSON.

        Args:
            run_id: Unique run identifier

        Returns:
            Metadata dictionary
        """
        import json

        run_dir = self._get_run_dir(run_id)
        manifest_path = f"{run_dir}/manifest.json"

        if not self.exists(manifest_path):
            raise FileNotFoundError(f"Metadata not found: {manifest_path}")

        data = self.read_bytes(manifest_path)
        return json.loads(data.decode('utf-8'))

    def write_json(self, run_id: str, filename: str, data: Dict[str, Any]):
        """
        Write arbitrary JSON file to run directory.

        Args:
            run_id: Unique run identifier
            filename: Name of JSON file (e.g., "topk_logits.json")
            data: Data to serialize as JSON
        """
        import json

        run_dir = self._get_run_dir(run_id)
        self.makedirs(run_dir, exist_ok=True)

        file_path = f"{run_dir}/{filename}"
        json_str = json.dumps(data, indent=2)
        self.write_bytes(file_path, json_str.encode('utf-8'))

    def read_json(self, run_id: str, filename: str) -> Dict[str, Any]:
        """
        Read arbitrary JSON file from run directory.

        Args:
            run_id: Unique run identifier
            filename: Name of JSON file

        Returns:
            Deserialized JSON data
        """
        import json

        run_dir = self._get_run_dir(run_id)
        file_path = f"{run_dir}/{filename}"

        if not self.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        data = self.read_bytes(file_path)
        return json.loads(data.decode('utf-8'))

    def write_feature_incidence(self, run_id: str, df):
        """
        Write feature incidence table as Parquet.

        Args:
            run_id: Unique run identifier
            df: pandas DataFrame with feature incidence data
        """
        tables_dir = self.get_full_path("tables/feature_incidence")
        self.makedirs(tables_dir, exist_ok=True)

        parquet_path = f"{tables_dir}/run={run_id}.parquet"
        df.to_parquet(parquet_path, index=False)

    def write_plt_activations(self, run_id: str, layer: int, df):
        """
        Write sparse PLT activations table as Parquet (one file per layer).

        Files are organized as: tables/plt_activations/run={run_id}/layer_{layer}.parquet
        This matches the raw activations hierarchy: runs/{date}/{run_id}/acts.zarr/raw/layer{layer}

        Args:
            run_id: Unique run identifier
            layer: Layer index
            df: pandas DataFrame with sparse PLT activations for this layer
                Expected columns: run_id, position, feature_idx, activation_value
        """
        run_dir = self.get_full_path(f"tables/plt_activations/run={run_id}")
        self.makedirs(run_dir, exist_ok=True)

        parquet_path = f"{run_dir}/layer_{layer}.parquet"
        df.to_parquet(parquet_path, index=False)

    def _get_run_dir(self, run_id: str) -> str:
        """
        Get run directory path with date-based partitioning.

        Args:
            run_id: Unique run identifier (format: YYYY-MM-DD_{uuid})

        Returns:
            Full path to run directory
        """
        import datetime

        # Extract date from run_id or use current date
        # run_id format: YYYY-MM-DD_{uuid}
        if '_' in run_id and len(run_id.split('_')[0]) == 10:  # Check if first part is date format
            date_part = run_id.split('_')[0]
        else:
            date_part = datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d')

        return self.get_full_path(f"runs/{date_part}/{run_id}")

    def get_registry_path(self) -> str:
        """Get path to SQLite registry database."""
        return self.get_full_path("registry.sqlite")
