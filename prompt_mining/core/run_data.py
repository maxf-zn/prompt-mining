"""
RunData: Accessor for per-run artifacts and metadata.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class RunData:
    """
    Accessor for run artifacts and metadata.

    This provides a convenient interface for loading run data during analysis.
    It lazy-loads artifacts on demand to minimize memory usage.

    Attributes:
        run_id: Unique run identifier
        storage: StorageBackend instance for loading artifacts
        registry: Registry instance for loading metadata
    """
    run_id: str
    storage: Any  # StorageBackend (avoid circular import)
    registry: Any  # Registry (avoid circular import)

    _compact_graph: Optional[Any] = None
    _metadata: Optional[Dict[str, Any]] = None

    def get_compact_graph(self):
        """Lazy-load CompactGraph."""
        if self._compact_graph is None:
            self._compact_graph = self.storage.read_compact_graph(self.run_id)
        return self._compact_graph

    def get_metadata(self) -> Dict[str, Any]:
        """Lazy-load run metadata."""
        if self._metadata is None:
            self._metadata = self.storage.read_metadata(self.run_id)
        return self._metadata

    def get_activations_path(self) -> Path:
        """Get path to Zarr activations."""
        return self.storage.get_activations_path(self.run_id)

    def get_registry_entry(self) -> Dict[str, Any]:
        """Get registry entry for this run."""
        return self.registry.get_run(self.run_id)
