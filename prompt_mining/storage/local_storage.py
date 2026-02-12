"""
LocalStorage: File system-based storage implementation.
"""
from pathlib import Path
from typing import Optional
from .base import StorageBackend


class LocalStorage(StorageBackend):
    """
    Local filesystem storage backend.

    Uses standard file I/O operations. Suitable for single-host deployments
    and development. For multi-host or large-scale deployments, use S3Storage.

    URI format: file:///absolute/path/to/storage
                or just /absolute/path/to/storage

    Example:
        storage = LocalStorage("file:///home/user/prompt_mining_data")
        storage = LocalStorage("/home/user/prompt_mining_data")
    """

    def __init__(self, uri: str):
        """
        Initialize local storage backend.

        Args:
            uri: Storage URI (file:///path or /path)
        """
        super().__init__(uri)

        # Strip file:// prefix if present
        if str(uri).startswith('file://'):
            self.root = Path(uri[7:])
        else:
            self.root = Path(uri)

        # Create root directory if it doesn't exist
        self.root.mkdir(parents=True, exist_ok=True)

    def exists(self, path: str) -> bool:
        """Check if path exists."""
        full_path = Path(path) if Path(path).is_absolute() else self.root / path
        return full_path.exists()

    def read_bytes(self, path: str) -> bytes:
        """Read file as bytes."""
        full_path = Path(path) if Path(path).is_absolute() else self.root / path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")
        return full_path.read_bytes()

    def write_bytes(self, path: str, data: bytes):
        """Write bytes to file."""
        full_path = Path(path) if Path(path).is_absolute() else self.root / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(data)

    def makedirs(self, path: str, exist_ok: bool = True):
        """Create directory."""
        full_path = Path(path) if Path(path).is_absolute() else self.root / path
        full_path.mkdir(parents=True, exist_ok=exist_ok)

    def get_full_path(self, relative_path: str) -> str:
        """
        Get full path from relative path.

        Args:
            relative_path: Relative path from storage root

        Returns:
            Absolute path as string
        """
        if Path(relative_path).is_absolute():
            return relative_path
        return str(self.root / relative_path)

    def list_runs(self, date: Optional[str] = None) -> list:
        """
        List all run IDs, optionally filtered by date.

        Args:
            date: Optional date string (YYYY-MM-DD) to filter runs

        Returns:
            List of run IDs
        """
        runs_dir = self.root / "runs"
        if not runs_dir.exists():
            return []

        run_ids = []
        if date:
            date_dir = runs_dir / date
            if date_dir.exists():
                for run_dir in date_dir.iterdir():
                    if run_dir.is_dir():
                        run_ids.append(run_dir.name)
        else:
            # List all dates
            for date_dir in runs_dir.iterdir():
                if date_dir.is_dir():
                    for run_dir in date_dir.iterdir():
                        if run_dir.is_dir():
                            run_ids.append(run_dir.name)

        return sorted(run_ids)

    def delete_run(self, run_id: str):
        """
        Delete all artifacts for a run.

        Args:
            run_id: Unique run identifier
        """
        import shutil

        run_dir = Path(self._get_run_dir(run_id))
        if run_dir.exists():
            shutil.rmtree(run_dir)

        # Delete feature incidence table
        feature_table = self.root / "tables" / "feature_incidence" / f"run={run_id}.parquet"
        if feature_table.exists():
            feature_table.unlink()

    def get_storage_stats(self) -> dict:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        import os

        def get_dir_size(path: Path) -> int:
            """Calculate total size of directory."""
            total = 0
            if path.exists():
                for entry in path.rglob('*'):
                    if entry.is_file():
                        total += entry.stat().st_size
            return total

        runs_dir = self.root / "runs"
        tables_dir = self.root / "tables"

        return {
            'root': str(self.root),
            'total_size_gb': get_dir_size(self.root) / (1024**3),
            'runs_size_gb': get_dir_size(runs_dir) / (1024**3),
            'tables_size_gb': get_dir_size(tables_dir) / (1024**3),
            'num_runs': len(self.list_runs()),
        }
