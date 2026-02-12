"""
SQLiteRegistry: SQLite-based run tracking and idempotency.
"""
import sqlite3
import json
import hashlib
from pathlib import Path
import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


class RunStatus(Enum):
    """Run status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SQLiteRegistry:
    """
    SQLite-based registry for run tracking and idempotency.

    The registry stores:
    - Run metadata and status
    - Idempotency keys (run_key, processing_fingerprint)
    - Artifact paths and sizes
    - Versioning (supersedes chain)

    For MVP, we use standard SQLite (no WAL mode). This is sufficient for
    single-host with multiple workers since each worker only updates its own runs.

    Future: Drop-in replacement with Postgres for multi-host orchestration.
    """

    def __init__(self, db_path: str):
        """
        Initialize SQLite registry.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema."""
        schema_path = Path(__file__).parent / "schema.sql"
        schema_sql = schema_path.read_text()

        with self._connect() as conn:
            conn.executescript(schema_sql)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        """Create database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn

    def create_run(
        self,
        run_id: str,
        prompt_id: str,
        dataset_id: str,
        model_hash: str,
        clt_hash: str,
        seed: int,
        run_key: str,
        processing_fingerprint: str,
        prompt_labels: Optional[Dict[str, Any]] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
        shard_idx: Optional[int] = None,
        num_shards: Optional[int] = None,
        worker_id: Optional[int] = None,
        device_id: Optional[int] = None,
        supersedes: Optional[str] = None,
    ) -> str:
        """
        Create a new run entry with minimal initial metadata.

        Args:
            run_id: Unique run identifier
            prompt_id: Prompt identifier
            dataset_id: Dataset identifier
            model_hash: Model weight hash
            clt_hash: Transcoder weight hash
            seed: Random seed
            run_key: Idempotency key (identifies unique prompt+model combo)
            processing_fingerprint: Config version hash
            prompt_labels: Prompt metadata labels
            config_snapshot: Full config used for this run
            shard_idx: Shard index (for multi-GPU)
            num_shards: Total number of shards
            worker_id: Worker ID
            device_id: Device ID
            supersedes: run_id of previous version (if config changed)

        Returns:
            run_id
        """
        timestamp = datetime.datetime.now(datetime.UTC).isoformat()

        # Use empty dicts by default so callers (like ingestion) can defer
        # storing full labels and config until finalize_run().
        if prompt_labels is None:
            prompt_labels = {}
        if config_snapshot is None:
            config_snapshot = {}

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, prompt_id, dataset_id, timestamp, status,
                    shard_idx, num_shards, worker_id, device_id,
                    model_hash, clt_hash, seed, run_key, processing_fingerprint,
                    supersedes, prompt_labels, config_snapshot
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, prompt_id, dataset_id, timestamp, RunStatus.PENDING.value,
                    shard_idx, num_shards, worker_id, device_id,
                    model_hash, clt_hash, seed, run_key, processing_fingerprint,
                    supersedes,
                    json.dumps(prompt_labels),
                    json.dumps(config_snapshot)
                )
            )
            conn.commit()

        return run_id

    def update_status(self, run_id: str, status: str, error_message: Optional[str] = None):
        """
        Update run status.

        Args:
            run_id: Unique run identifier
            status: New status (pending/running/completed/failed)
            error_message: Optional error message (for failed runs)
        """
        # Validate status
        try:
            RunStatus(status)
        except ValueError:
            raise ValueError(f"Invalid status: {status}")

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?, error_message = ?
                WHERE run_id = ?
                """,
                (status, error_message, run_id)
            )
            conn.commit()

    def finalize_run(
        self,
        run_id: str,
        status: str,
        prompt_labels: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Finalize a run after processing is complete.

        This performs a single atomic update of the run's terminal status,
        error_message (if any), and full prompt_labels snapshot.

        Args:
            run_id: Unique run identifier
            status: Final status ('completed', 'failed', etc.)
            prompt_labels: Final labels dict to persist (including eval results)
            error_message: Optional error message for failed runs
        """
        # Validate status using existing enum
        try:
            RunStatus(status)
        except ValueError:
            raise ValueError(f"Invalid status: {status}")

        with self._connect() as conn:
            if prompt_labels is not None:
                conn.execute(
                    """
                    UPDATE runs
                    SET status = ?, error_message = ?, prompt_labels = ?
                    WHERE run_id = ?
                    """,
                    (status, error_message, json.dumps(prompt_labels), run_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE runs
                    SET status = ?, error_message = ?
                    WHERE run_id = ?
                    """,
                    (status, error_message, run_id),
                )
            conn.commit()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get run by ID.

        Args:
            run_id: Unique run identifier

        Returns:
            Run record as dictionary, or None if not found
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,)
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_dict(row)

    def get_run_by_key(
        self,
        run_key: str,
        processing_fingerprint: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get run by idempotency key.

        Args:
            run_key: Idempotency key
            processing_fingerprint: Config version hash

        Returns:
            Run record as dictionary, or None if not found
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM runs
                WHERE run_key = ? AND processing_fingerprint = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (run_key, processing_fingerprint)
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_dict(row)

    def exists(
        self,
        run_key: str,
        processing_fingerprint: str,
        status: Optional[str] = None
    ) -> bool:
        """
        Check if run exists with given key and fingerprint.

        Args:
            run_key: Idempotency key
            processing_fingerprint: Config version hash
            status: Optional status filter (e.g., "completed")

        Returns:
            True if run exists, False otherwise
        """
        with self._connect() as conn:
            if status:
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) FROM runs
                    WHERE run_key = ? AND processing_fingerprint = ? AND status = ?
                    """,
                    (run_key, processing_fingerprint, status)
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) FROM runs
                    WHERE run_key = ? AND processing_fingerprint = ?
                    """,
                    (run_key, processing_fingerprint)
                )

            count = cursor.fetchone()[0]

        return count > 0

    def get_runs(
        self,
        dataset_id: Optional[str] = None,
        status: Optional[str] = None,
        labels: Optional[Dict[str, Any]] = None,
        latest_only: bool = False,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query runs with filters.

        Args:
            dataset_id: Filter by dataset
            status: Filter by status
            labels: Filter by label values (e.g., {"success": True})
            latest_only: Only return latest version per run_key
            limit: Maximum number of results

        Returns:
            List of run records
        """
        query = "SELECT * FROM runs WHERE 1=1"
        params = []

        if dataset_id:
            query += " AND dataset_id = ?"
            params.append(dataset_id)

        if status:
            query += " AND status = ?"
            params.append(status)

        # Label filtering (simple JSON string matching for MVP)
        if labels:
            for key, value in labels.items():
                # This is a simple approach; for complex queries use JSON functions
                query += f" AND prompt_labels LIKE ?"
                params.append(f'%"{key}": {json.dumps(value)}%')

        query += " ORDER BY timestamp DESC"

        if limit:
            query += f" LIMIT {limit}"

        with self._connect() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        runs = [self._row_to_dict(row) for row in rows]

        # Filter to latest version per run_key if requested
        if latest_only:
            seen_keys = set()
            latest_runs = []
            for run in runs:
                if run['run_key'] not in seen_keys:
                    seen_keys.add(run['run_key'])
                    latest_runs.append(run)
            runs = latest_runs

        return runs

    def register_artifact(
        self,
        run_id: str,
        kind: str,
        path: str,
        size_bytes: Optional[int] = None
    ):
        """
        Register an artifact for a run.

        Args:
            run_id: Unique run identifier
            kind: Artifact type (compact_graph, activations, feature_incidence, metadata)
            path: Path to artifact
            size_bytes: Size in bytes
        """
        created_at = datetime.datetime.now(datetime.UTC).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO artifacts (run_id, kind, path, size_bytes, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, kind, path, size_bytes, created_at)
            )
            conn.commit()

    def get_artifacts(self, run_id: str) -> List[Dict[str, Any]]:
        """
        Get all artifacts for a run.

        Args:
            run_id: Unique run identifier

        Returns:
            List of artifact records
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM artifacts WHERE run_id = ?",
                (run_id,)
            )
            rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def update_prompt_labels(self, run_id: str, prompt_labels: Dict[str, Any]) -> None:
        """
        Update the prompt_labels JSON blob for a run.

        This is used to persist evaluation results (e.g., InjecAgent eval labels)
        that are computed after the initial run record is created.
        """
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET prompt_labels = ?
                WHERE run_id = ?
                """,
                (json.dumps(prompt_labels), run_id),
            )
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with statistics
        """
        with self._connect() as conn:
            # Total runs by status
            cursor = conn.execute(
                """
                SELECT status, COUNT(*) as count
                FROM runs
                GROUP BY status
                """
            )
            status_counts = {row['status']: row['count'] for row in cursor.fetchall()}

            # Total runs by dataset
            cursor = conn.execute(
                """
                SELECT dataset_id, COUNT(*) as count
                FROM runs
                GROUP BY dataset_id
                """
            )
            dataset_counts = {row['dataset_id']: row['count'] for row in cursor.fetchall()}

            # Total artifact size
            cursor = conn.execute(
                "SELECT SUM(size_bytes) as total_size FROM artifacts"
            )
            total_size = cursor.fetchone()['total_size'] or 0

        return {
            'total_runs': sum(status_counts.values()),
            'by_status': status_counts,
            'by_dataset': dataset_counts,
            'total_artifact_size_gb': total_size / (1024**3),
        }

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to dictionary, parsing JSON fields."""
        d = dict(row)

        # Parse JSON fields
        if d.get('prompt_labels'):
            d['prompt_labels'] = json.loads(d['prompt_labels'])
        if d.get('config_snapshot'):
            d['config_snapshot'] = json.loads(d['config_snapshot'])

        return d


# Utility functions for computing keys and fingerprints

def compute_run_key(
    prompt_spec,
    model_hash: str,
    clt_hash: str,
    seed: int
) -> str:
    """
    Compute idempotency key for a run.

    This identifies a unique prompt + model + seed combination.

    Args:
        prompt_spec: PromptSpec instance
        model_hash: Model weight hash
        clt_hash: Transcoder weight hash
        seed: Random seed

    Returns:
        SHA1 hash as hex string
    """
    key_str = f"{prompt_spec.dataset_id}|{prompt_spec.prompt_id}|{model_hash}|{clt_hash}|{seed}"
    return hashlib.sha1(key_str.encode()).hexdigest()


def compute_processing_fingerprint(
    model_name: str,
    transcoder_or_sae_name: str,
    capture_layers: list,
    capture_positions: list,
    enable_attribution: bool,
    enable_generation: bool,
) -> str:
    """
    Compute processing fingerprint (config hash for idempotency).

    Only includes settings that affect the output artifacts.
    If any of these change, we need to reprocess.

    Args:
        model_name: Model name/path
        transcoder_or_sae_name: Transcoder set or SAE release name
        capture_layers: List of layers to capture activations from
        capture_positions: List of positions to capture
        enable_attribution: Whether attribution is enabled
        enable_generation: Whether text generation is enabled

    Returns:
        SHA1 hash as hex string (first 16 chars)
    """
    fingerprint_dict = {
        "model": model_name,
        "transcoder_or_sae": transcoder_or_sae_name,
        "capture_layers": sorted(capture_layers) if capture_layers else [],
        "capture_positions": sorted(capture_positions) if isinstance(capture_positions, list) and all(isinstance(p, int) for p in capture_positions) else capture_positions,
        "attribution": enable_attribution,
        "generation": enable_generation,
    }
    fingerprint_str = json.dumps(fingerprint_dict, sort_keys=True)
    return hashlib.sha1(fingerprint_str.encode()).hexdigest()[:16]
