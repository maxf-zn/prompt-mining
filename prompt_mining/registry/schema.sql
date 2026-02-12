-- Registry schema for prompt mining platform
-- Version: 1.0

CREATE TABLE IF NOT EXISTS runs (
    -- Primary identification
    run_id TEXT PRIMARY KEY,
    prompt_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    status TEXT NOT NULL,  -- 'pending', 'running', 'completed', 'failed'

    -- Sharding info
    shard_idx INTEGER,
    num_shards INTEGER,
    worker_id INTEGER,
    device_id INTEGER,

    -- Reproducibility
    model_hash TEXT NOT NULL,
    clt_hash TEXT NOT NULL,
    seed INTEGER NOT NULL,
    run_key TEXT NOT NULL,
    processing_fingerprint TEXT NOT NULL,

    -- Versioning
    supersedes TEXT,  -- run_id of previous version (if config changed)

    -- Metadata
    prompt_labels TEXT,  -- JSON string: {"success": true, "attack_type": "dh"}
    config_snapshot TEXT,  -- JSON string: full config used for this run

    -- Optional error info
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_run_key ON runs(run_key, processing_fingerprint);
CREATE INDEX IF NOT EXISTS idx_dataset ON runs(dataset_id);
CREATE INDEX IF NOT EXISTS idx_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_timestamp ON runs(timestamp);

CREATE TABLE IF NOT EXISTS artifacts (
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,  -- 'compact_graph', 'activations', 'feature_incidence', 'metadata'
    path TEXT NOT NULL,
    size_bytes INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    UNIQUE(run_id, kind)
);

CREATE INDEX IF NOT EXISTS idx_artifact_kind ON artifacts(kind);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
);

INSERT OR IGNORE INTO schema_version (version, applied_at)
VALUES ('1.0', datetime('now'));
