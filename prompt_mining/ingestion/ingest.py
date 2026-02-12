#!/usr/bin/env python3
"""
Generic ingestion script for prompt mining platform.

This script processes datasets using configuration from a YAML file and generates:
1. CompactGraphs (.pt)
2. Feature incidence tables (.parquet)
3. Activation matrices (.zarr)
4. Registry entries (.sqlite)

Usage:
    # Run with config file
    python -m prompt_mining.ingestion.ingest --config config.yaml --dataset-name injecagent_dh

    # Override number of prompts
    python -m prompt_mining.ingestion.ingest --config config.yaml --dataset-name injecagent_dh --num-prompts 100

    # Force rerun (ignore idempotency)
    python -m prompt_mining.ingestion.ingest --config config.yaml --dataset-name injecagent_dh --force
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import dotenv
import torch
import yaml

# Load HF credentials (and other env) from a local .env if present.
dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

from prompt_mining.model.model_wrapper import ModelConfig
from prompt_mining.ingestion.prompt_runner import RunConfig
from prompt_mining.ingestion.pipelines.base import IngestArgs
from prompt_mining.ingestion.pipelines.full import FullIngestionPipeline
from prompt_mining.ingestion.pipelines.labels_only import LabelsOnlyIngestionPipeline


def load_configs_from_yaml(config_path: Path, dataset_name: str) -> Tuple[ModelConfig, RunConfig, Dict[str, Any], Dict[str, Any]]:
    """
    Load ModelConfig and RunConfig from YAML file.

    Args:
        config_path: Path to YAML config file
        dataset_name: Name of dataset to find in config

    Returns:
        Tuple of (ModelConfig, RunConfig, dataset_config, global_config)
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Find the dataset config
    dataset_cfg = None
    for ds in raw.get("datasets", []):
        if ds.get("name") == dataset_name:
            dataset_cfg = ds
            break

    if dataset_cfg is None:
        available = [ds.get("name", ds.get("class", "unnamed")) for ds in raw.get("datasets", [])]
        raise ValueError(f"Dataset '{dataset_name}' not found in config. Available: {available}")

    # Parse model config
    model_raw = raw.get("model", {})
    backend = model_raw.get("backend", "circuit_tracer")

    # Parse SAE configs if present
    sae_configs = None
    if model_raw.get("sae_configs"):
        sae_configs = model_raw["sae_configs"]
        if isinstance(sae_configs, str):
            sae_configs = json.loads(sae_configs)

    model_config = ModelConfig(
        model_name=model_raw.get("name", ""),
        backend=backend,
        transcoder_set=model_raw.get("transcoder_set") if backend == "circuit_tracer" else None,
        sae_configs=sae_configs if backend in ("saelens", "huggingface") else None,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        dtype=model_raw.get("dtype", "float32"),
        enable_attribution=model_raw.get("enable_attribution", False),
        device_map=model_raw.get("device_map") if backend == "huggingface" else None,
    )

    # Parse run config
    run_raw = raw.get("run", {})

    # Handle capture config - support both nested and flat formats
    capture_acts_raw = run_raw.get("capture_acts_raw")
    capture_acts_plt = run_raw.get("capture_acts_plt")

    run_config = RunConfig(
        topk_logits=run_raw.get("topk_logits", 10),
        max_new_tokens=run_raw.get("max_new_tokens", 0),
        max_context_length=run_raw.get("max_context_length"),
        seed=run_raw.get("seed", 42),
        capture_acts_raw=capture_acts_raw,
        capture_acts_plt=capture_acts_plt,
        raw_hook_point=run_raw.get("raw_hook_point", "hook_resid_post"),
        features_max_nodes=run_raw.get("features_max_nodes", 4000),
        features_node_threshold=run_raw.get("features_node_threshold", 0.7),
        features_edge_threshold=run_raw.get("features_edge_threshold", 0.98),
    )

    return model_config, run_config, dataset_cfg, raw


def main():
    parser = argparse.ArgumentParser(
        description="Generic Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Config-based arguments
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    parser.add_argument("--dataset-name", required=True, help="Name of dataset from config to process")

    # Overrides
    parser.add_argument("--num-prompts", type=int, default=None, help="Override: number of prompts to process (default: from config or all)")
    parser.add_argument("--force", action="store_true", help="Force rerun existing prompts")

    args = parser.parse_args()

    # Validate config file exists
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        return 1

    # Load configs from YAML
    try:
        model_config, run_config, dataset_cfg, global_cfg = load_configs_from_yaml(args.config, args.dataset_name)
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1

    # Extract dataset parameters
    dataset_class_name = dataset_cfg.get("class")
    dataset_params = dataset_cfg.get("params", {})
    evaluator_class_name = dataset_cfg.get("evaluator_class")
    evaluator_params = dataset_cfg.get("evaluator_params", {})

    # Determine num_prompts: CLI override > dataset config > global config > all (-1)
    num_prompts = args.num_prompts
    if num_prompts is None:
        num_prompts = dataset_cfg.get("num_prompts")
    if num_prompts is None:
        num_prompts = global_cfg.get("num_prompts", -1)

    # Get output directory
    output_dir = Path(global_cfg.get("output_dir", "./output"))
    mode = str(global_cfg.get("mode", "ingest")).strip().lower()
    if mode not in ("ingest", "labels_only"):
        print(f"Error: invalid global mode '{mode}'. Expected 'ingest' or 'labels_only'.")
        return 1

    print("=" * 80)
    print("Generic Ingestion Pipeline")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Dataset: {args.dataset_name} ({dataset_class_name})")
    print(f"Dataset params: {json.dumps(dataset_params, indent=2)}")
    if evaluator_class_name:
        print(f"Evaluator: {evaluator_class_name}")
        print(f"Evaluator params: {json.dumps(evaluator_params, indent=2)}")
    print(f"Model: {model_config.model_name}")
    print(f"Backend: {model_config.backend}")
    if model_config.backend == "circuit_tracer":
        print(f"Transcoder set: {model_config.transcoder_set}")
    else:
        print(f"SAE configs: {json.dumps(model_config.sae_configs, indent=2)}")
    print(f"Attribution: {model_config.enable_attribution}")
    print(f"Generation: {run_config.max_new_tokens} tokens")
    print(f"Output directory: {output_dir}")
    print(f"Num prompts: {'ALL' if num_prompts == -1 else num_prompts}")
    print(f"Mode: {mode}")
    print()
    ingest_args = IngestArgs(
        config_path=args.config,
        dataset_name=args.dataset_name,
        num_prompts=args.num_prompts,
        force=args.force,
    )

    pipeline_cls = LabelsOnlyIngestionPipeline if mode == "labels_only" else FullIngestionPipeline
    pipeline = pipeline_cls(
        args=ingest_args,
        model_config=model_config,
        run_config=run_config,
        dataset_cfg=dataset_cfg,
        global_cfg=global_cfg,
    )
    return pipeline.run()

if __name__ == "__main__":
    sys.exit(main() or 0)
