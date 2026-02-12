"""
PromptRunner for single-pass prompt execution.

Handles generation, attribution, activation capture, and artifact persistence.
"""

import gc
import json
import hashlib
import time
import collections
import statistics
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch

from prompt_mining.core.prompt_spec import PromptSpec
from prompt_mining.core.compact_graph import CompactGraph, FeatureInfo
from prompt_mining.core.evaluator import Evaluator
from prompt_mining.model.model_wrapper import ModelWrapper
from prompt_mining.storage.base import StorageBackend
from prompt_mining.registry.sqlite_registry import SQLiteRegistry, compute_run_key, compute_processing_fingerprint
from prompt_mining.utils.inference import get_topk_logits, resolve_positions, resolve_layers
from prompt_mining.utils.gpu import clear_gpu_memory
from circuit_tracer import attribute
from circuit_tracer.graph import prune_graph, compute_node_influence


@dataclass
class RunConfig:
    """Configuration for a single run."""
    topk_logits: int = 10
    max_new_tokens: int = 100
    max_context_length: Optional[int] = None  # Skip prompts exceeding this token count
    seed: int = 42

    # Capture configuration
    capture_acts_raw: Dict[str, Any] = None
    capture_acts_plt: Dict[str, Any] = None
    raw_hook_point: str = "hook_resid_post"  # Hook point for raw activation capture

    # Feature selection / graph pruning
    # NOTE: `features_top_k` is kept for backwards compatibility with older
    # scripts/tests; current implementation uses pruning-based settings below.
    features_top_k: Optional[int] = None
    features_max_nodes: int = 4000  # Max nodes to include in full graph before pruning
    features_node_threshold: float = 0.7  # Keep nodes explaining this fraction of influence
    features_edge_threshold: float = 0.98  # Keep edges explaining this fraction of influence

    def __post_init__(self):
        if self.capture_acts_raw is None:
            self.capture_acts_raw = {"layers": [27, 31], "positions": [-5, -1]}
        if self.capture_acts_plt is None:
            self.capture_acts_plt = {"layers": [27], "positions": [-5, -1]}


@dataclass
class RunResult:
    """Result of a single prompt run."""
    run_id: str
    prompt_id: str
    status: str  # 'completed', 'skipped', or 'failed'
    error_message: Optional[str] = None
    artifacts_written: List[str] = None

    def __post_init__(self):
        if self.artifacts_written is None:
            self.artifacts_written = []


class PromptRunner:
    """
    Executes a single prompt through the model pipeline.

    Workflow:
    1. Check idempotency (skip if already processed)
    2. Tokenize and run forward pass
    3. Optionally generate text
    4. Compute attribution (if enabled)
    5. Extract CompactGraph
    6. Capture activations
    7. Write artifacts
    8. Export feature_incidence table
    9. Update registry
    """

    def __init__(
        self,
        model_wrapper: ModelWrapper,
        storage: StorageBackend,
        registry: SQLiteRegistry,
        run_config: RunConfig,
        evaluator: Optional[Evaluator] = None
    ):
        """
        Initialize PromptRunner.

        Args:
            model_wrapper: Loaded model wrapper
            storage: Storage backend for artifacts
            registry: Registry for run tracking
            run_config: Run configuration
            evaluator: Optional evaluator for dataset-specific evaluation
        """
        self.model_wrapper = model_wrapper
        self.storage = storage
        self.registry = registry
        self.run_config = run_config
        self.evaluator = evaluator

        # Compute model hashes (for run_key)
        model_info = model_wrapper.get_model_info()
        self.model_hash = self._hash_string(model_info["model_name"])
        self.clt_hash = self._hash_string(model_info["transcoder_set"])

        # Cache model dimensions (static across all runs)
        self.n_layers = model_info["n_layers"]

        # Pre-resolve static layer indices (don't depend on seq_len)
        from prompt_mining.utils.inference import resolve_layers

        self.layers_raw = []
        self.position_specs_raw = []
        if run_config.capture_acts_raw:
            layers_spec = run_config.capture_acts_raw.get("layers", [])
            self.layers_raw = resolve_layers(layers_spec, self.n_layers)
            self.position_specs_raw = run_config.capture_acts_raw.get("positions", [])

        self.layers_plt = []
        self.position_specs_plt = []
        if run_config.capture_acts_plt:
            layers_spec = run_config.capture_acts_plt.get("layers", [])
            self.layers_plt = resolve_layers(layers_spec, self.n_layers)
            self.position_specs_plt = run_config.capture_acts_plt.get("positions", [])

        # Get SAE hook point from adapter (where SAE was trained)
        sae_layer_info = model_wrapper.adapter.get_sae_layer_info()
        self.sae_hook_point = (
            sae_layer_info[0].hook_point
            if (isinstance(sae_layer_info, list) and len(sae_layer_info) > 0)
            else None
        )

        # Timing statistics
        self.stats = collections.defaultdict(list)

    def _hash_string(self, s: str) -> str:
        """Compute SHA1 hash of string."""
        return hashlib.sha1(s.encode()).hexdigest()[:12]

    def print_stats(self):
        """Print timing statistics."""
        print("\n" + "=" * 60)
        print(f"{'Timer Name':<30} | {'Avg (s)':<10} | {'Median':<10} | {'Max':<10} | {'Count':<5}")
        print("-" * 60)
        
        for name, times in sorted(self.stats.items()):
            if not times:
                continue
            avg = statistics.mean(times)
            median = statistics.median(times)
            max_val = max(times)
            count = len(times)
            print(f"{name:<30} | {avg:<10.4f} | {median:<10.4f} | {max_val:<10.4f} | {count:<5}")
        print("=" * 60 + "\n")

    def run(self, prompt_spec: PromptSpec) -> RunResult:
        """
        Execute prompt through full pipeline.

        Args:
            prompt_spec: Prompt specification

        Returns:
            RunResult with status and artifacts
        """
        # 1. Compute run identifiers
        run_key = compute_run_key(
            prompt_spec=prompt_spec,
            model_hash=self.model_hash,
            clt_hash=self.clt_hash,
            seed=self.run_config.seed
        )

        # Get transcoder/SAE name for fingerprint
        model_info = self.model_wrapper.get_model_info()
        transcoder_or_sae = model_info.get("transcoder_set", "")
        if not transcoder_or_sae or transcoder_or_sae == "N/A":
            # For SAELens/HuggingFace backends, use first SAE release name
            sae_layers = model_info.get("sae_layers", [])
            transcoder_or_sae = f"sae_layers_{sae_layers}" if sae_layers else "none"

        # Combine all capture layers and positions for fingerprint
        all_capture_layers = sorted(set(self.layers_raw) | set(self.layers_plt))
        all_capture_positions = self.position_specs_raw + self.position_specs_plt

        fingerprint = compute_processing_fingerprint(
            model_name=model_info["model_name"],
            transcoder_or_sae_name=transcoder_or_sae,
            capture_layers=all_capture_layers,
            capture_positions=all_capture_positions,
            enable_attribution=self.model_wrapper.config.enable_attribution,
            enable_generation=self.run_config.max_new_tokens > 0,
        )

        # 2. Check idempotency
        existing = self.registry.get_run_by_key(run_key, fingerprint)
        if existing and existing["status"] == "completed":
            return RunResult(
                run_id=existing["run_id"],
                prompt_id=prompt_spec.prompt_id,
                status="skipped"
            )

        # 3. Generate run_id
        run_id = self._generate_run_id(prompt_spec.prompt_id)

        # 4. Register run as 'pending/running' (phase 1)
        #    Use minimal metadata here; full labels (including eval) are written
        #    later via finalize_run().
        self.registry.create_run(
            run_id=run_id,
            prompt_id=prompt_spec.prompt_id,
            dataset_id=prompt_spec.dataset_id,
            run_key=run_key,
            processing_fingerprint=fingerprint,
            model_hash=self.model_hash,
            clt_hash=self.clt_hash,
            seed=self.run_config.seed,
            config_snapshot=self._config_to_dict()
        )

        # 5. Execute prompt (phase 2) and finalize run
        try:
            t0 = time.perf_counter()
            result = self._execute_prompt(run_id, prompt_spec)
            self.stats["total_execute_prompt"].append(time.perf_counter() - t0)

            # Finalize run with appropriate status
            self.registry.finalize_run(
                run_id=run_id,
                status=result.status,
                prompt_labels=prompt_spec.labels,
                error_message=result.error_message,
            )
            return result
        except Exception as e:
            # On failure, record failed status and error message, then return
            error_message = f"{type(e).__name__}: {e}"
            try:
                self.registry.finalize_run(
                    run_id=run_id,
                    status="failed",
                    prompt_labels=prompt_spec.labels,
                    error_message=error_message,
                )
            except Exception:
                # Avoid masking the original exception if finalize_run fails
                pass

            return RunResult(
                run_id=run_id,
                prompt_id=prompt_spec.prompt_id,
                status="failed",
                error_message=error_message,
            )

    def _execute_prompt(self, run_id: str, prompt_spec: PromptSpec) -> RunResult:
        """
        Execute the actual prompt processing.

        This is separated from run() for cleaner error handling.
        """
        artifacts = []

        # 1. Prepare text
        t0 = time.perf_counter()
        if prompt_spec.messages:
            # Apply chat template
            text = self.model_wrapper.apply_chat_template(
                prompt_spec.messages,
                tools=prompt_spec.tools,
            )
        else:
            text = prompt_spec.text

        # Update prompt_spec with flattened text
        prompt_spec.text = text
        self.stats["prepare_text"].append(time.perf_counter() - t0)

        # 2. Tokenize (for metadata only - model methods handle tokenization internally)
        t0 = time.perf_counter()
        input_ids = self.model_wrapper.tokenize(text)["input_ids"]
        input_len = len(input_ids[0])
        self.stats["tokenize"].append(time.perf_counter() - t0)

        # 2b. Skip if context length exceeds limit
        if self.run_config.max_context_length and input_len > self.run_config.max_context_length:
            # Clean up tokenization tensors before returning
            del input_ids
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all CUDA ops complete
            return RunResult(
                run_id=run_id,
                prompt_id=prompt_spec.prompt_id,
                status="skipped",
                error_message=f"Context length {input_len} exceeds max {self.run_config.max_context_length}",
            )

        # 3. Single forward pass with cache (OPTIMIZATION: gets logits + activations in one pass)
        t0 = time.perf_counter()
        # Get all layers needed for activation capture
        all_layers_needed = sorted(set(self.layers_raw) | set(self.layers_plt))

        # Determine which hook points to cache
        # For raw: use raw_hook_point
        # For PLT: use SAE hook point (where SAE was trained)
        hook_points_to_cache = [self.run_config.raw_hook_point]
        if self.layers_plt and self.sae_hook_point:
            # Add SAE hook point if we're capturing PLT activations
            if self.sae_hook_point not in hook_points_to_cache:
                hook_points_to_cache.append(self.sae_hook_point)

        # Run once and cache activations for all needed layers and hook points
        logits, activation_cache = self.model_wrapper.run_with_cache(
            text,
            layers=all_layers_needed,
            hook_points=hook_points_to_cache
        )
        self.stats["forward_with_cache"].append(time.perf_counter() - t0)

        # 4. Extract top-K logits
        t0 = time.perf_counter()
        topk_tokens, topk_probs = get_topk_logits(
            logits,
            k=self.run_config.topk_logits
        )

        topk_logits_data = {
            "tokens": topk_tokens.tolist(),
            "probabilities": topk_probs.tolist()
        }
        self.stats["extract_topk"].append(time.perf_counter() - t0)

        # 5. Optional: Generate text
        t0 = time.perf_counter()
        generated_text = None
        generated_continuation = None
        if self.run_config.max_new_tokens > 0:
            generated_output = self.model_wrapper.generate(
                text,
                max_new_tokens=self.run_config.max_new_tokens
            )

            generated_only_tokens = generated_output[:, input_len:]
            generated_continuation = self.model_wrapper.decode(generated_only_tokens)
        self.stats["generate_text"].append(time.perf_counter() - t0)

        # 5.5. Optional: Evaluate (if evaluator provided)
        # Evaluators may operate on prompt only, output only, or both
        t0 = time.perf_counter()
        if self.evaluator is not None:
            eval_result = self.evaluator.evaluate(prompt_spec, generated_continuation)
            if eval_result:
                # Update labels with evaluation results in-memory; they will be
                # persisted in a single shot via registry.finalize_run().
                prompt_spec.labels.update(eval_result)
        self.stats["evaluate"].append(time.perf_counter() - t0)

        # 6. Compute attribution (if enabled)
        t0_attr = time.perf_counter()
        compact_graph = None
        if self.model_wrapper.config.enable_attribution:
            # Check if backend supports attribution
            if not self.model_wrapper.adapter.supports_attribution():
                print(f"⚠ Attribution not supported for {self.model_wrapper.config.backend} backend, creating minimal graph")
                compact_graph = self._create_minimal_compact_graph(
                    prompt_spec=prompt_spec,
                    topk_tokens=topk_tokens,
                    topk_probs=topk_probs
                )
            else:
                # Run attribution via circuit_tracer
                try:
                    # Step 1: Build full graph with max_feature_nodes limit
                    t0 = time.perf_counter()
                    batch_size = 128 if input_len < 200 else 64
                    graph = attribute(
                        prompt=text,  # Pass text string
                        model=self.model_wrapper.adapter.get_raw_model(),
                        max_n_logits=self.run_config.topk_logits,
                        desired_logit_prob=0.95,
                        batch_size=batch_size,
                        max_feature_nodes=self.run_config.features_max_nodes,
                        offload="cpu",
                        verbose=False
                    )
                    self.stats["attribution_step1_attribute"].append(time.perf_counter() - t0)

                    # Step 2: Compute node influences once (to avoid recalculation)
                    t0 = time.perf_counter()
                    n_logits = len(graph.logit_tokens)
                    device = graph.adjacency_matrix.device
                    logit_weights = torch.zeros(
                        graph.adjacency_matrix.shape[0],
                        device=device
                    )
                    logit_weights[-n_logits:] = graph.logit_probabilities.to(device)
                    node_influence = compute_node_influence(graph.adjacency_matrix, logit_weights)
                    self.stats["attribution_step2_influence"].append(time.perf_counter() - t0)

                    # Step 3: Prune graph to get nodes explaining features_node_threshold of influence
                    t0 = time.perf_counter()
                    prune_result = prune_graph(
                        graph,
                        node_threshold=self.run_config.features_node_threshold,
                        edge_threshold=self.run_config.features_edge_threshold,
                    )
                    self.stats["attribution_step3_prune"].append(time.perf_counter() - t0)

                    # Step 4: Extract CompactGraph from pruned graph, passing pre-computed influences
                    t0 = time.perf_counter()
                    model_info = self.model_wrapper.get_model_info()
                    compact_graph = CompactGraph.from_graph(
                        graph=graph,
                        prompt_id=prompt_spec.prompt_id,
                        input_string=prompt_spec.text or "",
                        topk_tokens=topk_tokens,  # Already numpy array
                        topk_probs=topk_probs,  # Already numpy array
                        model_hash=self.model_hash,
                        clt_hash=self.clt_hash,
                        seed=self.run_config.seed,
                        n_layers=model_info["n_layers"],
                        node_mask=prune_result.node_mask,  # Use pruned node mask
                        node_influence=node_influence  # Pass pre-computed influences
                    )
                    self.stats["attribution_step4_compact"].append(time.perf_counter() - t0)

                except torch.cuda.OutOfMemoryError as e:
                    # GPU OOM during attribution - expected for very long sequences
                    # Create minimal graph and continue pipeline
                    compact_graph = self._create_minimal_compact_graph(
                        prompt_spec=prompt_spec,
                        input_ids=input_ids,
                        topk_tokens=topk_tokens,
                        topk_probs=topk_probs
                    )
                    print(f"⚠ GPU OOM during attribution - creating minimal graph")
                except Exception as e:
                    # Other attribution errors - log and continue
                    print(f"⚠ Attribution failed with error: {type(e).__name__}: {str(e)}")
                    compact_graph = self._create_minimal_compact_graph(
                        prompt_spec=prompt_spec,
                        input_ids=input_ids,
                        topk_tokens=topk_tokens,
                        topk_probs=topk_probs
                    )
                self.model_wrapper.to_original_device() # Attribution offloads to CPU, need to ensure the model is on the original device
        self.stats["attribution_total"].append(time.perf_counter() - t0_attr)

        # 7. Capture activations (independent of attribution)
        t0 = time.perf_counter()
        captured_acts = {}
        zarr_metadata = {}
        captured_acts, zarr_metadata = self._capture_activations(
            seq_len=len(input_ids[0]),
            cached_activations=activation_cache
        )
        self.stats["capture_activations"].append(time.perf_counter() - t0)

        # 8. Write artifacts
        t0 = time.perf_counter()
        if compact_graph:
            self.storage.write_compact_graph(run_id, compact_graph)
            artifacts.append("compact_graph.pt")

            # Export feature_incidence
            self._export_feature_incidence(run_id, compact_graph, prompt_spec)
            artifacts.append("feature_incidence.parquet")

        # Export sparse PLT activations (if captured)
        if captured_acts:
            plt_positions = zarr_metadata.get('plt_positions', []) if zarr_metadata else []
            self._export_plt_activations(run_id, captured_acts, plt_positions)
            artifacts.append("plt_activations.parquet")

        # Write topk_logits
        self.storage.write_json(
            run_id,
            "topk_logits.json",
            topk_logits_data
        )
        artifacts.append("topk_logits.json")

        # Write generation (if exists)
        if generated_text:
            # Store both the full text (for debugging) and the continuation
            # actually shown to the user / used for evaluation.
            self.storage.write_json(
                run_id,
                "generation.json",
                {
                    "text": generated_text,
                    "continuation": generated_continuation,
                },
            )
            artifacts.append("generation.json")

        # Write raw activations only to Zarr (PLT is now in parquet)
        if captured_acts:
            # Filter to only raw activations
            raw_acts = {k: v for k, v in captured_acts.items() if k.startswith('raw/')}
            if raw_acts:
                self.storage.write_activations(run_id, raw_acts, metadata=zarr_metadata)
                artifacts.append("acts.zarr")

        # Get model dimensions
        model_info = self.model_wrapper.get_model_info()
        d_model = model_info["d_model"]

        # Get d_transcoder from captured PLT activations (if available)
        d_transcoder = None
        for key, array in captured_acts.items():
            if key.startswith('plt/'):
                d_transcoder = array.shape[1]  # (n_positions, d_transcoder)
                break

        # Write metadata
        manifest = {
            "run_id": run_id,
            "prompt_id": prompt_spec.prompt_id,
            "dataset_id": prompt_spec.dataset_id,
            "model_hash": self.model_hash,
            "clt_hash": self.clt_hash,
            "seed": self.run_config.seed,
            "input_text": text,
            "input_length": len(input_ids[0]),
            "labels": prompt_spec.labels,
            "d_model": d_model,
            "d_transcoder": d_transcoder
        }
        self.storage.write_metadata(run_id, manifest)
        artifacts.append("manifest.json")
        self.stats["write_artifacts"].append(time.perf_counter() - t0)
        torch.cuda.empty_cache()

        return RunResult(
            run_id=run_id,
            prompt_id=prompt_spec.prompt_id,
            status="completed",
            artifacts_written=artifacts
        )

    def _create_minimal_compact_graph(
        self,
        prompt_spec: PromptSpec,
        input_ids: torch.Tensor,
        topk_tokens: np.ndarray,
        topk_probs: np.ndarray
    ) -> CompactGraph:
        """
        Create a minimal CompactGraph without full attribution.

        Used as fallback when attribution is disabled or fails.
        """
        # Create empty features list
        features = []

        model_info = self.model_wrapper.get_model_info()

        return CompactGraph(
            prompt_id=prompt_spec.prompt_id,
            input_string=prompt_spec.text or "",
            input_tokens=input_ids[0].detach().cpu().numpy(),
            logit_tokens=topk_tokens,  # Already numpy array
            logit_probabilities=topk_probs,  # Already numpy array
            features=features,
            n_layers=model_info["n_layers"],
            model_hash=self.model_hash,
            clt_hash=self.clt_hash,
            seed=self.run_config.seed
        )
    
    def _capture_activations(
        self,
        seq_len: int,
        cached_activations: Dict[int, torch.Tensor]
    ) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Extract activations from cached activations (no second forward pass needed).

        OPTIMIZATION: Uses cached activations from run_with_cache() instead of
        running a separate forward pass with hooks.

        Args:
            seq_len: Sequence length
            cached_activations: Pre-computed activations from run_with_cache()
        Returns:
            Tuple of (activations_dict, metadata_dict)
            - activations_dict: Maps layer names to arrays (e.g., 'raw/layer10': (n_pos, d_model))
            - metadata_dict: Zarr metadata with seq_len, raw_positions, plt_positions, etc.
        """
        captured_acts = {}

        # Resolve positions based on seq_len (only thing that varies per prompt)
        # Layers are already cached in __init__ (static)
        all_positions_raw = set()
        all_positions_plt = set()

        if self.layers_raw:
            # Resolve positions for raw space (depends on seq_len)
            position_indices_raw = resolve_positions(self.position_specs_raw, seq_len)
            all_positions_raw.update(position_indices_raw)

        if self.layers_plt:
            # Resolve positions for PLT space (depends on seq_len)
            position_indices_plt = resolve_positions(self.position_specs_plt, seq_len)
            all_positions_plt.update(position_indices_plt)

        # All layers we need (for both raw and plt) - use cached values
        all_layers_needed = sorted(set(self.layers_raw) | set(self.layers_plt))

        # Process cached activations (cache keys are full hook names: "blocks.5.hook_resid_pre")
        for layer_idx in all_layers_needed:
            # Capture raw activations if requested for this layer
            if layer_idx in self.layers_raw and all_positions_raw:
                # Get activations from raw hook point
                raw_hook_key = f"blocks.{layer_idx}.{self.run_config.raw_hook_point}"
                if raw_hook_key not in cached_activations:
                    continue

                raw_acts = cached_activations[raw_hook_key]  # (batch, seq_len, d_model)

                positions = sorted(all_positions_raw)
                # Slice all positions at once: (batch, seq_len, d_model) -> (n_positions, d_model)
                if raw_acts.ndim == 3:
                    layer_array = raw_acts[0, positions, :].float().detach().cpu().numpy()
                else:
                    layer_array = raw_acts[positions, :].float().detach().cpu().numpy()
                captured_acts[f'raw/layer{layer_idx}'] = layer_array

            # Capture PLT/SAE activations if requested for this layer
            if layer_idx in self.layers_plt and all_positions_plt:
                # Get activations from SAE hook point (where SAE was trained)
                sae_hook_key = f"blocks.{layer_idx}.{self.sae_hook_point}"
                if sae_hook_key not in cached_activations:
                    continue

                sae_input_acts = cached_activations[sae_hook_key]  # (batch, seq_len, d_model)

                # Encode through SAE (works for both transcoders and SAEs)
                sae_acts = self.model_wrapper.encode_layer_activations(
                    sae_input_acts,
                    layer_idx
                )  # (batch, seq_len, d_sae)

                positions = sorted(all_positions_plt)
                # Slice all positions at once: (batch, seq_len, d_sae) -> (n_positions, d_sae)
                if sae_acts.ndim == 3:
                    layer_array = sae_acts[0, positions, :].float().detach().cpu().numpy()
                else:
                    layer_array = sae_acts[positions, :].float().detach().cpu().numpy()
                captured_acts[f'plt/layer{layer_idx}'] = layer_array

        # Build metadata
        metadata = {
            "seq_len": seq_len,
            "layers_captured": all_layers_needed,
            # Per-space metadata for correct loading
            "raw_positions": sorted(all_positions_raw),
            "plt_positions": sorted(all_positions_plt),
            "raw_layers": self.layers_raw,  # Already sorted from cached resolution
            "plt_layers": self.layers_plt,  # Already sorted from cached resolution
        }

        return captured_acts, metadata

    def _export_feature_incidence(
        self,
        run_id: str,
        compact_graph: CompactGraph,
        prompt_spec: PromptSpec
    ) -> None:
        """
        Export feature_incidence table from CompactGraph.

        Flattens FeatureInfo list into tabular format for analysis.
        """
        if not compact_graph.features:
            # No features to export
            return

        rows = []
        seq_len = len(compact_graph.input_tokens)

        for feature in compact_graph.features:
            rows.append({
                "run_id": run_id,
                "prompt_id": prompt_spec.prompt_id,
                "layer": feature.layer,
                "position": feature.position,
                "feature_idx": feature.feature_idx,
                "activation_value": feature.activation_value,
                "influence": feature.influence,
                "logit_attribution": feature.logit_attribution,
                "is_last_pos": (feature.position == seq_len - 1),
                "prompt_labels": json.dumps(prompt_spec.labels)
            })

        df = pd.DataFrame(rows)
        self.storage.write_feature_incidence(run_id, df)

    def _export_plt_activations(
        self,
        run_id: str,
        captured_acts: Dict[str, np.ndarray],
        plt_positions: List[int]
    ) -> None:
        """
        Export sparse PLT activations to parquet (one file per layer).

        Extracts all non-zero PLT features from captured activations and saves
        in sparse format: (run_id, position, feature_idx, activation_value).

        Files are organized as: tables/plt_activations/run={run_id}/layer_{layer}.parquet
        This matches the raw activations hierarchy: runs/{date}/{run_id}/acts.zarr/raw/layer{layer}

        Args:
            run_id: Unique run identifier
            captured_acts: Dictionary of captured activations (from _capture_activations)
            plt_positions: List of actual token positions captured (e.g., [108] for "last")
        """
        # Process each PLT layer separately (one file per layer)
        for layer_key, activations in captured_acts.items():
            if not layer_key.startswith('plt/'):
                continue

            # Extract layer number from key like 'plt/layer10'
            layer_idx = int(layer_key.split('layer')[1])

            # activations shape: (n_positions, d_transcoder)
            # Find all non-zero elements at once (vectorized)
            pos_indices, feat_indices = np.nonzero(activations)
            activation_values = activations[pos_indices, feat_indices]

            # Map row indices to actual token positions
            # pos_indices are 0, 1, 2... (array rows)
            # plt_positions are actual token positions (e.g., [0, 108] for first and last)
            actual_positions = [plt_positions[idx] for idx in pos_indices if idx < len(plt_positions)]
            n_nonzero = len(actual_positions)

            if n_nonzero > 0:
                # Build dataframe for this layer
                df = pd.DataFrame({
                    "run_id": run_id,
                    "position": actual_positions,
                    "feature_idx": feat_indices[:n_nonzero].tolist(),
                    "activation_value": activation_values[:n_nonzero].tolist()
                })

                # Write one file per layer
                self.storage.write_plt_activations(run_id, layer_idx, df)

    def _generate_run_id(self, prompt_id: str) -> str:
        """Generate unique run ID with date prefix for directory structure."""
        import datetime
        import uuid
        
        # Use date prefix format: YYYY-MM-DD_{uuid}
        date_str = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
        unique_id = str(uuid.uuid4())[:8]
        return f"{date_str}_{unique_id}"

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert RunConfig to dictionary for storage."""
        return {
            "topk_logits": self.run_config.topk_logits,
            "max_new_tokens": self.run_config.max_new_tokens,
            "seed": self.run_config.seed,
            "capture_acts_raw": self.run_config.capture_acts_raw,
            "capture_acts_plt": self.run_config.capture_acts_plt,
            "features_max_nodes": self.run_config.features_max_nodes,
            "features_node_threshold": self.run_config.features_node_threshold
        }
