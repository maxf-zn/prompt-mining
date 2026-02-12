from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from tqdm import tqdm

from prompt_mining.registry.sqlite_registry import SQLiteRegistry, compute_run_key

from prompt_mining.ingestion.pipelines.base import BaseIngestionPipeline


class LabelsOnlyIngestionPipeline(BaseIngestionPipeline):
    """
    Labels-only pipeline:
    - iterates dataset PromptSpecs
    - runs evaluator
    - writes evaluator outputs into prompt_labels inside a standard registry.sqlite
    - does NOT run the model or write artifacts
    """

    def _config_snapshot(self) -> Dict[str, Any]:
        """
        JSON-safe snapshot for registry runs.config_snapshot (labels_only).

        Must contain only JSON-serializable types (e.g., stringify torch.device).
        """
        mc = self.model_config
        return {
            "mode": "labels_only",
            "dataset": {"class": self.dataset_class_name, "params": self.dataset_params},
            "evaluator": {"class": self.evaluator_class_name, "params": self.evaluator_params},
            "model": {
                "model_name": getattr(mc, "model_name", None),
                "backend": getattr(mc, "backend", None),
                "dtype": getattr(mc, "dtype", None),
                "device": str(getattr(mc, "device", None)),
                "device_map": getattr(mc, "device_map", None),
                "enable_attribution": getattr(mc, "enable_attribution", None),
                "transcoder_set": getattr(mc, "transcoder_set", None),
                "sae_configs": getattr(mc, "sae_configs", None),
            },
            # Keep run config minimal (labels_only ignores most of it).
            "run": {
                "seed": getattr(self.run_config, "seed", None),
                "max_new_tokens": getattr(self.run_config, "max_new_tokens", None),
            },
        }

    def run(self) -> int:
        if not self.evaluator_class_name:
            print("Error: labels_only mode requires evaluator_class to be set for the dataset.")
            return 1

        dataset = self.load_dataset()
        print(f"✓ Loaded {len(dataset)} examples")

        evaluator = self.load_evaluator()
        print("✓ Evaluator initialized")

        num_prompts = self.resolve_num_prompts()
        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        registry_path = out_dir / "registry.sqlite"
        registry = SQLiteRegistry(db_path=str(registry_path))
        print(f"✓ Registry at {registry_path}")

        # Sentinel values for required schema fields
        model_hash = "labels_only"
        clt_hash = "labels_only"
        seed = 0

        fp_payload: Dict[str, Any] = {
            "mode": "labels_only",
            "evaluator_class": self.evaluator_class_name,
            "evaluator_params": self.evaluator_params,
        }
        processing_fingerprint = hashlib.sha1(json.dumps(fp_payload, sort_keys=True).encode()).hexdigest()[:16]

        print("=" * 80)
        print("Starting labels-only ingestion (evaluator labels only)...")

        count = 0
        success = 0
        skipped = 0
        failed = 0
        first_errors_printed = 0

        for i, prompt_spec in tqdm(enumerate(dataset), total=len(dataset), miniters=10):
            if num_prompts != -1 and i >= num_prompts:
                break

            count += 1
            run_id = prompt_spec.prompt_id  # stable; you join on prompt_id
            run_key = compute_run_key(prompt_spec, model_hash=model_hash, clt_hash=clt_hash, seed=seed)

            existing = None if self.args.force else registry.get_run_by_key(run_key, processing_fingerprint)
            if existing and existing.get("status") == "completed":
                skipped += 1
                continue

            try:
                if existing and existing.get("run_id") != run_id:
                    run_id = existing["run_id"]

                # Avoid collision if prompt_id was already used as a run_id in this registry.
                if existing is None and registry.get_run(run_id) is not None:
                    run_id = hashlib.sha1(f"{prompt_spec.prompt_id}|{processing_fingerprint}".encode()).hexdigest()[:12]

                registry.create_run(
                    run_id=run_id,
                    prompt_id=prompt_spec.prompt_id,
                    dataset_id=prompt_spec.dataset_id,
                    model_hash=model_hash,
                    clt_hash=clt_hash,
                    seed=seed,
                    run_key=run_key,
                    processing_fingerprint=processing_fingerprint,
                    prompt_labels=prompt_spec.labels,
                    config_snapshot=self._config_snapshot(),
                )

                eval_result = evaluator.evaluate(prompt_spec, generated_text=None) or {}
                final_labels = dict(prompt_spec.labels)
                final_labels.update(eval_result)
                registry.finalize_run(run_id=run_id, status="completed", prompt_labels=final_labels, error_message=None)
                success += 1
            except Exception as e:
                failed += 1
                if first_errors_printed < 5:
                    first_errors_printed += 1
                    print(f"\nlabels_only error (sample {first_errors_printed}/5): {type(e).__name__}: {e}")
                try:
                    registry.finalize_run(
                        run_id=run_id,
                        status="failed",
                        prompt_labels=prompt_spec.labels,
                        error_message=f"{type(e).__name__}: {e}",
                    )
                except Exception:
                    pass

        print("\nLabels-only ingestion complete!")
        print(f"Total: {count}")
        print(f"Success: {success}")
        print(f"Skipped: {skipped}")
        print(f"Failed: {failed}")
        print(f"Output: {out_dir.absolute()}")
        return 0


