from __future__ import annotations

from tqdm import tqdm

from prompt_mining.ingestion.pipelines.base import BaseIngestionPipeline
from prompt_mining.ingestion.prompt_runner import PromptRunner
from prompt_mining.model.model_wrapper import ModelWrapper
from prompt_mining.registry.sqlite_registry import SQLiteRegistry
from prompt_mining.storage.local_storage import LocalStorage


class FullIngestionPipeline(BaseIngestionPipeline):
    """
    Full ingestion pipeline:
    - flattens prompt (via model wrapper)
    - runs model forward pass (+ optional generation)
    - writes artifacts and registry entries
    - optionally augments prompt_labels via evaluator
    """

    def run(self) -> int:
        dataset = self.load_dataset()
        print(f"✓ Loaded {len(dataset)} examples")

        evaluator = self.load_evaluator()
        if evaluator is not None:
            print("✓ Evaluator initialized")

        num_prompts = self.resolve_num_prompts()
        out_dir = self.output_dir

        storage = LocalStorage(uri=f"file://{out_dir.absolute()}")

        registry_path = out_dir / "registry.sqlite"
        registry = SQLiteRegistry(db_path=str(registry_path))
        print(f"✓ Registry at {registry_path}")

        print("Loading model (this may take a while)...")
        model_wrapper = ModelWrapper(self.model_config)
        model_wrapper.load()
        print("✓ Model loaded")

        runner = PromptRunner(
            model_wrapper=model_wrapper,
            storage=storage,
            registry=registry,
            run_config=self.run_config,
            evaluator=evaluator,
        )

        print("=" * 80)
        print("Starting ingestion...")

        count = 0
        success = 0
        skipped = 0
        failed = 0

        try:
            for i, prompt_spec in tqdm(enumerate(dataset), total=len(dataset), miniters=10):
                if num_prompts != -1 and i >= num_prompts:
                    break

                count += 1
                try:
                    result = runner.run(prompt_spec)
                    if result.status == "completed":
                        success += 1
                    elif result.status == "skipped":
                        skipped += 1
                    else:
                        failed += 1
                        print(f"\n  Failed: {result.error_message}")
                except Exception as e:
                    failed += 1
                    print(f"\n  Error: {e}")
                    import traceback

                    traceback.print_exc()
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            print(f"\n\nIngestion complete!")
            print(f"Total: {count}")
            print(f"Success: {success}")
            print(f"Skipped: {skipped}")
            print(f"Failed: {failed}")
            print(f"Output: {out_dir.absolute()}")

            runner.print_stats()

        return 0


