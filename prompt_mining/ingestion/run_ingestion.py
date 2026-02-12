#!/usr/bin/env python3
"""
GPU Pool Orchestrator for Dataset Ingestion.

Manages parallel ingestion jobs across multiple GPUs with automatic
job scheduling and live status display.

Usage:
    python -m prompt_mining.ingestion.run_ingestion config.yaml
    python -m prompt_mining.ingestion.run_ingestion config.yaml --dry-run
    python -m prompt_mining.ingestion.run_ingestion config.yaml --gpus 0,1,2,3
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: 'rich' not installed. Using basic output. Install with: pip install rich")


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """Represents a single dataset ingestion job."""
    name: str
    dataset_class: str
    dataset_params: dict
    evaluator_class: Optional[str] = None
    evaluator_params: Optional[dict] = None
    num_prompts: Optional[int] = None  # Override global setting

    # Runtime state
    status: JobStatus = JobStatus.PENDING
    gpus: list[int] = field(default_factory=list)
    process: Optional[asyncio.subprocess.Process] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    return_code: Optional[int] = None
    log_file: Optional[Path] = None

    def duration(self) -> Optional[timedelta]:
        if self.start_time is None:
            return None
        end = self.end_time or datetime.now()
        return end - self.start_time

    def duration_str(self) -> str:
        dur = self.duration()
        if dur is None:
            return "-"
        total_seconds = int(dur.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"


@dataclass
class Config:
    """Parsed configuration."""
    config_path: Path  # Path to YAML config (passed to ingest.py)
    gpus: list[int]
    output_dir: Path
    num_prompts: int
    num_gpus: int
    env: dict
    jobs: list[Job]
    log_dir: Path
    # For display only
    model_name: str = ""
    backend: str = ""


class GPUPool:
    """Manages available GPUs."""

    def __init__(self, gpus: list[int]):
        self.all_gpus = set(gpus)
        self.available = set(gpus)
        self._lock = asyncio.Lock()

    async def acquire(self, num_gpus: int = 1) -> Optional[list[int]]:
        """Acquire N available GPUs atomically. Returns None if insufficient GPUs."""
        if num_gpus <= 0:
            raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
        async with self._lock:
            if len(self.available) < num_gpus:
                return None
            gpus = sorted(self.available)[:num_gpus]
            for g in gpus:
                self.available.remove(g)
            return gpus

    async def release(self, gpus: list[int]):
        """Release multiple GPUs back to the pool."""
        async with self._lock:
            for gpu in gpus:
                if gpu in self.all_gpus:
                    self.available.add(gpu)

    def status(self) -> str:
        return f"{len(self.available)}/{len(self.all_gpus)} available"


class Orchestrator:
    """Main orchestrator for running ingestion jobs."""

    def __init__(self, config: Config, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.gpu_pool = GPUPool(config.gpus)
        self.jobs = config.jobs
        self.console = Console() if RICH_AVAILABLE else None
        self._shutdown = False
        self._running_tasks: set[asyncio.Task] = set()

    def _build_command(self, job: Job) -> list[str]:
        """Build the ingest.py command for a job."""
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "ingest.py"),
            "--config", str(self.config.config_path),
            "--dataset-name", job.name,
        ]

        # Num prompts override (job-specific or global)
        num_prompts = job.num_prompts if job.num_prompts is not None else self.config.num_prompts
        if num_prompts != -1:
            cmd.extend(["--num-prompts", str(num_prompts)])

        return cmd

    def _build_env(self, gpus: list[int]) -> dict:
        """Build environment variables for a job."""
        env = os.environ.copy()
        # Expose multiple GPUs for device_map='auto' / model parallel loads.
        # CUDA_VISIBLE_DEVICES accepts a comma-separated list.
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        env.update(self.config.env)
        return env

    async def _run_job(self, job: Job, gpus: list[int]):
        """Run a single job on the specified GPU(s)."""
        job.status = JobStatus.RUNNING
        job.gpus = gpus
        job.start_time = datetime.now()
        job.log_file = self.config.log_dir / f"{job.name}.log"

        cmd = self._build_command(job)
        env = self._build_env(gpus)

        if self.dry_run:
            # Simulate job completion
            await asyncio.sleep(0.5)
            job.status = JobStatus.COMPLETED
            job.end_time = datetime.now()
            job.return_code = 0
            return

        try:
            with open(job.log_file, "w") as log_f:
                # Write command to log for debugging
                log_f.write(f"# Command: {' '.join(cmd)}\n")
                log_f.write(f"# GPUs: {gpus}\n")
                log_f.write(f"# Started: {job.start_time.isoformat()}\n")
                log_f.write("=" * 80 + "\n\n")
                log_f.flush()

                job.process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=log_f,
                    stderr=asyncio.subprocess.STDOUT,
                    env=env,
                    cwd=Path(__file__).parent.parent.parent,  # prompt-mining root
                )

                job.return_code = await job.process.wait()
        except Exception as e:
            job.return_code = -1
            if job.log_file:
                with open(job.log_file, "a") as f:
                    f.write(f"\n\nOrchestrator error: {e}\n")

        job.end_time = datetime.now()
        job.status = JobStatus.COMPLETED if job.return_code == 0 else JobStatus.FAILED

    async def _job_worker(self, job: Job):
        """Worker that acquires GPU, runs job, releases GPU."""
        # Wait for available GPU(s)
        gpus: Optional[list[int]] = None
        while gpus is None and not self._shutdown:
            gpus = await self.gpu_pool.acquire(self.config.num_gpus)
            if gpus is None:
                await asyncio.sleep(0.5)

        if self._shutdown:
            return

        try:
            await self._run_job(job, gpus)
        finally:
            await self.gpu_pool.release(gpus)

    def _create_status_table(self) -> Table:
        """Create a rich table showing job status."""
        table = Table(
            title="Ingestion Jobs",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Dataset", style="white", min_width=20)
        table.add_column("Status", justify="center", min_width=10)
        table.add_column("GPUs", justify="center", min_width=8)
        table.add_column("Duration", justify="right", min_width=12)
        table.add_column("Log", style="dim", max_width=30)

        for job in self.jobs:
            # Status with color
            if job.status == JobStatus.PENDING:
                status = Text("PENDING", style="dim")
            elif job.status == JobStatus.RUNNING:
                status = Text("RUNNING", style="bold yellow")
            elif job.status == JobStatus.COMPLETED:
                status = Text("COMPLETED", style="bold green")
            else:
                status = Text("FAILED", style="bold red")

            gpu_str = ",".join(str(g) for g in job.gpus) if job.gpus else "-"
            log_str = job.log_file.name if job.log_file else "-"

            table.add_row(
                job.name,
                status,
                gpu_str,
                job.duration_str(),
                log_str,
            )

        return table

    def _create_summary_panel(self) -> Panel:
        """Create summary panel with counts."""
        pending = sum(1 for j in self.jobs if j.status == JobStatus.PENDING)
        running = sum(1 for j in self.jobs if j.status == JobStatus.RUNNING)
        completed = sum(1 for j in self.jobs if j.status == JobStatus.COMPLETED)
        failed = sum(1 for j in self.jobs if j.status == JobStatus.FAILED)

        text = Text()
        text.append(f"Pending: {pending}  ", style="dim")
        text.append(f"Running: {running}  ", style="yellow")
        text.append(f"Completed: {completed}  ", style="green")
        text.append(f"Failed: {failed}  ", style="red" if failed > 0 else "dim")
        text.append(f"| GPUs: {self.gpu_pool.status()}", style="cyan")

        return Panel(text, title="Summary", border_style="blue")

    def _print_basic_status(self):
        """Print status without rich library."""
        os.system('clear' if os.name == 'posix' else 'cls')
        print("=" * 60)
        print("INGESTION STATUS")
        print("=" * 60)
        print(f"{'Dataset':<25} {'Status':<12} {'GPUs':<10} {'Duration':<12}")
        print("-" * 60)
        for job in self.jobs:
            gpu_str = ",".join(str(g) for g in job.gpus) if job.gpus else "-"
            print(f"{job.name:<25} {job.status.value:<12} {gpu_str:<10} {job.duration_str():<12}")
        print("-" * 60)
        pending = sum(1 for j in self.jobs if j.status == JobStatus.PENDING)
        running = sum(1 for j in self.jobs if j.status == JobStatus.RUNNING)
        completed = sum(1 for j in self.jobs if j.status == JobStatus.COMPLETED)
        failed = sum(1 for j in self.jobs if j.status == JobStatus.FAILED)
        print(f"Pending: {pending} | Running: {running} | Completed: {completed} | Failed: {failed}")
        print(f"GPUs: {self.gpu_pool.status()}")

    async def _status_updater(self, live: Optional["Live"] = None):
        """Update status display periodically."""
        while not self._shutdown:
            if live:
                table = self._create_status_table()
                summary = self._create_summary_panel()
                from rich.console import Group
                live.update(Group(table, summary))
            else:
                self._print_basic_status()
            await asyncio.sleep(1)

    def _handle_signal(self, sig):
        """Handle interrupt signals."""
        self._shutdown = True
        if self.console:
            self.console.print("\n[yellow]Received interrupt, shutting down...[/yellow]")
        else:
            print("\nReceived interrupt, shutting down...")

        # Terminate running processes
        for job in self.jobs:
            if job.process and job.status == JobStatus.RUNNING:
                try:
                    job.process.terminate()
                except ProcessLookupError:
                    pass

    async def run(self):
        """Run all jobs with GPU scheduling."""
        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: self._handle_signal(s))

        # Create log directory
        self.config.log_dir.mkdir(parents=True, exist_ok=True)

        # Print initial info
        if self.console:
            self.console.print(Panel(
                f"[bold]Model:[/bold] {self.config.model_name}\n"
                f"[bold]Backend:[/bold] {self.config.backend}\n"
                f"[bold]Output:[/bold] {self.config.output_dir}\n"
                f"[bold]GPUs:[/bold] {self.config.gpus}\n"
                f"[bold]Jobs:[/bold] {len(self.jobs)}",
                title="Configuration",
                border_style="green"
            ))
        else:
            print(f"Model: {self.config.model_name}")
            print(f"Backend: {self.config.backend}")
            print(f"Output: {self.config.output_dir}")
            print(f"GPUs: {self.config.gpus}")
            print(f"Jobs: {len(self.jobs)}")

        if self.dry_run:
            msg = "[yellow]DRY RUN - No actual jobs will be executed[/yellow]"
            if self.console:
                self.console.print(msg)
            else:
                print("DRY RUN - No actual jobs will be executed")

        # Create job tasks
        job_tasks = [asyncio.create_task(self._job_worker(job)) for job in self.jobs]

        # Run with live status display
        if RICH_AVAILABLE and self.console:
            with Live(self._create_status_table(), console=self.console, refresh_per_second=1) as live:
                status_task = asyncio.create_task(self._status_updater(live))
                try:
                    await asyncio.gather(*job_tasks)
                finally:
                    self._shutdown = True
                    await status_task
        else:
            status_task = asyncio.create_task(self._status_updater())
            try:
                await asyncio.gather(*job_tasks)
            finally:
                self._shutdown = True
                status_task.cancel()
                try:
                    await status_task
                except asyncio.CancelledError:
                    pass

        # Print final summary
        self._print_final_summary()

    def _print_final_summary(self):
        """Print final summary after all jobs complete."""
        completed = [j for j in self.jobs if j.status == JobStatus.COMPLETED]
        failed = [j for j in self.jobs if j.status == JobStatus.FAILED]

        if self.console:
            self.console.print("\n")
            self.console.print(Panel(
                f"[green]Completed: {len(completed)}[/green]\n"
                f"[red]Failed: {len(failed)}[/red]\n"
                f"[dim]Logs: {self.config.log_dir}[/dim]",
                title="Final Summary",
                border_style="blue"
            ))

            if failed:
                self.console.print("\n[bold red]Failed Jobs:[/bold red]")
                for job in failed:
                    self.console.print(f"  - {job.name}: see {job.log_file}")
        else:
            print("\n" + "=" * 60)
            print("FINAL SUMMARY")
            print("=" * 60)
            print(f"Completed: {len(completed)}")
            print(f"Failed: {len(failed)}")
            print(f"Logs: {self.config.log_dir}")
            if failed:
                print("\nFailed Jobs:")
                for job in failed:
                    print(f"  - {job.name}: see {job.log_file}")


def load_config(config_path: Path, gpu_override: Optional[list[int]] = None) -> Config:
    """Load and validate configuration from YAML file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Validate required fields
    required = ["model", "output_dir", "datasets"]
    for field in required:
        if field not in raw:
            raise ValueError(f"Missing required config field: {field}")

    model = raw["model"]
    if "name" not in model:
        raise ValueError("Missing model.name in config")
    if "backend" not in model:
        raise ValueError("Missing model.backend in config")

    # Validate backend-specific config
    backend = model["backend"]
    if backend == "circuit_tracer" and "transcoder_set" not in model:
        raise ValueError("circuit_tracer backend requires model.transcoder_set")
    if (backend == "saelens" or backend == "huggingface") and "sae_configs" not in model:
        raise ValueError("saelens or huggingface backend requires model.sae_configs")

    # GPUs: CLI override > config > default
    gpus = gpu_override or raw.get("gpus", [0])
    if isinstance(gpus, str):
        gpus = [int(g.strip()) for g in gpus.split(",")]

    # Parse jobs from datasets
    jobs = []
    for ds in raw["datasets"]:
        if "class" not in ds:
            raise ValueError(f"Dataset missing 'class' field: {ds}")

        job = Job(
            name=ds.get("name", ds["class"].replace("Dataset", "").lower()),
            dataset_class=ds["class"],
            dataset_params=ds.get("params", {}),
            evaluator_class=ds.get("evaluator_class"),
            evaluator_params=ds.get("evaluator_params"),
            num_prompts=ds.get("num_prompts"),
        )
        jobs.append(job)

    # Build config
    output_dir = Path(raw["output_dir"])
    log_dir = Path(raw.get("log_dir", config_path.parent / "logs"))

    return Config(
        config_path=config_path.absolute(),
        gpus=gpus,
        output_dir=output_dir,
        num_prompts=raw.get("num_prompts", -1),
        num_gpus=int(raw.get("num_gpus", 1)),
        env=raw.get("env", {}),
        jobs=jobs,
        log_dir=log_dir,
        # For display only
        model_name=model["name"],
        backend=backend,
    )


def main():
    parser = argparse.ArgumentParser(
        description="GPU Pool Orchestrator for Dataset Ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    parser.add_argument("--gpus", type=str, help="Override GPU list (comma-separated, e.g., '0,1,2')")

    args = parser.parse_args()

    # Parse GPU override
    gpu_override = None
    if args.gpus:
        gpu_override = [int(g.strip()) for g in args.gpus.split(",")]

    # Load config
    try:
        config = load_config(args.config, gpu_override)
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1

    # Run orchestrator
    orchestrator = Orchestrator(config, dry_run=args.dry_run)
    asyncio.run(orchestrator.run())

    # Return non-zero if any jobs failed
    failed = sum(1 for j in orchestrator.jobs if j.status == JobStatus.FAILED)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
