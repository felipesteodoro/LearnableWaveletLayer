from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from queue import Queue  # stdlib queue
from typing import List, Optional

from .job import ExperimentJob
from .worker import GPUWorker

logger = logging.getLogger(__name__)

# Jobs in these states are considered permanently finished — never re-queued.
_TERMINAL_STATES = {"done", "failed"}


class StatusManager:
    """Thread-safe job state tracker with atomic JSON persistence."""

    def __init__(self, status_file: Path, log_dir: Path, start_time: Optional[float] = None):
        self.status_file = status_file
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._jobs: dict[str, ExperimentJob] = {}
        self._start_time = start_time or time.time()

    def register(self, job: ExperimentJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job
            self._persist()

    def update(self, job: ExperimentJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job
            self._persist()

    def _persist(self) -> None:
        jobs = list(self._jobs.values())
        statuses = [j.status for j in jobs]

        data = {
            "queue_start_time": self._start_time,
            "updated_at": time.time(),
            "summary": {
                "total": len(jobs),
                "pending": statuses.count("pending"),
                "running": statuses.count("running"),
                "retrying": statuses.count("retrying"),
                "done": statuses.count("done"),
                "failed": statuses.count("failed"),
            },
            "jobs": [j.to_dict() for j in jobs],
        }

        # Atomic write: write to .tmp then rename to avoid partial reads
        tmp = self.status_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        tmp.replace(self.status_file)


class GPUJobQueueManager:
    """
    Manages a pool of GPU workers that consume a shared job queue.

    Basic usage::

        manager = GPUJobQueueManager(n_gpus=7)
        manager.add_many(jobs)
        manager.run()

    Resume after crash (auto-detects queue_status.json)::

        manager = GPUJobQueueManager.resume_or_create(all_jobs, gpu_ids=[0,1,2,3,4,5,6])
        manager.run()
    """

    def __init__(
        self,
        results_dir: str = "results",
        retry_wait: int = 60,
        gpu_ids: Optional[List[int]] = None,
        n_gpus: int = 7,
        _start_time: Optional[float] = None,
    ):
        self._gpu_ids: List[int] = gpu_ids if gpu_ids is not None else list(range(n_gpus))
        self._retry_wait = retry_wait

        self._base = Path(__file__).parent.parent
        self._status_file = self._base / results_dir / "queue_status.json"
        log_dir = self._base / results_dir / "logs" / "queue"
        self._status_file.parent.mkdir(parents=True, exist_ok=True)

        self._queue: Queue[Optional[ExperimentJob]] = Queue()
        self._status = StatusManager(self._status_file, log_dir, _start_time)
        self._workers: List[GPUWorker] = []

        logger.info("Queue manager ready — status file: %s", self._status_file)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, job: ExperimentJob) -> "GPUJobQueueManager":
        self._status.register(job)
        self._queue.put(job)
        return self

    def add_many(self, jobs: List[ExperimentJob]) -> "GPUJobQueueManager":
        for job in jobs:
            self.add(job)
        return self

    def run(self) -> StatusManager:
        n = len(self._gpu_ids)
        total = self._queue.qsize()
        logger.info("Starting %d GPU workers for %d pending jobs.", n, total)

        for gpu_id in self._gpu_ids:
            w = GPUWorker(gpu_id, self._queue, self._status, self._retry_wait)
            w.start()
            self._workers.append(w)

        self._queue.join()

        for _ in self._workers:
            self._queue.put(None)
        for w in self._workers:
            w.join()

        logger.info("All jobs finished.")
        return self._status

    # ------------------------------------------------------------------
    # Resume after crash
    # ------------------------------------------------------------------

    @classmethod
    def resume_or_create(
        cls,
        all_jobs: List[ExperimentJob],
        gpu_ids: Optional[List[int]] = None,
        n_gpus: int = 7,
        results_dir: str = "results",
        retry_wait: int = 60,
    ) -> "GPUJobQueueManager":
        """
        If a previous queue_status.json exists, resumes from it:
          - 'done' jobs are skipped entirely.
          - 'failed' (exhausted retries) jobs are also skipped.
          - 'running' / 'retrying' / 'pending' jobs are re-queued
            (running jobs were killed by the crash — reset to pending).

        If no checkpoint exists, starts fresh with all_jobs.
        """
        base = Path(__file__).parent.parent
        status_file = base / results_dir / "queue_status.json"

        manager = cls(
            results_dir=results_dir,
            retry_wait=retry_wait,
            gpu_ids=gpu_ids,
            n_gpus=n_gpus,
        )

        if not status_file.exists():
            logger.info("No checkpoint found — starting fresh (%d jobs).", len(all_jobs))
            manager.add_many(all_jobs)
            return manager

        try:
            with open(status_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read checkpoint (%s) — starting fresh.", exc)
            manager.add_many(all_jobs)
            return manager

        # Preserve original queue start time so dashboard ETA stays accurate
        manager._status._start_time = data.get("queue_start_time", time.time())

        saved: dict[str, dict] = {j["job_id"]: j for j in data.get("jobs", [])}
        # Index full job list by (ticker, model_name, mode) for matching
        index: dict[tuple, ExperimentJob] = {
            (j.ticker, j.model_name, j.mode): j for j in all_jobs
        }

        skipped = resumed = fresh = 0

        for job in all_jobs:
            key = (job.ticker, job.model_name, job.mode)
            saved_job = next(
                (s for s in saved.values()
                 if s["ticker"] == job.ticker
                 and s["model_name"] == job.model_name
                 and s["mode"] == job.mode),
                None,
            )

            if saved_job is None:
                # New job not in checkpoint
                manager.add(job)
                fresh += 1
                continue

            status = saved_job.get("status", "pending")

            if status == "done":
                # Restore into StatusManager so dashboard shows full history
                restored = ExperimentJob.from_dict(saved_job)
                manager._status.register(restored)
                skipped += 1

            elif status == "failed" and saved_job.get("retry_count", 0) >= saved_job.get("max_retries", 2):
                restored = ExperimentJob.from_dict(saved_job)
                manager._status.register(restored)
                skipped += 1

            else:
                # Crashed mid-run or still pending: reset to pending and re-queue
                restored = ExperimentJob.from_dict(saved_job)
                restored.status = "pending"
                restored.pid = None
                restored.start_time = None
                restored.end_time = None
                manager._status.register(restored)
                manager._queue.put(restored)
                resumed += 1

        logger.info(
            "Checkpoint loaded — skipped %d done/failed, resumed %d, new %d.",
            skipped, resumed, fresh,
        )
        return manager
