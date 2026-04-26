from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import threading
from pathlib import Path
from queue import Queue  # stdlib queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .job_queue import StatusManager

from .job import ExperimentJob

logger = logging.getLogger(__name__)

RUNNER_SCRIPT = Path(__file__).parent / "experiment_runner.py"


class GPUWorker(threading.Thread):
    def __init__(
        self,
        gpu_id: int,
        queue: Queue,
        status: "StatusManager",
        retry_wait: int = 60,
    ):
        super().__init__(daemon=True, name=f"GPU-{gpu_id}")
        self.gpu_id = gpu_id
        self.queue = queue
        self.status = status
        self.retry_wait = retry_wait

    def run(self):
        while True:
            job: ExperimentJob | None = self.queue.get()
            if job is None:  # poison pill — shut down
                self.queue.task_done()
                break

            try:
                self._execute(job)
            except Exception as exc:
                self._handle_failure(job, str(exc))
            finally:
                self.queue.task_done()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute(self, job: ExperimentJob):
        config_file = f"/tmp/exp_{job.job_id}.json"
        with open(config_file, "w") as f:
            json.dump(job.to_dict(), f, indent=2)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        env["EXP_CONFIG_FILE"] = config_file
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"

        job.status = "running"
        job.gpu_id = self.gpu_id
        job.start_time = time.time()
        job.end_time = None
        job.error_msg = None
        self.status.update(job)

        log_path = self.status.log_dir / f"{job.job_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w") as log_fh:
            proc = subprocess.Popen(
                [sys.executable, str(RUNNER_SCRIPT)],
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                cwd=str(RUNNER_SCRIPT.parent.parent),
            )
            job.pid = proc.pid
            self.status.update(job)
            returncode = proc.wait()

        job.pid = None
        job.end_time = time.time()

        if returncode != 0:
            error_excerpt = _tail(log_path, n=30)
            raise RuntimeError(error_excerpt)

        job.status = "done"
        self.status.update(job)
        logger.info("Done: %s (GPU %d, %.1fs)", job.name, self.gpu_id, job.elapsed)

    def _handle_failure(self, job: ExperimentJob, error_msg: str):
        job.error_msg = error_msg[-600:]

        if job.retry_count < job.max_retries:
            job.retry_count += 1
            job.status = "retrying"
            self.status.update(job)
            logger.warning(
                "Retry %d/%d for %s in %ds",
                job.retry_count,
                job.max_retries,
                job.name,
                self.retry_wait,
            )
            time.sleep(self.retry_wait)
            job.status = "pending"
            job.start_time = None
            job.end_time = None
            # Re-queue BEFORE task_done so Queue.join() keeps waiting
            self.queue.put(job)
        else:
            job.status = "failed"
            job.end_time = time.time()
            self.status.update(job)
            logger.error("Failed (no retries left): %s\n%s", job.name, job.error_msg)


def _tail(path: Path, n: int = 30) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return ""
