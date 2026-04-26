from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ExperimentJob:
    ticker: str
    model_name: str          # CNN | LSTM | CNN_LSTM | Transformer
    mode: str                # raw | db4 | learned_wavelet
    config: dict = field(default_factory=dict)
    max_retries: int = 2

    # Runtime state — set by GPUJobQueueManager / GPUWorker
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    retry_count: int = 0
    status: str = "pending"  # pending | running | retrying | done | failed
    gpu_id: Optional[int] = None
    pid: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_msg: Optional[str] = None

    @property
    def name(self) -> str:
        return f"{self.ticker}/{self.model_name}/{self.mode}"

    @property
    def elapsed(self) -> Optional[float]:
        if self.start_time is None:
            return None
        return (self.end_time or time.time()) - self.start_time

    def to_dict(self) -> dict:
        d = asdict(self)
        d["elapsed"] = self.elapsed
        d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentJob":
        """Reconstruct from a persisted dict (e.g. queue_status.json)."""
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in allowed})
