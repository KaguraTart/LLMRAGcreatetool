"""
Job Store

In-memory tracking for async indexing jobs.
"""

import uuid
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class JobStatus:
    id: str
    workspace: str
    type: str  # "index"
    status: str = "pending"  # pending | running | done | failed
    progress: int = 0  # 0-100
    message: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


class JobStore:
    """Thread-safe (asyncio) in-memory job registry."""

    def __init__(self):
        self._jobs: dict[str, JobStatus] = {}

    def create(self, workspace: str, job_type: str = "index") -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = JobStatus(id=job_id, workspace=workspace, type=job_type)
        logger.debug(f"Created job {job_id} for workspace '{workspace}'")
        return job_id

    def update(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        result: Optional[dict] = None,
    ) -> None:
        job = self._get(job_id)
        if status is not None:
            job.status = status
            if status == "running" and job.started_at is None:
                job.started_at = datetime.now(timezone.utc).isoformat()
            if status in ("done", "failed"):
                job.finished_at = datetime.now(timezone.utc).isoformat()
        if progress is not None:
            job.progress = progress
        if message is not None:
            job.message = message
        if result is not None:
            job.result = result

    def get(self, job_id: str) -> JobStatus:
        return self._get(job_id)

    def _get(self, job_id: str) -> JobStatus:
        if job_id not in self._jobs:
            raise KeyError(f"Job '{job_id}' not found")
        return self._jobs[job_id]
