"""
llmragd FastAPI Application

REST API wrapping the RAG pipeline with workspace management.
Default port: 7474
"""

import asyncio
import logging
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .job_store import JobStore
from .pipeline_runner import PipelineRunner
from .workspace_manager import WorkspaceManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singletons (initialised at startup)
# ---------------------------------------------------------------------------

workspace_manager = WorkspaceManager()
job_store = JobStore()
pipeline_runner = PipelineRunner(job_store=job_store)

app = FastAPI(
    title="llmragd",
    description="LLM RAG Daemon — exposes the RAGPipeline as a local REST API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class WorkspaceCreateRequest(BaseModel):
    name: str
    config_overrides: dict[str, Any] = {}


class IndexRequest(BaseModel):
    path: str


class QueryRequest(BaseModel):
    text: str
    top_k: int = 5


class AnswerRequest(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}


@app.get("/api/v1/status", tags=["System"])
def status():
    workspaces = workspace_manager.list()
    workspace_infos = []
    for ws in workspaces:
        try:
            count = pipeline_runner.doc_count(ws)
        except Exception:
            count = 0
        workspace_infos.append({"name": ws.name, "doc_count": count})

    return {
        "daemon": "llmragd",
        "version": "1.0.0",
        "workspace_count": len(workspaces),
        "workspaces": workspace_infos,
    }


# -- Workspaces --

@app.post("/api/v1/workspaces", tags=["Workspaces"], status_code=201)
def create_workspace(req: WorkspaceCreateRequest):
    try:
        ws = workspace_manager.create(req.name, config_overrides=req.config_overrides)
        return ws.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/v1/workspaces", tags=["Workspaces"])
def list_workspaces():
    return [ws.to_dict() for ws in workspace_manager.list()]


@app.get("/api/v1/workspaces/{name}", tags=["Workspaces"])
def get_workspace(name: str):
    try:
        ws = workspace_manager.get(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    try:
        doc_count = pipeline_runner.doc_count(ws)
    except Exception:
        doc_count = 0

    info = ws.to_dict()
    info["doc_count"] = doc_count
    return info


@app.delete("/api/v1/workspaces/{name}", tags=["Workspaces"], status_code=204)
def delete_workspace(name: str):
    try:
        workspace_manager.delete(name)
        pipeline_runner.evict(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# -- Indexing --

@app.post("/api/v1/workspaces/{name}/index", tags=["Indexing"], status_code=202)
def start_index_job(name: str, req: IndexRequest):
    try:
        ws = workspace_manager.get(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    job_id = job_store.create(workspace=name, job_type="index")
    job_store.update(job_id, status="pending", message="Queued")

    async def _run():
        try:
            await pipeline_runner.index_path(ws, req.path, job_id=job_id)
        except Exception as exc:
            logger.error(f"Indexing job {job_id} failed: {exc}")
            job_store.update(job_id, status="failed", message=str(exc))

    asyncio.create_task(_run())

    return {"job_id": job_id}


@app.get("/api/v1/workspaces/{name}/jobs/{job_id}", tags=["Indexing"])
def get_job(name: str, job_id: str):
    try:
        return job_store.get(job_id).to_dict()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/v1/workspaces/{name}/jobs/{job_id}/stream", tags=["Indexing"])
async def stream_job(name: str, job_id: str):
    """Server-Sent Events stream for indexing job progress."""
    try:
        job_store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    async def _generator():
        last_progress = -1
        while True:
            try:
                job = job_store.get(job_id)
            except KeyError:
                break
            if job.progress != last_progress:
                last_progress = job.progress
                yield {
                    "event": "progress",
                    "data": f'{{"progress": {job.progress}, "message": "{job.message}", "status": "{job.status}"}}',
                }
            if job.status in ("done", "failed"):
                yield {"event": "end", "data": f'{{"status": "{job.status}"}}'}
                break
            await asyncio.sleep(0.5)

    return EventSourceResponse(_generator())


# -- Query / Answer --

@app.post("/api/v1/workspaces/{name}/query", tags=["Query"])
async def query_workspace(name: str, req: QueryRequest):
    try:
        ws = workspace_manager.get(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    try:
        results = await pipeline_runner.query(ws, req.text, top_k=req.top_k)
        return {"results": results}
    except Exception as exc:
        logger.error(f"Query failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/workspaces/{name}/answer", tags=["Query"])
async def answer_workspace(name: str, req: AnswerRequest):
    try:
        ws = workspace_manager.get(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    try:
        result = await pipeline_runner.query_and_answer(ws, req.text)
        return result
    except Exception as exc:
        logger.error(f"Answer generation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
