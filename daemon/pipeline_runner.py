"""
Pipeline Runner

Wraps RAGPipeline with workspace-aware ChromaDB routing.
Each workspace gets its own ChromaStore collection and persisted data directory.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from src.config import Config
from src.indexers.vector_store import ChromaStore
from src.indexers.retriever import Retriever
from src.pipeline import RAGPipeline

from .job_store import JobStore
from .workspace_manager import WorkspaceConfig

logger = logging.getLogger(__name__)

# Path to the global config.yaml (relative to repo root)
_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config.yaml"


class _WorkspaceRAGPipeline(RAGPipeline):
    """RAGPipeline subclass that routes the vector store to a workspace-specific ChromaDB directory."""

    def __init__(self, config: Config, chroma_dir: str, collection_name: str):
        self._ws_chroma_dir = chroma_dir
        self._ws_collection_name = collection_name
        super().__init__(config)

    def _init_indexers(self):
        self.vector_store = ChromaStore(
            persist_dir=self._ws_chroma_dir,
            collection=self._ws_collection_name,
        )
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
            provider=self.provider or self.minimax,
            retrieval_mode=self.config.qa.retrieval_mode,
            vector_weight=self.config.qa.vector_weight,
            bm25_weight=self.config.qa.bm25_weight,
            rerank_enabled=self.config.qa.rerank_enabled,
            rerank_top_n=self.config.qa.rerank_top_n,
            bm25_k1=self.config.qa.bm25_k1,
            bm25_b=self.config.qa.bm25_b,
        )


class PipelineRunner:
    """Manages per-workspace RAGPipeline instances and async indexing jobs."""

    def __init__(self, config_path: Optional[str] = None, job_store: Optional[JobStore] = None):
        self._config_path = config_path or str(_DEFAULT_CONFIG_PATH)
        self._job_store = job_store
        self._pipeline_cache: dict[str, _WorkspaceRAGPipeline] = {}
        self._base_config: Optional[Config] = None

    def _get_base_config(self) -> Config:
        if self._base_config is None:
            if Path(self._config_path).exists():
                self._base_config = Config.from_yaml(self._config_path)
            else:
                self._base_config = Config()
        return self._base_config

    def get_pipeline(self, workspace: WorkspaceConfig) -> _WorkspaceRAGPipeline:
        """Return cached pipeline for workspace, creating it on first call."""
        name = workspace.name
        if name not in self._pipeline_cache:
            config = self._get_base_config().model_copy(deep=True)
            # Apply workspace-level overrides (flat key=value pairs)
            for key, value in workspace.config_overrides.items():
                parts = key.split(".")
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            # Force Chroma for daemon mode
            config.vector_store.type = "chroma"
            self._pipeline_cache[name] = _WorkspaceRAGPipeline(
                config=config,
                chroma_dir=workspace.chroma_dir,
                collection_name=name,
            )
            logger.info(f"Initialized pipeline for workspace '{name}'")
        return self._pipeline_cache[name]

    def evict(self, workspace_name: str) -> None:
        """Remove pipeline from cache (e.g., after workspace deletion)."""
        self._pipeline_cache.pop(workspace_name, None)

    def doc_count(self, workspace: WorkspaceConfig) -> int:
        """Return number of indexed vectors for the workspace."""
        try:
            pipeline = self.get_pipeline(workspace)
            return pipeline.vector_store.count()
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index_path(
        self,
        workspace: WorkspaceConfig,
        path: str,
        job_id: Optional[str] = None,
    ) -> dict:
        """Index a file or directory into the workspace. Runs in a thread pool."""

        def _update(progress: int, message: str, status: str = "running"):
            if self._job_store and job_id:
                self._job_store.update(
                    job_id,
                    status=status,
                    progress=progress,
                    message=message,
                )

        pipeline = self.get_pipeline(workspace)
        p = Path(path)

        if not p.exists():
            _update(0, f"Path not found: {path}", status="failed")
            raise FileNotFoundError(f"Path not found: {path}")

        _update(0, "Starting indexing", status="running")

        loop = asyncio.get_event_loop()

        if p.is_file():
            _update(10, f"Indexing file: {p.name}")
            chunks = await pipeline.process(str(p))
            _update(100, f"Done. {len(chunks)} chunks indexed.", status="done")
            result = {"files": 1, "chunks": len(chunks)}
        else:
            # Directory — process_corpus uses ThreadPoolExecutor internally (sync wrapper)
            _update(10, f"Scanning directory: {p}")

            def _run_corpus():
                return asyncio.run(pipeline.process_corpus(str(p)))

            all_chunks = await loop.run_in_executor(None, _run_corpus)
            chunk_count = len(all_chunks) if isinstance(all_chunks, list) else 0
            _update(100, f"Done. {chunk_count} chunks indexed.", status="done")
            result = {"chunks": chunk_count}

        if self._job_store and job_id:
            self._job_store.update(job_id, result=result)

        return result

    # ------------------------------------------------------------------
    # Query / Answer
    # ------------------------------------------------------------------

    async def query(
        self,
        workspace: WorkspaceConfig,
        text: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Vector search query."""
        pipeline = self.get_pipeline(workspace)
        results = await pipeline.query(text, k=top_k)
        return results if isinstance(results, list) else []

    async def query_and_answer(
        self,
        workspace: WorkspaceConfig,
        text: str,
    ) -> dict:
        """Full RAG: query → retrieve → answer."""
        pipeline = self.get_pipeline(workspace)
        result = await pipeline.query_and_answer(text)
        if isinstance(result, dict):
            return result
        return {"answer": str(result)}
