"""
Workspace Manager

A workspace is a named, isolated RAG context backed by its own ChromaDB collection.
Workspace configs are persisted to ~/.llmrag/workspaces.json.
"""

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_BASE = Path.home() / ".llmrag"


@dataclass
class WorkspaceConfig:
    name: str
    chroma_dir: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config_overrides: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WorkspaceConfig":
        return cls(
            name=data["name"],
            chroma_dir=data["chroma_dir"],
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            config_overrides=data.get("config_overrides", {}),
        )


class WorkspaceManager:
    """Manages named RAG workspaces, persisted as JSON."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or _DEFAULT_BASE
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._persist_path = self.base_dir / "workspaces.json"
        self._workspaces: dict[str, WorkspaceConfig] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(self, name: str, config_overrides: dict | None = None) -> WorkspaceConfig:
        if name in self._workspaces:
            raise ValueError(f"Workspace '{name}' already exists")

        chroma_dir = str(self.base_dir / "workspaces" / name / "chroma")
        Path(chroma_dir).mkdir(parents=True, exist_ok=True)

        ws = WorkspaceConfig(
            name=name,
            chroma_dir=chroma_dir,
            config_overrides=config_overrides or {},
        )
        self._workspaces[name] = ws
        self._save()
        logger.info(f"Created workspace: {name}")
        return ws

    def get(self, name: str) -> WorkspaceConfig:
        if name not in self._workspaces:
            raise KeyError(f"Workspace '{name}' not found")
        return self._workspaces[name]

    def list(self) -> list[WorkspaceConfig]:
        return list(self._workspaces.values())

    def delete(self, name: str) -> None:
        if name not in self._workspaces:
            raise KeyError(f"Workspace '{name}' not found")
        ws = self._workspaces.pop(name)
        # Remove persisted data directory
        ws_data_dir = self.base_dir / "workspaces" / name
        if ws_data_dir.exists():
            shutil.rmtree(ws_data_dir)
        self._save()
        logger.info(f"Deleted workspace: {name}")

    def exists(self, name: str) -> bool:
        return name in self._workspaces

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        data = {name: ws.to_dict() for name, ws in self._workspaces.items()}
        self._persist_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        if not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text())
            self._workspaces = {
                name: WorkspaceConfig.from_dict(ws_data)
                for name, ws_data in data.items()
            }
            logger.info(f"Loaded {len(self._workspaces)} workspace(s)")
        except Exception as exc:
            logger.warning(f"Failed to load workspaces: {exc}")
