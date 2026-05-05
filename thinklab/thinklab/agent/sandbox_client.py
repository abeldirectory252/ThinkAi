"""
Sandbox client — HTTP interface to a code execution sandbox.

Provides methods for executing code, managing files and sessions,
creating workspaces, and snapshots.
"""
import logging
from typing import Optional

import requests

logger = logging.getLogger("thinklab.agent.sandbox")


class SandboxClient:
    """HTTP client for the ThinkLab sandbox execution environment."""

    def __init__(self, base_url: str = "http://localhost:8000",
                 api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"
        self.session.headers["Content-Type"] = "application/json"

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _post(self, path: str, **kwargs) -> dict:
        r = self.session.post(self._url(path), **kwargs)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str, **kwargs) -> dict:
        r = self.session.get(self._url(path), **kwargs)
        r.raise_for_status()
        return r.json()

    # ── Code execution ──────────────────────────────────────────────

    def execute(self, code: str, language: str = "python") -> dict:
        """Execute code in an isolated sandbox. Returns stdout/stderr."""
        return self._post("/execute", json={
            "code": code,
            "language": language,
        })

    # ── Persistent sessions ─────────────────────────────────────────

    def create_session(self, language: str = "python") -> dict:
        """Create a persistent execution session."""
        return self._post("/sessions", json={"language": language})

    def exec_in_session(self, session_id: str, code: str) -> dict:
        """Execute code in a persistent session (state carries over)."""
        return self._post(f"/sessions/{session_id}/execute", json={
            "code": code,
        })

    def close_session(self, session_id: str) -> dict:
        """Close a persistent session."""
        r = self.session.delete(self._url(f"/sessions/{session_id}"))
        r.raise_for_status()
        return r.json()

    # ── File management ─────────────────────────────────────────────

    def upload_file(self, filepath: str) -> dict:
        """Upload a file to the sandbox."""
        with open(filepath, "rb") as f:
            r = self.session.post(
                self._url("/files"),
                files={"file": f},
                headers={k: v for k, v in self.session.headers.items()
                         if k != "Content-Type"},
            )
        r.raise_for_status()
        return r.json()

    def list_files(self) -> dict:
        """List all files in the sandbox."""
        return self._get("/files")

    def download_file(self, file_id: str, local_path: str) -> None:
        """Download a file from the sandbox."""
        r = self.session.get(self._url(f"/files/{file_id}"), stream=True)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # ── Workspaces ──────────────────────────────────────────────────

    def create_workspace(self, name: str) -> dict:
        """Create a named workspace."""
        return self._post("/workspaces", json={"name": name})

    def snapshot_workspace(self, workspace_id: str) -> dict:
        """Snapshot a workspace for later restore."""
        return self._post(f"/workspaces/{workspace_id}/snapshot")
