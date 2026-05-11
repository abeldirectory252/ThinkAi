"""HTTP client for the ThinkLab sandbox API."""
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
import requests

logger = logging.getLogger("thinklab.agent.sandbox")


class SandboxClient:
    """Wraps every sandbox endpoint. Stateless — one instance per agent."""

    def __init__(self, base_url: str = "http://localhost:8000",
                 api_key: str = ""):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        if api_key:
            self.session.headers["X-API-Key"] = api_key
        self.session.headers["Content-Type"] = "application/json"

    # ── Health ──────────────────────────────────────────────────────
    def health(self) -> dict:
        return self.session.get(f"{self.base}/health").json()

    def ready(self) -> dict:
        return self.session.get(f"{self.base}/ready").json()

    # ── Execution ───────────────────────────────────────────────────
    def execute(self, code: str, language: str = "python",
                timeout: int = 30) -> dict:
        r = self.session.post(f"{self.base}/execute", json={
            "code": code, "language": language, "timeout": timeout,
        })
        r.raise_for_status()
        return r.json()

    def execute_batch(self, executions: List[dict]) -> dict:
        r = self.session.post(f"{self.base}/execute/batch", json={
            "executions": executions,
        })
        r.raise_for_status()
        return r.json()

    def execute_stream(self, code: str, language: str = "python"):
        r = self.session.post(f"{self.base}/execute/stream", json={
            "code": code, "language": language,
        }, stream=True)
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if line:
                yield line

    # ── Files ───────────────────────────────────────────────────────
    def upload_file(self, filepath: str) -> dict:
        with open(filepath, "rb") as f:
            r = self.session.post(
                f"{self.base}/files/upload",
                files={"file": f},
                headers={"X-API-Key": self.session.headers.get("X-API-Key", "")},
            )
        r.raise_for_status()
        return r.json()

    def list_files(self) -> dict:
        return self.session.get(f"{self.base}/files").json()

    def download_file(self, file_id: str, dest: str) -> Path:
        r = self.session.get(f"{self.base}/files/{file_id}")
        r.raise_for_status()
        p = Path(dest)
        p.write_bytes(r.content)
        return p

    def file_metadata(self, file_id: str) -> dict:
        return self.session.get(f"{self.base}/files/{file_id}/metadata").json()

    def update_file(self, file_id: str, filepath: str) -> dict:
        with open(filepath, "rb") as f:
            r = self.session.put(
                f"{self.base}/files/{file_id}",
                files={"file": f},
                headers={"X-API-Key": self.session.headers.get("X-API-Key", "")},
            )
        r.raise_for_status()
        return r.json()

    def delete_file(self, file_id: str) -> dict:
        r = self.session.delete(f"{self.base}/files/{file_id}")
        r.raise_for_status()
        return r.json()

    # ── Sessions ────────────────────────────────────────────────────
    def create_session(self, language: str = "python") -> dict:
        r = self.session.post(f"{self.base}/sessions", json={
            "language": language,
        })
        r.raise_for_status()
        return r.json()

    def get_session(self, session_id: str) -> dict:
        return self.session.get(f"{self.base}/sessions/{session_id}").json()

    def exec_in_session(self, session_id: str, code: str) -> dict:
        r = self.session.post(f"{self.base}/sessions/{session_id}/exec", json={
            "code": code,
        })
        r.raise_for_status()
        return r.json()

    def close_session(self, session_id: str) -> dict:
        r = self.session.delete(f"{self.base}/sessions/{session_id}")
        r.raise_for_status()
        return r.json()

    # ── Workspaces ──────────────────────────────────────────────────
    def create_workspace(self, name: str, ttl_seconds: int = 3600) -> dict:
        r = self.session.post(f"{self.base}/workspaces", json={
            "name": name, "ttl_seconds": ttl_seconds,
        })
        r.raise_for_status()
        return r.json()

    def list_workspaces(self) -> dict:
        return self.session.get(f"{self.base}/workspaces").json()

    def get_workspace(self, ws_id: str) -> dict:
        return self.session.get(f"{self.base}/workspaces/{ws_id}").json()

    def update_workspace(self, ws_id: str, ttl_seconds: int) -> dict:
        r = self.session.put(f"{self.base}/workspaces/{ws_id}", json={
            "ttl_seconds": ttl_seconds,
        })
        r.raise_for_status()
        return r.json()

    def delete_workspace(self, ws_id: str) -> dict:
        r = self.session.delete(f"{self.base}/workspaces/{ws_id}")
        r.raise_for_status()
        return r.json()

    def snapshot_workspace(self, ws_id: str) -> dict:
        r = self.session.post(f"{self.base}/workspaces/{ws_id}/snapshot")
        r.raise_for_status()
        return r.json()

    def restore_workspace(self, ws_id: str) -> dict:
        r = self.session.post(f"{self.base}/workspaces/{ws_id}/restore")
        r.raise_for_status()
        return r.json()

    # ── Languages ───────────────────────────────────────────────────
    def list_languages(self) -> dict:
        return self.session.get(f"{self.base}/languages").json()

    def list_packages(self, language: str = "python") -> dict:
        return self.session.get(f"{self.base}/languages/{language}/packages").json()
