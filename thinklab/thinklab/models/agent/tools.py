"""Tool definitions the agent LLM can invoke."""
import json
from typing import Any, Callable, Dict, List


SANDBOX_TOOLS = [
    {
        "name": "execute_code",
        "description": "Execute Python code in the sandbox. Returns stdout, stderr, exit_code.",
        "parameters": {
            "code": {"type": "string", "description": "Python code to execute", "required": True},
            "language": {"type": "string", "description": "Language (default: python)", "required": False},
        },
    },
    {
        "name": "execute_in_session",
        "description": "Execute code in a persistent session (variables are preserved between calls).",
        "parameters": {
            "code": {"type": "string", "description": "Code to execute", "required": True},
        },
    },
    {
        "name": "upload_file",
        "description": "Upload a local file to the sandbox.",
        "parameters": {
            "filepath": {"type": "string", "description": "Local file path", "required": True},
        },
    },
    {
        "name": "list_files",
        "description": "List all files in the sandbox.",
        "parameters": {},
    },
    {
        "name": "read_file",
        "description": "Download and read a file from sandbox by file_id.",
        "parameters": {
            "file_id": {"type": "string", "description": "File ID", "required": True},
        },
    },
    {
        "name": "create_workspace",
        "description": "Create a new isolated workspace.",
        "parameters": {
            "name": {"type": "string", "description": "Workspace name", "required": True},
        },
    },
    {
        "name": "snapshot_workspace",
        "description": "Save a snapshot of the current workspace state.",
        "parameters": {
            "workspace_id": {"type": "string", "description": "Workspace ID", "required": True},
        },
    },
    {
        "name": "analyze_image",
        "description": "Run the loaded vision model on an image with a prompt. Returns model output text.",
        "parameters": {
            "image_path": {"type": "string", "description": "Path to image file", "required": True},
            "prompt": {"type": "string", "description": "Analysis prompt", "required": True},
        },
    },
    {
        "name": "explain_image",
        "description": "Run vision model + Grad-CAM + LIME on an image. Returns text + heatmaps.",
        "parameters": {
            "image_path": {"type": "string", "description": "Path to image", "required": True},
            "prompt": {"type": "string", "description": "Analysis prompt", "required": True},
        },
    },
    {
        "name": "finish",
        "description": "Signal that the task is complete. Provide final answer.",
        "parameters": {
            "answer": {"type": "string", "description": "Final answer to the user", "required": True},
        },
    },
]


class ToolRegistry:
    """Maps tool names → callables."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}

    def register(self, name: str, fn: Callable):
        self._tools[name] = fn

    def call(self, name: str, params: dict) -> Any:
        if name not in self._tools:
            return {"error": f"Unknown tool: {name}"}
        try:
            return self._tools[name](**params)
        except Exception as e:
            return {"error": str(e)}

    def schema(self) -> List[dict]:
        return SANDBOX_TOOLS

    def format_for_prompt(self) -> str:
        lines = ["Available tools:"]
        for t in SANDBOX_TOOLS:
            params = ", ".join(
                f"{k}: {v['type']}" for k, v in t.get("parameters", {}).items()
            )
            lines.append(f"  - {t['name']}({params}): {t['description']}")
        return "\n".join(lines)
