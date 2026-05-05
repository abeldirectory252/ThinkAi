"""
Tool registry for the ThinkLab agent.

Each tool has a name, description, parameter spec, and a callable.
The ToolRegistry formats tool descriptions for the LLM system prompt
and dispatches calls by name.
"""
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("thinklab.agent.tools")


# ── Tool definitions (schema only — callables are wired in agent.py) ──

SANDBOX_TOOLS: List[dict] = [
    {
        "name": "execute_code",
        "description": "Execute Python code in an isolated sandbox. Returns stdout/stderr.",
        "params": {"code": "string (Python source)", "language": "string (default: python)"},
    },
    {
        "name": "execute_in_session",
        "description": "Execute code in a persistent session (state persists across calls).",
        "params": {"code": "string (Python source)"},
    },
    {
        "name": "upload_file",
        "description": "Upload a local file to the sandbox.",
        "params": {"filepath": "string (local path)"},
    },
    {
        "name": "list_files",
        "description": "List all files in the sandbox.",
        "params": {},
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file from the sandbox (first 10k chars).",
        "params": {"file_id": "string"},
    },
    {
        "name": "create_workspace",
        "description": "Create a named workspace in the sandbox.",
        "params": {"name": "string"},
    },
    {
        "name": "snapshot_workspace",
        "description": "Snapshot a workspace for later restore.",
        "params": {"workspace_id": "string"},
    },
    {
        "name": "analyze_image",
        "description": "Run the vision AI model on an image with a text prompt. Returns generated text.",
        "params": {"image_path": "string", "prompt": "string"},
    },
    {
        "name": "explain_image",
        "description": "Run the vision AI model with explainability (Grad-CAM + LIME). "
                       "Returns text, heatmap count, overlap score, and top segments.",
        "params": {"image_path": "string", "prompt": "string", "mode": "string (grad_cam|lime|both)"},
    },
    {
        "name": "finish",
        "description": "Signal that you are done and provide the final answer.",
        "params": {"answer": "string (your final answer to the user)"},
    },
]


class ToolRegistry:
    """Registry that maps tool names to callables and formats them for prompts."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}

    def register(self, name: str, fn: Callable, description: str = ""):
        """Register a tool callable by name."""
        self._tools[name] = fn
        logger.debug("Registered tool: %s", name)

    def call(self, name: str, params: dict) -> Any:
        """Execute a registered tool by name."""
        fn = self._tools.get(name)
        if fn is None:
            return {"error": f"Unknown tool: {name}. Available: {list(self._tools.keys())}"}
        try:
            return fn(**params)
        except TypeError as e:
            return {"error": f"Bad params for '{name}': {e}"}
        except Exception as e:
            logger.exception("Tool '%s' failed", name)
            return {"error": f"Tool '{name}' raised: {type(e).__name__}: {e}"}

    def format_for_prompt(self) -> str:
        """Format all tool definitions for the LLM system prompt."""
        lines = ["## Available Tools\n"]
        for tool_def in SANDBOX_TOOLS:
            name = tool_def["name"]
            desc = tool_def["description"]
            params = tool_def.get("params", {})
            param_str = ", ".join(f"{k}: {v}" for k, v in params.items()) if params else "none"
            lines.append(f"### `{name}`")
            lines.append(f"  {desc}")
            lines.append(f"  Parameters: {param_str}\n")
        return "\n".join(lines)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())
