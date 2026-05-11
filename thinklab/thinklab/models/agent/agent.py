"""
ThinkLab Agent — ReAct loop using LLM + Sandbox tools.

The agent:
  1. Receives a task from the user
  2. Reasons about what to do (LLM)
  3. Calls tools (sandbox execute, files, vision model, etc.)
  4. Observes results
  5. Repeats until done or max_steps
"""
import json
import logging
import re
import tempfile
from typing import Optional, List, Dict, Any

import torch

from .sandbox_client import SandboxClient
from .tools import ToolRegistry, SANDBOX_TOOLS

logger = logging.getLogger("thinklab.agent")

SYSTEM_PROMPT = """You are ThinkLab Agent — an AI that solves tasks by writing and executing code.

You have access to a sandbox environment where you can run Python code, manage files,
and use a vision AI model to analyze images.

{tools}

To use a tool, respond with a JSON block:
```tool
{{"tool": "tool_name", "params": {{"key": "value"}}}}
class ThinkLabAgent: """ReAct agent that uses an LLM + sandbox to solve tasks."""


def __init__(
    self,
    thinklab_model,
    sandbox_url: str = "http://localhost:8000",
    sandbox_api_key: str = "",
    mcp_endpoints: Optional[List[str]] = None,
    max_steps: int = 20,
    verbose: bool = True,
):
    self.tm = thinklab_model
    self.model = thinklab_model.model
    self.tokenizer = thinklab_model.tokenizer
    self.sandbox = SandboxClient(sandbox_url, sandbox_api_key)
    self.max_steps = max_steps
    self.verbose = verbose

    # Persistent session for stateful execution
    self._session_id: Optional[str] = None

    # Tool registry
    self.tools = ToolRegistry()
    self._register_tools()

    # Conversation history
    self.history: List[Dict[str, str]] = []

    # MCP endpoints (for future MCP protocol support)
    self.mcp_endpoints = mcp_endpoints or []

    logger.info("Agent initialized | sandbox=%s | max_steps=%d",
                 sandbox_url, max_steps)

def _register_tools(self):
    """Wire tool names to actual callables."""
    self.tools.register("execute_code", self._tool_execute)
    self.tools.register("execute_in_session", self._tool_session_exec)
    self.tools.register("upload_file", self._tool_upload)
    self.tools.register("list_files", self._tool_list_files)
    self.tools.register("read_file", self._tool_read_file)
    self.tools.register("create_workspace", self._tool_create_ws)
    self.tools.register("snapshot_workspace", self._tool_snapshot)
    self.tools.register("analyze_image", self._tool_analyze)
    self.tools.register("explain_image", self._tool_explain)
    self.tools.register("finish", self._tool_finish)

# ── Tool implementations ────────────────────────────────────────
def _tool_execute(self, code: str, language: str = "python") -> dict:
    return self.sandbox.execute(code, language)

def _tool_session_exec(self, code: str) -> dict:
    if not self._session_id:
        sess = self.sandbox.create_session("python")
        self._session_id = sess["session_id"]
        logger.info("Created session: %s", self._session_id)
    return self.sandbox.exec_in_session(self._session_id, code)

def _tool_upload(self, filepath: str) -> dict:
    return self.sandbox.upload_file(filepath)

def _tool_list_files(self) -> dict:
    return self.sandbox.list_files()

def _tool_read_file(self, file_id: str) -> dict:
    tmp = tempfile.mktemp(suffix=".txt")
    self.sandbox.download_file(file_id, tmp)
    with open(tmp) as f:
        content = f.read(10000)  # cap at 10k chars
    return {"file_id": file_id, "content": content}

def _tool_create_ws(self, name: str) -> dict:
    return self.sandbox.create_workspace(name)

def _tool_snapshot(self, workspace_id: str) -> dict:
    return self.sandbox.snapshot_workspace(workspace_id)

def _tool_analyze(self, image_path: str, prompt: str) -> dict:
    result = self.tm.inference(image_path, prompt, max_tokens=128)
    return {"text": result["text"]}

def _tool_explain(self, image_path: str, prompt: str) -> dict:
    result = self.tm.explain(image_path, prompt)
    return {
        "text": result["text"],
        "mean_overlap": result.get("correlation", {}).get("mean_overlap", 0),
        "tokens": [pt["token"] for pt in result.get("correlation", {}).get("per_token", [])],
    }

def _tool_finish(self, answer: str) -> dict:
    return {"finished": True, "answer": answer}

# ── Parse tool call from LLM output ─────────────────────────────
@staticmethod
def _parse_tool_call(text: str) -> Optional[dict]:
    """Extract ```tool { ... } ``` block from LLM output."""
    pattern = r"```tool\s*\n?(.*?)\n?```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None
    return None

# ── Build prompt for LLM ────────────────────────────────────────
def _build_prompt(self, task: str) -> str:
    system = SYSTEM_PROMPT.format(tools=self.tools.format_for_prompt())
    parts = [system, f"\n## Task\n{task}\n"]
    for entry in self.history:
        if entry["role"] == "assistant":
            parts.append(f"\n## Assistant\n{entry['content']}\n")
        elif entry["role"] == "observation":
            parts.append(f"\n## Observation\n{entry['content']}\n")
    parts.append("\n## Assistant\n")
    return "\n".join(parts)

# ── Main agent loop ─────────────────────────────────────────────
def run(self, task: str) -> dict:
    """
    Execute an agentic task.

    Args:
        task: natural language task description

    Returns:
        dict with: answer, steps, history
    """
    logger.info("Agent task: %s", task[:100])
    self.history = []
    steps = []

    for step_i in range(self.max_steps):
        # Build full prompt
        prompt = self._build_prompt(task)

        # Generate LLM response
        dev = next(self.model.parameters()).device
        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([input_ids], device=dev)

        # Create dummy pixel values (agent mode may not need vision)
        img_size = self.tm.image_size
        dummy_pv = torch.zeros(1, 3, img_size, img_size,
                               dtype=self.tm.dtype, device=dev)

        with torch.no_grad():
            gen_out = self.model.generate(
                dummy_pv, input_ids,
                max_new_tokens=512,
                temperature=0.3,
                top_k=20, top_p=0.9,
            )

        response = self.tokenizer.decode(gen_out["generated_ids"])
        self.history.append({"role": "assistant", "content": response})

        if self.verbose:
            logger.info("Step %d | LLM: %s", step_i, response[:200])

        # Parse tool call
        tool_call = self._parse_tool_call(response)
        if tool_call is None:
            # No tool call — treat as thinking, continue
            steps.append({"step": step_i, "type": "think", "text": response})
            continue

        tool_name = tool_call.get("tool", "")
        tool_params = tool_call.get("params", {})

        # Execute tool
        result = self.tools.call(tool_name, tool_params)
        result_str = json.dumps(result, indent=2, default=str)[:5000]

        if self.verbose:
            logger.info("Step %d | Tool: %s → %s",
                        step_i, tool_name, result_str[:200])

        steps.append({
            "step": step_i,
            "type": "tool",
            "tool": tool_name,
            "params": tool_params,
            "result": result,
        })

        # Check if finished
        if isinstance(result, dict) and result.get("finished"):
            return {
                "answer": result["answer"],
                "steps": steps,
                "history": self.history,
                "total_steps": step_i + 1,
            }

        # Feed observation back
        self.history.append({"role": "observation", "content": result_str})

    return {
        "answer": "[Agent reached max steps without finishing]",
        "steps": steps,
        "history": self.history,
        "total_steps": self.max_steps,
    }

# ── Cleanup ─────────────────────────────────────────────────────
def cleanup(self):
    if self._session_id:
        try:
            self.sandbox.close_session(self._session_id)
        except Exception:
            pass
        self._session_id = None
