"""
ThinkLab Agent — Agentic Workflow Examples

Demonstrates:
  1. Basic agent task (analyze + code)
  2. Multi-step research workflow
  3. Stateful session with persistent variables
  4. Standalone sandbox usage
  5. MCP-connected agent
  6. Explainability workflow (per-class Grad-CAM + LIME)
"""
import thinklab
from thinklab.agent import SandboxClient


# ═══════════════════════════════════════════════════════════════════
#  1. Basic Agent — Analyze Image + Write Code
# ═══════════════════════════════════════════════════════════════════
def workflow_basic():
    """
    The agent will:
      Step 1: Use analyze_image to read the X-ray
      Step 2: Write Python code to parse the findings
      Step 3: Execute that code in the sandbox
      Step 4: Return structured results via finish tool
    """
    model = thinklab.load_llm(
        model_name="google/medgemma-4b-it",
        token="hf_xxx",
        IsAgent=True,
        temperature=0.1,
        safety_filter="strict",
        device="auto",
    )

    agent = model.agent(
        sandbox_url="http://localhost:8000",
        sandbox_api_key="your-secret-key",
        max_steps=10,
    )

    result = agent.run(
        "Analyze the chest X-ray at /data/xray.png. "
        "Identify all pathological findings. "
        "Then write Python code to calculate the percentage of lung "
        "area affected by consolidation and save the result to /data/report.json."
    )

    print(f"Answer: {result['answer']}")
    print(f"Steps taken: {result['total_steps']}")

    for step in result["steps"]:
        if step["type"] == "tool":
            print(f"  Step {step['step']}: {step['tool']}({list(step['params'].keys())})")
        else:
            print(f"  Step {step['step']}: [thinking] {step['text'][:80]}...")

    agent.cleanup()


# ═══════════════════════════════════════════════════════════════════
#  2. Multi-Step Research — Compare Two Images
# ═══════════════════════════════════════════════════════════════════
def workflow_research():
    model = thinklab.load_llm(
        model_name="google/medgemma-4b-it",
        token="hf_xxx",
        IsAgent=True,
        device="auto",
    )

    agent = model.agent(
        sandbox_url="http://localhost:8000",
        sandbox_api_key="your-key",
        max_steps=20,
    )

    result = agent.run(
        "I have two chest X-rays:\n"
        "  - Baseline: /data/xray_baseline.png (taken 2025-10-15)\n"
        "  - Follow-up: /data/xray_followup.png (taken 2026-05-01)\n\n"
        "Do the following:\n"
        "1. Analyze both images for pathological findings\n"
        "2. Compare findings between baseline and follow-up\n"
        "3. Write Python code to quantify changes (opacity area, distribution)\n"
        "4. Generate a structured comparison report in JSON format\n"
        "5. Save the report to /data/comparison_report.json"
    )

    print(result["answer"])
    agent.cleanup()


# ═══════════════════════════════════════════════════════════════════
#  3. Stateful Session — Variables Persist Between Executions
# ═══════════════════════════════════════════════════════════════════
def workflow_stateful():
    model = thinklab.load_llm(
        model_name="google/medgemma-4b-it",
        token="hf_xxx",
        IsAgent=True,
        device="auto",
    )

    agent = model.agent(
        sandbox_url="http://localhost:8000",
        sandbox_api_key="your-key",
        max_steps=15,
    )

    result = agent.run(
        "Using a PERSISTENT session (execute_in_session), do the following:\n"
        "1. Load numpy and create a 512x512 random matrix\n"
        "2. Apply a Gaussian filter to it\n"
        "3. Compute the mean and std of the filtered result\n"
        "4. Plot it with matplotlib and save to /data/filtered.png\n"
        "Build each step incrementally — use the session so variables carry over."
    )

    print(result["answer"])
    agent.cleanup()


# ═══════════════════════════════════════════════════════════════════
#  4. Standalone Sandbox — Direct API Usage (No Agent)
# ═══════════════════════════════════════════════════════════════════
def workflow_sandbox_direct():
    sb = SandboxClient("http://localhost:8000", "your-key")

    print(sb.health())
    print(sb.ready())

    out = sb.execute("import sys; print(sys.version)")
    print(f"Python: {out['stdout']}")

    batch = sb.execute_batch([
        {"code": "x = [1, 2, 3, 4, 5]", "language": "python"},
        {"code": "print(sum(x))", "language": "python"},
    ])
    print(f"Batch results: {batch}")

    sess = sb.create_session("python")
    sid = sess["session_id"]
    sb.exec_in_session(sid, "import numpy as np")
    sb.exec_in_session(sid, "arr = np.random.randn(100)")
    result = sb.exec_in_session(sid, "print(f'Mean: {arr.mean():.4f}, Std: {arr.std():.4f}')")
    print(result["stdout"])
    sb.close_session(sid)


# ═══════════════════════════════════════════════════════════════════
#  5. MCP-Connected Agent
# ═══════════════════════════════════════════════════════════════════
def workflow_mcp():
    model = thinklab.load_llm(
        model_name="google/medgemma-4b-it",
        token="hf_xxx",
        IsAgent=True,
        device="auto",
    )

    agent = model.agent(
        sandbox_url="http://localhost:8000",
        sandbox_api_key="your-key",
        mcp_endpoints=[
            "http://localhost:3000/mcp",      # DICOM server
            "http://localhost:3001/mcp",      # FHIR integration
        ],
        max_steps=20,
    )

    result = agent.run(
        "Fetch the DICOM study XR20260503-001 from the PACS server, "
        "extract the chest X-ray image, analyze it for abnormalities, "
        "and create a FHIR DiagnosticReport resource with the findings."
    )

    print(result["answer"])
    agent.cleanup()


# ═══════════════════════════════════════════════════════════════════
#  6. Explainability — Agent uses explain_image for per-class heatmaps
# ═══════════════════════════════════════════════════════════════════
def workflow_explainability():
    """
    Agent calls explain_image tool which runs:
        model.inference(image, prompt,
            explainability={"enabled": True, "mode": "both", "per_class": True})

    Grad-CAM is class-specific → separate heatmap per class/token.
    LIME is class-specific → separate superpixel importance per class.
    Agent can then segment and analyze each class's attention independently.
    """
    model = thinklab.load_llm(
        model_name="google/medgemma-4b-it",
        token="hf_xxx",
        IsAgent=True,
        device="auto",
    )

    agent = model.agent(
        sandbox_url="http://localhost:8000",
        sandbox_api_key="your-key",
        max_steps=15,
    )

    result = agent.run(
        "Analyze /data/xray.png using the explain_image tool to get "
        "per-class Grad-CAM and LIME heatmaps. Then:\n"
        "1. Report which anatomical regions the model focused on for each class\n"
        "2. Write Python code to segment the Grad-CAM heatmaps by class\n"
        "3. Compute overlap between Grad-CAM and LIME attention for each class\n"
        "4. Output a per-class trust score (0-1) for the model's predictions"
    )

    print(result["answer"])
    agent.cleanup()


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("ThinkLab Agent Workflows")
    print("=" * 40)
    print("1. workflow_basic()")
    print("2. workflow_research()")
    print("3. workflow_stateful()")
    print("4. workflow_sandbox_direct()")
    print("5. workflow_mcp()")
    print("6. workflow_explainability()")
