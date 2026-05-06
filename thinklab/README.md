# ThinkLab

**Pure PyTorch multimodal AI research framework** with runtime explainability, production-grade inference, and agentic code execution.

> No `transformers`. No `huggingface_hub`. Just PyTorch + raw weights.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ✨ Features

| Feature | Description |
|---|---|
| **Unified Factory API** | `thinklab.load_llm()` — one call to load any registered model |
| **Pure PyTorch** | Custom SigLIP, Gemma 1/3 decoders, SafeTensors loader |
| **Runtime Explainability** | Grad-CAM + LIME controlled per `.inference()` call, not at load time |
| **Per-Class Heatmaps** | Grad-CAM produces separate visualization per class/token |
| **Clinical Context** | Patient demographics, vitals, labs — auto-skipped for non-medical models |
| **Structured Output** | `InferenceResult` with request_id, latency, token speed, HIPAA metadata |
| **ReAct Agent** | LLM reasons → executes code in sandbox → observes → repeats |
| **MCP Support** | Discover and invoke tools from external MCP servers |
| **Memory-Aware** | Automatic layer offloading when GPU memory is tight |

---

## 📦 Installation

```bash
git clone https://github.com/abeldirectory252/ThinkAi.git
cd ThinkAi
pip install -e .
```

---

## 🚀 Quick Start

### Basic Inference

```python
import thinklab

model = thinklab.load_llm(
    model_name="google/paligemma-3b-pt-224-128",
    token="hf_xxxxx",
    device="auto",
)

result = model.inference(
    prompt="describe this image",
    image_path="photo.jpg",
)
print(result.model_output)
print(result.to_json())
```

### Inference + Explainability

Explainability is a **runtime option** on `.inference()` — NOT on `load_llm()`:

```python
model = thinklab.load_llm(
    model_name="google/medgemma-4b-it",
    token="hf_xxxxx",
    device="auto",
)

result = model.inference(
    prompt="Analyze this chest X-ray for pneumonia",
    image_path="xray.jpg",
    explainability={
        "enabled": True,
        "mode": "grad_cam",     # "grad_cam", "lime", or "both"
        "per_class": True,      # separate heatmap per class
    },
)

print(result.model_output)

# result.explain contains ACTUAL IMAGES
for i, overlay in enumerate(result.explain.grad_cam_overlays):
    label = result.explain.grad_cam_labels[i]
    print(f"Class '{label}' → heatmap shape {overlay.shape}")
    # Image.fromarray(overlay).save(f"heatmap_{label}.png")
```

### Grad-CAM: Per-Class Visualization

Grad-CAM is **class-specific** — it produces a separate heatmap for every class/token the model generates:

```python
result = model.inference(
    prompt="identify all pathologies",
    image_path="xray.jpg",
    explainability={"enabled": True, "mode": "grad_cam", "per_class": True},
)

# Each class gets its own heatmap
# e.g. "pneumonia" → focuses on right lower lobe
#      "effusion"  → focuses on costophrenic angles
for hm, label in zip(result.explain.grad_cam_heatmaps,
                      result.explain.grad_cam_labels):
    print(f"  {label}: max_activation={hm.max():.3f}")
```

### LIME: Region Importance

```python
result = model.inference(
    prompt="What abnormalities do you see?",
    image_path="xray.jpg",
    explainability={"enabled": True, "mode": "lime", "lime_samples": 200},
)

# Green = supports prediction, Red = against prediction
positive = result.explain.lime_positive_overlay   # numpy H×W×3
negative = result.explain.lime_negative_overlay   # numpy H×W×3

# Which superpixels matter most
top = sorted(result.explain.lime_weights.items(), key=lambda x: -x[1])[:5]
print(f"Top 5 important regions: {top}")
```

### Both + Correlation

```python
result = model.inference(
    prompt="Describe all findings",
    image_path="xray.jpg",
    explainability={"enabled": True, "mode": "both"},
)

print(f"Grad-CAM ↔ LIME agreement: {result.explain.mean_overlap:.2f}")
```

---

## 🏥 Clinical Context (Medical Models Only)

Clinical context is auto-injected for medical models and **auto-skipped** for non-medical models:

```python
result = model.inference(
    prompt="Analyze this chest X-ray for pneumonia",
    image_path="xray.jpg",
    payload={
        "clinical_context": {
            "patient_demographics": {"age_years": 67, "sex": "M"},
            "symptoms": [
                {"symptom": "fever", "duration_days": 3, "severity": "moderate"},
            ],
            "vital_signs": {"temperature_celsius": 38.5, "oxygen_saturation": 94},
            "reason_for_exam": "Suspected pneumonia",
        },
        "inference_config": {
            "pathologies_to_check": ["pneumonia", "pleural_effusion"],
            "generate_differential": True,
        },
    },
    explainability={"enabled": True, "mode": "both"},
)

print(result.clinical_context_used)  # True for MedGemma, False for PaliGemma
```

<details>
<summary>📄 Sample JSON Output</summary>

```json
{
  "request_id": "medgemma4bit_req_20260503_103000_7f3a9b2c",
  "model": "google/medgemma-4b-it",
  "version": "gemma3",
  "inference_mode": "explain",
  "timestamp": "2026-05-03T10:30:00.123456+00:00",
  "processing_latency_ms": 287.3,
  "token_speed": "40tok/sec",
  "tokens_generated": 156,
  "hipaa_compliant": true,
  "model_output": "Findings: Focal consolidation in the right lower lobe...",
  "clinical_context_used": true,
  "explain": {
    "mode": "both",
    "total_heatmaps": 156,
    "mean_overlap": 0.73,
    "grad_cam": {
      "count": 156,
      "labels": ["pneumonia", "consolidation", "..."],
      "heatmap_shape": [224, 224]
    },
    "lime": {
      "mask_shape": [224, 224],
      "n_segments": 50,
      "top_positive_segments": [[12, 0.85], [7, 0.72]]
    }
  },
  "disclaimer": "AI-generated analysis. Must be reviewed by a qualified healthcare professional."
}
```

</details>

---

## 🤖 Agent Mode

The ReAct agent reasons → executes code in sandbox → observes → repeats. The agent's `explain_image` tool internally calls `.inference(explainability={"enabled": True, ...})` to get per-class heatmaps.

```python
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
    "Analyze /data/xray.png with Grad-CAM per-class heatmaps. "
    "Segment each class's attention and compute trust scores."
)
print(result["answer"])
agent.cleanup()
```

### Agent Tools

| Tool | Description |
|---|---|
| `execute_code` | Run Python in sandbox (stateless) |
| `execute_in_session` | Run in persistent session (stateful) |
| `upload_file` / `list_files` / `read_file` | Sandbox file management |
| `create_workspace` / `snapshot_workspace` | Isolated workspaces |
| `analyze_image` | Run model inference on an image |
| `explain_image` | Inference + per-class Grad-CAM + LIME heatmaps |
| `finish` | Signal task completion |

---

## 🏗️ Architecture

```
thinklab/
├── __init__.py              ← exports: load_llm, list_models, register_model
├── registry.py              ← model registry + ThinkLabModel wrapper
├── schema.py                ← ExplainConfig, ExplainResult, InferenceResult
├── model_builders.py        ← builder functions (paligemma, medgemma)
├── inference.py             ← InferenceEngine (clinical prompt, heatmap overlay)
├── trainer.py               ← Trainer (freeze vision, gradient accumulation)
│
├── core/
│   └── base_model.py        ← smart_device(), offload_layers(), memory utils
├── weights/
│   └── huggingface.py       ← raw HTTP downloader + SafeTensors loader
├── models/
│   ├── multimodal/           ← SigLIP, Gemma 1/3, PaliGemma, tokenizer
│   └── ModelExplain/         ← grad_cam.py, lime_explainer.py, correlator.py
└── agent/
    ├── agent.py             ← ReAct loop
    ├── sandbox_client.py    ← Sandbox HTTP client
    ├── tools.py             ← Tool definitions
    └── mcp_client.py        ← MCP tool discovery
```

---

## ⚙️ `load_llm()` Parameters

> **Note:** Explainability is NOT configured here — pass it at `.inference()` time.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | str | required | HuggingFace repo ID |
| `token` | str | None | HF token (required for gated models) |
| `device` | str | "auto" | "auto", "cuda", "cpu", "mps" |
| `IsAgent` | bool | False | Enable `.agent()` method |
| `temperature` | float | 0.7 | Sampling temperature |
| `safety_filter` | str | "strict" | "strict", "moderate", "none" |
| `radiology_specialization` | bool | False | Optimize for radiology |
| `anatomy_prior` | str | "general" | "chest_xray", "brain_mri", etc. |

## `.inference()` Parameters

| Parameter | Type | Description |
|---|---|---|
| `prompt` | str | Text prompt |
| `image_path` | str | Path to image file |
| `payload` | dict | Clinical context, metadata |
| `explainability` | dict | `{"enabled": True, "mode": "grad_cam"/"lime"/"both"}` |

## `result.explain` Fields

| Field | Type | Description |
|---|---|---|
| `grad_cam_heatmaps` | `List[numpy H×W]` | Raw per-class heatmaps |
| `grad_cam_overlays` | `List[numpy H×W×3]` | Heatmap overlaid on image |
| `grad_cam_labels` | `List[str]` | Class/token label per heatmap |
| `lime_positive_overlay` | `numpy H×W×3` | Green = supports prediction |
| `lime_negative_overlay` | `numpy H×W×3` | Red = against prediction |
| `lime_weights` | `Dict[int, float]` | Superpixel → importance weight |
| `mean_overlap` | `float` | Grad-CAM ↔ LIME correlation |

---

## 📄 License

MIT — see [LICENSE](LICENSE)
