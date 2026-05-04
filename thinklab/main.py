"""
ThinkLab — Usage Examples

Explainability is a RUNTIME option on .inference(), NOT on load_llm().
result.explain contains actual heatmap images (numpy arrays).
"""
import thinklab


# ═══════════════════════════════════════════════════════════════════
#  1. Simple — No explainability
# ═══════════════════════════════════════════════════════════════════
def example_simple():
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
    print(result.explain)  # None — explainability not requested


# ═══════════════════════════════════════════════════════════════════
#  2. Grad-CAM — Per-class heatmap images
# ═══════════════════════════════════════════════════════════════════
def example_grad_cam():
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
            "mode": "grad_cam",     # only Grad-CAM
            "per_class": True,      # separate heatmap per token/class
            "overlay_alpha": 0.5,
            "colormap": "jet",
        },
    )

    print(result.model_output)

    # result.explain contains ACTUAL IMAGES
    if result.explain:
        # Per-class overlays (numpy H×W×3 RGB arrays)
        for i, overlay in enumerate(result.explain.grad_cam_overlays):
            label = result.explain.grad_cam_labels[i]
            print(f"  Heatmap {i}: '{label}' → shape {overlay.shape}")
            # Save: Image.fromarray(overlay).save(f"heatmap_{label}.png")

        # Raw heatmaps (H×W float, 0-1)
        for hm in result.explain.grad_cam_heatmaps:
            print(f"  Raw heatmap: {hm.shape}, max={hm.max():.3f}")


# ═══════════════════════════════════════════════════════════════════
#  3. LIME — Superpixel regions
# ═══════════════════════════════════════════════════════════════════
def example_lime():
    model = thinklab.load_llm(
        model_name="google/medgemma-4b-it",
        token="hf_xxxxx",
        device="auto",
    )

    result = model.inference(
        prompt="What abnormalities do you see?",
        image_path="xray.jpg",
        explainability={
            "enabled": True,
            "mode": "lime",
            "lime_samples": 200,
        },
    )

    if result.explain:
        # Green overlay = regions supporting prediction
        print(f"Positive overlay: {result.explain.lime_positive_overlay.shape}")
        # Red overlay = regions against prediction
        if result.explain.lime_negative_overlay is not None:
            print(f"Negative overlay: {result.explain.lime_negative_overlay.shape}")
        # Superpixel weights
        top = sorted(result.explain.lime_weights.items(), key=lambda x: -x[1])[:5]
        print(f"Top 5 segments: {top}")


# ═══════════════════════════════════════════════════════════════════
#  4. Both + Correlation
# ═══════════════════════════════════════════════════════════════════
def example_both():
    model = thinklab.load_llm(
        model_name="google/medgemma-4b-it",
        token="hf_xxxxx",
        device="auto",
    )

    result = model.inference(
        prompt="Describe all pathological findings",
        image_path="xray.jpg",
        explainability={"enabled": True, "mode": "both"},
    )

    print(result.model_output)
    print(f"Grad-CAM ↔ LIME correlation: {result.explain.mean_overlap:.2f}")
    print(f"Total heatmaps: {result.explain.total_heatmaps}")
    print(result.to_json())


# ═══════════════════════════════════════════════════════════════════
#  5. Production — Clinical Context + Explainability
# ═══════════════════════════════════════════════════════════════════
def example_production():
    model = thinklab.load_llm(
        model_name="google/medgemma-4b-it",
        token="hf_xxxxx",
        device="auto",
        temperature=0.1,
        safety_filter="strict",
        radiology_specialization=True,
        anatomy_prior="chest_xray",
        confidence_threshold=0.7,
    )

    result = model.inference(
        prompt="Analyze this chest X-ray for pneumonia",
        image_path="xray.jpg",
        payload={
            "clinical_context": {
                "patient_demographics": {"age_years": 67, "sex": "M"},
                "symptoms": [
                    {"symptom": "fever", "duration_days": 3, "severity": "moderate"},
                ],
                "vital_signs": {
                    "temperature_celsius": 38.5,
                    "oxygen_saturation": 94,
                },
                "reason_for_exam": "Suspected pneumonia",
            },
            "inference_config": {
                "pathologies_to_check": ["pneumonia", "pleural_effusion"],
                "generate_differential": True,
                "num_differentials": 3,
            },
            "metadata": {"department": "radiology", "study_id": "XR20260503-001"},
        },
        explainability={"enabled": True, "mode": "both"},
    )

    print(result.to_json())
    print(f"Clinical context used: {result.clinical_context_used}")  # True


# ═══════════════════════════════════════════════════════════════════
#  6. PaliGemma — Clinical context auto-skipped
# ═══════════════════════════════════════════════════════════════════
def example_paligemma():
    model = thinklab.load_llm(
        model_name="google/paligemma-3b-pt-224-128",
        token="hf_xxxxx",
        device="auto",
    )

    result = model.inference(
        prompt="describe this image",
        image_path="photo.jpg",
        explainability={"enabled": True, "mode": "grad_cam"},
    )

    print(f"clinical_context_used = {result.clinical_context_used}")  # False
    print(f"explain = {result.explain.to_dict() if result.explain else None}")


# ═══════════════════════════════════════════════════════════════════
#  7. Agent mode
# ═══════════════════════════════════════════════════════════════════
def example_agent():
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
        "Analyze /data/xray.png with Grad-CAM to show which lung regions "
        "the model focuses on. Compute the attention area percentage."
    )
    print(result["answer"])
    agent.cleanup()


if __name__ == "__main__":
    print("Registered models:", thinklab.list_models())
