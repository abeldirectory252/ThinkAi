#!/usr/bin/env python3
"""
ThinkLab — PaliGemma 3B inference with Grad-CAM + LIME correlation.

Usage:
    python run_inference.py \
        --image photo.jpg \
        --prompt "describe this image" \
        --token YOUR_HF_TOKEN \
        --save-dir ./weights/paligemma-3b \
        --output explanation.png
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from thinklab.models.multimodal import PaliGemma, GemmaTokenizer, ImageProcessor
from thinklab.models.ModelExplain import GradCAM, LIMEExplainer, TextVisionCorrelator

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("thinklab")


def main():
    parser = argparse.ArgumentParser(description="PaliGemma 3B + Explainability")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", default="describe this image",
                        help="Text prompt")
    parser.add_argument("--token", default=None,
                        help="HuggingFace access token (for gated models)")
    parser.add_argument("--repo", default="google/paligemma-3b-pt-224-128",
                        help="HF repo ID")
    parser.add_argument("--save-dir", default="./weights/paligemma-3b",
                        help="Directory for downloaded weights")
    parser.add_argument("--output", default="explanation.png",
                        help="Output visualization path")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--lime-samples", type=int, default=128,
                        help="LIME perturbation samples (lower = faster)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # ── Step 1: Load model ─────────────────────────────────────────
    logger.info("Loading PaliGemma from %s ...", args.repo)
    model = PaliGemma.from_pretrained(
        repo_id=args.repo,
        save_dir=args.save_dir,
        token=args.token,
        dtype=dtype,
        device=args.device,
    )
    dev = next(model.parameters()).device
    logger.info("Model on device: %s | Params: %.1f M",
                dev, sum(p.numel() for p in model.parameters()) / 1e6)

    # ── Step 2: Load tokenizer + image ─────────────────────────────
    tok_path = Path(args.save_dir) / "tokenizer.model"
    tokenizer = GemmaTokenizer(tok_path)
    processor = ImageProcessor(image_size=224)

    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image.resize((224, 224)))
    pixel_values = processor(image, dtype=dtype).to(dev)

    input_ids_list = tokenizer.build_paligemma_input(args.prompt)
    input_ids = torch.tensor([input_ids_list], device=dev)

    # ── Step 3: Generate text ──────────────────────────────────────
    logger.info("Generating (max %d tokens) ...", args.max_tokens)
    gen_out = model.generate(
        pixel_values, input_ids,
        max_new_tokens=args.max_tokens,
        temperature=0.7,
    )
    generated_text = tokenizer.decode(gen_out["generated_ids"])
    logger.info("Generated: %s", generated_text)

    # ── Step 4: Grad-CAM (per token) ───────────────────────────────
    logger.info("Computing Grad-CAM per token ...")
    grad_cam = GradCAM(model, target_layer_idx=-1)
    gc_heatmaps = grad_cam.compute_per_token(
        pixel_values, input_ids,
        generated_ids=gen_out["generated_ids"],
        image_size=224, patch_size=14,
    )
    grad_cam.remove_hooks()

    # ── Step 5: LIME ───────────────────────────────────────────────
    logger.info("Running LIME (%d samples) ...", args.lime_samples)
    lime = LIMEExplainer(
        model, tokenizer, processor,
        num_samples=args.lime_samples,
        num_superpixels=50,
    )
    lime_result = lime.explain(image_np, args.prompt, max_tokens=args.max_tokens)
    logger.info("LIME original output: %s", lime_result["original_text"])

    # ── Step 6: Correlate ──────────────────────────────────────────
    logger.info("Correlating text ↔ vision ...")
    correlator = TextVisionCorrelator(tokenizer)
    correlation = correlator.correlate(
        generated_ids=gen_out["generated_ids"],
        grad_cam_heatmaps=gc_heatmaps,
        lime_result=lime_result,
        image_np=image_np,
    )

    # Print per-token correlation
    print("\n" + "=" * 60)
    print("TOKEN ↔ VISION CORRELATION")
    print("=" * 60)
    for pt in correlation["per_token"]:
        print(f"  Token: {pt['token']:>15s}  |  "
              f"LIME overlap: {pt['overlap_with_lime']:+.3f}  |  "
              f"Top regions: {pt['top_superpixels']}")
    print(f"\n  Mean Grad-CAM ↔ LIME overlap: "
          f"{correlation['mean_overlap']:.3f}")
    print("=" * 60)

    # ── Step 7: Visualize ──────────────────────────────────────────
    correlator.visualize(
        image_np, correlation,
        save_path=args.output,
        top_k_tokens=6,
    )
    logger.info("Done. Output saved to %s", args.output)


if __name__ == "__main__":
    main()
