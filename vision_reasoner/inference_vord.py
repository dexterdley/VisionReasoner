"""
Simple VORD inference script for quick output inspection.
Runs both VORD and regular decoding on the same image and compares outputs.

Usage:
    CUDA_VISIBLE_DEVICES=0 python vision_reasoner/inference_vord.py --image_path <image> --query <text>
"""
import argparse
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from vision_reasoner.models.vision_reasoner_model import VisionReasonerModel
from vision_reasoner.utils import visualize_results_enhanced


def main():
    parser = argparse.ArgumentParser(description="VORD inference comparison")
    parser.add_argument("--model_path", type=str, default="pretrained_models/VisionReasoner-7B")
    parser.add_argument("--task_router_model_path", type=str, default="pretrained_models/TaskRouter-1.5B")
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--image_path", type=str, default="assets/airplanes.png", help="Path to the input image")
    parser.add_argument("--query", type=str, default="How many airplanes are there in this image?", help="Query/instruction for the model")
    parser.add_argument("--visual_alpha", type=float, default=1.5)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="./results/result_visualization.png", help="Path to save the output visualization")
    args = parser.parse_args()

    # Seed
    import random, numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    image = Image.open(args.image_path).convert("RGB")
    print(f"Image: {args.image_path} ({image.size[0]}x{image.size[1]})")
    print(f"Query: {args.query}")
    print(f"Seed:  {args.seed}")
    print("=" * 60)

    # --- Regular decoding ---
    print("\n[1/2] Regular decoding (visual_alpha=0)...")
    model = VisionReasonerModel(
        reasoning_model_path=args.model_path,
        task_router_model_path=args.task_router_model_path,
        segmentation_model_path=args.segmentation_model_path,
        visual_alpha=0.0, mode="REGULAR",
    )
    # Re-seed so both runs have the same random state for sampling
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    result_regular = model.segment_objects(image, args.query)

    print(f"  Thinking: {result_regular['thinking']}")
    print(f"  Bboxes:   {result_regular['bboxes']}")
    print(f"  Mask sum: {result_regular['masks'].sum()}")

    # --- VORD decoding ---
    print(f"\n[2/2] VORD decoding (alpha={args.visual_alpha}, noise_step={args.noise_step})...")
    model.visual_alpha = args.visual_alpha
    model.mode = "VORD"
    model.noise_step = args.noise_step
    model.reasoning_model.noise_step = args.noise_step
    from evaluation.vcd_sample import evolve_guidance_sampling
    evolve_guidance_sampling(visual_alpha=args.visual_alpha, mode="VORD")

    # Re-seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    result_vord = model.segment_objects(image, args.query)

    print(f"  Thinking: {result_vord['thinking']}")
    print(f"  Bboxes:   {result_vord['bboxes']}")
    print(f"  Mask sum: {result_vord['masks'].sum()}")

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("COMPARISON")
    print(f"  Regular bboxes: {len(result_regular['bboxes'])}  |  VORD bboxes: {len(result_vord['bboxes'])}")
    print(f"  Regular mask px: {result_regular['masks'].sum()}  |  VORD mask px: {result_vord['masks'].sum()}")

    if result_regular['thinking'] == result_vord['thinking']:
        print("  ⚠ Thinking outputs are IDENTICAL (VORD had no effect on text)")
    else:
        print("  ✓ Thinking outputs DIFFER (VORD changed the decoding)")

    # --- Visualize ---
    base, ext = os.path.splitext(args.output_path)
    out_regular = f"{base}_regular{ext}"
    out_vord = f"{base}_vord{ext}"
    visualize_results_enhanced(image, result_regular, "segmentation", out_regular)
    print(f"\nRegular visualization saved: {out_regular}")
    visualize_results_enhanced(image, result_vord, "segmentation", out_vord)
    print(f"VORD visualization saved:    {out_vord}")


if __name__ == "__main__":
    main()
