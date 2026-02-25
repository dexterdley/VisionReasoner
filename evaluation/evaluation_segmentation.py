import argparse
import random
import torch
import json
import numpy as np
import os
from datasets import load_from_disk, load_dataset
from PIL import Image as PILImage
from tqdm import tqdm
import sys
from visualization import visualize_result
from vord_segmentation import run_vord_segmentation
from vcd_sample import evolve_guidance_sampling

# Add the parent directory to the Python path to import model module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision_reasoner.models.vision_reasoner_model import VisionReasonerModel
from vision_reasoner.models.qwen_vl import QwenVLModel
from vision_reasoner.models.qwen_vl_cot import QwenVLCoTModel
from vision_reasoner.models.visurf_model import ViSurfModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vision_reasoner")
    parser.add_argument("--model_path", type=str, default="pretrained_models/VisionReasoner-7B")
    parser.add_argument("--task_router_model_path", type=str, default="pretrained_models/TaskRouter-1.5B")
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)

    # for parallel evaluation
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--num_parts", type=int, required=True)

    # VORD options
    parser.add_argument("--use_vord", action="store_true", help="Enable VORD mask filtering")
    parser.add_argument("--vord_noise_step", type=int, default=50, help="VORD diffusion noise timestep (0-999)")
    parser.add_argument("--vord_threshold", type=float, default=0.5, help="VORD mask binarization threshold")

    # VGD options (token-level visual guidance decoding)
    parser.add_argument("--visual_alpha", type=float, default=0.0, help="Guidance alpha (0=disabled)")
    parser.add_argument("--mode", type=str, default="VORD", help="Contrastive decoding mode, example VORD or VCD")
    parser.add_argument("--noise_step", type=int, default=500, help="VORD diffusion noise timestep (0-999)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0, 0
    return intersection, union

def compute_bbox_iou(bbox1, bbox2):
    # Calculate the intersection area of two bboxes
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate areas of the two bboxes
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # Calculate union area
    union = area1 + area2 - intersection
    
    # Avoid division by zero
    if union == 0:
        return 0
    
    return intersection / union

def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] All random states set to {seed}")

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Install appropriate sampling strategy (alpha=0 → regular _sample, alpha>0 → VGD)
    # evolve_guidance_sampling(visual_alpha=args.visual_alpha, mode=args.mode)
    
    # Initialize model
    if args.model == "qwen2vl":
        model = QwenVLModel(model_path=args.model_path)
    elif args.model == "qwen25vl":
        model = QwenVLModel(model_path=args.model_path)
    elif args.model == "qwen25vl_cot":
        model = QwenVLCoTModel(model_path=args.model_path)
    elif args.model == "vision_reasoner":
        model = VisionReasonerModel(reasoning_model_path=args.model_path, 
                                    task_router_model_path=args.task_router_model_path, 
                                    segmentation_model_path=args.segmentation_model_path,
                                    visual_alpha=args.visual_alpha, mode=args.mode,
                                    noise_step=args.noise_step)
    elif args.model == "visurf":
        model = ViSurfModel(reasoning_model_path=args.model_path, 
                                    task_router_model_path=args.task_router_model_path, 
                                    segmentation_model_path=args.segmentation_model_path)
    
    # Load dataset
    dataset = load_dataset(args.test_data_path, split="test")
    total_len = len(dataset)
    part_size = total_len // args.num_parts
    start_idx = args.idx * part_size
    end_idx = start_idx + part_size if args.idx < args.num_parts - 1 else total_len
    
    dataset = dataset.select(range(start_idx, end_idx))
    
    # Check if dataset has bbox information
    has_bbox = 'bbox' in dataset[0]
    
    all_outputs = []
    
    # Prepare batches
    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Processing batches"):
        batch_data = [dataset[j] for j in range(i, min(i + args.batch_size, len(dataset)))]
        
        batch_images = [item["image"].convert("RGB") for item in batch_data]
        batch_questions = [item["text"].lower().strip(".\"?!") for item in batch_data]
        id_list = [{
            "image_id": item["image_id"],
            "ann_id": item["ann_id"],
            "mask": item["mask"],
            "img_height": item["img_height"],
            "img_width": item["img_width"],
            "bbox": item["bbox"] if has_bbox else None
        } for item in batch_data]
        
        process_batch(model, batch_images, batch_questions, id_list, all_outputs, has_bbox,
                      use_vord=args.use_vord, vord_noise_step=args.vord_noise_step, 
                      vord_threshold=args.vord_threshold)
    
    # Save results
    output_file = os.path.join(args.output_path, f"output_{args.idx}.json")
    with open(output_file, "w") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)

def merge_bboxes(bboxes):
    """合并所有bboxes，返回最左上和最右下的坐标"""
    if not bboxes:
        return None
    
    # 初始化最左上和最右下坐标
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        min_x = min(min_x, x1)
        min_y = min(min_y, y1)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)
    
    return [min_x, min_y, max_x, max_y]

def get_bbox(mask):
    # Convert mask to numpy array if not already
    mask = np.array(mask)
    
    # Find non-zero points
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # If mask is all zeros, return empty list
    if not np.any(mask):
        return []
    
    # Get boundaries
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def process_batch(model, batch_images, batch_questions, id_list, all_outputs, has_bbox,
                  use_vord=False, vord_noise_step=50, vord_threshold=0.5):
    """Process a batch of images and questions"""
    if use_vord:
        # VORD: process one image at a time with perturbation-based mask filtering
        batch_results = []
        for image, question in zip(batch_images, batch_questions):
            result = run_vord_segmentation(
                model, image, question,
                noise_step=vord_noise_step, threshold=vord_threshold
            )
            batch_results.append(result)
    else:
        batch_results = model.segment_objects_batch(batch_images, batch_questions)
    
    for i, result in enumerate(batch_results):
        try:
            thinking = result["thinking"]
            bboxes = result["bboxes"]
            mask_all = result["masks"]
            gt_mask = np.array(id_list[i]["mask"])
            
            # visualize_result(batch_images[i], mask_all, gt_mask, thinking, batch_questions[i])
            
            intersection, union = compute_iou(mask_all, gt_mask)
            
            bbox_iou = 0.0
            
            try:     
                if has_bbox:
                    gt_bbox = id_list[i]["bbox"]
                else:
                    gt_bbox = get_bbox(gt_mask)
                
                if gt_bbox == []:
                    if len(bboxes) == 0:
                        bbox_iou = 1.0
                    else:
                        bbox_iou = 0.0
                else:
                    # adapt to multi-object detection
                    for pred_bbox in bboxes:
                        if compute_bbox_iou(pred_bbox, gt_bbox) > 0.5:
                            bbox_iou = 1.0
                            break
                    merged_bbox = merge_bboxes(bboxes)
                    if merged_bbox:
                        if compute_bbox_iou(merged_bbox, gt_bbox) > 0.5:
                            bbox_iou = 1.0
                    
            except Exception as e:
                print(f"Bbox error: {e}, Image ID: {id_list[i]['image_id']}, Ann ID: {id_list[i]['ann_id']}")
                bbox_iou = 0.0
            
            all_outputs.append({
                "image_id": id_list[i]["image_id"],
                "ann_id": id_list[i]["ann_id"],
                "think": thinking,
                "intersection": int(intersection),
                "union": int(union),
                "bbox_iou": bbox_iou,
                "non_object": int(gt_mask.sum()==0 and len(bboxes)==0),   # for grefcoco where there is no object in the image
                "non_object_GT": int(gt_mask.sum()==0)
            })
            
        except Exception as e:
            print(f"Error processing result: {e}")
            # Add penalty in this situation
            all_outputs.append({
                "image_id": id_list[i]["image_id"],
                "ann_id": id_list[i]["ann_id"],
                "think": "",
                "intersection": 0,
                "union": np.array(id_list[i]["mask"]).sum(),
                "bbox_iou": 0.0,
                "non_object": 0.0,   # for grefcoco where there is no object in the image
                "non_object_GT": int(np.array(id_list[i]["mask"]).sum()==0)
            })
        

if __name__ == "__main__":
    main()