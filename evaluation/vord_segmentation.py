"""
VORD (Visual Ordinal Reasoning Decoding) applied to segmentation masks.

Instead of filtering text tokens, VORD is applied at the pixel level on
SAM2 soft mask scores. Pixels whose segmentation is not visually grounded
(i.e., not robust to image perturbation) are filtered out.

Algorithm:
    1. Run full pipeline on original image v -> soft mask S_orig
    2. Run full pipeline on noisy image v_hat -> soft mask S_noisy
    3. Compute similarity margin m from VLM vision encoder
    4. Per pixel p: delta(p) = S_orig(p) + m - S_noisy(p)
    5. Soft blending via sigmoid(delta / temperature)
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from PIL import Image as PILImage
from qwen_vl_utils import process_vision_info


def add_diffusion_noise(image_tensor, noise_step):
    """Add diffusion noise to an image tensor at a given timestep."""
    num_steps = 1000

    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t * x_0 + alphas_1_m_t * noise)

    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image, noise_step)
    return image_tensor_cd


def add_noise_to_image(image, noise_step=50):
    """Add diffusion noise to a PIL image at a given diffusion timestep."""
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    noisy_tensor = add_diffusion_noise(img_tensor, noise_step)
    noisy_tensor = noisy_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1)  # [H, W, C]
    noisy_array = (noisy_tensor.numpy() * 255).astype(np.uint8)
    return PILImage.fromarray(noisy_array)


def compute_similarity_margin(model, processor, original_image, noisy_image):
    """
    Compute VORD similarity margin: m = (1/pi) * arccos(cos_sim(f(v), f(v_hat)))
    
    Uses the VLM's vision encoder to extract visual features from original
    and noisy images, then computes a normalized angular distance.
    """
    msg_orig = [{"role": "user", "content": [
        {"type": "image", "image": original_image},
        {"type": "text", "text": "describe"}
    ]}]
    msg_noisy = [{"role": "user", "content": [
        {"type": "image", "image": noisy_image},
        {"type": "text", "text": "describe"}
    ]}]

    proc = model.processor
    
    text_orig = proc.apply_chat_template(msg_orig, tokenize=False, add_generation_prompt=True)
    text_noisy = proc.apply_chat_template(msg_noisy, tokenize=False, add_generation_prompt=True)

    img_orig, _ = process_vision_info(msg_orig)
    img_noisy, _ = process_vision_info(msg_noisy)

    inputs_orig = proc(text=[text_orig], images=img_orig, return_tensors="pt").to("cuda")
    inputs_noisy = proc(text=[text_noisy], images=img_noisy, return_tensors="pt").to("cuda")

    with torch.no_grad():
        vlm = model.reasoning_model
        visual_orig = vlm.visual(inputs_orig.pixel_values, grid_thw=inputs_orig.image_grid_thw)
        visual_noisy = vlm.visual(inputs_noisy.pixel_values, grid_thw=inputs_noisy.image_grid_thw)

    f_v = visual_orig.mean(dim=0 if visual_orig.dim() == 2 else 1).flatten()
    f_v_hat = visual_noisy.mean(dim=0 if visual_noisy.dim() == 2 else 1).flatten()

    cos_sim = F.cosine_similarity(f_v.unsqueeze(0), f_v_hat.unsqueeze(0)).item()
    cos_sim = max(min(cos_sim, 1.0), -1.0)
    margin = (1.0 / math.pi) * math.acos(cos_sim)

    return margin


def generate_soft_masks(sam_model, image, bboxes, points=None):
    """
    Generate soft (probability) segmentation masks using SAM2's logit outputs.
    
    Returns the sigmoid of SAM2's mask logits, giving per-pixel confidence
    scores in [0, 1].
    """
    img_height, img_width = image.height, image.width
    soft_mask_all = np.zeros((img_height, img_width), dtype=np.float32)
    
    if not bboxes:
        return soft_mask_all
    
    if points and len(points) != len(bboxes):
        return soft_mask_all
    
    try:
        sam_model.set_image(image)
        
        if points:
            for bbox, point in zip(bboxes, points):
                masks, scores, logits = sam_model.predict(
                    point_coords=[point],
                    point_labels=[1],
                    box=bbox
                )
                best_idx = np.argmax(scores)
                mask_logit = logits[best_idx]  # [256, 256]
                
                mask_logit_tensor = torch.from_numpy(mask_logit).unsqueeze(0).unsqueeze(0).float()
                mask_logit_upscaled = F.interpolate(
                    mask_logit_tensor,
                    size=(img_height, img_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                soft_mask = torch.sigmoid(mask_logit_upscaled).numpy()
                
                soft_mask_all = np.maximum(soft_mask_all, soft_mask)
        else:
            for bbox in bboxes:
                masks, scores, logits = sam_model.predict(box=bbox)
                best_idx = np.argmax(scores)
                mask_logit = logits[best_idx]
                
                mask_logit_tensor = torch.from_numpy(mask_logit).unsqueeze(0).unsqueeze(0).float()
                mask_logit_upscaled = F.interpolate(
                    mask_logit_tensor,
                    size=(img_height, img_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                soft_mask = torch.sigmoid(mask_logit_upscaled).numpy()
                
                soft_mask_all = np.maximum(soft_mask_all, soft_mask)
        
        return soft_mask_all
        
    except Exception as e:
        print(f"Error generating soft masks: {e}")
        return soft_mask_all


def apply_vord_mask_filtering(soft_mask_orig, soft_mask_noisy, margin, threshold=0.5):
    """
    Apply VORD filtering at the pixel level on soft segmentation masks.
    
    For each pixel p:
        delta(p) = S_orig(p) + margin - S_noisy(p)
        If delta(p) < 0, set pixel to 0 (not visually grounded)
    """
    # VORD: delta = S_orig + margin - S_noisy
    delta = soft_mask_orig + margin - soft_mask_noisy
    
    # Zero out pixels where delta < 0 (not visually grounded)
    filtered_soft_mask = soft_mask_orig.copy()
    filtered_soft_mask[delta < 0] = 0.0
    
    # Binarize
    binary_mask = filtered_soft_mask > threshold
    
    return binary_mask


def run_vord_segmentation(model, image, query, noise_step=50, threshold=0.5):
    """
    VORD segmentation pipeline (single-pass, aligned approach).
    
    1. Run VLM on original image -> get bboxes/points (once)
    2. Create noisy image via diffusion noise
    3. Run SAM2 on both images with SAME bboxes -> soft masks
    4. Compute similarity margin
    5. Apply soft VORD blending
    """
    # Run VLM on original image to get bboxes/points (only once)
    result_orig = model.segment_objects(image, query)
    bboxes_orig = result_orig["bboxes"]
    points_orig = result_orig["points"]
    thinking = result_orig["thinking"]
    full_response = result_orig["full_response"]
    pred_answer = result_orig["pred_answer"]
    
    if not bboxes_orig:
        return result_orig
    
    # Create noisy image via diffusion noise
    noisy_image = add_noise_to_image(image, noise_step=noise_step)
    
    # Get soft masks from SAM2 using the SAME bboxes/points
    sam_model = model.segmentation_model
    soft_mask_orig = generate_soft_masks(sam_model, image, bboxes_orig, points_orig)
    soft_mask_noisy = generate_soft_masks(sam_model, noisy_image, bboxes_orig, points_orig)
    
    # Compute similarity margin
    margin = compute_similarity_margin(model, None, image, noisy_image)
    
    # Apply soft VORD blending
    vord_mask = apply_vord_mask_filtering(
        soft_mask_orig, soft_mask_noisy, margin, threshold=threshold
    )
    
    return {
        "masks": vord_mask,
        "bboxes": bboxes_orig,
        "points": points_orig,
        "thinking": thinking,
        "full_response": full_response,
        "pred_answer": pred_answer,
        "vord_margin": margin
    }
