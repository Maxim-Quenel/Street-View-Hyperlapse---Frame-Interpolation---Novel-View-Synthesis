import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from config_loader import get_depth_model_id


# ============================================================
# PARTIE DEPTH ANYTHING V2 (identique fichier inspiration, avec config)
# ============================================================

def get_depth_map(image_path, model_path=None):
    print(f"   Traitement profondeur pour : {os.path.basename(image_path)}")
    model_id = model_path or get_depth_model_id()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    
    with torch.no_grad():
        depth = model(**inputs).predicted_depth
    
    depth = torch.nn.functional.interpolate(depth.unsqueeze(1), size=img.size[::-1], mode="bicubic", align_corners=False)
    depth_np = depth.squeeze().cpu().numpy()
    
    # Normalisation propre
    depth_min, depth_max = depth_np.min(), depth_np.max()
    depth_normalized = (depth_np - depth_min) / (depth_max - depth_min + 1e-6)
    
    del model, processor, inputs, depth
    torch.cuda.empty_cache()
    
    return depth_normalized, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ============================================================
# PARTIE WARPING 3D (copie inspiration)
# ============================================================

def warp_image_3d(img_rgb, depth_norm, shift_z):
    h, w = img_rgb.shape[:2]
    fx, fy = w / 1.0, w / 1.0 
    cx, cy = w / 2.0, h / 2.0
    
    z_metric = 1.0 / (depth_norm + 0.1) 
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    x3d = (u - cx) * z_metric / fx
    y3d = (v - cy) * z_metric / fy
    z3d = z_metric
    
    z3d_new = z3d - shift_z
    valid_mask = z3d_new > 0.1
    
    u_new = (x3d[valid_mask] * fx / z3d_new[valid_mask]) + cx
    v_new = (y3d[valid_mask] * fy / z3d_new[valid_mask]) + cy
    
    u_new = np.clip(np.round(u_new).astype(int), 0, w - 1)
    v_new = np.clip(np.round(v_new).astype(int), 0, h - 1)
    
    warped_img = np.zeros((h, w, 3), dtype=np.uint8)
    warped_depth = np.full((h, w), 1000.0, dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    colors = img_rgb[valid_mask]
    depths = z3d_new[valid_mask]
    
    # Painter's Algorithm
    indices = np.argsort(depths)[::-1] 
    u_sorted = u_new[indices]
    v_sorted = v_new[indices]
    c_sorted = colors[indices]
    d_sorted = depths[indices]
    
    warped_img[v_sorted, u_sorted] = c_sorted
    warped_depth[v_sorted, u_sorted] = d_sorted
    mask[v_sorted, u_sorted] = 255
    
    return warped_img, warped_depth, mask


def consensus_merge(warp_a, depth_a, mask_a, warp_b, depth_b, mask_b, t):
    h, w = warp_a.shape[:2]
    valid_a = mask_a > 0
    valid_b = mask_b > 0
    final_img = np.zeros_like(warp_a)
    
    mask_only_a = valid_a & (~valid_b)
    final_img[mask_only_a] = warp_a[mask_only_a]
    
    mask_only_b = valid_b & (~valid_a)
    final_img[mask_only_b] = warp_b[mask_only_b]
    
    both = valid_a & valid_b
    if np.any(both):
        da = depth_a[both]
        db = depth_b[both]
        diff = np.abs(da - db)
        threshold = 0.5 
        alpha = t 
        weights_b = np.full(da.shape, alpha)
        
        mask_a_closer = (da < (db - threshold))
        weights_b[mask_a_closer] = 0.0
        mask_b_closer = (db < (da - threshold))
        weights_b[mask_b_closer] = 1.0
        
        wa = warp_a[both].astype(np.float32)
        wb = warp_b[both].astype(np.float32)
        w_b_exp = weights_b[:, None]
        
        blended = wa * (1.0 - w_b_exp) + wb * w_b_exp
        final_img[both] = blended.astype(np.uint8)
        
    mask_holes = (~(valid_a | valid_b)).astype(np.uint8) * 255
    if np.any(mask_holes):
        final_img = cv2.inpaint(final_img, mask_holes, 3, cv2.INPAINT_TELEA)
        
    return final_img


# ============================================================
# PIPELINE (copie inspiration, signature conservée pour l'app)
# ============================================================

def process_interpolation(
    img_a_path,
    img_b_path,
    output_frames_folder,
    num_frames=30,
    start_index=0,
    skip_first_frame=False,
    distance_hint_m=None,  # ignoré pour rester fidèle à la méthode inspiration
    depth_model_id=None,
):
    print(">>> [Interpolate] 1. Calcul des Depth Maps...")
    depth_a, img_a_cv = get_depth_map(img_a_path, depth_model_id)
    depth_b, img_b_cv = get_depth_map(img_b_path, depth_model_id)
    
    # 2. Scale Auto via SIFT
    print(">>> [Interpolate] 2. Calcul du scale via SIFT...")
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_a_cv, None)
    kp2, des2 = sift.detectAndCompute(img_b_cv, None)
    matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
    if len(good) > 10:
        pts_a = np.float32([kp1[m.queryIdx].pt for m in good])
        pts_b = np.float32([kp2[m.trainIdx].pt for m in good])
        displacement = np.mean(np.linalg.norm(pts_b - pts_a, axis=1))
        scale = displacement / 25.0 
    else:
        scale = 2.0
    print(f"    Scale calculé : {scale:.2f}")

    # 3. Boucle d'Animation
    print(">>> [Interpolate] 3. Génération des frames...")
    
    frames_written = 0
    current_index = start_index

    for i in range(num_frames):
        if skip_first_frame and i == 0:
            continue

        t = i / (num_frames - 1)
        
        # Mouvement virtuel
        shift_a = scale * t 
        shift_b = -(scale * (1 - t))
        
        # Projection
        wa, da, ma = warp_image_3d(img_a_cv, depth_a, shift_a)
        wb, db, mb = warp_image_3d(img_b_cv, depth_b, shift_b)
        
        # Fusion
        final = consensus_merge(wa, da, ma, wb, db, mb, t)
        
        path = os.path.join(output_frames_folder, f"frame_{current_index:03d}.jpg")
        cv2.imwrite(path, final)
        frames_written += 1
        current_index += 1
        print(f"    Frame {frames_written}/{num_frames if not skip_first_frame else num_frames-1} générée.")

    torch.cuda.empty_cache()
    return frames_written
