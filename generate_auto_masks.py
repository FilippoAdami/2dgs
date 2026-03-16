import os
import torch
import numpy as np
import cv2
import argparse
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# --- 1. Configuration ---
parser = argparse.ArgumentParser(description="Generate SAM2 multi-view masks")
parser.add_argument("-s", "--source_path", type=str, required=True, help="Path to the dataset directory")
parser.add_argument("--max_tracks", type=int, default=15, help="Strict limit to prevent VRAM overflow")
args = parser.parse_args()

# Dynamically construct sub-directories based on the input path
IMAGE_DIR = os.path.join(args.source_path, "images")
OUTPUT_DIR = os.path.join(args.source_path, "masks_auto")
SAM2_SANDBOX_DIR = os.path.join(args.source_path, "images_sam2_sandbox")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAM2_SANDBOX_DIR, exist_ok=True)

MODEL_CFG = "sam2_hiera_l.yaml"
CHECKPOINT = "/workspace/weights/sam2_hiera_large.pt"
MAX_TRACKS = args.max_tracks

# --- 2. The Symlink Sandbox ---
print("Building integer-named symlink sandbox to bypass SAM 2 naming bug...")
original_frames = sorted(os.listdir(IMAGE_DIR))
frame_mapping = {}

for i, orig_name in enumerate(original_frames):
    # SAM expects purely integers
    sam_name = f"{i:05d}.jpg" 
    src = os.path.join(IMAGE_DIR, orig_name)
    dst = os.path.join(SAM2_SANDBOX_DIR, sam_name)
    
    if not os.path.exists(dst):
        os.symlink(src, dst)
        
    frame_mapping[sam_name] = orig_name

first_frame_path = os.path.join(IMAGE_DIR, original_frames[0])

# --- 3. Auto-Mask the First Frame ---
print(f"Loading first frame: {first_frame_path}")
image = Image.open(first_frame_path)
image_np = np.array(image.convert("RGB"))

print("Initializing SAM 2 Base Model for Automatic Grid Generation...")
sam2_base = build_sam2(MODEL_CFG, CHECKPOINT, device="cuda", apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_base,
    points_per_side=16,          
    points_per_batch=16,         
    pred_iou_thresh=0.6,         
    stability_score_thresh=0.9,
    min_mask_region_area=200     
)

print("Scanning image geometry... (This takes a moment)")
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    auto_masks = mask_generator.generate(image_np)

auto_masks = sorted(auto_masks, key=(lambda x: x['area']), reverse=True)
tracked_masks = auto_masks[:MAX_TRACKS]

print(f"Discovered {len(auto_masks)} total objects. Tracking the top {len(tracked_masks)} largest objects.")

np.random.seed(42) 
object_colors = {i + 1: np.random.randint(0, 255, size=(3,), dtype=np.uint8) for i in range(len(tracked_masks))}

del sam2_base
del mask_generator
torch.cuda.empty_cache()

# --- 4. Initialize the Video Predictor on the Sandbox ---
print("\nInitializing SAM 2 Video Predictor on the Sandbox directory...")
# IMPORTANT: Pointing to the sandbox, not the original dir!
predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT, device="cuda")
inference_state = predictor.init_state(video_path=SAM2_SANDBOX_DIR)

# --- 5. Inject the Auto-Masks ---
print("Injecting initial masks into the tracking engine...")
for i, mask_data in enumerate(tracked_masks):
    obj_id = i + 1
    mask_bool = mask_data["segmentation"]
    _, _, _ = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=obj_id,
        mask=mask_bool
    )

# --- 6. Propagate and Save with Original Naming ---
print("Propagating all objects across the 3D capture sequence...")
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    
    # THE FIX: Directly map the sequence index to our original filename array
    orig_img_name = original_frames[out_frame_idx]
    
    h, w = image_np.shape[:2]
    combined_color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i, obj_id in enumerate(out_obj_ids):
        mask = (out_mask_logits[i, 0] > 0.0).cpu().numpy()
        combined_color_mask[mask] = object_colors[obj_id]
        
    # Save using the ORIGINAL filename so 2DGS can read it!
    save_path = os.path.join(OUTPUT_DIR, f"{orig_img_name.split('.')[0]}.png")
    cv2.imwrite(save_path, combined_color_mask)
    
    if out_frame_idx % 10 == 0:
        print(f"Successfully processed frame {out_frame_idx} / {len(original_frames)}")

print(f"\nPhase 1 Complete! All exhaustive multi-view masks safely written to {OUTPUT_DIR}")
