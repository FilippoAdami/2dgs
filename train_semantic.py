import os
import sys
import torch
import cv2
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser
from utils.loss_utils import l1_loss
from gaussian_renderer import render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams

def semantic_training():
    # Config
    parser = ArgumentParser(description="Semantic Geometry Freeze")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--sam_masks_dir", type=str, default="/workspace/mip_nerf_360_data/360_v2/garden/masks_auto")
    
    # Parse standard 2DGS arguments alongside our custom ones
    args = parser.parse_args(sys.argv[1:])
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    # 1. Load the locked 5K geometry
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=5000)
    train_cameras = scene.getTrainCameras()
    
    # 2. Override the original images with our new SAM 2 masks
    print("Swapping COLMAP RGB data for SAM 2 Semantic Masks...")
    for cam in train_cameras:
        mask_path = os.path.join(args.sam_masks_dir, cam.image_name + ".png")
        mask_img = cv2.imread(mask_path)
        if mask_img is not None:
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
            
            # --- THE VRAM & PROJECTION FIX ---
            # Extract the exact height and width the 2DGS backend is expecting
            _, target_h, target_w = cam.original_image.shape
            
            # Resize the SAM 2 mask to match the camera perfectly, using INTER_NEAREST so the distinct semantic hex colors do not blend
            mask_img = cv2.resize(mask_img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            
            mask_tensor = torch.from_numpy(mask_img).float().cuda() / 255.0
            cam.original_image = mask_tensor.permute(2, 0, 1)
        else:
            print(f"Warning: Mask not found for {cam.image_name}")
            
    # Free up system memory before starting the heavy PyTorch loop
    torch.cuda.empty_cache()

    # 3. The Geometry Freeze
    print("Locking 3D Geometry (XYZ, Scaling, Rotation, Opacity)...")
    gaussians.training_setup(opt)
    
    # Mathematically sever the gradient graphs for spatial parameters leaving only _features_dc (the base colors) active in the optimizer
    gaussians._xyz.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)
    
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print("Initiating Semantic Paint Job (2000 Iterations)...")
    progress_bar = tqdm(range(1, 2001), desc="Painting Semantics")
    viewpoint_stack = None
    
    # training loop to paint the geometry with the SAM 2 segmentations
    for iteration in range(1, 2001):
        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        
        # Pure L1 loss to match the exact SAM 2 hex colors
        loss = l1_loss(image, gt_image) 
        loss.backward()

        with torch.no_grad():
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.{5}f}"})
                progress_bar.update(10)
            
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

    progress_bar.close()
    
    # 4. Save the Semantic Point Cloud
    print("\nSaving Segmented 3D Point Cloud...")
    out_dir = os.path.join(dataset.model_path, "point_cloud", "iteration_7000_semantic")
    os.makedirs(out_dir, exist_ok=True)
    gaussians.save_ply(os.path.join(out_dir, "point_cloud.ply"))
    print("Success! Geometry painted and saved.")

if __name__ == "__main__":
    semantic_training()
