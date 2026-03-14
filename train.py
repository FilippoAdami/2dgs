#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training_original_function(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        # --- AMD ROCm NaN Sanitization ---
        if "rend_normal" in render_pkg and render_pkg["rend_normal"] is not None:
            render_pkg["rend_normal"] = torch.nan_to_num(render_pkg["rend_normal"], nan=0.0)
        if "surf_normal" in render_pkg and render_pkg["surf_normal"] is not None:
            render_pkg["surf_normal"] = torch.nan_to_num(render_pkg["surf_normal"], nan=0.0)
        # ---------------------------------
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        rend_alpha = render_pkg.get('rend_alpha', None)
        if rend_alpha is None:
            rend_alpha = torch.ones_like(render_pkg['rend_normal'][0:1])
        rend_alpha = torch.nan_to_num(rend_alpha, nan=0.0)
        
        # --- AMD ROCm Isolation ---
        # Create a boolean mask of solid objects to physically exclude the infinite sky
        valid_mask = rend_alpha > 0.05
        
        if opt.lambda_normal > 0.0 and iteration > 7000:
            rend_normal  = torch.nan_to_num(render_pkg['rend_normal'], nan=0.0, posinf=0.0, neginf=0.0)
            surf_normal = torch.nan_to_num(render_pkg['surf_normal'], nan=0.0, posinf=0.0, neginf=0.0)
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            # Only calculate the mean on valid, finite geometry
            normal_loss = opt.lambda_normal * normal_error[valid_mask].mean()
            normal_loss = torch.nan_to_num(normal_loss, nan=0.0)
        else:
            # Completely bypass the graph to prevent NaN poisoning
            normal_loss = torch.tensor(0.0).cuda()

        if opt.lambda_dist > 0.0 and iteration > 3000:
            rend_dist = torch.nan_to_num(render_pkg["rend_dist"], nan=0.0)
            # The valid_mask automatically drops the posinf values from the empty sky
            dist_loss = opt.lambda_dist * rend_dist[valid_mask].mean()
            dist_loss = torch.nan_to_num(dist_loss, nan=0.0)
        else:
            dist_loss = torch.tensor(0.0).cuda()
        # --------------------------

        # loss
        total_loss = loss + dist_loss + normal_loss
        
        total_loss.backward()

        iter_end.record()

        """ with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log 

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close() """ 

        # ------------------------------------------------------------------
        # --- PHASE 1: TAMING 3DGS DYNAMIC ASYMPTOTIC BUDGETING ---
        # ------------------------------------------------------------------
        with torch.no_grad():
            # Progress bar & EMA Loss
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            
            # Initialization of the State Machine
            if iteration == 1 or 'is_densifying' not in locals():
                B = 1000000  # Exact Budget
                S = len(scene.getTrainCameras()[0].pointcloud) if hasattr(scene.getTrainCameras()[0], 'pointcloud') else 138766
                DENSIFY_FREQ = 500  # Taming 3DGS Discovery
                
                densify_step = 0
                is_densifying = True
                best_loss = float('inf')
                patience_counter = 0
            
            """ # The Asymptotic Densification Loop
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                if is_densifying and iteration > opt.densify_from_iter and iteration % DENSIFY_FREQ == 0:
                    densify_step += 1
                    
                    # Inverse Exponential Curve: Approaches B infinitely
                    # 0.15 decay means it reaches ~95% of budget in 20 steps (10,000 iterations)
                    import math
                    progress = 1.0 - math.exp(-0.15 * densify_step)
                    target_points = S + (B - S) * progress
                    current_points = gaussians.get_xyz.shape[0]
                    
                    allowed_additions = int(max(0, target_points - current_points))
                    target_selected = allowed_additions // 2
                    
                    if target_selected > 0 and current_points < B:
                        grads = gaussians.xyz_gradient_accum / gaussians.denom
                        grads[grads.isnan()] = 0.0
                        sorted_grads, _ = torch.sort(grads.squeeze(), descending=True)
                        
                        idx = min(target_selected, len(sorted_grads) - 1)
                        dynamic_threshold = max(sorted_grads[idx].item(), opt.densify_grad_threshold)
                        
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(dynamic_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold) """
            # The Asymptotic Densification Loop
            if iteration < opt.densify_until_iter:
                
                # --- AMD ROCm SILENT TRUNCATION FAILSAFE ---
                if visibility_filter.shape[0] != gaussians.max_radii2D.shape[0]:
                    print(f"\n[ROCm FAILSAVE] Array desync caught at iteration {iteration}! Healing tensor graph...")
                    diff = gaussians.max_radii2D.shape[0] - visibility_filter.shape[0]
                    if diff > 0:
                        # Pad the truncated masks with zeros (making dropped points safely invisible)
                        pad_mask = torch.zeros(diff, dtype=torch.bool, device="cuda")
                        visibility_filter = torch.cat([visibility_filter, pad_mask])
                        
                        pad_radii = torch.zeros(diff, dtype=radii.dtype, device="cuda")
                        radii = torch.cat([radii, pad_radii])
                    else:
                        # Safety truncation in case of inverse desync
                        visibility_filter = visibility_filter[:gaussians.max_radii2D.shape[0]]
                        radii = radii[:gaussians.max_radii2D.shape[0]]
                # -------------------------------------------

                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                if is_densifying and iteration > opt.densify_from_iter and iteration % DENSIFY_FREQ == 0:
                    densify_step += 1
                    
                    import math
                    progress = 1.0 - math.exp(-0.15 * densify_step)
                    target_points = S + (B - S) * progress
                    current_points = gaussians.get_xyz.shape[0]
                    
                    allowed_additions = int(max(0, target_points - current_points))
                    target_selected = allowed_additions // 2
                    
                    if target_selected > 0 and current_points < B:
                        grads = gaussians.xyz_gradient_accum / gaussians.denom
                        grads[grads.isnan()] = 0.0
                        sorted_grads, _ = torch.sort(grads.squeeze(), descending=True)
                        
                        idx = min(target_selected, len(sorted_grads) - 1)
                        dynamic_threshold = max(sorted_grads[idx].item(), opt.densify_grad_threshold)
                        
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(dynamic_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                    """------------------------------------------------------------------------------------------------------"""
                    
                    # --- DYNAMIC DISCOVERY: DENSIFICATION PLATEAU ---
                    # Calculate a 1% relative improvement threshold to absorb stochastic camera noise
                    improvement_threshold = best_loss * 0.99 
                    
                    if ema_loss_for_log < improvement_threshold:
                        best_loss = ema_loss_for_log
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    # Wait 10 densification cycles (5,000 iterations) to mathematically prove the plateau
                    if patience_counter >= 10:
                        print(f"\n[DYNAMIC DISCOVERY] Geometry plateaued at iteration {iteration} ({current_points} points). Halting densification.")
                        is_densifying = False
                        best_refine_loss = ema_loss_for_log
                        refine_patience = 0

            # ------------------------------------------------------------------
            # --- PHASE 2: STRICT REFINEMENT AND EARLY STOPPING ---
            # ------------------------------------------------------------------
            if not is_densifying:
                if ema_loss_for_log < best_refine_loss - 0.00005:
                    best_refine_loss = ema_loss_for_log
                    refine_patience = 0
                else:
                    refine_patience += 1
                
                # If 1500 iterations of pure refinement yield no color/normal improvement, we are done.
                if refine_patience >= 1500:
                    print(f"\n[EARLY STOPPING] Scene fully converged at iteration {iteration}. Halting training to save compute.")
                    scene.save(iteration)
                    progress_bar.close()
                    break

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{gaussians.get_xyz.shape[0]}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            
            if iteration == opt.iterations:
                progress_bar.close()
            # ------------------------------------------------------------------

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            """ # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity() """
            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    # --- TAMING 3DGS: PROGRAMMATIC BUDGETING ---
                    B = 1000000  # Final exact budget constraint (1M)
                    S = 138766   # Initial SfM point count
                    
                    # Calculate current step in the densification schedule
                    total_densify_steps = (opt.densify_until_iter - opt.densify_from_iter) // opt.densification_interval
                    current_step = (iteration - opt.densify_from_iter) // opt.densification_interval
                    
                    # Parabolic curve (quadratic decrease) targeting the exact budget B
                    target_points = S + (B - S) * (1.0 - (1.0 - current_step / total_densify_steps)**2)
                    current_points = gaussians.get_xyz.shape[0]
                    
                    # Calculate how many points we are allowed to add this cycle
                    allowed_additions = int(max(0, target_points - current_points))
                    
                    # 2DGS splitting/cloning yields an average of 2 primitives per selected Gaussian
                    target_selected = allowed_additions // 2
                    
                    if target_selected > 0 and current_points < B:
                        # Extract and sort the current gradient magnitudes
                        grads = gaussians.xyz_gradient_accum / gaussians.denom
                        grads[grads.isnan()] = 0.0
                        sorted_grads, _ = torch.sort(grads.squeeze(), descending=True)
                        
                        # Find the precise gradient threshold that isolates only the top 'target_selected' Gaussians
                        idx = min(target_selected, len(sorted_grads) - 1)
                        dynamic_threshold = sorted_grads[idx].item()
                        
                        # Apply a safety floor to prevent cloning zero-gradient background noise
                        dynamic_threshold = max(dynamic_threshold, opt.densify_grad_threshold)
                    else:
                        # Budget reached: clamp threshold to infinity to halt densification
                        dynamic_threshold = float('inf')

                    gaussians.densify_and_prune(dynamic_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                    # -------------------------------------------

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

#edited training function
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # --- 5K Timeline Variables ---
    B = 1000000
    S = len(scene.getTrainCameras()[0].pointcloud) if hasattr(scene.getTrainCameras()[0], 'pointcloud') else 138766
    DENSIFY_FREQ = 200 # Accelerated cloning cycles
    densify_step = 0

    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        rend_alpha = render_pkg.get('rend_alpha', None)
        if rend_alpha is None:
            rend_alpha = torch.ones_like(render_pkg['rend_normal'][0:1])
        rend_alpha = torch.nan_to_num(rend_alpha, nan=0.0)
        valid_mask = rend_alpha > 0.05
        
        # --- Timeline Shifts ---
        if opt.lambda_normal > 0.0 and iteration > 3500: # Activates AFTER pruning
            rend_normal  = torch.nan_to_num(render_pkg['rend_normal'], nan=0.0, posinf=0.0, neginf=0.0)
            surf_normal = torch.nan_to_num(render_pkg['surf_normal'], nan=0.0, posinf=0.0, neginf=0.0)
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = opt.lambda_normal * normal_error[valid_mask].mean()
            normal_loss = torch.nan_to_num(normal_loss, nan=0.0)
        else:
            normal_loss = torch.tensor(0.0).cuda()

        if opt.lambda_dist > 0.0 and iteration > 2000:
            rend_dist = torch.nan_to_num(render_pkg["rend_dist"], nan=0.0)
            dist_loss = opt.lambda_dist * rend_dist[valid_mask].mean()
            dist_loss = torch.nan_to_num(dist_loss, nan=0.0)
        else:
            dist_loss = torch.tensor(0.0).cuda()

        total_loss = loss + dist_loss + normal_loss
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            # --- Compressed Densification Window ---
            if iteration < 3500:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                if iteration > 500 and iteration % DENSIFY_FREQ == 0:
                    densify_step += 1
                    import math
                    # Steeper curve to reach budget in ~15 steps
                    progress = 1.0 - math.exp(-0.25 * densify_step)
                    target_points = S + (B - S) * progress
                    current_points = gaussians.get_xyz.shape[0]
                    
                    allowed_additions = int(max(0, target_points - current_points))
                    target_selected = allowed_additions // 2
                    
                    if target_selected > 0 and current_points < B:
                        grads = gaussians.xyz_gradient_accum / gaussians.denom
                        grads[grads.isnan()] = 0.0
                        sorted_grads, _ = torch.sort(grads.squeeze(), descending=True)
                        
                        idx = min(target_selected, len(sorted_grads) - 1)
                        dynamic_threshold = max(sorted_grads[idx].item(), opt.densify_grad_threshold)
                        
                        gaussians.densify_and_prune(dynamic_threshold, opt.opacity_cull, scene.cameras_extent, None)

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{gaussians.get_xyz.shape[0]}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            
            if iteration == opt.iterations:
                progress_bar.close()

            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                scene.save(iteration)

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
