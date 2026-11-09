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
# Modified by Jiazhong Cen, 2025

import torch
from random import randint
from gaussian_renderer import network_gui, render_contrastive_feature
from scene import Scene, GaussianModel, FeatureGaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import wandb

def visualize_cosine_heatmaps(multi_lvl_cos_seg, multi_lvl_masks, iteration, level=0):
    """Return paired cosine-similarity and mask visualizations."""
    if level >= len(multi_lvl_cos_seg):
        print(f"[Viz] Skipping heatmap: level {level} is unavailable.")
        return {}

    if level >= len(multi_lvl_masks):
        print(f"[Viz] Skipping masks: level {level} is unavailable.")
        return {}

    cos_seg = multi_lvl_cos_seg[level]  # [H, W, num_mask]
    masks = multi_lvl_masks[level]  # [H, W, num_mask]

    if cos_seg.numel() == 0 or masks.numel() == 0:
        print(f"[Viz] Skipping heatmap: level {level} is empty.")
        return {}

    cos_seg_np = cos_seg.detach().cpu().numpy()  # [H, W, num_mask]
    masks_np = masks.detach().cpu().numpy()  # [H, W, num_mask]
    H, W, num_mask = cos_seg_np.shape

    vmin = cos_seg_np.min()
    vmax = cos_seg_np.max()

    heatmap_images = {}

    for mask_idx in range(num_mask):
        heatmap = cos_seg_np[:, :, mask_idx]  # [H, W]
        mask = masks_np[:, :, mask_idx]  # [H, W]

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        im1 = axes[0].imshow(heatmap, cmap='jet', vmin=vmin, vmax=vmax, interpolation='nearest')
        axes[0].set_title(f'Cosine Similarity Heatmap\nIter {iteration}, Level {level}, Mask {mask_idx}')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], label='Cosine Similarity')

        im2 = axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[1].set_title(f'Ground Truth Mask\nIter {iteration}, Level {level}, Mask {mask_idx}')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], label='Mask Value')

        plt.tight_layout()

        fig.canvas.draw()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        heatmap_images[mask_idx] = image_array

        plt.close()

    print(f"[Viz] Generated {num_mask} heatmaps.")
    return heatmap_images

def visualize_cosine_distributions(pos_values, neg_values, iteration, bins=50):
    """Return a histogram of positive and negative cosine similarities."""
    if pos_values.numel() == 0 or neg_values.numel() == 0:
        print("[Viz] Skipping histogram: positive or negative samples are empty.")
        return None

    pos_np = pos_values.detach().float().cpu().view(-1).numpy()
    neg_np = neg_values.detach().float().cpu().view(-1).numpy()

    plt.figure(figsize=(6, 4))
    plt.hist(pos_np, bins=bins, alpha=0.6, label="Positives", color="#1f77b4", density=True)
    plt.hist(neg_np, bins=bins, alpha=0.6, label="Negatives", color="#ff7f0e", density=True)
    plt.title(f"Cosine Distribution @ Iter {iteration}")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    fig = plt.gcf()
    fig.canvas.draw()
    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return image_array

def training_report(iteration, loss, multi_lvl_loss, rendered_feature_norm_reg, cosine_pos, cosine_neg,
                   rendered_feature_norm, testing_iterations, scene, feature_gaussians, pipe,
                   background, MULTI_LVL_DIM, num_lvl, use_wandb=False, downsample=False, downsample_scale=2):
    """Log training metrics and visualize held-out views at evaluation iterations."""

    if use_wandb:
        wandb.log({
            'train_loss/total_loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
            'train_loss/multi_lvl_loss': multi_lvl_loss.item() if isinstance(multi_lvl_loss, torch.Tensor) else multi_lvl_loss,
            'train_loss/rendered_feature_norm_reg': rendered_feature_norm_reg.item() if isinstance(rendered_feature_norm_reg, torch.Tensor) else rendered_feature_norm_reg,
            'metrics/cosine_pos': cosine_pos.item() if isinstance(cosine_pos, torch.Tensor) else cosine_pos,
            'metrics/cosine_neg': cosine_neg.item() if isinstance(cosine_neg, torch.Tensor) else cosine_neg,
            'metrics/rendered_feature_norm': rendered_feature_norm.item() if isinstance(rendered_feature_norm, torch.Tensor) else rendered_feature_norm,
        })

    if testing_iterations and iteration in testing_iterations:
        torch.cuda.empty_cache()

        test_cameras = [camera for camera in scene.getTestCameras()
                        if camera.original_masks is not None]
        if test_cameras and len(test_cameras) > 0:
            print(f"\n[ITER {iteration}] Visualizing {len(test_cameras)} test views")

            for cam_idx, test_cam in enumerate(test_cameras):
                with torch.no_grad():

                    sam_masks = test_cam.original_masks.cuda().float()

                    multi_lvl_masks = []
                    multi_lvl_masks_before_filt = []
                    MASK_SIZE_THRESH = 400

                    sam_masks[sam_masks == -1] = -1000
                    if num_lvl == 1:  # OR dataset
                        tmp_masks = sam_masks[0].clone()
                        tmp_masks[tmp_masks < 0] = 0
                        tmp_lvl_masks = torch.nn.functional.one_hot(tmp_masks.long(), num_classes=tmp_masks.max().int().item()+1)[:,:,1:].float()
                        multi_lvl_masks_before_filt.append(tmp_lvl_masks)
                    else:
                        for lvl in range(1, num_lvl+1):
                            tmp_masks = sam_masks[lvl].clone()
                            smallest_index = sam_masks[lvl-1].int().max()+1
                            tmp_masks -= smallest_index-1
                            tmp_masks[tmp_masks < 0] = 0
                            tmp_lvl_masks = torch.nn.functional.one_hot(tmp_masks.long(), num_classes=tmp_masks.max().int().item()+1)[:,:,1:].float()
                            multi_lvl_masks_before_filt.append(tmp_lvl_masks)

                    for lvl in range(num_lvl):
                        tmp_mask = multi_lvl_masks_before_filt[lvl]
                        mask_non_zero_count = tmp_mask.sum(dim=(0,1))
                        fi = mask_non_zero_count > MASK_SIZE_THRESH

                        for sub_lvl in range(0, lvl):
                            tmp_mask2 = multi_lvl_masks_before_filt[sub_lvl]
                            intersection = torch.einsum('hwc,hwf->cf', tmp_mask, tmp_mask2)
                            union = tmp_mask.sum(dim=0).sum(dim=0).unsqueeze(-1) + tmp_mask2.sum(dim=0).sum(dim=0) - intersection
                            inter_over_union = intersection / union
                            t_fi = torch.logical_and(fi, inter_over_union.max(dim=1)[0] < 0.8)
                            fi = fi if t_fi.count_nonzero() == 0 else t_fi

                        tmp_mask = tmp_mask[:,:,fi]
                        multi_lvl_masks.append(tmp_mask)

                    test_cam.feature_height, test_cam.feature_width = test_cam.image_height, test_cam.image_width

                    render_pkg_feat = render_contrastive_feature(test_cam, feature_gaussians, pipe, background,
                                                                 norm_point_features=True, multi_lvl_norm=True,
                                                                 multi_lvl_dim=MULTI_LVL_DIM, smooth_type=None, smooth_weights=None)

                    rendered_features = render_pkg_feat["render"]
                    rendered_features = torch.nn.functional.interpolate(rendered_features.unsqueeze(0),
                                                                        test_cam.original_masks.shape[-2:],
                                                                        mode='bilinear').squeeze(0)

                    if downsample:
                        rendered_features = torch.nn.functional.interpolate(rendered_features.unsqueeze(0),
                                                                            (rendered_features.shape[-2]//downsample_scale,
                                                                             rendered_features.shape[-1]//downsample_scale),
                                                                            mode='bilinear').squeeze(0)
                        multi_lvl_masks = [torch.nn.functional.interpolate(mlm.unsqueeze(0).permute([0,3,1,2]),
                                                                            rendered_features.shape[-2:],
                                                                            mode='nearest').squeeze(0).permute([1,2,0])
                                          for mlm in multi_lvl_masks]

                    rendered_features = rendered_features.permute([1,2,0])
                    multi_lvl_features = []
                    pre_dim = 0
                    for dim in MULTI_LVL_DIM:
                        cur_lvl_features = rendered_features[:,:,0:pre_dim+dim]
                        multi_lvl_features.insert(0, cur_lvl_features)
                        pre_dim += dim

                    # Pool mask features into unit-length prototypes.
                    multi_lvl_prototypes = [torch.nn.functional.normalize(torch.einsum('hwc,hwf->fc', mlf, mlm), dim=-1, p=2)
                                          for mlf, mlm in zip(multi_lvl_features, multi_lvl_masks)]
                    multi_lvl_cos_seg = [torch.einsum('fc,hwc->hwf', mlp, torch.nn.functional.normalize(mlf, dim=-1, p=2))
                                        for mlp, mlf in zip(multi_lvl_prototypes, multi_lvl_features)]

                    if multi_lvl_cos_seg and len(multi_lvl_cos_seg) > 0 and multi_lvl_masks and len(multi_lvl_masks) > 0:
                        pos_mask = multi_lvl_masks[0] == 1
                        neg_mask = multi_lvl_masks[0] == 0

                        if pos_mask.any() and neg_mask.any():
                            pos_scores = multi_lvl_cos_seg[0][pos_mask]
                            neg_scores = multi_lvl_cos_seg[0][neg_mask]

                            if pos_scores.numel() > 0 and neg_scores.numel() > 0:

                                if use_wandb:

                                    cosine_dist_image = visualize_cosine_distributions(
                                        pos_scores,
                                        neg_scores,
                                        iteration
                                    )
                                    if cosine_dist_image is not None:
                                        wandb.log({f'test_cam_{cam_idx}/cosine_distribution': wandb.Image(cosine_dist_image)})

                                    heatmap_images = visualize_cosine_heatmaps(
                                        multi_lvl_cos_seg,
                                        multi_lvl_masks,
                                        iteration,
                                        level=0
                                    )
                                    for mask_idx, heatmap_image in heatmap_images.items():
                                        wandb.log({f'test_cam_{cam_idx}/heatmap_mask_{mask_idx:03d}': wandb.Image(heatmap_image)})

        print(f"\n[ITER {iteration}] Training metrics: Loss={loss.item() if isinstance(loss, torch.Tensor) else loss:.6f}, "
              f"Multi Lvl Loss={multi_lvl_loss.item() if isinstance(multi_lvl_loss, torch.Tensor) else multi_lvl_loss:.6f}, "
              f"Pos Cos={cosine_pos.item() if isinstance(cosine_pos, torch.Tensor) else cosine_pos:.4f}, "
              f"Neg Cos={cosine_neg.item() if isinstance(cosine_neg, torch.Tensor) else cosine_neg:.4f}")

        torch.cuda.empty_cache()

def training(dataset, opt, pipe, iteration, downsample = False, use_wandb=False, test_iterations=None, save_iterations=None, downsample_scale=2):

    dataset.need_features = False
    dataset.need_masks = True

    gaussians = GaussianModel(dataset.sh_degree)

    feature_gaussians = FeatureGaussianModel(dataset.feature_dim)

    sample_rate = 0.2 if 'Replica' in dataset.source_path else 1.0
    scene = Scene(dataset, gaussians, feature_gaussians, load_iteration=iteration, shuffle=False, target='contrastive_feature', mode='train', sample_rate=sample_rate)

    training_cameras = [camera for camera in scene.getTrainCameras()
                        if camera.original_masks is not None]
    if not training_cameras:
        raise ValueError(
            "Feature training requires at least one training view with _s.npy masks "
            "in language_features. Check the training/test split in test.txt."
        )

    print('Number of 3D Gaussians:', gaussians.get_xyz.shape[0])

    num_lvl = 1
    ADJUSTABLE = 'scannet' in dataset.source_path
    print('Gaussian adjustable:', ADJUSTABLE)
    feature_gaussians.change_to_segmentation_mode(opt, "contrastive_feature", fixed_feature=False, adjustable= ADJUSTABLE, opacity_adjustable=True)

    MULTI_LVL_DIM = [16, 8, 8] if num_lvl == 3 else [32]

    smooth_weights = None

    del gaussians
    torch.cuda.empty_cache()
    # feature_gaussians.training_setup(opt)

    background = torch.ones([dataset.feature_dim], dtype=torch.float32, device="cuda") if dataset.white_background else torch.zeros([dataset.feature_dim], dtype=torch.float32, device="cuda")

    first_iter = 0
    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):

        if not viewpoint_stack:
            viewpoint_stack = training_cameras.copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        with torch.no_grad():
            sam_masks = viewpoint_cam.original_masks.cuda().float()

            multi_lvl_masks = []
            multi_lvl_masks_before_filt = []
            MASK_SIZE_THRESH = 400

            sam_masks[sam_masks == -1] = -1000
            if num_lvl == 1: # OR dataset
                tmp_masks = sam_masks[0].clone()
                tmp_masks[tmp_masks < 0] = 0

                tmp_lvl_masks = torch.nn.functional.one_hot(tmp_masks.long(), num_classes=tmp_masks.max().int().item()+1)[:, :,1:].float()
                multi_lvl_masks_before_filt.append(tmp_lvl_masks)
            else:
                for lvl in range(1,num_lvl+1):
                    tmp_masks = sam_masks[lvl].clone()
                    smallest_index = sam_masks[lvl-1].int().max()+1
                    tmp_masks -= smallest_index-1
                    tmp_masks[tmp_masks < 0] = 0
                    tmp_lvl_masks = torch.nn.functional.one_hot(tmp_masks.long(), num_classes=tmp_masks.max().int().item()+1)[:,:,1:].float()
                    multi_lvl_masks_before_filt.append(tmp_lvl_masks)

            for lvl in range(num_lvl):
                tmp_mask = multi_lvl_masks_before_filt[lvl]
                mask_non_zero_count = tmp_mask.sum(dim = (0,1))
                fi = mask_non_zero_count > MASK_SIZE_THRESH

                for sub_lvl in range(0, lvl):
                    tmp_mask2 = multi_lvl_masks_before_filt[sub_lvl]

                    intersection = torch.einsum('hwc,hwf->cf', tmp_mask, tmp_mask2)
                    union = tmp_mask.sum(dim = 0).sum(dim = 0).unsqueeze(-1) + tmp_mask2.sum(dim = 0).sum(dim = 0) - intersection
                    inter_over_union = intersection / union

                    t_fi = torch.logical_and(fi, inter_over_union.max(dim = 1)[0] < 0.8)
                    fi = fi if t_fi.count_nonzero() == 0 else t_fi

                tmp_mask = tmp_mask[:,:,fi]
                multi_lvl_masks.append(tmp_mask)

            viewpoint_cam.feature_height, viewpoint_cam.feature_width = viewpoint_cam.image_height, viewpoint_cam.image_width

        torch.cuda.empty_cache()

        render_pkg_feat = render_contrastive_feature(viewpoint_cam, feature_gaussians, pipe, background, norm_point_features=True,
                                                     multi_lvl_norm=True, multi_lvl_dim=MULTI_LVL_DIM, smooth_type = None, smooth_weights=None)

        rendered_features = render_pkg_feat["render"]
        rendered_features = torch.nn.functional.interpolate(rendered_features.unsqueeze(0), viewpoint_cam.original_masks.shape[-2:], mode='bilinear').squeeze(0)

        if downsample:
            rendered_features = torch.nn.functional.interpolate(rendered_features.unsqueeze(0),(rendered_features.shape[-2]//downsample_scale, rendered_features.shape[-1]//downsample_scale), mode='bilinear').squeeze(0)
            multi_lvl_masks = [torch.nn.functional.interpolate(mlm.unsqueeze(0).permute([0,3,1,2]), rendered_features.shape[-2:], mode='nearest').squeeze(0).permute([1,2,0]) for mlm in multi_lvl_masks]

        multi_lvl_rendered_feature_norm = 0
        pre_dim = 0
        for dim in MULTI_LVL_DIM:
            multi_lvl_rendered_feature_norm += rendered_features[pre_dim:pre_dim+dim, :,:].norm(dim = 0).mean()
            pre_dim += dim

        rendered_feature_norm = multi_lvl_rendered_feature_norm / len(MULTI_LVL_DIM)
        rendered_feature_norm_reg = (1-rendered_feature_norm)**2

        rendered_features = rendered_features.permute([1,2,0])
        multi_lvl_features = []
        pre_dim = 0
        for dim in MULTI_LVL_DIM:
            # NOTE: FINEST -> MEDIAN -> COARSEST
            cur_lvl_features = rendered_features[:,:,0:pre_dim+dim]

            multi_lvl_features.insert(0, cur_lvl_features)
            pre_dim += dim

        # Pool mask features into unit-length prototypes.
        multi_lvl_prototypes = [torch.nn.functional.normalize(torch.einsum('hwc,hwf->fc', mlf, mlm), dim = -1, p = 2) for mlf, mlm in zip(multi_lvl_features, multi_lvl_masks)]

        multi_lvl_cos_seg = [torch.einsum('fc,hwc->hwf', mlp, torch.nn.functional.normalize(mlf, dim = -1, p = 2)) for mlp, mlf in zip(multi_lvl_prototypes, multi_lvl_features)]

        # Enforce cross-level consistency using fine positives and coarse negatives.

        multi_lvl_loss = 0

        if num_lvl > 1:
            fine_mask_on_coarse_prototypes = [torch.nn.functional.normalize(torch.einsum('hwc,hwf->fc', mlf, mlm), dim = -1, p = 2) for mlf, mlm in zip(multi_lvl_features[1:], multi_lvl_masks[:-1])]
            fine_mask_on_coarse_cos_seg = [torch.einsum('fc,hwc->hwf', mlp, torch.nn.functional.normalize(mlf, dim = -1, p = 2)) for mlp, mlf in zip(fine_mask_on_coarse_prototypes, multi_lvl_features[1:])]
            coarse_mask_on_fine_prototypes = [torch.nn.functional.normalize(torch.einsum('hwc,hwf->fc', mlf, mlm), dim = -1, p = 2) for mlf, mlm in zip(multi_lvl_features[:-1], multi_lvl_masks[1:])]
            coarse_mask_on_fine_cos_seg = [torch.einsum('fc,hwc->hwf', mlp, torch.nn.functional.normalize(mlf, dim = -1, p = 2)) for mlp, mlf in zip(coarse_mask_on_fine_prototypes, multi_lvl_features[:-1])]
            for lvl in range(num_lvl):
                pos_mask = multi_lvl_masks[lvl] == 1
                neg_mask = multi_lvl_masks[lvl] == 0

                neg_sample_rate = pos_mask.sum(dim = (0,1)) / neg_mask.sum(dim = (0,1))
                random_num = torch.rand_like(multi_lvl_cos_seg[lvl])
                sampled_neg_mask = torch.logical_and(neg_mask, random_num < neg_sample_rate)
                sampled_neg_mask = torch.logical_or(sampled_neg_mask, torch.logical_and(multi_lvl_cos_seg[lvl] > 0.5, neg_mask))

                if lvl == 0:
                    multi_lvl_loss += -torch.min(multi_lvl_cos_seg[lvl], fine_mask_on_coarse_cos_seg[lvl])[pos_mask].mean() + torch.relu((multi_lvl_cos_seg[lvl])[sampled_neg_mask]).mean()

                elif lvl == num_lvl-1:
                    multi_lvl_loss += -(multi_lvl_cos_seg[lvl])[pos_mask].mean() + torch.relu(torch.max(multi_lvl_cos_seg[lvl], coarse_mask_on_fine_cos_seg[lvl-1])[sampled_neg_mask]).mean()

                else:
                    multi_lvl_loss += -torch.min(multi_lvl_cos_seg[lvl], fine_mask_on_coarse_cos_seg[lvl])[pos_mask].mean() + torch.relu(torch.max(multi_lvl_cos_seg[lvl], coarse_mask_on_fine_cos_seg[lvl-1])[sampled_neg_mask]).mean()

        else:
            pos_mask = multi_lvl_masks[0] == 1 # [H, W, num_mask]
            neg_mask = multi_lvl_masks[0] == 0 # [H, W, num_mask]

            neg_sample_rate = pos_mask.sum(dim = (0,1)) / neg_mask.sum(dim = (0,1))
            random_num = torch.rand_like(multi_lvl_cos_seg[0])

            sampled_neg_mask = neg_mask

            # (multi_lvl_cos_seg[0])[pos_mask]: [pos_num]
            pos_scores = (multi_lvl_cos_seg[0])[pos_mask]
            neg_scores = (multi_lvl_cos_seg[0])[sampled_neg_mask]
            multi_lvl_loss += -pos_scores.mean() * 0.2 + torch.relu(neg_scores).mean()

        loss = multi_lvl_loss + rendered_feature_norm_reg

        with torch.no_grad():

            multi_lvl_cos_seg_cat = torch.cat(multi_lvl_cos_seg, dim = -1)
            multi_lvl_lbl = torch.cat(multi_lvl_masks, dim = -1)
            cosine_pos = multi_lvl_cos_seg_cat[multi_lvl_lbl == 1].mean()
            cosine_neg = multi_lvl_cos_seg_cat[multi_lvl_lbl == 0].mean()

        loss.backward()

        feature_gaussians.optimizer.step()
        feature_gaussians.optimizer.zero_grad(set_to_none = True)

        if iteration % 10 == 0:
            progress_bar.set_postfix({
                "RFN": f"{rendered_feature_norm.item():.{3}f}",
                "Pos cos": f"{cosine_pos.item():.{3}f}",
                "Neg cos": f"{cosine_neg.item():.{3}f}",
                "Multi Lvl Loss": f"{multi_lvl_loss.item():.{3}f}",
                "Loss": f"{loss.item():.{3}f}",
            })
            progress_bar.update(10)

        with torch.no_grad():

            training_report(
                iteration=iteration,
                loss=loss,
                multi_lvl_loss=multi_lvl_loss,
                rendered_feature_norm_reg=rendered_feature_norm_reg,
                cosine_pos=cosine_pos,
                cosine_neg=cosine_neg,
                rendered_feature_norm=rendered_feature_norm,
                testing_iterations=test_iterations if test_iterations is not None else [],
                scene=scene,
                feature_gaussians=feature_gaussians,
                pipe=pipe,
                background=background,
                MULTI_LVL_DIM=MULTI_LVL_DIM,
                num_lvl=num_lvl,
                use_wandb=use_wandb,
                downsample=downsample,
                downsample_scale=downsample_scale
            )

        if save_iterations and iteration in save_iterations:
            scene.save_feature(iteration, target = 'contrastive_feature', smooth_weights = torch.softmax(smooth_weights, dim = -1) if smooth_weights is not None else None, smooth_type = None)

    scene.save_feature(iteration, target = 'contrastive_feature', smooth_weights = torch.softmax(smooth_weights, dim = -1) if smooth_weights is not None else None, smooth_type = None)

if __name__ == "__main__":

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=np.random.randint(10000, 20000))
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--downsample", action="store_true")
    parser.add_argument("--downsample_scale",default=2, type=int)
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Log training metrics to Weights & Biases")

    args = get_combined_args(parser, target_cfg_file = 'cfg_args')
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.use_wandb:
        dataset_name = os.path.basename(os.path.normpath(args.source_path)) if hasattr(args, 'source_path') else "unknown"
        wandb.init(project="laga_affinity_features", name=f"{dataset_name}")

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print(f"\nTraining started: {start_time_str}")

    training(lp.extract(args), op.extract(args), pp.extract(args), args.iteration, args.downsample,
             args.use_wandb, args.test_iterations, args.save_iterations, args.downsample_scale)

    end_time = time.time()
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    total_time = end_time - start_time

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    print(f"\nTraining finished: {end_time_str}")
    print(f"Total training time: {time_str} ({total_time:.2f} seconds)")

    os.makedirs(args.model_path, exist_ok=True)
    time_log_file = os.path.join(args.model_path, "training_time.txt")
    with open(time_log_file, 'w', encoding='utf-8') as f:
        f.write("Training duration\n")
        f.write("=" * 50 + "\n")
        f.write(f"Start time: {start_time_str}\n")
        f.write(f"End time: {end_time_str}\n")
        f.write(f"Total training time: {time_str}\n")
        f.write(f"Total seconds: {total_time:.2f} seconds\n")
        f.write(f"Total hours: {total_time / 3600:.2f} hours\n")
    print(f"Training duration saved to: {time_log_file}")

    if args.use_wandb:
        wandb.finish()

    print("\nTraining complete.")
