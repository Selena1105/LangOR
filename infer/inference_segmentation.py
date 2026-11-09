"""Query Gaussian instances, render masks, and evaluate segmentation."""
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Tuple, List

from gaussian_renderer import render
from inference_utils import vis_mask_save, smooth_mask
from utils.general_utils import build_scaling_rotation
from utils.sh_utils import SH2RGB


def export_labeled_gaussians(
    scene_gaussians,
    accumulator: dict,
    output_path: Path,
    logger=None,
):
    """Export Gaussian attributes and text labels, not embedding vectors.

    The accumulator contains query_list and query_point_labels; output_path
    specifies the destination .pt file."""
    assert scene_gaussians is not None
    log = logger.log if logger else print


    points_xyz = scene_gaussians.get_xyz  # (N, 3)
    scaling = scene_gaussians.get_scaling  # (N, 3)
    rotation = scene_gaussians.get_rotation  # (N, 4), normalized quaternion


    L = build_scaling_rotation(scaling, rotation)  # (N, 3, 3)
    # Compose covariance as R S S^T R^T.
    covariances = L @ L.transpose(1, 2)  # (N, 3, 3)


    opacities = scene_gaussians.get_opacity  # (N, 1) or (N,)


    features_dc = scene_gaussians._features_dc  # (N, 3, 1)
    features_dc_squeeze = features_dc.squeeze(-1)  # (N, 3, 1) -> (N, 3)
    colors = SH2RGB(features_dc_squeeze)
    colors = torch.clamp(colors, 0.0, 1.0)

    query_list = accumulator.get('query_list', [])
    query_point_labels_list = accumulator.get('query_point_labels', [])


    num_points = points_xyz.shape[0]
    if query_list and query_point_labels_list:

        labels_mat = torch.stack(query_point_labels_list, dim=0)  # (N_query, N_point)
        is_foreground = (labels_mat != -1)  # (N_query, N_point)

        # Resolve overlapping queries in input order; guard unmatched points below.
        first_fg = is_foreground.long().argmax(dim=0)
        has_fg = is_foreground.any(dim=0)  # (N_point)
        query_prompts = [
            query_list[first_fg[i].item()] if has_fg[i] else "unmatched"
            for i in range(num_points)
        ]
    else:
        query_prompts = ["unmatched"] * num_points

    data = {
        'features': query_prompts,
        'points': points_xyz.detach().cpu(),  # (N, 3)
        'covariances': covariances.detach().cpu(),  # (N, 3, 3)
        'opacities': opacities.detach().cpu(),  # (N,) or (N, 1)
        'colors': colors.detach().cpu(),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)

    log(f"Gaussian features saved to: {output_path} ({num_points} points, {len(query_list)} queries)")


@torch.no_grad()
def select_gaussians_by_text(
    text_encoder,
    multi_lvl_cluster_semantics: list,
    multi_lvl_cluster_semantic_weights: list,
    multi_lvl_point_labels: list,
    num_per_cluster_semantics: int,
    prompt: str = "frog cup",
    spatial_outlier_std: float = 0.5,
    feature_gaussians=None,
    scene_gaussians=None,
    gaussian_accumulator: Optional[dict] = None,
    segment_cosine_thresh: float = 0.8,
    logger=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match a text query to cluster descriptors across feature levels.

    Return the query selection mask and background exclusion mask."""
    log = logger.log if logger else print
    assert feature_gaussians is not None and scene_gaussians is not None


    query_emb = text_encoder.encode(
        prompt,
        convert_to_numpy=True,
        show_progress_bar=False,
    )  # (D,)
    if query_emb.ndim == 1:
        query_emb = query_emb[None, :]
    query_emb = torch.from_numpy(query_emb).float().cuda()  # (1, D)
    query_emb = torch.nn.functional.normalize(query_emb, dim=-1, p=2)[0]


    lvl_best_cluster_ids: List[int] = []
    lvl_best_cluster_scores: List[float] = []
    lvl_cluster_cosine_sims: List[torch.Tensor] = []


    num_lvls = len(multi_lvl_cluster_semantics)
    lvl_indices = list(range(num_lvls))
    for lvl in lvl_indices:
        cluster_semantics = multi_lvl_cluster_semantics[lvl].cuda()  # (N_desc, D)
        cluster_semantics_weights = multi_lvl_cluster_semantic_weights[lvl].clone().cuda() # (N_desc)


        cluster_semantics_norm = torch.nn.functional.normalize(cluster_semantics, dim=-1, p=2)


        cluster_sem_query_rel = torch.matmul(cluster_semantics_norm, query_emb)  # (N_desc,)

        weighted_cluster_query_scores = (cluster_sem_query_rel * cluster_semantics_weights).reshape([-1, num_per_cluster_semantics]) # (N_obj, N_desc_per_obj)


        weighted_cluster_query_scores, best_cluster_semantic_idx = weighted_cluster_query_scores.max(dim=1)  # (N_obj,), (N_obj,)

        best_cluster_score, best_cluster_id = weighted_cluster_query_scores.max(dim=0)
        lvl_best_cluster_ids.append(best_cluster_id.item())
        lvl_best_cluster_scores.append(best_cluster_score.item())



        rel_score = cluster_sem_query_rel.reshape([-1, num_per_cluster_semantics])
        cluster_indices = torch.arange(rel_score.shape[0], device=rel_score.device)  # (N_obj)
        best_des_rel_score = rel_score[cluster_indices, best_cluster_semantic_idx]  # (N_obj)

        lvl_cluster_cosine_sims.append(best_des_rel_score)


    selected_idx = int(np.argmax(lvl_best_cluster_scores))

    cluster_cosine_sims = lvl_cluster_cosine_sims[selected_idx]

    valid_cluster_mask = cluster_cosine_sims > segment_cosine_thresh
    valid_cluster_ids = torch.where(valid_cluster_mask)[0].cpu().numpy().tolist()

    if len(valid_cluster_ids) == 0:
        log(f"Warning: no cluster exceeds similarity {segment_cosine_thresh} for query {prompt!r}; using the best match.")
        valid_cluster_ids = [lvl_best_cluster_ids[selected_idx]]


    point_labels = multi_lvl_point_labels[selected_idx]
    points_xyz = scene_gaussians.get_xyz  # (N_point, 3)
    if points_xyz.device != point_labels.device:
        points_xyz = points_xyz.to(point_labels.device)


    selected_mask = torch.zeros_like(point_labels, dtype=torch.bool)
    for cluster_id in valid_cluster_ids:
        point_mask = (point_labels == cluster_id)
        if point_mask.any():
            selected_xyz = points_xyz[point_mask]
            if selected_xyz.shape[0] > 2:
                centroid = selected_xyz.mean(dim=0)
                dists = torch.norm(selected_xyz - centroid, dim=1)
                dist_mean = dists.mean()
                dist_std = dists.std()
                dist_thresh = dist_mean + spatial_outlier_std * (dist_std + 1e-6)

                keep_point = torch.zeros_like(point_mask, dtype=torch.bool)
                keep_point[point_mask] = dists <= dist_thresh
                selected_mask = selected_mask | keep_point
            else:
                selected_mask = selected_mask | point_mask
        else:
            selected_mask = selected_mask | point_mask


    gaussian_selection_mask = selected_mask.float().unsqueeze(-1)  # (N_point, 1), CUDA tensor

    background_mask = (point_labels != -1).unsqueeze(-1).float()  # (N_point, 1), CUDA tensor
    background_mask = (background_mask[:, 0] <= 0.0)


    if gaussian_accumulator is not None:
        filtered_point_labels = point_labels.clone()
        filtered_point_labels[~selected_mask] = -1
        gaussian_accumulator.setdefault('query_list', []).append(prompt)
        gaussian_accumulator.setdefault('query_point_labels', []).append(
            filtered_point_labels.detach().cpu()
        )

    return gaussian_selection_mask, background_mask

def render_query_mask(target_view, scene_gaussians, pipeline_args, gaussian_selection_mask,
                           background_mask):
    """Render the query selection mask while excluding background Gaussians."""
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    render_res = render(
        target_view,
        scene_gaussians,
        pipeline_args,
        background,
        override_color=gaussian_selection_mask.repeat(1, 3),
        filtered_mask=background_mask,
    )["render"]
    return render_res


def calculate_iou(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    """Return intersection over union for binary masks."""
    intersection = np.sum(np.logical_and(mask_gt, mask_pred))
    union = np.sum(np.logical_or(mask_gt, mask_pred))
    iou = intersection / union if union > 0 else 0.0
    return float(iou)


def calculate_foreground_recall(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    """Return foreground recall (TP / ground-truth positives), or NaN for an empty target."""

    gt_tensor = torch.from_numpy(mask_gt).long()
    pred_tensor = torch.from_numpy(mask_pred).long()

    correct = torch.sum((gt_tensor == pred_tensor) & (gt_tensor == 1.0)).item()
    total = torch.sum(gt_tensor == 1.0).item()


    if total > 0:
        class_acc = correct / total
        return float(class_acc)
    else:
        return float('nan')


def calculate_localization_accuracy(
    rendered_mask_channels: torch.Tensor,
    mask_gt: np.ndarray,
    scale: int = 30
) -> Tuple[float, np.ndarray, int, np.ndarray]:
    """Locate peak responses after smoothing and test whether they hit the target.

    Return accuracy, (x, y) coordinates, the selected channel, and filtered maps."""
    num_channels, h, w = rendered_mask_channels.shape


    if mask_gt.shape[0] != h or mask_gt.shape[1] != w:
        mask_gt = cv2.resize(
            mask_gt,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)


    kernel = np.ones((scale, scale)) / (scale ** 2)
    np_relev = rendered_mask_channels.detach().cpu().numpy()
    avg_filtered = cv2.filter2D(np_relev.transpose(1, 2, 0), -1, kernel)

    channel_scores = np.zeros((num_channels,))
    channel_coordinates = []
    for i in range(num_channels):
        score = avg_filtered[..., i].max()
        coord = np.nonzero(avg_filtered[..., i] == score)
        channel_scores[i] = score

        channel_coordinates.append(np.asarray(coord).transpose(1, 0)[..., ::-1])


    selected_channel = np.argmax(channel_scores)
    coord_final = channel_coordinates[selected_channel]


    loc_acc = 0.0
    for coord_list in coord_final:
        x, y = int(coord_list[0]), int(coord_list[1])

        if 0 <= x < w and 0 <= y < h:
            if mask_gt[y, x] == 1:
                loc_acc = 1.0
                break

    return loc_acc, coord_final, selected_channel, avg_filtered


def save_localization_visualization(
    rendered_mask_channels: torch.Tensor,
    image: torch.Tensor,
    coord_final: np.ndarray,
    selected_channel: int,
    avg_filtered: np.ndarray,
    mask_gt: np.ndarray,
    save_path: Path,
    thresh: float = 0.5,
):
    """Save response maps with target contours and predicted locations."""

    avg_filtered_torch = torch.from_numpy(avg_filtered[..., selected_channel]).unsqueeze(-1).to(rendered_mask_channels.device)
    select_output = rendered_mask_channels[selected_channel].unsqueeze(-1)
    torch_relev = 0.5 * (avg_filtered_torch + select_output)


    p_i = torch.clip(torch_relev - thresh, 0, 1)
    p_i_normalized = p_i / (p_i.max() + 1e-6)
    cmap = plt.get_cmap("turbo")
    p_i_np = p_i_normalized.detach().cpu().numpy().squeeze()
    valid_composited = cmap(p_i_np)[..., :3]


    mask = (torch_relev <= thresh).squeeze().cpu().numpy()
    image_np = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else image
    valid_composited[mask, :] = image_np[mask, :] * 0.3


    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(valid_composited)
    ax.axis('off')


    mask_contours, _ = cv2.findContours(mask_gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    has_label = False
    for i, contour in enumerate(mask_contours):
        label = 'GT Mask' if i == 0 else ''
        ax.plot(contour[:, 0, 0], contour[:, 0, 1], 'g-', linewidth=2, label=label)
        if i == 0:
            has_label = True


    if len(coord_final) > 0:
        ax.scatter(coord_final[:, 0], coord_final[:, 1], color='firebrick', marker='o',
                  s=200, edgecolor='black', linewidth=2.5, alpha=1, label='Max Activation Point')
        has_label = True


    if has_label:
        ax.legend(loc='upper right', fontsize=10)

    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=200)
    plt.close()


def load_mask_annotations(mask_dir: Path) -> Tuple[Dict, List]:
    """Load per-view object masks and their query names.

    Return annotations and view identifiers."""
    mask_paths = []
    eval_index_list = []
    prompts = []
    frame_names = []

    for frame_id in os.listdir(mask_dir):
        mask_paths_one_frame = []
        if 'txt' in frame_id:
            continue
        mask_dir_one_frame = os.path.join(mask_dir, frame_id)
        eval_index_list.append(frame_id)

        for obj in os.listdir(mask_dir_one_frame):
            mask_path = os.path.join(mask_dir_one_frame, obj)
            mask_paths_one_frame.append(mask_path)

        mask_paths.append(mask_paths_one_frame)
        prompts.append(list(map(lambda x: os.path.split(x)[-1].split('.')[0], mask_paths_one_frame)))

    annotations = {}
    for i, idx in enumerate(eval_index_list):
        frame_names.append(idx)
        frame_annotations = defaultdict(dict)
        prompts_one_frame = prompts[i]
        mask_paths_one_frame = mask_paths[i]
        for j, prompt in enumerate(prompts_one_frame):
            mask_gt = cv2.imread(mask_paths_one_frame[j])
            mask_gt[mask_gt == 255] = 1
            mask_gt = mask_gt[:, :, 0]
            frame_annotations[prompt]['mask'] = mask_gt

        annotations[f'{idx}'] = frame_annotations

    return annotations, frame_names


def evaluate_and_save_query_masks(
    rendered_masks: torch.Tensor,
    image: torch.Tensor,
    frame_annotations: Dict,
    thresh: float = 0.5,
    output_dir: Optional[Path] = None,
    show_results: bool = True
) -> Tuple[List[float], List[float], List[float]]:
    """Evaluate rendered query masks and optionally save masks and localization plots.

    rendered_masks has shape (channels, queries, height, width). Return per-query
    IoU, foreground recall, and localization accuracy."""
    num_channels, num_queries, h, w = rendered_masks.shape
    num_channels = 1  # Evaluate the first RGB channel; the rendered mask is repeated across RGB.
    positives = list(frame_annotations.keys())

    query_ious, query_recalls, query_localization_accuracies = [], [], []


    if image.shape[0] != h or image.shape[1] != w:
        image = torch.nn.functional.interpolate(
            image.permute(2, 0, 1).unsqueeze(0),  # (1, 3, H, W)
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)  # (h, w, 3)


    for k in range(num_queries):

        if positives[k] == "background":
            continue

        channel_ious = np.zeros(num_channels)
        channel_recalls = np.zeros(num_channels)
        channel_masks = np.zeros((num_channels, h, w))


        mask_gt = frame_annotations[positives[k]]['mask'].astype(np.uint8)
        if mask_gt.shape[0] != h or mask_gt.shape[1] != w:
            mask_gt = cv2.resize(
                mask_gt,
                (w, h),  # (width, height)
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)


        for i in range(num_channels):

            output = rendered_masks[i][k]
            mask_pred = (output.detach().cpu().numpy() > thresh).astype(np.uint8)
            mask_pred = smooth_mask(mask_pred)
            channel_masks[i] = mask_pred


            channel_ious[i] = calculate_iou(mask_gt, mask_pred)


            channel_recalls[i] = calculate_foreground_recall(mask_gt, mask_pred)


        channel_scores = torch.zeros((num_channels,), device=rendered_masks.device)
        for i in range(num_channels):
            score = rendered_masks[i, k].max()
            channel_scores[i] = score
        evaluation_channel = torch.argmax(channel_scores)


        rendered_mask_channels = rendered_masks[:, k]  # (channels, height, width)
        loc_acc, coord_final, selected_channel, avg_filtered = calculate_localization_accuracy(rendered_mask_channels, mask_gt)


        query_ious.append(channel_ious[evaluation_channel])
        query_recalls.append(channel_recalls[evaluation_channel])
        query_localization_accuracies.append(loc_acc)


        if show_results and output_dir is not None:
            save_path = output_dir / f'chosen_{positives[k]}.png'
            vis_mask_save(channel_masks[evaluation_channel], save_path)


            output_path_loca = output_dir / 'localization'
            loca_save_path = output_path_loca / f'{positives[k]}.png'
            save_localization_visualization(
                rendered_mask_channels,
                image,
                coord_final,
                selected_channel,
                avg_filtered,
                mask_gt,
                loca_save_path,
                thresh=thresh,
            )

    return query_ious, query_recalls, query_localization_accuracies



def save_ground_truth_masks(annotations: Dict, output_dir: Path):
    """Save ground-truth masks separately from annotation loading."""
    for frame_name, objects in annotations.items():
        for query_text, annotation in objects.items():
            vis_mask_save(annotation["mask"], output_dir / "gt" / frame_name / f"{query_text}.jpg")
