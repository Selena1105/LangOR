import torch
import numpy as np
import hdbscan
from copy import deepcopy
from typing import List, Tuple, Dict

from gaussian_renderer import render_contrastive_feature
from inference_config import InferenceConfig
from inference_utils import normalize_multi_level_features


def _sample_pixels_uniform(
    num_foreground_pixels: int,
    max_pixels_per_view: int,
    pixel_mask_indices_pre: np.ndarray,
    valid_pixel_mask_pre: np.ndarray
) -> np.ndarray:
    """Sample foreground pixels uniformly using the common sampling interface."""
    return np.random.choice(num_foreground_pixels, max_pixels_per_view, replace=False)


def _sample_pixels_by_mask_area(
    num_foreground_pixels: int,
    max_pixels_per_view: int,
    pixel_mask_indices_pre: np.ndarray,
    valid_pixel_mask_pre: np.ndarray,
    last_mask_idx: int,
    foreground_coords: np.ndarray
) -> np.ndarray:
    """Sample foreground pixels with mask-area weights."""
    pixel_weights = np.zeros(num_foreground_pixels, dtype=np.float64)
    for mask_idx in range(last_mask_idx):
        mask_pixels = (pixel_mask_indices_pre == mask_idx) & valid_pixel_mask_pre
        area = mask_pixels.sum()
        if area > 0:
            mask_pixel_indices = np.where(mask_pixels)[0]
            pixel_weights[mask_pixel_indices] = 1.0 / area

    if pixel_weights.sum() > 0:
        pixel_weights = pixel_weights / pixel_weights.sum()
    else:
        pixel_weights = np.ones(num_foreground_pixels) / num_foreground_pixels

    return np.random.choice(
        num_foreground_pixels,
        max_pixels_per_view,
        replace=False,
        p=pixel_weights
    )

def _sample_pixels_by_mask_area_and_center(
    num_foreground_pixels: int,
    max_pixels_per_view: int,
    pixel_mask_indices_pre: np.ndarray,
    valid_pixel_mask_pre: np.ndarray,
    last_mask_idx: int,
    foreground_coords: np.ndarray,
    sigma: float
) -> np.ndarray:
    """Sample pixels using mask-area and distance-to-center weights."""
    pixel_weights = np.zeros(num_foreground_pixels, dtype=np.float64)

    for mask_idx in range(last_mask_idx):
        mask_pixels = (pixel_mask_indices_pre == mask_idx) & valid_pixel_mask_pre
        area = mask_pixels.sum()
        if area == 0:
            continue

        mask_pixel_indices = np.where(mask_pixels)[0]
        mask_coords = foreground_coords[mask_pixel_indices]

        center_y = mask_coords[:, 0].mean()
        center_x = mask_coords[:, 1].mean()

        distances = np.sqrt(
            (mask_coords[:, 0] - center_y) ** 2 +
            (mask_coords[:, 1] - center_x) ** 2
        )

        normalized_distances = distances / (distances.max() + 1e-6)
        distance_weights = np.exp(-normalized_distances ** 2 / (2 * sigma ** 2))

        pixel_weights[mask_pixel_indices] = (1.0 / area) * distance_weights

    if pixel_weights.sum() > 0:
        pixel_weights = pixel_weights / pixel_weights.sum()
    else:
        pixel_weights = np.ones(num_foreground_pixels) / num_foreground_pixels

    return np.random.choice(
        num_foreground_pixels,
        max_pixels_per_view,
        replace=False,
        p=pixel_weights
    )


def sample_pixels(
    num_foreground_pixels: int,
    max_pixels_per_view: int,
    pixel_mask_indices_pre: np.ndarray,
    valid_pixel_mask_pre: np.ndarray,
    last_mask_idx: int,
    foreground_coords: np.ndarray,
    sampling_strategy: int = 2,
    sigma: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample foreground indices using uniform, area, or area-and-center weights.

    Return sampled offsets and their corresponding valid pixel indices."""
    if num_foreground_pixels <= max_pixels_per_view:
        return np.arange(num_foreground_pixels), valid_pixel_mask_pre[:num_foreground_pixels]

    if sampling_strategy == 0:
        sample_indices = _sample_pixels_uniform(
            num_foreground_pixels, max_pixels_per_view,
            pixel_mask_indices_pre, valid_pixel_mask_pre
        )
    elif sampling_strategy == 1:
        sample_indices = _sample_pixels_by_mask_area(
            num_foreground_pixels, max_pixels_per_view,
            pixel_mask_indices_pre, valid_pixel_mask_pre,
            last_mask_idx, foreground_coords
        )
    else:
        sample_indices = _sample_pixels_by_mask_area_and_center(
            num_foreground_pixels, max_pixels_per_view,
            pixel_mask_indices_pre, valid_pixel_mask_pre,
            last_mask_idx, foreground_coords, sigma
        )

    return sample_indices, valid_pixel_mask_pre[sample_indices]


def collect_foreground_features_from_view(
    view,
    tview,
    view_idx: int,
    feature_gaussians,
    pipeline_args,
    config: InferenceConfig,
    multi_lvl_mask_info: List[List[dict]],
    sampling_strategy: int = 2
) -> Dict[int, Dict]:
    """Collect sampled foreground features and mask metadata for each level."""
    max_pixels_per_view = config.max_pixels_per_view


    bg_color = [0 for _ in range(config.feature_dim)]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    rendered_feature = render_contrastive_feature(
        view, feature_gaussians, pipeline_args, background,
        norm_point_features=True, multi_lvl_norm=True,
        multi_lvl_dim=config.multi_lvl_dim, smooth_type=None
    )['render'].permute([1, 2, 0])  # (H, W, C)

    multi_lvl_features = normalize_multi_level_features(rendered_feature, config.multi_lvl_dim)
    sam_masks = view.original_masks.cuda()

    view_data = {}


    for lvl in range(1, config.num_lvl + 1) if config.num_lvl > 1 else [0]:
        tmp_masks = sam_masks[lvl].clone()


        smallest_index = sam_masks[lvl - 1].int().max() + 1 if lvl > 0 else 0
        tmp_masks -= smallest_index - 1
        tmp_masks[tmp_masks < 0] = 0


        tmp_lvl_masks = torch.nn.functional.one_hot(
            tmp_masks.long(),
            num_classes=tmp_masks.max().int().item() + 1
        )[:, :, 1:].float()

        if tmp_lvl_masks.shape[2] == 0:
            continue


        last_mask_idx = tmp_lvl_masks.shape[2] - 1


        mask_global_idx_start = len(multi_lvl_mask_info[lvl])
        for mask_idx in range(last_mask_idx):
            mask_info = {
                'view_idx': view_idx,
                'image_name': tview.image_name,
                'mask_idx': mask_idx,
            }
            multi_lvl_mask_info[lvl].append(mask_info)


        mask_global_idx_map = {i: mask_global_idx_start + i for i in range(last_mask_idx)}

        foreground_mask = (tmp_lvl_masks.sum(dim=2) > 0) & (tmp_lvl_masks[:, :, last_mask_idx] == 0)


        current_feature = multi_lvl_features[lvl]
        if current_feature.shape[:2] != foreground_mask.shape[:2]:
            current_feature = torch.nn.functional.interpolate(
                current_feature.permute([2, 0, 1]).unsqueeze(0),
                foreground_mask.shape[:2], mode='bilinear'
            ).squeeze(0).permute([1, 2, 0])

        foreground_features = current_feature[foreground_mask]
        if foreground_features.shape[0] == 0:
            continue


        foreground_mask_np = foreground_mask.cpu().numpy()
        foreground_coords = np.column_stack(np.where(foreground_mask_np))
        num_foreground_pixels = foreground_features.shape[0]


        y_coords = foreground_coords[:, 0]
        x_coords = foreground_coords[:, 1]


        pixel_mask_values = tmp_lvl_masks[y_coords, x_coords, :last_mask_idx]


        pixel_mask_idx_tensor = torch.argmax(pixel_mask_values, dim=1)


        mask_belongs = pixel_mask_values.max(dim=1)[0] > 0.5
        pixel_mask_idx_tensor[~mask_belongs] = -1


        pixel_mask_idx_np = pixel_mask_idx_tensor.cpu().numpy()
        valid_pixel_mask_np = pixel_mask_idx_np >= 0


        sample_indices, sampled_valid_mask = sample_pixels(
            num_foreground_pixels, max_pixels_per_view,
            pixel_mask_idx_np, valid_pixel_mask_np,
            last_mask_idx, foreground_coords, sampling_strategy, config.sigma
        )

        foreground_features = foreground_features[sample_indices]
        foreground_coords = foreground_coords[sample_indices]
        pixel_mask_idx_np = pixel_mask_idx_np[sample_indices]
        valid_pixel_mask_np = sampled_valid_mask

        if valid_pixel_mask_np.sum() == 0:
            continue


        foreground_features = foreground_features[valid_pixel_mask_np]
        pixel_mask_idx_np = pixel_mask_idx_np[valid_pixel_mask_np]
        foreground_coords = foreground_coords[valid_pixel_mask_np]


        pixel_global_mask_indices = np.array([mask_global_idx_map[idx] for idx in pixel_mask_idx_np])

        view_data[lvl] = {
            'view_idx': view_idx,
            'image_name': tview.image_name,
            'features': foreground_features.detach().cpu().numpy(),
            'pixel_mask_indices': pixel_global_mask_indices,
            'coords': foreground_coords,
            'mask_shape': foreground_mask.shape,
            'num_masks_in_view': last_mask_idx
        }

    return view_data


def perform_hdbscan_clustering(
    lvl_data: List[Dict],
    config: InferenceConfig,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, np.ndarray]:
    """Cluster sampled multi-view features and assign remaining pixels.

    Return global labels, pixel-to-mask indices, cluster centers, and valid IDs."""
    max_sample_size = config.max_sample_size
    if len(lvl_data) == 0:
        return None, None, None, None


    all_features = np.vstack([vd['features'] for vd in lvl_data])
    all_pixel_mask_indices = np.concatenate([vd['pixel_mask_indices'] for vd in lvl_data])
    num_total = all_features.shape[0]


    if num_total > max_sample_size:
        sample_indices = np.random.choice(num_total, max_sample_size, replace=False)
        sample_features = all_features[sample_indices]
    else:
        sample_indices = np.arange(num_total)
        sample_features = all_features


    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        cluster_selection_epsilon=config.hdbscan_epsilon_base
    )
    sample_labels = clusterer.fit_predict(sample_features)
    unique_ids = np.unique(sample_labels[sample_labels != -1])

    if len(unique_ids) == 0:
        global_labels = np.full(num_total, -1, dtype=sample_labels.dtype)
        cluster_centers = torch.zeros(0, all_features.shape[1], device='cuda')
        valid_ids = np.array([], dtype=np.int32)
    else:

        cluster_centers_list = []
        for cid in unique_ids:
            mask = sample_labels == cid
            center = sample_features[mask].mean(axis=0)
            cluster_centers_list.append(center / (np.linalg.norm(center) + 1e-8))
        cluster_centers = torch.from_numpy(np.array(cluster_centers_list)).float().cuda()
        valid_ids = unique_ids.copy()


        if num_total > max_sample_size:
            remaining_indices = np.setdiff1d(np.arange(num_total), sample_indices)
            remaining_features = all_features[remaining_indices]
            remaining_norm = remaining_features / (np.linalg.norm(remaining_features, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(remaining_norm, cluster_centers.cpu().numpy().T)


            nearest = np.argmax(similarities, axis=1)
            max_sim = np.max(similarities, axis=1)
            remaining_labels = valid_ids[nearest].copy()

            remaining_labels[max_sim < config.remaining_sim_thresh] = -1

            global_labels = np.full(num_total, -1, dtype=sample_labels.dtype)
            global_labels[sample_indices] = sample_labels
            global_labels[remaining_indices] = remaining_labels
        else:
            global_labels = sample_labels

    return global_labels, all_pixel_mask_indices, cluster_centers, valid_ids

# main
def cluster_multiview_features(
    scene,
    feature_gaussians,
    pipeline_args,
    config: InferenceConfig
) -> Tuple[Dict[int, List[Dict]], List[List[dict]],
           List[np.ndarray], List[np.ndarray], List[np.ndarray],
           List[torch.Tensor]]:
    """Lift sampled foreground features into cross-view clusters at each level.

    Collect per-view features, then cluster them jointly with HDBSCAN."""
    sampling_strategy = config.sampling_strategy

    with torch.no_grad():

        all_view_data = {lvl: [] for lvl in range(config.num_lvl)}
        multi_lvl_mask_info = [[] for _ in range(config.num_lvl)]

        for view_idx, tview in enumerate(scene.getSemanticTrainCameras()):
            torch.cuda.empty_cache()
            view = deepcopy(tview)
            view.feature_height, view.feature_width = view.image_height, view.image_width

            view_data = collect_foreground_features_from_view(
                view, tview, view_idx,
                feature_gaussians, pipeline_args, config,
                multi_lvl_mask_info, sampling_strategy
            )


            for lvl in view_data:
                all_view_data[lvl].append(view_data[lvl])


        multi_lvl_global_labels = []
        multi_lvl_pixel_mask_indices = []
        multi_lvl_valid_ids = []
        multi_lvl_cluster_centers = []

        for lvl in range(config.num_lvl):
            if len(all_view_data[lvl]) == 0:
                multi_lvl_global_labels.append(None)
                multi_lvl_pixel_mask_indices.append(None)
                multi_lvl_valid_ids.append(None)
                multi_lvl_cluster_centers.append(None)
                continue

            global_labels, all_pixel_mask_indices, cluster_centers, valid_ids = perform_hdbscan_clustering(
                all_view_data[lvl], config
            )
            multi_lvl_global_labels.append(global_labels)
            multi_lvl_pixel_mask_indices.append(all_pixel_mask_indices)
            multi_lvl_valid_ids.append(valid_ids)
            multi_lvl_cluster_centers.append(cluster_centers)

    return multi_lvl_mask_info, multi_lvl_global_labels, multi_lvl_pixel_mask_indices, multi_lvl_valid_ids, multi_lvl_cluster_centers


