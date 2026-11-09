import torch
import numpy as np
from typing import List, Tuple

from inference_config import InferenceConfig
from inference_utils import normalize_multi_level_features


def vote_cluster_for_masks(
    num_masks: int,
    all_pixel_mask_indices: np.ndarray,
    global_labels: np.ndarray,
    cluster_centers: torch.Tensor,
    valid_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Assign masks by pixel votes and retain the corresponding cluster centers."""
    mask_cluster_labels = np.full(num_masks, -1, dtype=global_labels.dtype)


    for mask_global_idx in range(num_masks):
        mask_pixel_mask = all_pixel_mask_indices == mask_global_idx
        if mask_pixel_mask.sum() == 0:
            continue

        mask_pixel_labels = global_labels[mask_pixel_mask]


        valid_labels = mask_pixel_labels[mask_pixel_labels != -1]
        if len(valid_labels) > 0:
            unique, counts = np.unique(valid_labels, return_counts=True)
            most_common_label = unique[np.argmax(counts)]
            mask_cluster_labels[mask_global_idx] = most_common_label


    unique_cluster_ids = np.unique(mask_cluster_labels[mask_cluster_labels != -1])

    if len(unique_cluster_ids) == 0:
        feature_dim = cluster_centers.shape[1] if len(cluster_centers) > 0 else 0
        filtered_cluster_centers = torch.zeros(0, feature_dim, device='cuda')
        valid_ids = np.array([], dtype=np.int32)
    else:

        mask = np.isin(valid_ids, unique_cluster_ids)
        cluster_indices = np.where(mask)[0]

        if len(cluster_indices) > 0:
            filtered_cluster_centers = cluster_centers[cluster_indices]
            valid_ids = valid_ids[cluster_indices]
        else:
            feature_dim = cluster_centers.shape[1] if len(cluster_centers) > 0 else 0
            filtered_cluster_centers = torch.zeros(0, feature_dim, device='cuda')
            valid_ids = np.array([], dtype=np.int32)

    return mask_cluster_labels, unique_cluster_ids, filtered_cluster_centers


def build_semantic_mask_filter(
    scene,
    config: InferenceConfig,
    lvl: int
) -> torch.Tensor:
    """Build a per-level mask filter aligned with semantic embeddings."""
    view_feature_counts = []

    for view in scene.getSemanticTrainCameras():
        if config.num_lvl > 1:

            lvl_to_mask_id = []
            lastone = -1
            for i in range(config.num_lvl + 1):
                curone = view.original_masks[i].max().int().item()
                lvl_to_mask_id.append([lastone + 1, curone])
                lastone = curone

            interval = lvl_to_mask_id[lvl + 1]
            num_features_in_view = interval[1] - interval[0] + 1
        else:
            num_features_in_view = view.original_masks[0].max().int().item() + 1

        view_feature_counts.append(num_features_in_view)

    total_feature_count = sum(view_feature_counts)
    mask_filter = torch.ones(total_feature_count, dtype=torch.bool, device='cuda')


    current_idx = 0
    for num_features_in_view in view_feature_counts:
        if num_features_in_view > 0:
            mask_filter[current_idx + num_features_in_view - 1] = False
        current_idx += num_features_in_view

    return mask_filter


def match_masks_to_clusters(
    multi_lvl_mask_info: List[List[dict]],
    multi_lvl_global_labels: List[np.ndarray],
    multi_lvl_pixel_mask_indices: List[np.ndarray],
    multi_lvl_valid_ids: List[np.ndarray],
    multi_lvl_cluster_centers: List[torch.Tensor],
    scene,
    config: InferenceConfig,
    logger=None
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]]]:
    """Match multi-view masks to clusters by majority vote at each level."""
    multi_lvl_cluster_to_masks = []
    multi_lvl_mask_filter = []
    multi_lvl_filtered_centers = []

    for lvl in range(config.num_lvl):

        if multi_lvl_global_labels[lvl] is None:
            feature_dim = sum(config.multi_lvl_dim[:lvl+1]) if lvl < len(config.multi_lvl_dim) else config.multi_lvl_dim[0]
            multi_lvl_cluster_to_masks.append([])
            multi_lvl_mask_filter.append(torch.zeros(0, dtype=torch.bool, device='cuda'))
            multi_lvl_filtered_centers.append(torch.zeros(0, feature_dim, device='cuda'))
            continue

        global_labels = multi_lvl_global_labels[lvl]
        all_pixel_mask_indices = multi_lvl_pixel_mask_indices[lvl]
        valid_ids = multi_lvl_valid_ids[lvl]
        cluster_centers = multi_lvl_cluster_centers[lvl]


        num_masks = len(multi_lvl_mask_info[lvl])
        mask_cluster_labels, unique_cluster_ids, filtered_cluster_centers = vote_cluster_for_masks(
            num_masks, all_pixel_mask_indices, global_labels, cluster_centers, valid_ids
        )

        multi_lvl_filtered_centers.append(filtered_cluster_centers)


        mask_cluster_labels_tensor = torch.from_numpy(mask_cluster_labels).cuda()
        cluster_to_masks = []
        for cluster_id in unique_cluster_ids:
            mask_bool = mask_cluster_labels_tensor == cluster_id
            cluster_to_masks.append(mask_bool)
        multi_lvl_cluster_to_masks.append(cluster_to_masks)


        mask_filter = build_semantic_mask_filter(scene, config, lvl)
        multi_lvl_mask_filter.append(mask_filter)


    return multi_lvl_mask_filter, multi_lvl_cluster_to_masks, multi_lvl_filtered_centers


def assign_gaussians_to_clusters(
    feature_gaussians,
    multi_lvl_cluster_centers: List[torch.Tensor],
    config: InferenceConfig,
) -> List[torch.Tensor]:
    """Score Gaussians against multi-level cluster centers and assign labels.

    Points below the similarity threshold are treated as background."""

    with torch.no_grad():
        point_features = feature_gaussians.get_point_features


        multi_lvl_point_features = normalize_multi_level_features(
            point_features, config.multi_lvl_dim
        )


        multi_lvl_point_labels = []

        for lvl in range(config.num_lvl):
            gs_feature = multi_lvl_point_features[lvl]  # (N_points, dim)
            cluster_centers = multi_lvl_cluster_centers[lvl]  # (N_clusters, dim)

            gs_instance_score = torch.einsum('nc,bc->bn', cluster_centers, gs_feature)



            max_gs_instance_score, point_labels = torch.max(gs_instance_score, dim=1)
            is_background = max_gs_instance_score < config.background_thresh
            point_labels[is_background] = -1


            total_points = point_labels.shape[0]
            threshold_ratio = config.large_cluster_thresh
            threshold_count = total_points * threshold_ratio
            unique_labels, counts = torch.unique(point_labels, return_counts=True)

            large_clusters = []
            for label, count in zip(unique_labels.tolist(), counts.tolist()):
                if label != -1 and count > threshold_count:
                    large_clusters.append(label)


            if large_clusters:
                for cluster_id in large_clusters:
                    point_labels[point_labels == cluster_id] = -1

            multi_lvl_point_labels.append(point_labels)


    return multi_lvl_point_labels
