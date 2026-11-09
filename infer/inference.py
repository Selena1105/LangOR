"""Run legacy multi-level semantic segmentation and evaluation."""
import os
import json
import torch
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from sentence_transformers import SentenceTransformer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from scene import Scene, GaussianModel, FeatureGaussianModel

from inference_config import InferenceConfig, add_inference_args, load_cfg_args
from inference_utils import Logger
from semantics import collect_mask_embeddings, build_instance_descriptors
from lifting import cluster_multiview_features
from matching import match_masks_to_clusters, assign_gaussians_to_clusters
from inference_segmentation import select_gaussians_by_text, render_query_mask, load_mask_annotations, save_ground_truth_masks, evaluate_and_save_query_masks, export_labeled_gaussians


def build_image_metrics(frame_name, foreground_queries, query_ious, query_recalls, query_localization_accuracies):
    mean_iou = sum(query_ious) / len(query_ious) if query_ious else 0.0

    valid_macc = [m for m in query_recalls if not np.isnan(m)]
    mean_macc = sum(valid_macc) / len(valid_macc) if valid_macc else float('nan')

    mean_loc_acc = sum(query_localization_accuracies) / len(query_localization_accuracies) if query_localization_accuracies else 0.0

    image_metrics = {
        "image_index": frame_name,
        "per_object_iou": {},
        "mean_iou": float(mean_iou),
        "per_object_macc": {},
        "mean_macc": float(mean_macc) if not np.isnan(mean_macc) else None,
        "per_object_loc_acc": {},
        "mean_loc_acc": float(mean_loc_acc)
    }
    for obj_name, iou_val, macc_val, loc_acc_val in zip(foreground_queries, query_ious, query_recalls, query_localization_accuracies):
        image_metrics["per_object_iou"][obj_name] = float(iou_val)
        image_metrics["per_object_macc"][obj_name] = float(macc_val) if not np.isnan(macc_val) else None
        image_metrics["per_object_loc_acc"][obj_name] = float(loc_acc_val)

    return image_metrics


def load_models(config, scene_config):
    """Load scene and feature Gaussians at the configured iterations."""
    scene_gaussians = GaussianModel(scene_config.sh_degree)


    feature_gaussians = FeatureGaussianModel(config.feature_dim)


    scene = Scene(
        scene_config, scene_gaussians, feature_gaussians,
        load_iteration=-1,
        feature_load_iteration=config.feature_gaussian_iteration,
        shuffle=False, mode='eval',
        target='contrastive_feature'
    )

    return scene, scene_gaussians, feature_gaussians

def main():
    parser = ArgumentParser(description="Legacy semantic segmentation inference")
    add_inference_args(parser)
    args = parser.parse_args()
    config = InferenceConfig.from_args(args)


    output_dir = Path(config.output_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = Logger(output_dir / "experiment_log.txt")


    cfg_args = load_cfg_args(config)
    scene_config = config.build_scene_config(cfg_args)
    pipeline_config = config.build_pipeline_config(cfg_args)


    scene, scene_gaussians, feature_gaussians = load_models(config, scene_config)


    multi_lvl_mask_info, multi_lvl_global_labels, multi_lvl_pixel_mask_indices, multi_lvl_valid_ids, multi_lvl_cluster_centers =\
        cluster_multiview_features(scene, feature_gaussians, pipeline_config, config)

    # instance-mask matching
    multi_lvl_mask_filter, multi_lvl_cluster_to_masks, multi_lvl_cluster_centers =\
        match_masks_to_clusters(multi_lvl_mask_info, multi_lvl_global_labels, multi_lvl_pixel_mask_indices, multi_lvl_valid_ids,
                           multi_lvl_cluster_centers, scene, config, logger)

    # instance-gs matching
    multi_lvl_point_labels = assign_gaussians_to_clusters(feature_gaussians, multi_lvl_cluster_centers, config)


    multi_lvl_mask_semantics = collect_mask_embeddings(scene, config)


    multi_lvl_cluster_semantics, multi_lvl_cluster_semantic_weights =\
        build_instance_descriptors(multi_lvl_cluster_to_masks, multi_lvl_mask_semantics, multi_lvl_mask_filter, config,logger)


    text_encoder = SentenceTransformer(config.text_encoder_name)
    text_encoder.max_seq_length = config.text_encoder_max_seq_length


    annotations, frame_names = load_mask_annotations(Path(config.mask_folder))
    save_ground_truth_masks(annotations, output_dir)

    all_query_ious, all_query_recalls, all_query_localization_accuracies = [], [], []
    all_image_results = []

    train_views = scene.getTrainCameras()
    test_views = scene.getTestCameras()
    views = train_views + test_views
    frame_to_camera_idx = {}
    for cam_idx, cam in enumerate(views):
        frame_to_camera_idx[cam.image_name] = cam_idx


    gaussian_accumulator = {} if config.gaussian_save_path is not None else None
    for frame_index, frame_name in enumerate(frame_names):
        rendered_masks = []
        query_names = list(annotations[frame_name].keys())
        frame_annotations = annotations[frame_name]


        if frame_name not in frame_to_camera_idx:
            logger.log(f"Warning: skipping frame '{frame_name}'; no matching camera was found.")
            continue
        camera_idx = frame_to_camera_idx[frame_name]
        target_view = views[camera_idx]


        foreground_queries = [key for key in query_names if key != "background"]


        for query_text in query_names:

            if query_text == "background":
                continue


            gaussian_selection_mask, background_mask = select_gaussians_by_text(
                text_encoder,
                multi_lvl_cluster_semantics,
                multi_lvl_cluster_semantic_weights,
                multi_lvl_point_labels,
                num_per_cluster_semantics=config.num_per_cluster_semantics,
                prompt=query_text,
                feature_gaussians=feature_gaussians,
                scene_gaussians=scene_gaussians,
                spatial_outlier_std=config.spatial_outlier_std,
                segment_cosine_thresh=config.segment_cosine_thresh,
                gaussian_accumulator=gaussian_accumulator,
                logger=logger,
            )


            rendered_mask = render_query_mask(
                target_view,
                scene_gaussians,
                pipeline_config,
                gaussian_selection_mask,
                background_mask,
            )


            rendered_masks.append(rendered_mask)

        rendered_mask_stack = torch.stack(rendered_masks, dim=1)

        assert rendered_mask_stack.dim() == 4,\
            f"Expected 4D tensor (channels, queries, height, width), got {rendered_mask_stack.dim()}D tensor with shape {rendered_mask_stack.shape}"


        frame_output_dir = Path(config.output_path) / f"{frame_index + 1:0>5}"
        frame_output_dir.mkdir(exist_ok=True, parents=True)


        # Annotation-only inputs use a black background for overlays.
        any_mask = next(iter(frame_annotations.values()))["mask"]
        h, w = any_mask.shape
        rgb_img_np = np.zeros((h, w, 3), dtype=np.float32)

        rgb_img = torch.from_numpy(rgb_img_np).to("cuda")


        filtered_img_ann = {key: frame_annotations[key] for key in foreground_queries}


        query_ious, query_recalls, query_localization_accuracies = evaluate_and_save_query_masks(
            rendered_mask_stack,
            rgb_img,
            filtered_img_ann,
            thresh=config.mask_thresh,
            output_dir=frame_output_dir,
            show_results=config.show_results,
        )
        all_query_ious.extend(query_ious)
        all_query_recalls.extend(query_recalls)
        all_query_localization_accuracies.extend(query_localization_accuracies)

        image_metrics = build_image_metrics(
            frame_name, foreground_queries, query_ious, query_recalls, query_localization_accuracies
        )
        all_image_results.append(image_metrics)


    if all_image_results:

        scene_results = {
            "scene_output_path": str(config.output_path),
            "total_images": len(all_image_results),
            "images": all_image_results
        }


        output_dir = Path(config.output_path)
        output_dir.mkdir(exist_ok=True, parents=True)
        scene_results_file = output_dir / "iou_results.json"

        with open(scene_results_file, 'w', encoding='utf-8') as f:
            json.dump(scene_results, f, indent=2, ensure_ascii=False)

        logger.log(f"Metrics saved to: {scene_results_file}")


    if config.gaussian_save_path is not None:
        gaussian_save_file = Path(config.gaussian_save_path)
        export_labeled_gaussians(
            scene_gaussians=scene_gaussians,
            accumulator=gaussian_accumulator,
            output_path=gaussian_save_file,
            logger=logger,
        )

    if all_image_results:
        valid_macc = [value for value in all_query_recalls if not np.isnan(value)]
        logger.log(f"Inference complete: {len(all_image_results)} images evaluated.")
        logger.log(
            f"Mean IoU: {np.mean(all_query_ious):.4f} | "
            f"Mean class accuracy: {np.mean(valid_macc) if valid_macc else float('nan'):.4f} | "
            f"Mean localization accuracy: {np.mean(all_query_localization_accuracies):.4f}"
        )
        logger.log(f"Results directory: {output_dir}")
    else:
        logger.log("Warning: no images were evaluated; no segmentation metrics were produced.")
    logger.close()


if __name__ == "__main__":
    main()

