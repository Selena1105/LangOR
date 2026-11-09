"""Aggregate mask embeddings into multi-view instance descriptors."""

import torch
from typing import List, Tuple
from inference_config import InferenceConfig


def collect_mask_embeddings(scene, config: InferenceConfig) -> List[torch.Tensor]:
    """Collect mask embeddings from paired training views for each level."""
    multi_lvl_mask_semantics = []

    with torch.no_grad():
        for lvl in range(config.num_lvl):
            mask_semantics = []
            for view in scene.getSemanticTrainCameras():
                torch.cuda.empty_cache()
                f = view.original_features.view(-1, config.text_embedding_dim)


                # Mask IDs are contiguous across levels in the stored embeddings.
                lvl_to_mask_id = []
                lastone = -1

                for i in range(0, 4) if config.num_lvl == 3 else range(0, 1):
                    tm = view.original_masks[i]
                    curone = tm.max().int().item()
                    lvl_to_mask_id.append([lastone + 1, curone])
                    lastone = curone


                if config.num_lvl > 1:
                    interval = lvl_to_mask_id[lvl + 1]
                    f = f[interval[0]:interval[1] + 1, :]
                mask_semantics.append(f)


            multi_lvl_mask_semantics.append(torch.cat(mask_semantics, dim=0))

    return multi_lvl_mask_semantics


def build_instance_descriptors(
    multi_lvl_cluster_to_masks: List[List[torch.Tensor]],
    multi_lvl_mask_semantics: List[torch.Tensor],
    multi_lvl_mask_filter: List[torch.Tensor],
    config: InferenceConfig,
    logger=None
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Aggregate multi-view mask embeddings into instance descriptors.

    Support mean pooling (avg) and correlation-based selection (rule).
    Return per-level descriptors and descriptor weights."""

    multi_lvl_cluster_semantics = []
    multi_lvl_cluster_semantic_weights = []

    with torch.no_grad():
        for lvl in range(config.num_lvl):
            cluster_to_masks = multi_lvl_cluster_to_masks[lvl]

            feats = multi_lvl_mask_semantics[lvl]
            mask = multi_lvl_mask_filter[lvl]
            if not torch.is_tensor(mask):
                mask = torch.as_tensor(mask, device=feats.device)
            else:
                mask = mask.to(feats.device)
            mask_semantics = feats[mask]


            cluster_semantics = []
            cluster_weights = []


            cluster_info = []
            determined_descriptors = []

            for idx, clus in enumerate(cluster_to_masks): # cluster_to_mask [num_cluster, num_masks-4]

                clus = clus.to(mask_semantics.device)

                if clus.nonzero().shape[0] == 0:

                    cluster_info.append((idx, clus, 0, None, None))
                else:

                    clus_feats = torch.nn.functional.normalize(
                        mask_semantics[clus.nonzero()].squeeze(1), dim=-1, p=2
                    ).cuda()

                    avg_clus_feats = clus_feats.mean(dim=0)
                    avg_clus_feats = torch.nn.functional.normalize(avg_clus_feats, dim=-1, p=2)
                    num_feats = clus_feats.shape[0]
                    cluster_info.append((idx, clus, num_feats, clus_feats, avg_clus_feats))


            if config.descriptor_mode == "rule":

                for idx, clus, num_feats, clus_feats, avg_clus_feats in cluster_info:
                    if num_feats > 2:

                        correlations_with_avg = torch.einsum('nc,c->n', clus_feats, avg_clus_feats)


                        sorted_indices = torch.argsort(correlations_with_avg, descending=True)
                        sorted_feats = clus_feats[sorted_indices]


                        pairwise_correlations = torch.einsum('nc,mc->nm', sorted_feats, sorted_feats)

                        mask = torch.eye(num_feats, device=pairwise_correlations.device, dtype=torch.bool)
                        pairwise_correlations_masked = pairwise_correlations.clone()
                        pairwise_correlations_masked[mask] = 0
                        avg_pairwise_correlations = pairwise_correlations_masked.sum(dim=1) / (num_feats - 1)


                        best_idx = torch.argmax(avg_pairwise_correlations)
                        descriptor_feat = sorted_feats[best_idx]
                        weight = 1.0

                        determined_descriptors.append((idx, descriptor_feat, weight))


                for idx, clus, num_feats, clus_feats, avg_clus_feats in cluster_info:
                    if num_feats == 2:

                        if len(determined_descriptors) > 0:

                            determined_feats = torch.stack([desc[1] for desc in determined_descriptors])  # (N_determined, dim)


                            sim_a = torch.einsum('c,nc->n', clus_feats[0], determined_feats).max()
                            sim_b = torch.einsum('c,nc->n', clus_feats[1], determined_feats).max()


                            if sim_a > sim_b and sim_a > 0.8:
                                descriptor_feat = clus_feats[1]
                            elif sim_b > sim_a and sim_b > 0.8:
                                descriptor_feat = clus_feats[0]
                            else:
                                descriptor_feat = (clus_feats[0] + clus_feats[1]) / 2

                        else:
                            descriptor_feat = (clus_feats[0] + clus_feats[1]) / 2

                        descriptor_feat = torch.nn.functional.normalize(descriptor_feat, dim=-1, p=2)
                        weight = 1.0

                        determined_descriptors.append((idx, descriptor_feat, weight))


                for idx, clus, num_feats, clus_feats, avg_clus_feats in cluster_info:
                    if num_feats == 1:
                        descriptor_feat = clus_feats[0]
                        weight = 1.0

                        determined_descriptors.append((idx, descriptor_feat, weight))


                determined_descriptors.sort(key=lambda x: x[0])
                descriptor_dict = {idx: (feat, w) for idx, feat, w in determined_descriptors}


                for idx, clus, num_feats, clus_feats, avg_clus_feats in cluster_info:
                    tmp_features = torch.rand(
                        config.num_per_cluster_semantics, mask_semantics.shape[-1]
                    ).cuda()
                    tmp_weights = torch.zeros(config.num_per_cluster_semantics).cuda()

                    if num_feats == 0:

                        pass
                    elif idx in descriptor_dict:
                        tmp_features[0], weight = descriptor_dict[idx]
                        tmp_weights[0] = 1.0

                    cluster_semantics.append(tmp_features)
                    cluster_weights.append(tmp_weights)

            elif config.descriptor_mode == "avg":

                for idx, clus, num_feats, clus_feats, avg_clus_feats in cluster_info:
                    tmp_features = torch.rand(
                        config.num_per_cluster_semantics, mask_semantics.shape[-1]
                    ).cuda()
                    tmp_weights = torch.zeros(config.num_per_cluster_semantics).cuda()

                    if num_feats == 0:

                        pass
                    else:

                        tmp_features[0] = avg_clus_feats
                        tmp_weights[0] = 1.0

                    cluster_semantics.append(tmp_features)
                    cluster_weights.append(tmp_weights)

            else:
                raise ValueError(f"Unsupported descriptor mode: {config.descriptor_mode}")


            cluster_semantics = torch.cat(cluster_semantics, 0)
            cluster_weights = torch.cat(cluster_weights, 0)

            multi_lvl_cluster_semantics.append(cluster_semantics)
            multi_lvl_cluster_semantic_weights.append(cluster_weights)

    return multi_lvl_cluster_semantics, multi_lvl_cluster_semantic_weights

