"""Logging, feature normalization, and mask visualization utilities."""
import torch
import numpy as np
from pathlib import Path
import cv2
from datetime import datetime
import re


class Logger:
    """Write messages to the console and a log file."""
    def __init__(self, log_file_path: Path):
        self.log_file_path = log_file_path
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.log_file.write(f"Experiment log - Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write("=" * 80 + "\n\n")
        self.log_file.flush()

        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def _remove_ansi_codes(self, text: str) -> str:
        """Strip ANSI escape sequences before writing to disk."""
        return self.ansi_escape.sub('', text)

    def log(self, *args, **kwargs):
        """Write a message to the console and log file."""

        message = ' '.join(str(arg) for arg in args)
        if kwargs:
            message += ' ' + ' '.join(f'{k}={v}' for k, v in kwargs.items())


        print(*args, **kwargs)


        clean_message = self._remove_ansi_codes(message)
        self.log_file.write(clean_message + '\n')
        self.log_file.flush()

    def close(self):
        """Close the log file."""
        if self.log_file:
            self.log_file.write("\n" + "=" * 80 + "\n")
            self.log_file.close()
            self.log_file = None

    def __del__(self):
        """Close the log file when the logger is released."""
        if hasattr(self, 'log_file') and self.log_file:
            self.close()


def normalize_multi_level_features(features: torch.Tensor, multi_lvl_dim: list) -> list:
    """Normalize feature blocks and return cumulative features in reverse level order."""
    normed_features = []
    pre_dim = 0
    for dim in multi_lvl_dim:
        normed_features.append(
            torch.nn.functional.normalize(features[..., pre_dim:pre_dim + dim], dim=-1))
        pre_dim += dim
    normed_features = torch.cat(normed_features, dim=-1)


    # Accumulate normalized blocks, ordered from finest to coarsest.
    pre_dim = 0
    multi_lvl_features = []
    for i in range(len(multi_lvl_dim)):
        multi_lvl_features.insert(0, torch.nn.functional.normalize(
            normed_features[..., :multi_lvl_dim[i] + pre_dim], dim=-1))
        pre_dim += multi_lvl_dim[i]

    return multi_lvl_features


def smooth_mask(mask: np.ndarray, kernel_size: int = 5, min_area: int = 200, fill_holes: bool = True) -> np.ndarray:
    """Smooth a binary mask, remove small components, and optionally fill holes."""

    if mask.max() > 1:
        mask_binary = (mask > 127).astype(np.uint8)
    else:
        mask_binary = (mask > 0.5).astype(np.uint8)


    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)


    optimized_mask = np.zeros_like(mask_binary)


    areas = stats[:, cv2.CC_STAT_AREA]


    valid_labels = []
    for label_id in range(1, num_labels):
        if areas[label_id] >= min_area:
            valid_labels.append(label_id)


    if len(valid_labels) == 0:

        if num_labels > 1:
            largest_label = np.argmax(areas[1:]) + 1
            optimized_mask[labels == largest_label] = 1
    else:

        for label_id in valid_labels:
            optimized_mask[labels == label_id] = 1


    if fill_holes:

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        optimized_mask = cv2.morphologyEx(optimized_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)


        h, w = optimized_mask.shape
        mask_filled = optimized_mask.copy()


        corner_points = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]
        for x, y in corner_points:
            if mask_filled[y, x] == 0:
                cv2.floodFill(mask_filled, None, (x, y), 2)

        optimized_mask = (mask_filled < 2).astype(np.uint8)


    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    smoothed_mask = cv2.filter2D(optimized_mask.astype(np.float32), -1, kernel)

    return np.clip(smoothed_mask, 0, 1).astype(np.uint8)


def vis_mask_save(mask: np.ndarray, save_path: Path):
    """Save a binary mask with foreground pixels set to 255."""
    mask_save = mask.copy()
    mask_save[mask == 1] = 255
    save_path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(save_path), mask_save)


