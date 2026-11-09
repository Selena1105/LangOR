"""Configuration for multi-level semantic segmentation inference."""
import ast
from pathlib import Path
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, fields
from typing import List, Optional


@dataclass
class InferenceConfig:
    """Parameters used by the segmentation pipeline."""

    model_path: Optional[str] = None
    mask_folder: Optional[str] = None
    text_encoder_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    text_encoder_max_seq_length: int = 384
    text_embedding_dim: int = 384
    feature_dim: int = 32
    feature_gaussian_iteration: int = 10
    num_lvl: int = 1
    descriptor_mode: str = 'rule'
    sampling_strategy: int = 2
    max_sample_size: int = 100000
    max_pixels_per_view: int = 50000
    sigma: float = 0.5
    remaining_sim_thresh: float = 0.3
    hdbscan_min_cluster_size: int = 300
    hdbscan_epsilon_base: float = 0.05
    num_per_cluster_semantics: int = 1
    mask_thresh: float = 0.5
    spatial_outlier_std: float = 0.5
    segment_cosine_thresh: float = 0.8
    background_thresh: float = 0.3
    large_cluster_thresh: float = 0.15
    output_path: Optional[str] = None
    gaussian_save_path: Optional[str] = None
    show_results: bool = False

    @classmethod
    def from_args(cls, args: Namespace):
        """Build and validate configuration from parsed command-line values."""
        values = {
            item.name: getattr(args, item.name)
            for item in fields(cls)
            if getattr(args, item.name, None) is not None
        }
        config = cls(**values)
        config.validate()
        return config

    @property
    def multi_lvl_dim(self) -> List[int]:
        """Derive feature blocks from the selected level layout."""
        if self.num_lvl not in (1, 3):
            raise ValueError("num_lvl must be 1 or 3")
        return [32] if self.num_lvl == 1 else [16, 8, 8]

    def validate(self):
        """Require model, annotation, and output paths."""
        if self.feature_dim != sum(self.multi_lvl_dim):
            raise ValueError("feature_dim must match the total level dimensions (32)")
        if not self.model_path:
            raise ValueError("--model_path is required")
        if not self.mask_folder:
            raise ValueError("--mask_folder is required")
        if not self.output_path:
            raise ValueError("--output_path is required")

    def build_scene_config(self, cfg_args: Namespace) -> Namespace:
        """Construct the scene configuration from saved training arguments."""
        return Namespace(
            model_path=self.model_path,
            source_path=cfg_args.source_path,
            sh_degree=cfg_args.sh_degree,
            need_features=True,
            need_masks=True,
            allow_principle_point_shift=getattr(cfg_args, 'allow_principle_point_shift', False),
            white_background=getattr(cfg_args, 'white_background', False),
            eval=getattr(cfg_args, 'eval', False),
            images=getattr(cfg_args, 'images', 'images'),
            data_device=getattr(cfg_args, 'data_device', 'cuda'),
            resolution=getattr(cfg_args, 'resolution', -1),
        )

    def build_pipeline_config(self, cfg_args: Namespace) -> Namespace:
        """Construct the rendering pipeline configuration."""
        return Namespace(
            convert_SHs_python=getattr(cfg_args, 'convert_SHs_python', False),
            compute_cov3D_python=getattr(cfg_args, 'compute_cov3D_python', False),
            debug=getattr(cfg_args, 'debug', False),
        )


def add_inference_args(parser: ArgumentParser):
    """Register parameters used by the inference pipeline."""
    group = parser.add_argument_group('Paths and output')
    group.add_argument('--model_path', type=str, required=True, help='Trained model directory.')
    group.add_argument('--mask_folder', type=str, required=True, help='Ground-truth object mask directory.')
    group.add_argument('--output_path', type=str, required=True, help='Directory for segmentation results and metrics.')
    group.add_argument('--gaussian_save_path', type=str, default=None, help='Optional .pt file for Gaussian attributes and query labels.')
    group.add_argument('--show_results', action='store_true', help='Save evaluation visualizations.')

    group = parser.add_argument_group('Encoder and feature layout')
    group.add_argument('--text_encoder_name', type=str, help='SentenceTransformer model name or local path.')
    group.add_argument('--text_encoder_max_seq_length', type=int, help='Maximum encoder sequence length.')
    group.add_argument('--text_embedding_dim', type=int, help='Dimension of stored mask embeddings.')
    group.add_argument('--feature_gaussian_iteration', type=int, help='Feature checkpoint iteration (default: 10).')
    group.add_argument('--num_lvl', type=int, help='Feature levels: 1 or 3 (default: 1).', choices=[1, 3])
    group.add_argument('--descriptor_mode', type=str, choices=['rule', 'avg'], help='Instance descriptor aggregation method.')

    group = parser.add_argument_group('Sampling and clustering')
    group.add_argument('--sampling_strategy', type=int, choices=[0, 1, 2], help='Pixel sampling: 0=uniform, 1=area, 2=area and center.')
    group.add_argument('--max_sample_size', type=int, help='Maximum feature samples used for clustering.')
    group.add_argument('--max_pixels_per_view', type=int, help='Maximum sampled foreground pixels per view.')
    group.add_argument('--sigma', type=float, help='Spatial sampling Gaussian width.')
    group.add_argument('--remaining_sim_thresh', type=float, help='Similarity threshold for assigning unsampled pixels.')
    group.add_argument('--hdbscan_min_cluster_size', type=int, help='Minimum HDBSCAN cluster size.')
    group.add_argument('--hdbscan_epsilon_base', type=float, help='HDBSCAN cluster selection epsilon.')
    group.add_argument('--num_per_cluster_semantics', type=int, help='Descriptor slots per cluster.')

    group = parser.add_argument_group('Segmentation')
    group.add_argument('--mask_thresh', type=float, help='Threshold for rendered masks (default: 0.5).')
    group.add_argument('--spatial_outlier_std', type=float, help='Distance standard-deviation multiplier for spatial filtering.')
    group.add_argument('--segment_cosine_thresh', type=float, help='Minimum query-to-cluster cosine similarity.')
    group.add_argument('--background_thresh', type=float, help='Minimum point-to-cluster similarity.')
    group.add_argument('--large_cluster_thresh', type=float, help='Maximum cluster fraction before background exclusion.')


def _parse_saved_namespace(text: str) -> Namespace:
    """Read saved Namespace literals without executing configuration code."""
    try:
        expression = ast.parse(text.strip(), mode='eval').body
        if (not isinstance(expression, ast.Call)
                or not isinstance(expression.func, ast.Name)
                or expression.func.id != 'Namespace'
                or expression.args
                or any(item.arg is None for item in expression.keywords)):
            raise ValueError("Expected Namespace with literal keyword values")
        return Namespace(**{
            item.arg: ast.literal_eval(item.value) for item in expression.keywords
        })
    except (SyntaxError, ValueError, TypeError) as error:
        raise ValueError("Invalid saved configuration: expected Namespace literals") from error


def load_cfg_args(config: InferenceConfig) -> Namespace:
    """Read cfg_args written by the scene training pipeline."""
    cfg_path = Path(config.model_path) / "cfg_args"
    try:
        return _parse_saved_namespace(cfg_path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise FileNotFoundError(f"Scene config file not found: {cfg_path}") from None
