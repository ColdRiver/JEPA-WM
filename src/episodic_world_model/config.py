"""
episodic_world_model/config.py
Zero-hardcoding configuration engine for Embodied Latent World Model.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import torch


@dataclass
class CameraIntrinsicsConfig:
    """Optical parameters for metric 3D pinhole deprojection."""
    fx: float = 615.2
    fy: float = 615.4
    cx: float = 320.0
    cy: float = 240.0
    depth_scale: float = 0.001  # Convert mm to meters if raw RealSense uint16
    min_depth: float = 0.2     # Near clipping plane (meters)
    max_depth: float = 3.0     # Far clipping plane (meters)


@dataclass
class IngestionConfig:
    """Configures input video sourcing and streaming constraints."""
    source_mode: str = "kaggle"  # Options: 'kaggle' (local file/dir) or 'huggingface'
    video_path: Optional[str] = "/kaggle/input/test-video/sample.mp4"
    hf_dataset_name: str = "sayakpaul/charades-sample"  # Standard Charades subset
    hf_split: str = "train"
    hf_video_index: int = 0
    target_fps: float = 15.0
    input_resolution: Tuple[int, int] = (224, 224)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1


@dataclass
class ModelArchitectureConfig:
    """Latent space, slot-attention, and backbone dimensions."""
    backbone_name: str = "vjepa_2.1"
    pretrained_weights_path: Optional[str] = None
    embedding_dim: int = 512
    patch_size: int = 16
    grid_size: Tuple[int, int] = (14, 14)  # 224 // 16
    num_patches: int = 196
    
    # Slot-Attention parameters (Hand vs Entity disentanglement)
    num_slots: int = 4  # [Agent/Hand, Manipulated Entity, Support Surface, Background]
    slot_dim: int = 512
    slot_iters: int = 3
    slot_temp: float = 0.07

    # Dynamic Boundary Router (H-Net Dynamic Chunking)
    router_hidden_dim: int = 128
    router_boundary_threshold: float = 0.5  # Sigmoid decision boundary
    sparsity_loss_weight: float = 0.01

    # Intentionality Filter (P-JEPA Persistent Topological Shift)
    min_intentionality_ratio: float = 0.35  # persistent_shift / total_path_length
    relaxation_settle_frames: int = 4
    absolute_surprise_floor: float = 1e-4   # Prevents small-sigma z-score division trap


@dataclass
class EpisodicMemoryConfig:
    """Parameters governing memory nodes, edge graph, and temporal search."""
    temporal_history_window: int = 40       # Causal stats window for baseline tracking
    pre_roll_buffer_frames: int = 8          # Pre-contact approach preservation
    max_active_episodes: int = 1000
    occlusion_iou_threshold: float = 0.45    # 3D spatial overlap to declare occlusion
    text_embedding_dim: int = 512


@dataclass
class SystemPipelineConfig:
    """Master pipeline configuration."""
    camera: CameraIntrinsicsConfig = field(default_factory=CameraIntrinsicsConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    model: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    memory: EpisodicMemoryConfig = field(default_factory=EpisodicMemoryConfig)
    debug_mode: bool = True
