"""
episodic_world_model/spatial_memory.py
Metric 3D Pinhole Deprojection and Relational Episodic Memory Graph.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F
from episodic_world_model.config import CameraIntrinsicsConfig, EpisodicMemoryConfig


class EdgeType(Enum):
    TEMPORAL_SEQUENCE = "temporal_sequence"  # Episode A happened before Episode B
    SPATIAL_PROXIMITY = "spatial_proximity"  # Episodes occurred within interaction radius
    OCCLUDES = "occludes"                    # Episode B physically covered Episode A's entity


@dataclass
class BoundingBox2D:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def centroid(self) -> Tuple[int, int]:
        return int((self.xmin + self.xmax) / 2), int((self.ymin + self.ymax) / 2)

    @property
    def area(self) -> int:
        return max(0, self.xmax - self.xmin) * max(0, self.ymax - self.ymin)


@dataclass
class MetricPose3D:
    x: float  # Lateral distance (meters)
    y: float  # Height relative to optical center (meters)
    z: float  # Depth along camera optical axis (meters)
    confidence: float

    def distance_to(self, other: 'MetricPose3D') -> float:
        return float(np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2))


@dataclass
class EpisodeNode:
    """A verified, intentional action chunk committed to episodic memory."""
    episode_id: str
    t_start: float
    t_end: float
    action_primitive: str
    action_embedding: torch.Tensor          # Delta z_action [D]
    entity_prototype: torch.Tensor          # Manipulated object slot embedding [D]
    agent_prototype: torch.Tensor           # Hand/forearm slot embedding [D]
    resting_box_2d: BoundingBox2D
    resting_pose_3d: MetricPose3D
    trajectory_3d: List[MetricPose3D]
    temporal_mean_z: torch.Tensor
    is_occluded: bool = False
    occluded_by_id: Optional[str] = None


@dataclass
class MemoryEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricSpatialDeprojector:
    """
    Transforms 2D slot-attention heatmaps into metric 3D Cartesian coordinates
    via pinhole deprojection with median-depth patch filtering.
    """
    def __init__(self, config: CameraIntrinsicsConfig):
        self.config = config

    def slot_mask_to_bbox(self, slot_mask: torch.Tensor, image_shape: Tuple[int, int]) -> BoundingBox2D:
        H, W = image_shape
        if slot_mask.dim() == 1:
            mask_2d = slot_mask.view(14, 14).unsqueeze(0).unsqueeze(0)
        else:
            mask_2d = slot_mask.unsqueeze(0).unsqueeze(0)

        upsampled = F.interpolate(mask_2d, size=(H, W), mode="bilinear", align_corners=False).squeeze().detach().cpu().numpy()
        threshold = np.percentile(upsampled, 65)
        binary_mask = upsampled >= threshold

        y_indices, x_indices = np.where(binary_mask)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return BoundingBox2D(int(W * 0.4), int(H * 0.4), int(W * 0.6), int(H * 0.6))

        return BoundingBox2D(
            xmin=int(np.min(x_indices)),
            ymin=int(np.min(y_indices)),
            xmax=int(np.max(x_indices)),
            ymax=int(np.max(y_indices))
        )

    def deproject_box_to_3d(self, bbox: BoundingBox2D, depth_map: torch.Tensor) -> MetricPose3D:
        depth_np = depth_map.squeeze().detach().cpu().numpy()
        H, W = depth_np.shape

        u_c, v_c = bbox.centroid
        u_c = np.clip(u_c, 0, W - 1)
        v_c = np.clip(v_c, 0, H - 1)

        box_w = bbox.xmax - bbox.xmin
        box_h = bbox.ymax - bbox.ymin
        crop_xmin = int(np.clip(bbox.xmin + 0.25 * box_w, 0, W - 1))
        crop_xmax = int(np.clip(bbox.xmax - 0.25 * box_w, crop_xmin + 1, W))
        crop_ymin = int(np.clip(bbox.ymin + 0.25 * box_h, 0, H - 1))
        crop_ymax = int(np.clip(bbox.ymax - 0.25 * box_h, crop_ymin + 1, H))

        depth_crop = depth_np[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        valid_mask = (depth_crop >= self.config.min_depth) & (depth_crop <= self.config.max_depth) & (~np.isnan(depth_crop))

        if np.sum(valid_mask) > 0:
            z_metric = float(np.median(depth_crop[valid_mask]))
            confidence = float(np.sum(valid_mask) / depth_crop.size)
        else:
            z_metric = 1.0
            confidence = 0.1

        x_metric = float((u_c - self.config.cx) * z_metric / self.config.fx)
        y_metric = float((v_c - self.config.cy) * z_metric / self.config.fy)

        return MetricPose3D(x=x_metric, y=y_metric, z=z_metric, confidence=confidence)


class RelationalEpisodicMemoryGraph:
    """Spatio-Temporal Scene Memory Graph tracking structural sequence and occlusions."""
    def __init__(self, config: EpisodicMemoryConfig):
        self.config = config
        self.nodes: Dict[str, EpisodeNode] = {}
        self.edges: List[MemoryEdge] = []
        self._last_committed_id: Optional[str] = None

    def commit_episode(self, episode: EpisodeNode):
        self.nodes[episode.episode_id] = episode

        if self._last_committed_id is not None:
            dt = episode.t_start - self.nodes[self._last_committed_id].t_end
            self.edges.append(MemoryEdge(
                source_id=self._last_committed_id,
                target_id=episode.episode_id,
                edge_type=EdgeType.TEMPORAL_SEQUENCE,
                weight=float(dt)
            ))

        for prior_id, prior_node in self.nodes.items():
            if prior_id == episode.episode_id:
                continue

            dist_3d = episode.resting_pose_3d.distance_to(prior_node.resting_pose_3d)

            if dist_3d <= 0.35:
                self.edges.append(MemoryEdge(
                    source_id=prior_id,
                    target_id=episode.episode_id,
                    edge_type=EdgeType.SPATIAL_PROXIMITY,
                    weight=dist_3d
                ))

                if dist_3d <= 0.12 and episode.action_primitive in ["place", "cover", "drop"]:
                    prior_node.is_occluded = True
                    prior_node.occluded_by_id = episode.episode_id
                    self.edges.append(MemoryEdge(
                        source_id=episode.episode_id,
                        target_id=prior_id,
                        edge_type=EdgeType.OCCLUDES,
                        weight=dist_3d,
                        metadata={"occluding_action": episode.action_primitive}
                    ))

        self._last_committed_id = episode.episode_id

    def get_unoccluded_target(self, node_id: str) -> Tuple[MetricPose3D, bool, Optional[str]]:
        node = self.nodes.get(node_id)
        if not node:
            raise KeyError(f"Episode {node_id} not found in memory.")

        if node.is_occluded:
            return node.resting_pose_3d, True, node.occluded_by_id
        return node.resting_pose_3d, False, None
