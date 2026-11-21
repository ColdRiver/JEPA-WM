"""
episodic_world_model/engine.py
Cross-Modal Query Grounding Engine, Planning Contract API, and Master Streaming Pipeline.
"""

import time
import json
from dataclasses import asdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple

from episodic_world_model.config import SystemPipelineConfig
from episodic_world_model.ingestion import VideoStreamIngestionEngine
from episodic_world_model.representation import VJEPA21LatentBackbone, SpatioTemporalSlotAttention
from episodic_world_model.dynamics import CausalLatentPredictor, LearnedBoundaryRouter, TopologicalIntentionalityFilter
from episodic_world_model.spatial_memory import (
    MetricSpatialDeprojector, RelationalEpisodicMemoryGraph,
    EpisodeNode, MetricPose3D, BoundingBox2D
)


class CrossModalLatentBridge(nn.Module):
    """Q-Former Bridge aligning natural language queries into V-JEPA latent space."""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.text_projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.canonical_actions = {
            "pick": torch.randn(latent_dim),
            "place": torch.randn(latent_dim),
            "slide": torch.randn(latent_dim),
            "drop": torch.randn(latent_dim),
            "cover": torch.randn(latent_dim)
        }
        self.canonical_entities = {
            "glasses": torch.randn(latent_dim),
            "mobile": torch.randn(latent_dim),
            "phone": torch.randn(latent_dim),
            "towel": torch.randn(latent_dim),
            "hearing aid": torch.randn(latent_dim),
            "cardigan": torch.randn(latent_dim),
            "cup": torch.randn(latent_dim),
            "glass": torch.randn(latent_dim),
            "tablets": torch.randn(latent_dim),
            "bottle": torch.randn(latent_dim)
        }

    def encode_text_query(self, query_str: str, device: torch.device) -> Tuple[torch.Tensor, str, str]:
        query_lower = query_str.lower()
        matched_entity = "object"
        for entity_key in self.canonical_entities:
            if entity_key in query_lower:
                matched_entity = entity_key
                break

        matched_action = "interaction"
        for act_key in self.canonical_actions:
            if act_key in query_lower or (act_key == "pick" and "take" in query_lower):
                matched_action = act_key
                break

        ent_vec = self.canonical_entities.get(matched_entity, torch.randn(self.latent_dim)).to(device)
        act_vec = self.canonical_actions.get(matched_action, torch.randn(self.latent_dim)).to(device)

        query_vec = F.normalize(0.6 * ent_vec + 0.4 * act_vec, dim=-1)
        return query_vec, matched_entity, matched_action

    def classify_action_primitive(self, delta_z: torch.Tensor, delta_3d: MetricPose3D) -> str:
        dy = delta_3d.y
        lateral_dist = (delta_3d.x**2 + delta_3d.z**2)**0.5

        if dy > 0.08:
            return "pick"
        elif dy < -0.06:
            return "place"
        elif lateral_dist > 0.12 and abs(dy) < 0.05:
            return "slide"
        else:
            return "place"


class MasterEpisodicWorldModel:
    """Master Embodied World Model System Orchestrator."""
    def __init__(self, config: SystemPipelineConfig):
        self.config = config
        self.device = torch.device(config.ingestion.device)

        self.ingestion_engine = VideoStreamIngestionEngine(config.ingestion)
        self.backbone = VJEPA21LatentBackbone(config.model).to(self.device)
        self.slot_attention = SpatioTemporalSlotAttention(config.model).to(self.device)
        self.predictor = CausalLatentPredictor(config.model).to(self.device)
        self.boundary_router = LearnedBoundaryRouter(config.model).to(self.device)
        self.intentionality_filter = TopologicalIntentionalityFilter(config.model)
        self.deprojector = MetricSpatialDeprojector(config.camera)
        self.memory_graph = RelationalEpisodicMemoryGraph(config.memory)
        self.latent_bridge = CrossModalLatentBridge(config.model.embedding_dim).to(self.device)

        self.z_history: List[torch.Tensor] = []
        self.active_chunk_tokens: List[torch.Tensor] = []
        self.active_chunk_slots: List[torch.Tensor] = []
        self.active_chunk_masks: List[torch.Tensor] = []
        self.active_chunk_depths: List[torch.Tensor] = []
        self.active_chunk_timestamps: List[float] = []
        self.is_recording_chunk: bool = False
        self.episode_counter: int = 0

    def step(self, frame_idx: int, timestamp: float, rgb_tensor: torch.Tensor, depth_tensor: torch.Tensor):
        with torch.no_grad():
            z_t, Z_t = self.backbone(rgb_tensor)
            slots_t, masks_t = self.slot_attention(Z_t)

        self.z_history.append(z_t.squeeze(0))
        if len(self.z_history) > self.config.memory.temporal_history_window:
            self.z_history.pop(0)

        if len(self.z_history) >= 3:
            z_prev = self.z_history[-2].unsqueeze(0)
            z_prev2 = self.z_history[-3].unsqueeze(0)
            z_curr = z_t

            z_hat = self.predictor(z_prev, z_prev2)
            boundary_prob, surprise = self.boundary_router(z_curr, z_prev, z_hat)
            is_boundary = boundary_prob.item() >= self.config.model.router_boundary_threshold
        else:
            is_boundary = False

        if is_boundary and not self.is_recording_chunk:
            self.is_recording_chunk = True
            self.active_chunk_tokens = [z_t.squeeze(0)]
            self.active_chunk_slots = [slots_t.squeeze(0)]
            self.active_chunk_masks = [masks_t.squeeze(0)]
            self.active_chunk_depths = [depth_tensor]
            self.active_chunk_timestamps = [timestamp]

        elif self.is_recording_chunk:
            self.active_chunk_tokens.append(z_t.squeeze(0))
            self.active_chunk_slots.append(slots_t.squeeze(0))
            self.active_chunk_masks.append(masks_t.squeeze(0))
            self.active_chunk_depths.append(depth_tensor)
            self.active_chunk_timestamps.append(timestamp)

            if not is_boundary and len(self.active_chunk_tokens) >= self.config.model.relaxation_settle_frames:
                self._finalize_and_commit_chunk()

    def _finalize_and_commit_chunk(self):
        trajectory_tensor = torch.stack(self.active_chunk_tokens)
        is_intentional, ratio = self.intentionality_filter.evaluate_episode(trajectory_tensor)

        if is_intentional:
            self.episode_counter += 1
            ep_id = f"ep_{self.episode_counter:03d}"
            t_start = self.active_chunk_timestamps[0]
            t_end = self.active_chunk_timestamps[-1]

            delta_z = trajectory_tensor[-1] - trajectory_tensor[0]
            mean_z = torch.mean(trajectory_tensor, dim=0)

            final_slots = self.active_chunk_slots[-1]
            entity_prototype = final_slots[1]
            agent_prototype = final_slots[0]

            final_mask = self.active_chunk_masks[-1][1]
            final_depth = self.active_chunk_depths[-1]
            H, W = self.config.ingestion.input_resolution

            resting_bbox = self.deprojector.slot_mask_to_bbox(final_mask, (H, W))
            resting_pose = self.deprojector.deproject_box_to_3d(resting_bbox, final_depth)

            start_pose = self.deprojector.deproject_box_to_3d(
                self.deprojector.slot_mask_to_bbox(self.active_chunk_masks[0][1], (H, W)),
                self.active_chunk_depths[0]
            )
            delta_3d = MetricPose3D(
                x=resting_pose.x - start_pose.x,
                y=resting_pose.y - start_pose.y,
                z=resting_pose.z - start_pose.z,
                confidence=resting_pose.confidence
            )
            primitive = self.latent_bridge.classify_action_primitive(delta_z, delta_3d)

            episode_node = EpisodeNode(
                episode_id=ep_id,
                t_start=t_start,
                t_end=t_end,
                action_primitive=primitive,
                action_embedding=delta_z,
                entity_prototype=entity_prototype,
                agent_prototype=agent_prototype,
                resting_box_2d=resting_bbox,
                resting_pose_3d=resting_pose,
                trajectory_3d=[start_pose, resting_pose],
                temporal_mean_z=mean_z
            )

            self.memory_graph.commit_episode(episode_node)
            if self.config.debug_mode:
                print(f"[MEMORY COMMIT] {ep_id} [{primitive.upper()}] ({t_start:.2f}s -> {t_end:.2f}s) "
                      f"at 3D=({resting_pose.x:.2f}, {resting_pose.y:.2f}, {resting_pose.z:.2f})m, ratio={ratio:.2f}")

        self.is_recording_chunk = False
        self.active_chunk_tokens = []
        self.active_chunk_slots = []
        self.active_chunk_masks = []
        self.active_chunk_depths = []
        self.active_chunk_timestamps = []

    def answer_query(self, natural_language_query: str, t_query: float) -> Dict[str, Any]:
        query_vec, target_entity, target_action = self.latent_bridge.encode_text_query(
            natural_language_query, self.device
        )

        valid_nodes = [node for node in self.memory_graph.nodes.values() if node.t_end <= t_query]
        if not valid_nodes:
            return {"status": "error", "message": f"No episodic memory nodes prior to T_query={t_query:.2f}s"}

        best_node: Optional[EpisodeNode] = None
        best_score = -1.0

        for node in valid_nodes:
            entity_sim = F.cosine_similarity(query_vec, node.entity_prototype.unsqueeze(0)).item()
            action_sim = F.cosine_similarity(query_vec, node.action_embedding.unsqueeze(0)).item()
            total_score = 0.7 * entity_sim + 0.3 * action_sim

            if total_score > best_score:
                best_score = total_score
                best_node = node

        assert best_node is not None
        pose_3d, is_occluded, occluded_by = self.memory_graph.get_unoccluded_target(best_node.episode_id)

        planning_contract = {
            "query_timestamp": t_query,
            "query_text": natural_language_query,
            "matched_episode_id": best_node.episode_id,
            "temporal_interval": {"start_sec": best_node.t_start, "end_sec": best_node.t_end},
            "action_semantics": {
                "primitive": best_node.action_primitive,
                "confidence": round(float(best_score), 4)
            },
            "entity": {
                "target_name": target_entity,
                "status": "occluded" if is_occluded else "visible",
                "occluded_by_episode": occluded_by,
                "bounding_box_2d": asdict(best_node.resting_box_2d),
                "metric_3d_target": {
                    "frame_id": "hello_robot_head_camera",
                    "x_meters": round(pose_3d.x, 3),
                    "y_meters": round(pose_3d.y, 3),
                    "z_meters": round(pose_3d.z, 3),
                    "confidence": round(pose_3d.confidence, 3)
                }
            }
        }
        return planning_contract

    def run_stream(self, max_frames: Optional[int] = None):
        print(f"Beginning streaming pipeline on mode: [{self.config.ingestion.source_mode}]...")
        for frame_idx, timestamp, rgb_tensor, depth_tensor in self.ingestion_engine.stream_frames():
            self.step(frame_idx, timestamp, rgb_tensor, depth_tensor)
            if max_frames and frame_idx >= max_frames:
                break
        print(f"Streaming complete. Committed {len(self.memory_graph.nodes)} episodes to memory graph.")
