"""
episodic_world_model/dynamics.py
Causal Latent Predictor, Learned Boundary Router (H-Net), and Intentionality Filter (P-JEPA).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from episodic_world_model.config import ModelArchitectureConfig


class CausalLatentPredictor(nn.Module):
    """Autoregressive predictor forecasting expected next-step latent state."""
    def __init__(self, config: ModelArchitectureConfig):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )

    def forward(self, z_prev: torch.Tensor, z_prev2: torch.Tensor) -> torch.Tensor:
        state_context = torch.cat([z_prev, z_prev - z_prev2], dim=-1)
        delta_hat = self.predictor(state_context)
        z_hat = z_prev + delta_hat
        return F.normalize(z_hat, dim=-1)


class LearnedBoundaryRouter(nn.Module):
    """H-Net dynamic chunking router (Hwang et al., 2025)."""
    def __init__(self, config: ModelArchitectureConfig):
        super().__init__()
        self.config = config
        input_dim = config.embedding_dim * 3

        self.router_net = nn.Sequential(
            nn.Linear(input_dim, config.router_hidden_dim),
            nn.LayerNorm(config.router_hidden_dim),
            nn.GELU(),
            nn.Linear(config.router_hidden_dim, 1)
        )

    def forward(self, z_t: torch.Tensor, z_prev: torch.Tensor, z_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        innovation = z_t - z_hat
        velocity = z_t - z_prev

        cos_sim = F.cosine_similarity(z_t, z_hat, dim=-1).unsqueeze(-1)
        raw_surprise = torch.clamp(1.0 - cos_sim, min=0.0)

        router_input = torch.cat([innovation, velocity, z_t], dim=-1)
        logits = self.router_net(router_input)
        boundary_prob = torch.sigmoid(logits)

        return boundary_prob, raw_surprise


class TopologicalIntentionalityFilter:
    """Filters out reversible environmental noise using P-JEPA persistence ratio."""
    def __init__(self, config: ModelArchitectureConfig):
        self.config = config
        self.min_ratio = config.min_intentionality_ratio
        self.floor = config.absolute_surprise_floor

    def evaluate_episode(self, latent_trajectory: torch.Tensor) -> Tuple[bool, float]:
        T_steps = latent_trajectory.shape[0]
        if T_steps < 2:
            return False, 0.0

        persistent_displacement = torch.norm(latent_trajectory[-1] - latent_trajectory[0], p=2).item()
        step_differences = latent_trajectory[1:] - latent_trajectory[:-1]
        cumulative_path_length = torch.sum(torch.norm(step_differences, p=2, dim=-1)).item()

        if cumulative_path_length < self.floor:
            return False, 0.0

        ratio = persistent_displacement / (cumulative_path_length + 1e-8)
        is_intentional = ratio >= self.min_ratio
        return is_intentional, ratio
