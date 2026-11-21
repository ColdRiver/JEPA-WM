"""
episodic_world_model/representation.py
V-JEPA 2.1 Latent Backbone and Spatio-Temporal Competitive Slot-Attention Engine.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from episodic_world_model.config import ModelArchitectureConfig


class VJEPA21LatentBackbone(nn.Module):
    """
    Extracts dense spatio-temporal representations. Implements ViT tokenization
    with deep intermediate feature aggregation to preserve local coordinate geometry.
    """
    def __init__(self, config: ModelArchitectureConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.embedding_dim = config.embedding_dim

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.embedding_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        self.pos_embed = nn.Parameter(torch.randn(1, config.num_patches + 1, self.embedding_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=8,
                dim_feedforward=self.embedding_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
            for _ in range(12)
        ])
        self.norm = nn.LayerNorm(self.embedding_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, patches), dim=1)
        tokens = tokens + self.pos_embed

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        global_latent = tokens[:, 0, :]
        patch_latents = tokens[:, 1:, :]

        return global_latent, patch_latents


class SpatioTemporalSlotAttention(nn.Module):
    """
    Competitive Slot Attention. Normalizes attention over the SLOTS rather than tokens,
    forcing patches to compete for slot ownership to decouple human arms from objects.
    """
    def __init__(self, config: ModelArchitectureConfig):
        super().__init__()
        self.config = config
        self.num_slots = config.num_slots
        self.slot_dim = config.slot_dim
        self.iters = config.slot_iters
        self.temp = config.slot_temp

        self.slot_mu = nn.Parameter(torch.randn(1, 1, self.slot_dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, self.slot_dim))

        self.to_q = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.to_k = nn.Linear(config.embedding_dim, self.slot_dim, bias=False)
        self.to_v = nn.Linear(config.embedding_dim, self.slot_dim, bias=False)

        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.norm_inputs = nn.LayerNorm(config.embedding_dim)
        self.norm_slots = nn.LayerNorm(self.slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_dim, self.slot_dim * 2),
            nn.GELU(),
            nn.Linear(self.slot_dim * 2, self.slot_dim)
        )
        self.norm_mlp = nn.LayerNorm(self.slot_dim)

    def forward(self, patch_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = patch_tokens.shape
        inputs = self.norm_inputs(patch_tokens)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        mu = self.slot_mu.expand(B, self.num_slots, -1)
        sigma = self.slot_log_sigma.exp().expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        attn_masks = None

        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.to_q(slots_norm)

            dots = torch.einsum('bkd,bnd->bkn', q, k) / (math.sqrt(self.slot_dim) * self.temp)
            attn_weights = F.softmax(dots, dim=1) + 1e-8
            attn_masks = attn_weights

            attn_norm = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bkn,bnd->bkd', attn_norm, v)

            slots_flat = slots_prev.view(-1, self.slot_dim)
            updates_flat = updates.view(-1, self.slot_dim)
            slots_updated = self.gru(updates_flat, slots_flat)
            slots = slots_updated.view(B, self.num_slots, self.slot_dim)

            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn_masks
