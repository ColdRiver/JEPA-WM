# Surprise-Gated Episodic Memory & Spatio-Temporal Query Grounding

A modular perception and episodic memory pipeline for assistive mobile manipulators (**Hello Robot Stretch**). Given a continuous RGB video stream, the system autonomously flags variable-length human action timeframes, omits idle non-action periods, isolates and tracks the manipulated object in 2D and metric 3D space, and answers natural language temporal queries (e.g., *"Where did I keep my mobile?"*, *"Which is the last newspaper I used?"*) at query time $T_{\text{query}}$.

This repository implements the **perception, episodic memory, and action-embedding planning interface**—outputting continuous action embeddings, canonical verb-object tuples, 2D bounding boxes, and camera-frame 3D coordinates $(X, Y, Z)$ for downstream robot actuation agents.

---

## Key Capabilities

* **Sparse, Event-Driven Memory Gating**: Uses latent surprise with sustained-onset confirmation and cooldown hysteresis to record only active procedural intervals $[t_{\text{start}}, t_{\text{end}}]$, discarding redundant idle frames.
* **Dynamic Action Windows**: Adapts segment lengths organically to match micro-interactions and extended tasks without fixed sliding-window truncation.
* **Spatial Surprise + ProxyCLIP Object Localization**: Combines $14 \times 14$ patch-level temporal innovation with DINO-guided ProxyCLIP / ClearCLIP dense semantic grounding to isolate the manipulated entity while suppressing background distractors (floor, walls, green screens, furniture).
* **Metric 3D Spatial Grounding**: Estimates pixel-aligned physical depth via Depth-Anything-V2 and back-projects object centroids using Intel RealSense D435i pinhole intrinsics.
* **Query-Conditioned Action Embeddings**: Retrieves relevant historical timeframes via cross-attention over memory nodes and processes the retrieved tokens through a Query-Conditioned Vision Transformer (`QueryConditionedActionViT`) to generate a 512-dim action embedding and live-frame spatial specification.


link to kaggle notebook: https://www.kaggle.com/code/steelbeast/vjepa-model
---

## Theoretical Foundations

This architecture synthesizes mechanisms from the following research papers:

| Paper | Mechanism Used in Pipeline |
| :--- | :--- |
| **Dynamic Chunking for End-to-End Hierarchical Sequence Modeling** (*Hwang, Wang, Gu — arXiv:2507.07955*) | Content-dependent temporal boundary gating for variable-length procedural segments. |
| **P-JEPA: Procedural Video Representation Learning via Joint Embedding Predictive Architecture** (*Tristram et al. — arXiv:2606.23256*) | Frame-aligned dense latent trajectories, relational state-transition vectors ($\Delta \mathbf{z} = \mathbf{z}_{t_{\text{end}}} - \mathbf{z}_{t_{\text{start}}}$), and FCL action space embeddings. |
| **Worth Remembering: Surprise-Gated Robot Episodic Memory** (*Gorlo et al. — arXiv:2606.03787*) | Predictive surprise gating to trigger episodic memory writes, omit static intervals, and index 4D spatio-temporal memory nodes. |
| **ProxyCLIP & ClearCLIP** (*Lan et al. — ECCV 2024*) | Self-supervised ViT spatial affinity ($\mathbf{A}_{\text{DINO}}$) applied as proxy attention over value-projected semantic patch tokens for zero-shot object localization. |

---

## System Architecture

```text
Continuous RGB Stream (640 x 480)
         │
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Sensory Ingestion & Spatio-Temporal Tokenization            │
│ • RealSensePhysicalDepthEstimator (Depth-Anything-V2-Small -> D_t)   │
│ • ViT-B/16 Encoder -> Global token z_t ∈ R^768 & Patches Z_t ∈ R^196x768 │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 2: High-Vividness Surprise-Gated Episodic Segmentation         │
│ • Autoregressive momentum surprise S_t & adaptive Z-score gating     │
│ • Sustained onset (3 frames) + cooldown hysteresis -> [t_s, t_e]     │
│ • Drops idle/subtle frames from memory                               │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Spatial Patch Surprise & ProxyCLIP Object Tracking          │
│ • Patch-level surprise S_{t,p} x ProxyCLIP Objectness^2              │
│ • Peak-energy contour fitting -> flexible 2D bbox + metric (X, Y, Z) │
│ • P-JEPA transition Δz = Normalize(z_post - z_pre) & box-pooled p_obj│
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 4: 4D Episodic Memory Graph & FCL Action Space Projector       │
│ • FCL Projector: [Δz || p_obj || (X,Y,Z)] -> a_emb ∈ R^512           │
│ • Closed-form Ridge Alignment Bridge (768 -> 512)                    │
│ • Stores Node {[t_s, t_e], verb, entity, p_obj, pose, chunk_tokens}  │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼  ◄── Natural Language Query @ T_query
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 5: Cross-Attention Retrieval, Query-ViT & Live Grounding       │
│ • CLIP text query -> attention over Stage 4 memory nodes             │
│ • Retrieved timeframe passed to QueryConditionedActionViT -> a_final │
│ • Live ProxyCLIP grounding at T_query -> 2D bbox + metric (X, Y, Z)  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Stage-by-Stage Breakdown

### Stage 1: Sensory Ingestion & Tokenization
* **Stream Bridge**: Reads continuous `.mp4` video streams (`EPICKitchensStreamBridge`), resizing frames to $640 \times 480$ with factory RealSense D435i intrinsics ($f_x = 615.2, f_y = 615.4, c_x = 320.0, c_y = 240.0$).
* **Metric Depth**: `RealSensePhysicalDepthEstimator` runs `Depth-Anything-V2-Small` to estimate aligned depth maps $\mathbf{D}_t$ scaled to the tabletop/workspace range ($0.65\text{m}\text{--}1.45\text{m}$).
* **Spatial-Temporal Encoder**: Extracts L2-normalized global state tokens $\mathbf{z}_t \in \mathbb{R}^{768}$ and a $14 \times 14$ spatial grid $\mathbf{Z}_t \in \mathbb{R}^{196 \times 768}$.

### Stage 2: High-Vividness Surprise Gating
* **Latent Surprise**: Predicts $\hat{\mathbf{z}}_t = \text{Normalize}(2\mathbf{z}_{t-1} - \mathbf{z}_{t-2})$ and computes angular surprise $\mathcal{S}_t = 1 - \cos(\mathbf{z}_t, \hat{\mathbf{z}}_t)$.
* **Hysteresis State Machine**: `HighVividnessEpisodicSegmenter` remains `IDLE` until the adaptive Z-score exceeds `z_start` ($2.70$) for `sustained_frames=3` consecutive frames. It accumulates frames (including an 8-frame pre-roll buffer) until surprise decays below `z_end` ($0.60$) for `cooldown_frames=15` (subject to `min_chunk_len=24`).

### Stage 3: Spatial Surprise & ProxyCLIP Object Localization
* **ProxyCLIP Dense Grounding**: Computes ClearCLIP value-projected patch tokens (`v_proj -> out_proj -> post_layernorm -> visual_projection`, omitting final residual/FFN) and propagates them across Stage 1's $14 \times 14$ self-supervised spatial affinity matrix.
* **Background Distractor Suppression**: Computes per-patch objectness probability against manipulable targets (e.g., `cup`, `newspaper`, `mobile phone`, `water bottle`, `plate`) vs. room distractors (e.g., `floor tiles`, `green screen curtain`, `ceiling lights`, `office chair`).
* **Flexible Contour Bounding**: Thresholds the fused map $\mathcal{S}_{t, p} \times \text{ObjProb}^2$ using `max(Otsu, 88th percentile)` and selects the contour with maximum `Peak * Mean` intensity—avoiding large-area background bias—to fit an adaptive bounding box and deproject $(u_c, v_c)$ to $(X, Y, Z)$.

### Stage 4: Structured 4D Episodic Memory Graph
* **FCL Action Space**: `PJEPAActionSpaceProjector` fuses $[\Delta \mathbf{z} \,\|\, \mathbf{p}_{\text{obj}} \,\|\, \text{pose}_{3\text{D}}] \in \mathbb{R}^{1539}$ into a 512-dim unit-normalized action embedding $\mathbf{a}_{\text{emb}}$.
* **Subspace Alignment**: `CrossModalSubspaceBridge` fits a closed-form Dual Ridge Regression matrix $\mathbf{W}_{\text{align}} \in \mathbb{R}^{768 \times 512}$ over observed episode keyframes to align visual latents with CLIP text embeddings without random projection noise.
* **Canonical Taxonomy**: Indexes each node with canonical verbs (`TAKE`, `PUT_DOWN`, `MOVE`, `READ`, `DRINK`, `HOLD`, `USE_PHONE`) and the verified `target_entity` from Stage 3.

### Stage 5: Query-Conditioned ViT & Live Grounding
* **Temporal Retrieval**: Encodes the natural language query via CLIP text encoder, applies logit-scaled attention over Stage 4 memory nodes prior to $T_{\text{query}}$, and retrieves the best-matching episode timeframe $[t_s, t_e]$.
* **Query-Conditioned Action ViT**: `QueryConditionedActionViT` projects the query into visual space ($\mathbf{q}_{\text{cond}} \in \mathbb{R}^{768}$), cross-attends across the retrieved episode's frame tokens and peak-frame spatial patches, and outputs the final 512-dim query-conditioned action embedding.
* **Live-Frame Grounding**: Reads the frame at $T_{\text{query}}$ and localizes the queried object via ProxyCLIP similarity, outputting the live 2D bounding box and metric 3D coordinates $(X, Y, Z)$.

---

## Requirements & Installation

Designed to run in **FP16 mixed precision** on a single **16 GB NVIDIA T4 GPU** (e.g., Kaggle or Google Colab).

```bash
pip install -q torch torchvision transformers huggingface_hub opencv-python-headless timm matplotlib numpy
```

---

## Quickstart & Running on Custom Video

1. **Mount or place your video file** in the working/input directory (e.g., `/kaggle/input/datasets/steelbeast/lab-video/lab.mp4` or `./input_video.mp4`).
2. **Update `video_path`** in Stage 1 and Stage 2:
   ```python
   video_path = "/path/to/your/video.mp4"
   ```
3. **Execute the 5 stage cells sequentially**:
   * **Stage 1**: Initializes `depth_engine`, `stream_bridge`, and `vjepa_encoder`.
   * **Stage 2**: Runs `HighVividnessEpisodicSegmenter` across the full video (`max_frames = 999999`), populating `chunks` and plotting the surprise timeline and Start/Mid/End storyboard.
   * **Stage 3**: Runs `ProxyCLIPSpatialSurpriseTracker` on `chunks`, generating `tracked_episodes` and rendering tracked bounding boxes.
   * **Stage 4**: Builds `memory_graph` (`StructuredEpisodicMemoryGraph`) and plots the pairwise FCL action similarity matrix.
   * **Stage 5**: Initializes `Stage5EpisodicQueryEngine`, executes queries at $T_{\text{query}}$, prints the planning manifests, and visualizes the live-frame 2D/3D grounding.

---

## Output Payload Specification (Downstream Agent Interface)

Stage 5 returns a structured dictionary ready for downstream Hello Robot Stretch manipulation and navigation agents:

```json
{
  "query": "Which is the last newspaper or paper I used?",
  "T_query": 30.0,
  "retrieved_episode_id": 3,
  "retrieved_timeframe": [28.119, 29.42],
  "canonical_action": ["READ"],
  "target_entity": "newspaper",
  "attention_score": 0.6842,
  "final_action_embedding_dim": [512],
  "final_action_embedding": [0.0072, -0.0219, 0.0674, "..."],
  "output_2d_bbox": [312, 165, 488, 255],
  "output_metric_3d_xyz": [0.145, -0.042, 0.918]
}
```

---

## Supported Datasets

* **Custom Lab / Assistive Videos**: Stationary RGB videos of tabletop object manipulation (`.mp4`).
* **EPIC-KITCHENS-100**: Egocentric continuous kitchen tasks with canonical verb-noun action labels (`lightly-ai/epic-kitchens-100-clips`).
* **Charades-STA**: Room-scale multi-action human household videos with temporal natural language annotations (`jwnt4/charades-sta-test`).

---

## References

1. Hwang, S., Wang, B., & Gu, A. (2025). *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling*. arXiv:2507.07955.
2. Tristram, F., et al. (2026). *P-JEPA: Procedural Video Representation Learning via Joint Embedding Predictive Architecture*. arXiv:2606.23256.
3. Gorlo, N., Wise, D. K., Speranzon, A., & Carlone, L. (2026). *Worth Remembering: Surprise-Gated Robot Episodic Memory*. arXiv:2606.03787.
4. Lan, M., et al. (2024). *ProxyCLIP: Proxy Attention Improves CLIP for Open-Vocabulary Segmentation* & *ClearCLIP*. ECCV 2024.
5. Yang, L., et al. (2024). *Depth Anything V2*.
