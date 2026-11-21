"""
episodic_world_model/ingestion.py
Dual-source streaming video ingest: Kaggle Local Path vs Hugging Face Charades.
"""

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from typing import Generator, Tuple, Optional
from PIL import Image

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

from episodic_world_model.config import IngestionConfig


class VideoStreamIngestionEngine:
    """
    Normalizes inputs from arbitrary sources into a uniform stream of:
    (frame_idx, timestamp_sec, rgb_tensor [1, 3, H, W], depth_tensor [1, 1, H, W])
    """
    def __init__(self, config: IngestionConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.transform = T.Compose([
            T.Resize(self.config.input_resolution),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def stream_frames(self) -> Generator[Tuple[int, float, torch.Tensor, Optional[torch.Tensor]], None, None]:
        if self.config.source_mode.lower() == "kaggle":
            yield from self._stream_from_local_file(self.config.video_path)
        elif self.config.source_mode.lower() == "huggingface":
            yield from self._stream_from_huggingface()
        else:
            raise ValueError(f"Unknown source mode: {self.config.source_mode}. Must be 'kaggle' or 'huggingface'.")

    def _stream_from_local_file(self, video_path: str):
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"Local video file not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        native_fps = native_fps if native_fps > 0 else self.config.target_fps
        frame_interval = max(1, int(round(native_fps / self.config.target_fps)))

        frame_idx = 0
        emitted_idx = 0

        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / native_fps
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                rgb_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

                # Synthetic baseline depth map if streaming pure RGB video
                synthetic_depth = torch.ones(
                    (1, 1, self.config.input_resolution[0], self.config.input_resolution[1]),
                    device=self.device
                ) * 1.0  # 1.0 meter default baseline plane

                yield emitted_idx, timestamp, rgb_tensor, synthetic_depth
                emitted_idx += 1

            frame_idx += 1

        cap.release()

    def _stream_from_huggingface(self):
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("huggingface 'datasets' library is required to stream from Charades.")

        print(f"Loading Hugging Face Dataset: {self.config.hf_dataset_name} [{self.config.hf_split}]...")
        dataset = load_dataset(self.config.hf_dataset_name, split=self.config.hf_split, streaming=True)
        
        # Advance to specified index
        sample = None
        for i, item in enumerate(dataset):
            if i == self.config.hf_video_index:
                sample = item
                break

        if sample is None:
            raise RuntimeError(f"Could not retrieve sample at index {self.config.hf_video_index} from HuggingFace.")

        if "video" in sample:
            temp_path = "/tmp/hf_stream_temp.mp4"
            with open(temp_path, "wb") as f:
                f.write(sample["video"])
            yield from self._stream_from_local_file(temp_path)
            if os.path.exists(temp_path):
                os.remove(temp_path)
        elif "frames" in sample:
            for idx, pil_frame in enumerate(sample["frames"]):
                timestamp = idx / self.config.target_fps
                rgb_tensor = self.transform(pil_frame.convert("RGB")).unsqueeze(0).to(self.device)
                synthetic_depth = torch.ones(
                    (1, 1, self.config.input_resolution[0], self.config.input_resolution[1]),
                    device=self.device
                )
                yield idx, timestamp, rgb_tensor, synthetic_depth
        else:
            raise KeyError("HuggingFace sample missing 'video' bytes or 'frames' list.")
