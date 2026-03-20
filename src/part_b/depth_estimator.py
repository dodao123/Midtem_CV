"""Monocular depth estimation using MiDaS.

Estimates a relative depth map from a single RGB image
using the MiDaS v2.1 Small model (Intel ISL).
"""

import cv2
import numpy as np
import torch


class MonocularDepthEstimator:
    """Estimates depth from a single image using MiDaS.

    Uses MiDaS Small for lightweight, fast inference.
    Model is loaded lazily on first call and cached.

    Attributes:
        _model: MiDaS neural network model.
        _transform: Input preprocessing transform.
        _device: Torch device (cuda or cpu).
    """

    _instance = None

    def __init__(self):
        """Initialize estimator (model loaded lazily)."""
        self._model = None
        self._transform = None
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @classmethod
    def get_instance(cls) -> "MonocularDepthEstimator":
        """Return singleton instance for model reuse.

        Returns:
            MonocularDepthEstimator: Shared instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self) -> None:
        """Load MiDaS Small model and transforms."""
        self._model = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small",
            trust_repo=True,
        )
        self._model.to(self._device)
        self._model.eval()

        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms",
            trust_repo=True,
        )
        self._transform = midas_transforms.small_transform

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth from a single BGR image.

        MiDaS outputs inverse depth (higher = closer).
        This method returns a normalized depth map where
        higher values mean CLOSER to camera.

        Args:
            image: BGR image (H, W, 3), uint8.

        Returns:
            np.ndarray: Depth map (H, W), float32,
                0.0 = farthest, 1.0 = nearest.
        """
        if self._model is None:
            self._load_model()

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_batch = self._transform(rgb).to(self._device)

        with torch.no_grad():
            prediction = self._model(input_batch)

        raw_depth = prediction.squeeze().cpu().numpy()
        raw_depth = cv2.resize(
            raw_depth, (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return self._normalize_depth(raw_depth)

    @staticmethod
    def _normalize_depth(raw_depth: np.ndarray) -> np.ndarray:
        """Normalize MiDaS raw output to 0.0-1.0 range.

        MiDaS outputs relative inverse depth, but the
        direction can vary by model variant. We normalize
        and then ensure 1.0 = nearest by checking if the
        center region has higher values than edges.

        Args:
            raw_depth: Raw MiDaS prediction.

        Returns:
            np.ndarray: Normalized depth, 1.0 = nearest.
        """
        depth_min = raw_depth.min()
        depth_max = raw_depth.max()
        if depth_max - depth_min < 1e-6:
            return np.zeros_like(raw_depth, dtype=np.float32)
        normalized = (raw_depth - depth_min) / (depth_max - depth_min)
        normalized = normalized.astype(np.float32)

        # Auto-detect direction: center should be "near"
        # If center < edge mean, the map is inverted
        h, w = normalized.shape
        center = normalized[h//3:2*h//3, w//3:2*w//3].mean()
        edge = np.concatenate([
            normalized[0, :], normalized[-1, :],
            normalized[:, 0], normalized[:, -1],
        ]).mean()
        if center < edge:
            normalized = 1.0 - normalized

        return normalized
