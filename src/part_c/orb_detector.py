"""ORB feature detector implementation.

Uses Oriented FAST and Rotated BRIEF for efficient binary
keypoint detection and descriptor computation.
"""

from typing import List, Tuple

import cv2
import numpy as np

from src.part_c.detector_interface import FeatureDetectorInterface


class OrbDetector(FeatureDetectorInterface):
    """ORB-based feature detector.

    Attributes:
        _n_features: Maximum number of keypoints to retain.
        _scale_factor: Pyramid decimation ratio.
        _detector: OpenCV ORB instance.
    """

    def __init__(
        self,
        n_features: int = 1000,
        scale_factor: float = 1.2,
    ):
        """Initialize ORB detector.

        Args:
            n_features: Max keypoints to keep. Default 1000.
            scale_factor: Pyramid scale factor. Default 1.2.
        """
        self._n_features = n_features
        self._scale_factor = scale_factor
        self._detector = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
        )

    @property
    def name(self) -> str:
        """Return detector name.

        Returns:
            str: 'ORB' with config parameters.
        """
        return f"ORB (n={self._n_features}, sf={self._scale_factor})"

    @property
    def norm_type(self) -> int:
        """Return Hamming norm for binary descriptors.

        Returns:
            int: cv2.NORM_HAMMING.
        """
        return cv2.NORM_HAMMING

    def detect_and_compute(
        self,
        image: np.ndarray,
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Detect keypoints and compute ORB descriptors.

        Args:
            image: Input image (grayscale recommended).

        Returns:
            Tuple of (keypoints, 32-byte binary descriptors).
        """
        gray = self._ensure_grayscale(image)
        return self._detector.detectAndCompute(gray, None)

    @staticmethod
    def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert to grayscale if needed.

        Args:
            image: Input image.

        Returns:
            np.ndarray: Grayscale image.
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
