"""SIFT feature detector implementation.

Uses Scale-Invariant Feature Transform for robust keypoint
detection and descriptor computation.
"""

from typing import List, Tuple

import cv2
import numpy as np

from src.part_c.detector_interface import FeatureDetectorInterface


class SiftDetector(FeatureDetectorInterface):
    """SIFT-based feature detector.

    Attributes:
        _n_features: Maximum number of keypoints to retain.
        _contrast_threshold: Filter for low-contrast keypoints.
        _detector: OpenCV SIFT instance.
    """

    def __init__(
        self,
        n_features: int = 0,
        contrast_threshold: float = 0.04,
    ):
        """Initialize SIFT detector.

        Args:
            n_features: Max features (0 = unlimited). Default 0.
            contrast_threshold: Contrast filter threshold.
        """
        self._n_features = n_features
        self._contrast_threshold = contrast_threshold
        self._detector = cv2.SIFT_create(
            nfeatures=n_features,
            contrastThreshold=contrast_threshold,
        )

    @property
    def name(self) -> str:
        """Return detector name.

        Returns:
            str: 'SIFT' with config parameters.
        """
        return f"SIFT (n={self._n_features}, ct={self._contrast_threshold})"

    @property
    def norm_type(self) -> int:
        """Return L2 norm for float descriptors.

        Returns:
            int: cv2.NORM_L2.
        """
        return cv2.NORM_L2

    def detect_and_compute(
        self,
        image: np.ndarray,
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Detect keypoints and compute SIFT descriptors.

        Args:
            image: Input image (grayscale recommended).

        Returns:
            Tuple of (keypoints, 128-d float descriptors).
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
