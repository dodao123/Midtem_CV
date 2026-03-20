"""Abstract base class for feature detectors.

Defines the contract for all feature detection implementations
used in the image stitching pipeline.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import cv2
import numpy as np


class FeatureDetectorInterface(ABC):
    """Abstract base class for feature detection algorithms.

    All concrete detectors must implement detect_and_compute
    to extract keypoints and descriptors from an image.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the human-readable detector name.

        Returns:
            str: Detector name for display purposes.
        """
        pass

    @abstractmethod
    def detect_and_compute(
        self,
        image: np.ndarray,
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Detect keypoints and compute descriptors.

        Args:
            image: Input image (grayscale or BGR).

        Returns:
            Tuple of (keypoints list, descriptors ndarray).
        """
        pass

    @property
    @abstractmethod
    def norm_type(self) -> int:
        """Return the norm type for descriptor matching.

        Returns:
            int: cv2.NORM_L2 for float descriptors,
                 cv2.NORM_HAMMING for binary descriptors.
        """
        pass
