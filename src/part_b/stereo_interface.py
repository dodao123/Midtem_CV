"""Abstract base class for stereo matching algorithms.

Defines the interface for disparity computation from stereo image pairs.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class StereoMatcherInterface(ABC):
    """Abstract base class for stereo matching algorithms.

    All stereo matcher implementations must inherit from this class
    and implement the compute_disparity method.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the algorithm name.

        Returns:
            str: Human-readable algorithm name.
        """
        pass

    @abstractmethod
    def compute_disparity(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
    ) -> np.ndarray:
        """Compute disparity map from stereo image pair.

        Args:
            left_image: Left camera image (grayscale).
            right_image: Right camera image (grayscale).

        Returns:
            np.ndarray: Disparity map (float32).
        """
        pass

    @staticmethod
    def disparity_to_depth(
        disparity: np.ndarray,
        focal_length: float,
        baseline: float,
    ) -> np.ndarray:
        """Convert disparity map to depth map.

        Depth formula: Z = (f * B) / d
        Where f = focal length, B = baseline, d = disparity.

        Args:
            disparity: Disparity map.
            focal_length: Camera focal length in pixels.
            baseline: Distance between camera centers in same units as depth.

        Returns:
            np.ndarray: Depth map.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = (focal_length * baseline) / disparity
            depth[disparity <= 0] = 0
        return depth
