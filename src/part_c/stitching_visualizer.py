"""Visualization utilities for image stitching results.

Provides functions to draw keypoints, feature matches,
and save panorama outputs with annotations.
"""

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


class StitchingVisualizer:
    """Generates visualizations for the stitching pipeline.

    Attributes:
        _output_dir: Directory to save visualization images.
    """

    def __init__(self, output_dir: str):
        """Initialize visualizer.

        Args:
            output_dir: Path to save output images.
        """
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def draw_keypoints(
        self,
        image: np.ndarray,
        keypoints: list,
        filename: str,
    ) -> str:
        """Draw detected keypoints on an image.

        Args:
            image: Source image (BGR).
            keypoints: List of cv2.KeyPoint objects.
            filename: Output filename (without extension).

        Returns:
            str: Path to saved image.
        """
        result = cv2.drawKeypoints(
            image, keypoints, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        path = os.path.join(self._output_dir, f"{filename}.png")
        cv2.imwrite(path, result)
        return path

    def draw_matches(
        self,
        img1: np.ndarray,
        kp1: list,
        img2: np.ndarray,
        kp2: list,
        matches: list,
        filename: str,
        max_display: int = 50,
    ) -> str:
        """Draw feature matches between two images.

        Args:
            img1: First image (BGR).
            kp1: Keypoints from first image.
            img2: Second image (BGR).
            kp2: Keypoints from second image.
            matches: List of DMatch objects.
            filename: Output filename (without extension).
            max_display: Max matches to draw. Default 50.

        Returns:
            str: Path to saved image.
        """
        display_matches = sorted(
            matches, key=lambda m: m.distance,
        )[:max_display]
        result = cv2.drawMatches(
            img1, kp1, img2, kp2, display_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        path = os.path.join(self._output_dir, f"{filename}.png")
        cv2.imwrite(path, result)
        return path

    def save_panorama(
        self,
        panorama: np.ndarray,
        filename: str,
    ) -> str:
        """Save the final panorama image.

        Args:
            panorama: Stitched panorama (BGR).
            filename: Output filename (without extension).

        Returns:
            str: Path to saved image.
        """
        path = os.path.join(self._output_dir, f"{filename}.png")
        cv2.imwrite(path, panorama)
        return path
