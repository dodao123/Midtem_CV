"""Homography estimation using RANSAC.

Estimates the perspective transformation matrix (3x3 homography)
from matched point correspondences.
"""

from typing import Tuple, Optional

import cv2
import numpy as np


class HomographyEstimator:
    """Estimates homography matrix with RANSAC outlier rejection.

    The homography H maps points from image src to image dst
    such that: dst_point = H @ src_point (in homogeneous coords).

    Attributes:
        _ransac_threshold: Max reprojection error for inliers.
        _confidence: RANSAC confidence level.
    """

    def __init__(
        self,
        ransac_threshold: float = 4.0,
        confidence: float = 0.995,
    ):
        """Initialize homography estimator.

        Args:
            ransac_threshold: Max pixel error to count as inlier.
            confidence: RANSAC confidence, between 0 and 1.
        """
        self._ransac_threshold = ransac_threshold
        self._confidence = confidence

    def estimate(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """Estimate homography matrix using RANSAC.

        Args:
            src_points: Source point array, shape (N, 1, 2) float32.
            dst_points: Destination point array, shape (N, 1, 2) float32.

        Returns:
            Tuple of:
                - Homography matrix (3x3) or None if failed.
                - Inlier mask array or None if failed.
                - Number of inliers found.
        """
        if len(src_points) < 4:
            return None, None, 0

        homography, mask = cv2.findHomography(
            src_points,
            dst_points,
            cv2.RANSAC,
            self._ransac_threshold,
            confidence=self._confidence,
        )

        if homography is None:
            return None, None, 0

        num_inliers = int(mask.sum()) if mask is not None else 0
        return homography, mask, num_inliers

    def estimate_inverse(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """Estimate inverse homography (dst -> src direction).

        Args:
            src_points: Source point array.
            dst_points: Destination point array.

        Returns:
            Tuple of (inverse_H, mask, num_inliers).
        """
        homography, mask, num_inliers = self.estimate(src_points, dst_points)
        if homography is None:
            return None, None, 0
        return np.linalg.inv(homography), mask, num_inliers
