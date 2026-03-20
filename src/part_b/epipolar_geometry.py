"""Epipolar Geometry utilities for stereo vision.

Computes fundamental matrix and draws epipolar lines.
"""

from typing import Tuple, List

import cv2
import numpy as np


class EpipolarGeometry:
    """Handles epipolar geometry computations for stereo pairs.

    Computes fundamental matrix and visualizes epipolar lines.
    """

    def __init__(self, method: int = cv2.FM_RANSAC, threshold: float = 3.0):
        """Initialize Epipolar Geometry calculator.

        Args:
            method: cv2.FM_RANSAC, cv2.FM_LMEDS, or cv2.FM_8POINT.
            threshold: RANSAC threshold for outlier rejection.
        """
        self._method = method
        self._threshold = threshold
        self._fundamental_matrix = None
        self._inlier_mask = None

    def compute_fundamental_matrix(
        self,
        points_left: np.ndarray,
        points_right: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute fundamental matrix from point correspondences.

        F satisfies the epipolar constraint: x'.T @ F @ x = 0

        Args:
            points_left: Nx2 array of points in left image.
            points_right: Nx2 array of corresponding points in right.

        Returns:
            Tuple of (F, mask) where:
                F: 3x3 fundamental matrix.
                mask: Inlier mask array.
        """
        points_left = np.float32(points_left)
        points_right = np.float32(points_right)

        self._fundamental_matrix, self._inlier_mask = cv2.findFundamentalMat(
            points_left,
            points_right,
            method=self._method,
            ransacReprojThreshold=self._threshold,
        )

        return self._fundamental_matrix, self._inlier_mask

    def compute_epipolar_lines(
        self,
        points: np.ndarray,
        which_image: int,
    ) -> np.ndarray:
        """Compute epipolar lines for given points.

        Args:
            points: Nx2 array of points.
            which_image: 1 for lines in image 2, 2 for lines in image 1.

        Returns:
            np.ndarray: Nx3 array of line coefficients [a, b, c].
        """
        if self._fundamental_matrix is None:
            raise ValueError("Fundamental matrix not computed yet.")

        points = np.float32(points).reshape(-1, 1, 2)
        lines = cv2.computeCorrespondEpilines(
            points,
            which_image,
            self._fundamental_matrix,
        )
        return lines.reshape(-1, 3)

    @staticmethod
    def draw_epipolar_lines(
        image: np.ndarray,
        lines: np.ndarray,
        points: np.ndarray,
        colors: List[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Draw epipolar lines and corresponding points on image.

        Args:
            image: Input image (will be converted to color).
            lines: Nx3 array of line coefficients.
            points: Nx2 array of points.
            colors: List of BGR colors for each line/point.

        Returns:
            np.ndarray: Image with drawn lines and points.
        """
        if len(image.shape) == 2:
            output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            output = image.copy()

        height, width = output.shape[:2]

        for idx, (line, point) in enumerate(zip(lines, points)):
            if colors is not None:
                color = colors[idx]
            else:
                color = tuple(np.random.randint(0, 255, 3).tolist())

            a, b, c = line
            # Line equation: ax + by + c = 0
            # Find intersection with image borders
            x0, y0 = 0, int(-c / b) if b != 0 else 0
            x1, y1 = width, int(-(c + a * width) / b) if b != 0 else 0

            cv2.line(output, (x0, y0), (x1, y1), color, 1)
            cv2.circle(output, tuple(map(int, point)), 5, color, -1)

        return output

    @property
    def fundamental_matrix(self) -> np.ndarray:
        """Get the computed fundamental matrix.

        Returns:
            np.ndarray: 3x3 fundamental matrix.
        """
        return self._fundamental_matrix

    @property
    def inlier_ratio(self) -> float:
        """Calculate the inlier ratio from RANSAC.

        Returns:
            float: Ratio of inliers to total points.
        """
        if self._inlier_mask is None:
            return 0.0
        return np.sum(self._inlier_mask) / len(self._inlier_mask)
