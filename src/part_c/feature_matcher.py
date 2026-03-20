"""Feature matching between image pairs.

Matches descriptors using BFMatcher with Lowe's ratio test
to filter out ambiguous matches.
"""

from typing import List

import cv2
import numpy as np


class FeatureMatcher:
    """Matches feature descriptors between two images.

    Uses BFMatcher with KNN and Lowe's ratio test to produce
    high-quality correspondences.

    Attributes:
        _ratio_threshold: Max ratio for Lowe's test (lower = stricter).
        _norm_type: Distance norm (L2 for SIFT, Hamming for ORB).
    """

    def __init__(
        self,
        norm_type: int = cv2.NORM_L2,
        ratio_threshold: float = 0.75,
    ):
        """Initialize feature matcher.

        Args:
            norm_type: cv2.NORM_L2 or cv2.NORM_HAMMING.
            ratio_threshold: Lowe's ratio threshold. Default 0.75.
        """
        self._ratio_threshold = ratio_threshold
        self._norm_type = norm_type
        self._matcher = cv2.BFMatcher(norm_type)

    def match(
        self,
        descriptors_a: np.ndarray,
        descriptors_b: np.ndarray,
    ) -> List[cv2.DMatch]:
        """Match descriptors using KNN + Lowe's ratio test.

        Args:
            descriptors_a: Descriptors from first image.
            descriptors_b: Descriptors from second image.

        Returns:
            List of good DMatch objects passing ratio test.
        """
        if descriptors_a is None or descriptors_b is None:
            return []

        if len(descriptors_a) < 2 or len(descriptors_b) < 2:
            return []

        knn_matches = self._matcher.knnMatch(
            descriptors_a, descriptors_b, k=2,
        )
        return self._apply_ratio_test(knn_matches)

    def _apply_ratio_test(
        self,
        knn_matches: List,
    ) -> List[cv2.DMatch]:
        """Apply Lowe's ratio test to filter ambiguous matches.

        Args:
            knn_matches: Raw KNN match pairs from BFMatcher.

        Returns:
            List of good matches passing the ratio threshold.
        """
        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) < 2:
                continue
            best, second = match_pair
            if best.distance < self._ratio_threshold * second.distance:
                good_matches.append(best)
        return good_matches

    @staticmethod
    def extract_matched_points(
        keypoints_a: list,
        keypoints_b: list,
        matches: List[cv2.DMatch],
    ) -> tuple:
        """Extract matched point coordinates.

        Args:
            keypoints_a: Keypoints from image A.
            keypoints_b: Keypoints from image B.
            matches: List of DMatch objects.

        Returns:
            Tuple of (src_points, dst_points) as float32 arrays.
        """
        src = np.float32(
            [keypoints_a[m.queryIdx].pt for m in matches],
        ).reshape(-1, 1, 2)
        dst = np.float32(
            [keypoints_b[m.trainIdx].pt for m in matches],
        ).reshape(-1, 1, 2)
        return src, dst
