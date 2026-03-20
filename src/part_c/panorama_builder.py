"""Panorama builder orchestrating the full stitching pipeline.

Chains feature detection, matching, homography estimation,
warping, and blending to produce a final panorama image.
"""

from typing import List

import cv2
import numpy as np

from src.part_c.detector_interface import FeatureDetectorInterface
from src.part_c.feature_matcher import FeatureMatcher
from src.part_c.homography_estimator import HomographyEstimator
from src.part_c.image_warper import ImageWarper


class PanoramaBuilder:
    """Builds a panorama from multiple overlapping images.

    Attributes:
        _detector: Feature detector instance.
        _matcher: Feature matcher instance.
        _estimator: Homography estimator instance.
        _warper: Image warper instance.
    """

    def __init__(
        self,
        detector: FeatureDetectorInterface,
        ratio_threshold: float = 0.75,
        ransac_threshold: float = 4.0,
    ):
        """Initialize panorama builder.

        Args:
            detector: Feature detector to use.
            ratio_threshold: Lowe's ratio test threshold.
            ransac_threshold: RANSAC reprojection error threshold.
        """
        self._detector = detector
        self._matcher = FeatureMatcher(
            norm_type=detector.norm_type,
            ratio_threshold=ratio_threshold,
        )
        self._estimator = HomographyEstimator(
            ransac_threshold=ransac_threshold,
        )
        self._warper = ImageWarper(blend_mode="alpha")

    def stitch(self, images: List[np.ndarray]) -> np.ndarray:
        """Stitch multiple images into a panorama.

        Stitches sequentially from left to right, accumulating
        into a single canvas.

        Args:
            images: List of BGR images with overlap.

        Returns:
            np.ndarray: Final stitched panorama (BGR).

        Raises:
            ValueError: If fewer than 2 images provided.
        """
        if len(images) < 2:
            raise ValueError("Need at least 2 images to stitch.")

        result = images[0]
        for i in range(1, len(images)):
            result = self._stitch_pair(result, images[i])
        return self._crop_black_borders(result)

    def _stitch_pair(
        self,
        base: np.ndarray,
        next_image: np.ndarray,
    ) -> np.ndarray:
        """Stitch two images together.

        Args:
            base: Current accumulated panorama.
            next_image: Next image to stitch on.

        Returns:
            np.ndarray: Combined panorama.
        """
        kp1, des1 = self._detector.detect_and_compute(base)
        kp2, des2 = self._detector.detect_and_compute(next_image)

        matches = self._matcher.match(des1, des2)
        if len(matches) < 4:
            return base

        src, dst = FeatureMatcher.extract_matched_points(kp1, kp2, matches)
        homography, _, _ = self._estimator.estimate(dst, src)
        if homography is None:
            return base

        canvas_size, offset = self._warper.compute_canvas_size(
            [base, next_image],
            [offset_identity(base), homography],
        )
        base_warped = self._warper.warp_image(base, offset, canvas_size)
        next_warped = self._warper.warp_image(
            next_image, offset @ homography, canvas_size,
        )
        return self._warper.blend_two(base_warped, next_warped)

    @staticmethod
    def _crop_black_borders(image: np.ndarray) -> np.ndarray:
        """Remove black borders from stitched panorama.

        Args:
            image: Panorama with potential black borders.

        Returns:
            np.ndarray: Cropped image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return image
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return image[y:y + h, x:x + w]


def offset_identity(image: np.ndarray) -> np.ndarray:
    """Create identity homography for the base image.

    Args:
        image: The base image (unused, identity transform).

    Returns:
        np.ndarray: 3x3 identity matrix.
    """
    return np.eye(3, dtype=np.float64)
