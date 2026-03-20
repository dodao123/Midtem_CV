"""Stereo view synthesis from monocular image + depth.

Generates a synthetic right stereo view by shifting pixels
horizontally based on estimated depth values.
"""

import cv2
import numpy as np


class StereoViewSynthesizer:
    """Synthesizes a right stereo view from left image + depth.

    Shifts pixels horizontally based on depth: near objects
    are shifted more, far objects are shifted less.

    Attributes:
        _baseline_px: Maximum pixel shift for nearest objects.
    """

    def __init__(self, baseline_px: int = 40):
        """Initialize synthesizer with baseline shift.

        Args:
            baseline_px: Max horizontal shift in pixels.
                Higher values create more 3D depth effect.
        """
        self._baseline_px = baseline_px

    def synthesize_right_view(
        self,
        left_image: np.ndarray,
        depth_map: np.ndarray,
    ) -> np.ndarray:
        """Generate right stereo view from left + depth.

        Args:
            left_image: BGR image (H, W, 3), uint8.
            depth_map: Normalized depth (H, W), float32,
                0.0 = far, 1.0 = near.

        Returns:
            np.ndarray: Synthesized right view (H, W, 3).
        """
        disparity = self._depth_to_disparity(depth_map)
        warped = self._forward_warp(left_image, disparity)
        filled = self._fill_holes(warped)
        return filled

    def _depth_to_disparity(
        self, depth_map: np.ndarray,
    ) -> np.ndarray:
        """Convert normalized depth to pixel disparity.

        Args:
            depth_map: Depth values in 0.0-1.0 range.

        Returns:
            np.ndarray: Disparity in pixels (float32).
        """
        return (depth_map * self._baseline_px).astype(
            np.float32
        )

    @staticmethod
    def _forward_warp(
        image: np.ndarray,
        disparity: np.ndarray,
    ) -> np.ndarray:
        """Warp left image to right view via remap.

        Uses cv2.remap for smooth sub-pixel interpolation
        instead of manual integer-based pixel shifting.

        Args:
            image: Source BGR image.
            disparity: Per-pixel horizontal shift.

        Returns:
            np.ndarray: Warped right-view image.
        """
        height, width = image.shape[:2]

        # Build remap: for each right pixel (u, v),
        # find source pixel at (u + disparity, v)
        map_x = np.zeros(
            (height, width), dtype=np.float32,
        )
        map_y = np.zeros(
            (height, width), dtype=np.float32,
        )

        col_grid = np.tile(
            np.arange(width, dtype=np.float32), (height, 1),
        )
        row_grid = np.tile(
            np.arange(height, dtype=np.float32).reshape(
                -1, 1,
            ), (1, width),
        )

        map_x = col_grid + disparity
        map_y = row_grid

        warped = cv2.remap(
            image, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return warped

    @staticmethod
    def _fill_holes(image: np.ndarray) -> np.ndarray:
        """Fill any remaining dark artifacts.

        Args:
            image: Warped image possibly with artifacts.

        Returns:
            np.ndarray: Cleaned image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hole_mask = (gray < 5).astype(np.uint8) * 255

        if hole_mask.sum() < 100:
            return image

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3),
        )
        hole_mask = cv2.dilate(hole_mask, kernel)

        return cv2.inpaint(
            image, hole_mask, 5, cv2.INPAINT_TELEA,
        )
