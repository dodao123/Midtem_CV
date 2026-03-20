"""Image warping and blending for panorama construction.

Handles perspective warping of images using computed homography
matrices and multi-band blending for seamless stitching.
"""

from typing import Tuple

import cv2
import numpy as np


class ImageWarper:
    """Warps images using homography and blends overlapping regions.

    Supports alpha blending and feathering to reduce visible seams
    at image boundaries.

    Attributes:
        _blend_mode: Blending strategy ('alpha' or 'average').
    """

    def __init__(self, blend_mode: str = "alpha"):
        """Initialize ImageWarper.

        Args:
            blend_mode: 'alpha' for linear feathering, 'average' for
                simple average blending. Default is 'alpha'.
        """
        self._blend_mode = blend_mode

    def warp_image(
        self,
        image: np.ndarray,
        homography: np.ndarray,
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        """Warp image using a homography matrix.

        Args:
            image: Source image to warp (BGR).
            homography: 3x3 homography matrix.
            output_size: Target canvas size as (width, height).

        Returns:
            np.ndarray: Warped image on a black canvas.
        """
        width, height = output_size
        warped = cv2.warpPerspective(image, homography, (width, height))
        return warped

    def blend_two(
        self,
        base: np.ndarray,
        overlay: np.ndarray,
    ) -> np.ndarray:
        """Blend two images where overlay is laid on top of base.

        Uses alpha feathering in overlapping regions to reduce seams.

        Args:
            base: Background image (BGR, same size as overlay).
            overlay: Foreground image (BGR, same size as base).

        Returns:
            np.ndarray: Blended BGR image.
        """
        base_mask = (base.sum(axis=2) > 0).astype(np.float32)
        overlay_mask = (overlay.sum(axis=2) > 0).astype(np.float32)
        overlap = (base_mask * overlay_mask)[..., np.newaxis]

        if self._blend_mode == "alpha":
            return self._alpha_blend(base, overlay, overlap)

        # Fallback: overlay takes precedence
        result = base.copy()
        mask = overlay_mask.astype(bool)
        result[mask] = overlay[mask]
        return result

    @staticmethod
    def _alpha_blend(
        base: np.ndarray,
        overlay: np.ndarray,
        overlap_mask: np.ndarray,
    ) -> np.ndarray:
        """Apply 50/50 blending in overlapping regions.

        Args:
            base: Background image.
            overlay: Foreground image.
            overlap_mask: Binary mask of overlapping pixels (H,W,1).

        Returns:
            np.ndarray: Blended image.
        """
        base_f = base.astype(np.float32)
        overlay_f = overlay.astype(np.float32)

        # Where both images have content: average them
        blended = np.where(
            overlap_mask > 0,
            (base_f * 0.5 + overlay_f * 0.5),
            np.where(overlay_f.sum(axis=2, keepdims=True) > 0, overlay_f, base_f),
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    def compute_canvas_size(
        self,
        images: list,
        homographies: list,
    ) -> Tuple[Tuple[int, int], np.ndarray]:
        """Compute panorama canvas size fitting all warped images.

        Args:
            images: List of source images.
            homographies: List of homography matrices (same length as images).

        Returns:
            Tuple of ((canvas_width, canvas_height), offset_translation).
        """
        corners_all = []
        for img, h_matrix in zip(images, homographies):
            img_h, img_w = img.shape[:2]
            corners = np.array([
                [0, 0], [img_w, 0], [img_w, img_h], [0, img_h]
            ], dtype=np.float32).reshape(-1, 1, 2)
            warped_corners = cv2.perspectiveTransform(corners, h_matrix)
            corners_all.append(warped_corners)

        all_corners = np.concatenate(corners_all, axis=0)
        x_min, y_min = all_corners[:, 0, 0].min(), all_corners[:, 0, 1].min()
        x_max, y_max = all_corners[:, 0, 0].max(), all_corners[:, 0, 1].max()

        offset = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1],
        ], dtype=np.float64)

        canvas_w = int(np.ceil(x_max - x_min))
        canvas_h = int(np.ceil(y_max - y_min))
        return (canvas_w, canvas_h), offset
