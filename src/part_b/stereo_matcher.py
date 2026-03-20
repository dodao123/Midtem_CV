"""Stereo Matching implementations using Block Matching and SGBM.

Provides two algorithms for computing disparity maps from stereo pairs.
"""

from typing import Tuple

import cv2
import numpy as np

from src.part_b.stereo_interface import StereoMatcherInterface


class StereoMatcherBM(StereoMatcherInterface):
    """Block Matching stereo algorithm.

    Simple and fast algorithm using Sum of Absolute Differences (SAD).

    Attributes:
        _num_disparities: Maximum disparity search range.
        _block_size: Size of the matching block.
    """

    def __init__(
        self,
        num_disparities: int = 64,
        block_size: int = 15,
    ):
        """Initialize Block Matching stereo matcher.

        Args:
            num_disparities: Must be divisible by 16. Default is 64.
            block_size: Must be odd, in range 5-255. Default is 15.

        Raises:
            ValueError: If parameters are invalid.
        """
        if num_disparities % 16 != 0:
            raise ValueError("num_disparities must be divisible by 16.")
        if block_size % 2 == 0 or block_size < 5:
            raise ValueError("block_size must be odd and >= 5.")

        self._num_disparities = num_disparities
        self._block_size = block_size
        self._stereo = cv2.StereoBM_create(
            numDisparities=num_disparities,
            blockSize=block_size,
        )

    @property
    def name(self) -> str:
        """Return algorithm name.

        Returns:
            str: 'Block Matching' with parameters.
        """
        return f"Block Matching (nDisp={self._num_disparities}, block={self._block_size})"

    def compute_disparity(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
    ) -> np.ndarray:
        """Compute disparity using Block Matching.

        Args:
            left_image: Left grayscale image.
            right_image: Right grayscale image.

        Returns:
            np.ndarray: Disparity map normalized to float32.
        """
        disparity = self._stereo.compute(left_image, right_image)
        return disparity.astype(np.float32) / 16.0


class StereoMatcherSGBM(StereoMatcherInterface):
    """Semi-Global Block Matching (SGBM) stereo algorithm.

    More accurate than BM, uses path-wise cost aggregation.

    Attributes:
        _num_disparities: Maximum disparity search range.
        _block_size: Size of the matching block.
        _mode: SGBM mode (3WAY, HH, etc.).
    """

    def __init__(
        self,
        num_disparities: int = 80,
        block_size: int = 5,
        mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    ):
        """Initialize SGBM stereo matcher.

        Args:
            num_disparities: Must be divisible by 16. Default is 80.
            block_size: Must be odd, typically 3-11. Default is 5.
            mode: SGBM mode. Default is SGBM_3WAY.
        """
        if num_disparities % 16 != 0:
            raise ValueError("num_disparities must be divisible by 16.")

        self._num_disparities = num_disparities
        self._block_size = block_size
        self._mode = mode

        # P1 and P2 control smoothness
        p1 = 8 * 3 * block_size ** 2
        p2 = 32 * 3 * block_size ** 2

        self._stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=mode,
        )

    @property
    def name(self) -> str:
        """Return algorithm name.

        Returns:
            str: 'SGBM' with parameters.
        """
        return f"SGBM (nDisp={self._num_disparities}, block={self._block_size})"

    def compute_disparity(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
    ) -> np.ndarray:
        """Compute disparity using SGBM.

        Args:
            left_image: Left grayscale image.
            right_image: Right grayscale image.

        Returns:
            np.ndarray: Disparity map normalized to float32.
        """
        disparity = self._stereo.compute(left_image, right_image)
        return disparity.astype(np.float32) / 16.0
