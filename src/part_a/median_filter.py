"""Median Filter implementation.

Applies median filtering - excellent for salt-and-pepper noise removal.
"""

from typing import Tuple

import cv2
import numpy as np

from src.part_a.filter_interface import ImageFilterInterface


class MedianFilter(ImageFilterInterface):
    """Median Filter for noise removal.

    This filter replaces each pixel with the median value of its
    neighbors. Highly effective for salt-and-pepper noise removal
    while preserving edges.

    Attributes:
        _kernel_size: Size of the median kernel.
    """

    def __init__(self, kernel_size: int = 5):
        """Initialize the Median Filter.

        Args:
            kernel_size: Size of the square kernel. Must be odd. Default is 5.

        Raises:
            ValueError: If kernel_size is not a positive odd integer.
        """
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be a positive odd integer.")
        self._kernel_size = kernel_size

    @property
    def name(self) -> str:
        """Return the filter name.

        Returns:
            str: 'Median Filter'.
        """
        return "Median Filter"

    @property
    def kernel_size(self) -> Tuple[int, int]:
        """Return the kernel dimensions.

        Returns:
            Tuple[int, int]: Square kernel size.
        """
        return (self._kernel_size, self._kernel_size)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply median filtering to the image.

        Args:
            image: Input image (grayscale or BGR).

        Returns:
            np.ndarray: Median-filtered image.
        """
        return cv2.medianBlur(image, self._kernel_size)
