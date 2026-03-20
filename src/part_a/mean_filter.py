"""Mean Filter implementation.

Applies a simple averaging filter to smooth the image.
"""

from typing import Tuple

import cv2
import numpy as np

from src.part_a.filter_interface import ImageFilterInterface


class MeanFilter(ImageFilterInterface):
    """Mean (Box) Filter for image smoothing.

    This filter replaces each pixel with the average value
    of its neighboring pixels within the kernel window.

    Attributes:
        _kernel_size: Size of the averaging kernel.
    """

    def __init__(self, kernel_size: int = 5):
        """Initialize the Mean Filter.

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
            str: 'Mean Filter'.
        """
        return "Mean Filter"

    @property
    def kernel_size(self) -> Tuple[int, int]:
        """Return the kernel dimensions.

        Returns:
            Tuple[int, int]: Square kernel size.
        """
        return (self._kernel_size, self._kernel_size)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply mean filtering to the image.

        Args:
            image: Input image (grayscale or BGR).

        Returns:
            np.ndarray: Smoothed image.
        """
        return cv2.blur(image, self.kernel_size)
