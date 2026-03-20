"""Gaussian Filter implementation.

Applies Gaussian blur for smooth noise reduction with edge preservation.
"""

from typing import Tuple

import cv2
import numpy as np

from src.part_a.filter_interface import ImageFilterInterface


class GaussianFilter(ImageFilterInterface):
    """Gaussian Filter for image smoothing.

    This filter applies a Gaussian-weighted average to reduce noise
    while better preserving edges compared to mean filtering.

    Attributes:
        _kernel_size: Size of the Gaussian kernel.
        _sigma: Standard deviation of the Gaussian distribution.
    """

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        """Initialize the Gaussian Filter.

        Args:
            kernel_size: Size of the square kernel. Must be odd. Default is 5.
            sigma: Standard deviation (σ) of Gaussian. Default is 1.0.

        Raises:
            ValueError: If kernel_size is not a positive odd integer.
        """
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be a positive odd integer.")
        self._kernel_size = kernel_size
        self._sigma = sigma

    @property
    def name(self) -> str:
        """Return the filter name.

        Returns:
            str: 'Gaussian Filter' with sigma value.
        """
        return f"Gaussian Filter (σ={self._sigma})"

    @property
    def kernel_size(self) -> Tuple[int, int]:
        """Return the kernel dimensions.

        Returns:
            Tuple[int, int]: Square kernel size.
        """
        return (self._kernel_size, self._kernel_size)

    @property
    def sigma(self) -> float:
        """Return the sigma value.

        Returns:
            float: Standard deviation of the Gaussian.
        """
        return self._sigma

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian filtering to the image.

        Args:
            image: Input image (grayscale or BGR).

        Returns:
            np.ndarray: Gaussian-blurred image.
        """
        return cv2.GaussianBlur(image, self.kernel_size, self._sigma)
