"""Laplacian Sharpening Filter implementation.

Enhances edges using Laplacian second-derivative operator.
"""

from typing import Tuple

import cv2
import numpy as np

from src.part_a.filter_interface import ImageFilterInterface


class LaplacianSharpener(ImageFilterInterface):
    """Laplacian Sharpening Filter for edge enhancement.

    This filter uses the Laplacian operator to detect edges and
    subtracts them from the original image to enhance sharpness.

    Formula: sharpened = original - alpha * laplacian(original)

    Attributes:
        _kernel_size: Size of the Laplacian kernel.
        _alpha: Sharpening intensity factor.
    """

    def __init__(self, kernel_size: int = 3, alpha: float = 1.0):
        """Initialize the Laplacian Sharpener.

        Args:
            kernel_size: Size of the Laplacian kernel. Must be 1, 3, 5, or 7.
            alpha: Sharpening intensity. Higher = more sharpening.

        Raises:
            ValueError: If kernel_size is not 1, 3, 5, or 7.
        """
        valid_sizes = [1, 3, 5, 7]
        if kernel_size not in valid_sizes:
            raise ValueError(f"Kernel size must be one of {valid_sizes}.")
        self._kernel_size = kernel_size
        self._alpha = alpha

    @property
    def name(self) -> str:
        """Return the filter name.

        Returns:
            str: 'Laplacian Sharpener' with alpha value.
        """
        return f"Laplacian Sharpener (α={self._alpha})"

    @property
    def kernel_size(self) -> Tuple[int, int]:
        """Return the kernel dimensions.

        Returns:
            Tuple[int, int]: Square kernel size.
        """
        return (self._kernel_size, self._kernel_size)

    @property
    def alpha(self) -> float:
        """Return the sharpening intensity.

        Returns:
            float: Alpha coefficient.
        """
        return self._alpha

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Laplacian sharpening to the image.

        Args:
            image: Input image (grayscale or BGR).

        Returns:
            np.ndarray: Sharpened image.
        """
        laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=self._kernel_size)
        sharpened = image.astype(np.float64) - self._alpha * laplacian # core code
        return np.clip(sharpened, 0, 255).astype(np.uint8)
