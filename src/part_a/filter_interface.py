"""Abstract base class for all image filters.

This module defines the interface that all filter implementations must follow.
Following the SOLID principles - specifically Interface Segregation.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class ImageFilterInterface(ABC):
    """Abstract base class defining the contract for image filters.

    All concrete filter implementations must inherit from this class
    and implement the apply method.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the human-readable name of the filter.

        Returns:
            str: The filter name for display purposes.
        """
        pass

    @property
    @abstractmethod
    def kernel_size(self) -> Tuple[int, int]:
        """Return the kernel size used by the filter.

        Returns:
            Tuple[int, int]: The (width, height) of the kernel.
        """
        pass

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the filter to the input image.

        Args:
            image: Input image as numpy array (grayscale or BGR).

        Returns:
            np.ndarray: Filtered image with same shape as input.
        """
        pass

    def __repr__(self) -> str:
        """Return string representation of the filter.

        Returns:
            str: Class name with kernel size.
        """
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size})"
