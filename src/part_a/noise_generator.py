"""Noise Generator for testing image filters.

Provides functions to add various types of noise to images.
"""

from enum import Enum
from typing import Optional

import numpy as np


class NoiseType(Enum):
    """Enumeration of supported noise types."""

    GAUSSIAN = "gaussian"
    SALT_AND_PEPPER = "salt_and_pepper"
    SPECKLE = "speckle"


class NoiseGenerator:
    """Utility class for generating various types of image noise.

    This class provides static methods to add different noise types
    to images for testing filter effectiveness.
    """

    @staticmethod
    def add_gaussian_noise(
        image: np.ndarray,
        mean: float = 0.0,
        sigma: float = 25.0,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Add Gaussian noise to an image.

        Args:
            image: Input image as numpy array.
            mean: Mean of the Gaussian distribution. Default is 0.
            sigma: Standard deviation of the noise. Default is 25.
            seed: Random seed for reproducibility. Optional.

        Returns:
            np.ndarray: Image with added Gaussian noise.
        """
        if seed is not None:
            np.random.seed(seed)

        noise = np.random.normal(mean, sigma, image.shape)
        noisy_image = image.astype(np.float64) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    @staticmethod
    def add_salt_and_pepper_noise(
        image: np.ndarray,
        salt_probability: float = 0.02,
        pepper_probability: float = 0.02,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Add salt and pepper noise to an image.

        Args:
            image: Input image as numpy array.
            salt_probability: Probability of salt (white) pixels.
            pepper_probability: Probability of pepper (black) pixels.
            seed: Random seed for reproducibility. Optional.

        Returns:
            np.ndarray: Image with salt and pepper noise.
        """
        if seed is not None:
            np.random.seed(seed)

        noisy_image = image.copy()
        total_pixels = image.size

        # Add salt (white pixels)
        num_salt = int(total_pixels * salt_probability)
        salt_coords = [
            np.random.randint(0, dim, num_salt) for dim in image.shape[:2]
        ]
        noisy_image[salt_coords[0], salt_coords[1]] = 255

        # Add pepper (black pixels)
        num_pepper = int(total_pixels * pepper_probability)
        pepper_coords = [
            np.random.randint(0, dim, num_pepper) for dim in image.shape[:2]
        ]
        noisy_image[pepper_coords[0], pepper_coords[1]] = 0

        return noisy_image

    @staticmethod
    def add_speckle_noise(
        image: np.ndarray,
        variance: float = 0.04,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Add speckle (multiplicative) noise to an image.

        Args:
            image: Input image as numpy array.
            variance: Variance of the speckle noise. Default is 0.04.
            seed: Random seed for reproducibility. Optional.

        Returns:
            np.ndarray: Image with speckle noise.
        """
        if seed is not None:
            np.random.seed(seed)

        noise = np.random.randn(*image.shape) * np.sqrt(variance)
        noisy_image = image.astype(np.float64) * (1 + noise)
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
