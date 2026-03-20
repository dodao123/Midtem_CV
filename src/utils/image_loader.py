"""Utility module for image I/O operations."""

import os
from typing import Optional

import cv2
import numpy as np


class ImageLoader:
    """Utility class for loading and saving images."""

    @staticmethod
    def load_image(
        path: str,
        grayscale: bool = False,
    ) -> Optional[np.ndarray]:
        """Load an image from disk.

        Args:
            path: Path to the image file.
            grayscale: If True, load as grayscale. Default is False.

        Returns:
            np.ndarray or None: Loaded image, or None if failed.
        """
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        image = cv2.imread(path, flag)

        if image is None:
            print(f"Warning: Could not load image from {path}")

        return image

    @staticmethod
    def save_image(image: np.ndarray, path: str) -> bool:
        """Save an image to disk.

        Args:
            image: Image array to save.
            path: Destination file path.

        Returns:
            bool: True if successful, False otherwise.
        """
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        return cv2.imwrite(path, image)

    @staticmethod
    def create_sample_image(
        width: int = 256,
        height: int = 256,
        pattern: str = "gradient",
    ) -> np.ndarray:
        """Create a sample test image.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            pattern: Pattern type ('gradient', 'checkerboard', 'circles').

        Returns:
            np.ndarray: Generated grayscale test image.
        """
        if pattern == "gradient":
            return np.tile(
                np.linspace(0, 255, width, dtype=np.uint8),
                (height, 1),
            )

        if pattern == "checkerboard":
            block_size = 32
            img = np.zeros((height, width), dtype=np.uint8)
            for i in range(0, height, block_size):
                for j in range(0, width, block_size):
                    if (i // block_size + j // block_size) % 2 == 0:
                        img[i:i + block_size, j:j + block_size] = 255
            return img

        if pattern == "circles":
            img = np.zeros((height, width), dtype=np.uint8)
            center = (width // 2, height // 2)
            for radius in range(20, min(width, height) // 2, 30):
                cv2.circle(img, center, radius, 255, 2)
            return img

        return np.zeros((height, width), dtype=np.uint8)
