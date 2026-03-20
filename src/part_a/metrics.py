"""Image quality metrics for filter comparison.

Provides PSNR and SSIM calculations for quantitative analysis.
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim


class ImageMetrics:
    """Utility class for computing image quality metrics.

    Provides methods to calculate PSNR and SSIM for comparing
    filtered images against original references.
    """

    @staticmethod
    def calculate_psnr(
        original: np.ndarray,
        processed: np.ndarray,
        max_pixel_value: float = 255.0,
    ) -> float:
        """Calculate Peak Signal-to-Noise Ratio (PSNR).

        PSNR = 10 * log10(MAX^2 / MSE)

        Args:
            original: Reference image.
            processed: Processed/filtered image.
            max_pixel_value: Maximum possible pixel value. Default is 255.

        Returns:
            float: PSNR value in decibels (dB). Higher is better.
        """
        mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)

        if mse == 0:
            return float("inf")

        psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
        return psnr

    @staticmethod
    def calculate_mse(original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate Mean Squared Error (MSE).

        Args:
            original: Reference image.
            processed: Processed/filtered image.

        Returns:
            float: MSE value. Lower is better.
        """
        return np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)

    @staticmethod
    def calculate_ssim(
        original: np.ndarray,
        processed: np.ndarray,
        multichannel: bool = False,
    ) -> float:
        """Calculate Structural Similarity Index (SSIM).

        SSIM measures perceived quality considering luminance,
        contrast, and structure.

        Args:
            original: Reference image.
            processed: Processed/filtered image.
            multichannel: True if images are color (BGR). Default is False.

        Returns:
            float: SSIM value between -1 and 1. Higher is better.
        """
        is_color = len(original.shape) == 3 and original.shape[2] == 3

        if is_color or multichannel:
            return ssim(
                original,
                processed,
                channel_axis=2,
                data_range=original.max() - original.min(),
            )

        return ssim(
            original,
            processed,
            data_range=original.max() - original.min(),
        )
