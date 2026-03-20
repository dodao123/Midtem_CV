"""Filter Experiment Runner for comparative analysis.

Runs all filter experiments and collects metrics for Part A.
"""

from typing import List, Dict, Any

import numpy as np

from src.part_a.filters import (
    ImageFilterInterface,
    MeanFilter,
    GaussianFilter,
    MedianFilter,
    LaplacianSharpener,
)
from src.part_a.metrics import ImageMetrics


class FilterExperiment:
    """Runs filter experiments and collects comparison data.

    Attributes:
        original_image: The clean reference image.
        noisy_image: The image with added noise.
        filters: List of filters to test.
    """

    def __init__(
        self,
        original_image: np.ndarray,
        noisy_image: np.ndarray,
        filters: List[ImageFilterInterface] = None,
    ):
        """Initialize the experiment.

        Args:
            original_image: Clean reference image.
            noisy_image: Image with noise to be filtered.
            filters: List of filters to test. Uses default set if None.
        """
        self._original = original_image
        self._noisy = noisy_image
        self._filters = filters or self._get_default_filters()
        self._results: List[Dict[str, Any]] = []

    def _get_default_filters(self) -> List[ImageFilterInterface]:
        """Get the default set of filters to test.

        Returns:
            List of filter instances.
        """
        return [
            MeanFilter(kernel_size=3),
            MeanFilter(kernel_size=5),
            GaussianFilter(kernel_size=5, sigma=1.0),
            GaussianFilter(kernel_size=5, sigma=2.0),
            MedianFilter(kernel_size=3),
            MedianFilter(kernel_size=5),
            LaplacianSharpener(kernel_size=3, alpha=0.5),
        ]

    def run(self) -> List[Dict[str, Any]]:
        """Execute all filter experiments.

        Returns:
            List of result dictionaries with metrics.
        """
        self._results = []

        for filter_instance in self._filters:
            filtered_image = filter_instance.apply(self._noisy)

            psnr = ImageMetrics.calculate_psnr(self._original, filtered_image)
            ssim = ImageMetrics.calculate_ssim(self._original, filtered_image)

            self._results.append({
                "filter_name": filter_instance.name,
                "kernel_size": filter_instance.kernel_size,
                "filtered_image": filtered_image,
                "psnr": psnr,
                "ssim": ssim,
            })

        return self._results

    @property
    def results(self) -> List[Dict[str, Any]]:
        """Get experiment results.

        Returns:
            List of result dictionaries.
        """
        return self._results

    def get_best_filter_by_psnr(self) -> Dict[str, Any]:
        """Get the filter with highest PSNR.

        Returns:
            Result dictionary for best filter.
        """
        return max(self._results, key=lambda x: x["psnr"])

    def get_best_filter_by_ssim(self) -> Dict[str, Any]:
        """Get the filter with highest SSIM.

        Returns:
            Result dictionary for best filter.
        """
        return max(self._results, key=lambda x: x["ssim"])
