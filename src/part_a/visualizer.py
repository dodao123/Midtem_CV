"""Visualization utilities for image filtering results.

Provides functions to create side-by-side comparisons and analysis plots.
"""

from typing import List, Tuple, Optional
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


class FilterVisualizer:
    """Visualization utility for filter comparison results.

    Provides methods to create publication-quality comparison images
    and metric plots.
    """

    def __init__(self, output_directory: str = "outputs/part_a"):
        """Initialize the visualizer.

        Args:
            output_directory: Directory to save output images.
        """
        self._output_dir = output_directory
        os.makedirs(self._output_dir, exist_ok=True)

    def create_comparison_grid(
        self,
        images: List[np.ndarray],
        titles: List[str],
        filename: str,
        figsize: Tuple[int, int] = (15, 10),
        columns: int = 3,
    ) -> str:
        """Create a grid comparison of multiple images.

        Args:
            images: List of images to display.
            titles: List of titles for each image.
            filename: Output filename (without extension).
            figsize: Figure size in inches. Default is (15, 10).
            columns: Number of columns in the grid. Default is 3.

        Returns:
            str: Path to the saved figure.
        """
        num_images = len(images)
        rows = (num_images + columns - 1) // columns

        fig, axes = plt.subplots(rows, columns, figsize=figsize)
        axes = np.array(axes).flatten() if num_images > 1 else [axes]

        for idx, (img, title) in enumerate(zip(images, titles)):
            ax = axes[idx]
            display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
            cmap = "gray" if len(img.shape) == 2 else None
            ax.imshow(display_img, cmap=cmap)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        # Hide unused subplots
        for idx in range(num_images, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        output_path = os.path.join(self._output_dir, f"{filename}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path

    def create_metrics_bar_chart(
        self,
        filter_names: List[str],
        psnr_values: List[float],
        ssim_values: List[float],
        filename: str,
    ) -> str:
        """Create a bar chart comparing filter metrics.

        Args:
            filter_names: Names of the filters.
            psnr_values: PSNR values for each filter.
            ssim_values: SSIM values for each filter.
            filename: Output filename (without extension).

        Returns:
            str: Path to the saved figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        x_positions = np.arange(len(filter_names))
        bar_width = 0.6

        # PSNR Chart
        ax1.bar(x_positions, psnr_values, bar_width, color="steelblue")
        ax1.set_xlabel("Filter")
        ax1.set_ylabel("PSNR (dB)")
        ax1.set_title("PSNR Comparison")
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(filter_names, rotation=45, ha="right")

        # SSIM Chart
        ax2.bar(x_positions, ssim_values, bar_width, color="coral")
        ax2.set_xlabel("Filter")
        ax2.set_ylabel("SSIM")
        ax2.set_title("SSIM Comparison")
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(filter_names, rotation=45, ha="right")

        plt.tight_layout()
        output_path = os.path.join(self._output_dir, f"{filename}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path
