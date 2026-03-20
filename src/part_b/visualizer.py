"""Visualization utilities for Part B - 3D Reconstruction.

Creates disparity maps, point cloud previews, and epipolar visualizations.
"""

from typing import Tuple
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


class StereoVisualizer:
    """Visualization tools for stereo reconstruction results."""

    def __init__(self, output_directory: str = "outputs/part_b"):
        """Initialize visualizer.

        Args:
            output_directory: Directory to save output images.
        """
        self._output_dir = output_directory
        os.makedirs(self._output_dir, exist_ok=True)

    def visualize_disparity(
        self,
        disparity: np.ndarray,
        filename: str,
        colormap: int = cv2.COLORMAP_JET,
    ) -> str:
        """Save disparity map as colored heatmap.

        Args:
            disparity: Disparity map (float32).
            filename: Output filename (without extension).
            colormap: OpenCV colormap to use.

        Returns:
            str: Path to saved image.
        """
        # Normalize to 0-255
        disp_normalized = cv2.normalize(
            disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )

        # Apply colormap
        disp_colored = cv2.applyColorMap(disp_normalized, colormap)

        output_path = os.path.join(self._output_dir, f"{filename}.png")
        cv2.imwrite(output_path, disp_colored)

        return output_path

    def compare_disparities(
        self,
        disparities: list,
        names: list,
        filename: str,
    ) -> str:
        """Create side-by-side comparison of disparity maps.

        Args:
            disparities: List of disparity maps.
            names: List of algorithm names.
            filename: Output filename.

        Returns:
            str: Path to saved figure.
        """
        num_maps = len(disparities)
        fig, axes = plt.subplots(1, num_maps, figsize=(5 * num_maps, 4))

        if num_maps == 1:
            axes = [axes]

        for ax, disp, name in zip(axes, disparities, names):
            im = ax.imshow(disp, cmap='jet')
            ax.set_title(name, fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        output_path = os.path.join(self._output_dir, f"{filename}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def visualize_epipolar(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        left_with_lines: np.ndarray,
        right_with_lines: np.ndarray,
        filename: str,
    ) -> str:
        """Create epipolar lines visualization.

        Args:
            left_image: Original left image.
            right_image: Original right image.
            left_with_lines: Left image with epipolar lines.
            right_with_lines: Right image with epipolar lines.
            filename: Output filename.

        Returns:
            str: Path to saved figure.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Convert BGR to RGB for matplotlib
        axes[0, 0].imshow(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                          if len(left_image.shape) == 3 else left_image, cmap='gray')
        axes[0, 0].set_title("Ảnh trái gốc")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                          if len(right_image.shape) == 3 else right_image, cmap='gray')
        axes[0, 1].set_title("Ảnh phải gốc")
        axes[0, 1].axis('off')

        axes[1, 0].imshow(cv2.cvtColor(left_with_lines, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Đường epipolar trên ảnh trái")
        axes[1, 0].axis('off')

        axes[1, 1].imshow(cv2.cvtColor(right_with_lines, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Đường epipolar trên ảnh phải")
        axes[1, 1].axis('off')

        plt.suptitle("Hình học Epipolar", fontsize=14)
        plt.tight_layout()

        output_path = os.path.join(self._output_dir, f"{filename}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path
