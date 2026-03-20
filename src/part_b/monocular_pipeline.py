"""Monocular-to-3D reconstruction pipeline orchestrator.

Chains: depth estimation → stereo synthesis → 3D reconstruction.
Uses MiDaS depth directly for 3D via grid-based mesh.
"""

import os

import cv2
import numpy as np

from src.part_b.depth_estimator import MonocularDepthEstimator
from src.part_b.stereo_synthesizer import StereoViewSynthesizer
from src.part_b.mesh_builder import build_grid_mesh


class MonocularTo3DPipeline:
    """Full pipeline: single image → depth → stereo → 3D.

    Attributes:
        _depth_estimator: MiDaS depth estimation model.
        _synthesizer: Right-view synthesizer.
    """

    def __init__(self, baseline_px: int = 40):
        """Initialize pipeline components.

        Args:
            baseline_px: Pixel shift for stereo synthesis.
        """
        self._depth_estimator = (
            MonocularDepthEstimator.get_instance()
        )
        self._synthesizer = StereoViewSynthesizer(baseline_px)

    def run(
        self, image: np.ndarray, output_dir: str,
    ) -> dict:
        """Execute the full monocular-to-3D pipeline.

        Args:
            image: Input BGR image (H, W, 3), uint8.
            output_dir: Directory to save all outputs.

        Returns:
            dict: Paths to all generated output files.
        """
        os.makedirs(output_dir, exist_ok=True)

        depth = self._estimate_depth(image, output_dir)
        self._synthesize_stereo(image, depth, output_dir)
        self._reconstruct_from_depth(image, depth, output_dir)

        return self._build_result_paths(output_dir)

    def _estimate_depth(
        self, image: np.ndarray, output_dir: str,
    ) -> np.ndarray:
        """Step 1: Estimate depth from single image."""
        depth = self._depth_estimator.estimate(image)
        depth_vis = (depth * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(
            depth_vis, cv2.COLORMAP_INFERNO,
        )
        cv2.imwrite(f"{output_dir}/depth_map.png", depth_color)
        return depth

    def _synthesize_stereo(
        self,
        left: np.ndarray,
        depth: np.ndarray,
        output_dir: str,
    ) -> None:
        """Step 2: Generate stereo pair for visualization."""
        right = self._synthesizer.synthesize_right_view(
            left, depth,
        )
        cv2.imwrite(f"{output_dir}/left.png", left)
        cv2.imwrite(f"{output_dir}/right.png", right)

    def _reconstruct_from_depth(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        output_dir: str,
    ) -> None:
        """Step 3: Build 3D relief mesh from depth grid."""
        import open3d as o3d

        mesh = build_grid_mesh(color, depth, step=2)
        o3d.io.write_triangle_mesh(
            f"{output_dir}/mesh.ply", mesh,
        )

        pcd = mesh.sample_points_uniformly(
            number_of_points=100000,
        )
        o3d.io.write_point_cloud(
            f"{output_dir}/pointcloud.ply", pcd,
        )

    @staticmethod
    def _build_result_paths(output_dir: str) -> dict:
        """Build dict of output file paths."""
        return {
            "depth_map": f"{output_dir}/depth_map.png",
            "left_image": f"{output_dir}/left.png",
            "right_image": f"{output_dir}/right.png",
            "pointcloud": f"{output_dir}/pointcloud.ply",
            "mesh": f"{output_dir}/mesh.ply",
        }
