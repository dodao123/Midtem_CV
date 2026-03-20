"""Point Cloud Generator from disparity maps.

Converts 2D disparity to 3D point cloud for visualization.
"""

from typing import Tuple, Optional
import os

import cv2
import numpy as np


class PointCloudGenerator:
    """Generates 3D point clouds from disparity maps.

    Uses camera parameters to reproject 2D image points to 3D space.

    Attributes:
        _q_matrix: 4x4 reprojection matrix.
    """

    def __init__(
        self,
        focal_length: float = 718.856,
        baseline: float = 0.54,
        cx: float = 607.1928,
        cy: float = 185.2157,
    ):
        """Initialize Point Cloud Generator with camera parameters.

        Default values are from KITTI dataset calibration.

        Args:
            focal_length: Camera focal length in pixels.
            baseline: Distance between stereo cameras in meters.
            cx: Principal point x-coordinate.
            cy: Principal point y-coordinate.
        """
        self._focal_length = focal_length
        self._baseline = baseline
        self._cx = cx
        self._cy = cy
        self._q_matrix = self._create_q_matrix()

    def _create_q_matrix(self) -> np.ndarray:
        """Create the 4x4 reprojection matrix Q.

        Q matrix is used by cv2.reprojectImageTo3D().

        Returns:
            np.ndarray: 4x4 Q matrix.
        """
        q = np.float32([
            [1, 0, 0, -self._cx],
            [0, -1, 0, self._cy],
            [0, 0, 0, -self._focal_length],
            [0, 0, -1.0 / self._baseline, 0],
        ])
        return q

    def generate_point_cloud(
        self,
        disparity: np.ndarray,
        color_image: Optional[np.ndarray] = None,
        max_depth: float = 50.0,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate 3D point cloud from disparity map.

        Args:
            disparity: Disparity map from stereo matching.
            color_image: Optional BGR image for coloring points.
            max_depth: Maximum depth to include (filters far points).

        Returns:
            Tuple of (points_3d, colors) where:
                points_3d: Nx3 array of 3D coordinates.
                colors: Nx3 array of RGB colors (or None).
        """
        # Reproject to 3D
        points_3d = cv2.reprojectImageTo3D(disparity, self._q_matrix)

        # Create mask for valid points
        mask = disparity > 0
        mask &= np.isfinite(points_3d[:, :, 2])
        mask &= points_3d[:, :, 2] < max_depth
        mask &= points_3d[:, :, 2] > 0

        # Extract valid points
        valid_points = points_3d[mask]

        # Extract colors if provided
        colors = None
        if color_image is not None:
            if len(color_image.shape) == 3:
                colors = color_image[mask]
                colors = colors[:, ::-1]  # BGR to RGB
            else:
                gray = color_image[mask]
                colors = np.stack([gray, gray, gray], axis=1)

        return valid_points, colors

    def save_ply(
        self,
        filepath: str,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
    ) -> bool:
        """Save point cloud to PLY file format.

        Args:
            filepath: Output file path (.ply).
            points: Nx3 array of 3D points.
            colors: Optional Nx3 array of RGB colors (0-255).

        Returns:
            bool: True if successful.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")

            if colors is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")

            f.write("end_header\n")

            # Write vertex data
            for i in range(len(points)):
                x, y, z = points[i]
                if colors is not None:
                    r, g, b = colors[i].astype(int)
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
                else:
                    f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

        return True
