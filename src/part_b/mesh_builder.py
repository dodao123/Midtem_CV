"""Grid-based 3D mesh builder from depth map.

Creates a triangle mesh directly from a depth grid by
connecting neighboring pixels. Produces a clean 2.5D
relief surface. Filters background to avoid edge wings.
"""

import cv2
import numpy as np


def build_grid_mesh(
    color: np.ndarray,
    depth: np.ndarray,
    step: int = 2,
):
    """Build triangle mesh from depth map grid.

    Each pixel becomes a vertex. Neighboring pixels are
    connected into triangles forming a relief surface.
    Background pixels (low depth) are excluded.

    Args:
        color: BGR image (H, W, 3), uint8.
        depth: Normalized depth (0=far, 1=near), float32.
        step: Pixel step for downsampling (2 = half res).

    Returns:
        open3d.geometry.TriangleMesh: Colored mesh.
    """
    import open3d as o3d

    height, width = depth.shape
    # Large focal = less perspective distortion
    focal = width * 1.5
    cx, cy = width / 2.0, height / 2.0

    # Downsample for performance
    rows = np.arange(0, height, step)
    cols = np.arange(0, width, step)
    grid_h, grid_w = len(rows), len(cols)

    # Compute foreground mask: keep only objects with
    # meaningful depth (not flat background)
    fg_mask = _compute_foreground_mask(depth, rows, cols)

    vertices, colors_out = _create_vertices(
        depth, color, rows, cols, focal, cx, cy,
    )
    triangles = _create_triangles(
        depth, rows, cols, grid_h, grid_w, fg_mask,
    )

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        colors_out,
    )
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()

    return mesh


def _compute_foreground_mask(depth, rows, cols):
    """Create mask of foreground (non-background) pixels.

    Uses depth threshold + edge margin to exclude
    flat background at image borders.

    Args:
        depth: Normalized depth map.
        rows: Row indices.
        cols: Column indices.

    Returns:
        np.ndarray: Boolean mask (grid_h, grid_w).
    """
    grid_h, grid_w = len(rows), len(cols)
    mask = np.ones((grid_h, grid_w), dtype=bool)

    # Exclude pixels with very low depth (far background)
    min_depth = 0.15
    for i, v in enumerate(rows):
        for j, u in enumerate(cols):
            if depth[v, u] < min_depth:
                mask[i, j] = False

    # Shrink margins: exclude outer 5% of image
    margin_h = max(1, grid_h // 20)
    margin_w = max(1, grid_w // 20)
    mask[:margin_h, :] = False
    mask[-margin_h:, :] = False
    mask[:, :margin_w] = False
    mask[:, -margin_w:] = False

    return mask


def _create_vertices(
    depth, color, rows, cols, focal, cx, cy,
):
    """Create 3D vertices from depth grid.

    Args:
        depth: Normalized depth map.
        color: BGR color image.
        rows: Row indices to sample.
        cols: Column indices to sample.
        focal: Focal length estimate.
        cx: Principal point x.
        cy: Principal point y.

    Returns:
        Tuple of (vertices Nx3, colors Nx3).
    """
    grid_h, grid_w = len(rows), len(cols)
    total = grid_h * grid_w
    vertices = np.zeros((total, 3))
    colors_out = np.zeros((total, 3))

    # Tight Z range for relief-like appearance
    z_near, z_far = 2.0, 5.0

    for i, v in enumerate(rows):
        for j, u in enumerate(cols):
            idx = i * grid_w + j
            d = depth[v, u]

            z = z_far - d * (z_far - z_near)
            x = (u - cx) * z / focal
            y = (cy - v) * z / focal

            vertices[idx] = [x, y, z]
            bgr = color[v, u]
            colors_out[idx] = [
                bgr[2] / 255.0,
                bgr[1] / 255.0,
                bgr[0] / 255.0,
            ]

    return vertices, colors_out


def _create_triangles(
    depth, rows, cols, grid_h, grid_w, fg_mask,
):
    """Create triangle faces for foreground grid cells.

    Skips triangles at depth discontinuities and
    background regions.

    Args:
        depth: Normalized depth map.
        rows: Row indices.
        cols: Column indices.
        grid_h: Grid height.
        grid_w: Grid width.
        fg_mask: Foreground boolean mask.

    Returns:
        np.ndarray: Mx3 triangle index array.
    """
    triangles = []
    max_depth_diff = 0.12

    for i in range(grid_h - 1):
        for j in range(grid_w - 1):
            # Skip if any corner is background
            if not (fg_mask[i, j] and fg_mask[i, j + 1]
                    and fg_mask[i + 1, j]
                    and fg_mask[i + 1, j + 1]):
                continue

            tl = i * grid_w + j
            tr = i * grid_w + (j + 1)
            bl = (i + 1) * grid_w + j
            br = (i + 1) * grid_w + (j + 1)

            d_tl = depth[rows[i], cols[j]]
            d_tr = depth[rows[i], cols[j + 1]]
            d_bl = depth[rows[i + 1], cols[j]]
            d_br = depth[rows[i + 1], cols[j + 1]]

            diffs = [
                abs(d_tl - d_tr), abs(d_tl - d_bl),
                abs(d_tr - d_br), abs(d_bl - d_br),
            ]

            if max(diffs) < max_depth_diff:
                triangles.append([tl, bl, tr])
                triangles.append([tr, bl, br])

    if not triangles:
        triangles = [[0, 1, 2]]
    return np.array(triangles, dtype=np.int32)
