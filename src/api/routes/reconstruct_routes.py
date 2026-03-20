"""Part B: 3D reconstruction API route.

Replicates the exact logic from clean_reconstruction.py:
- StereoMatcherBM/SGBM for computed disparity
- Manual pixel-by-pixel projection (same as clean_reconstruction)
- Open3D Poisson mesh with identical parameters
"""

import os
import time

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse

from src.api.image_utils import decode_upload
from src.part_b import StereoMatcherBM, StereoMatcherSGBM

router = APIRouter(prefix="/api/stereo", tags=["Part B - 3D Reconstruction"])


@router.post("/reconstruct")
async def reconstruct_3d(
    left_image: UploadFile = File(...),
    right_image: UploadFile = File(...),
    method: str = Form("sgbm"),
    num_disparities: int = Form(64),
    block_size: int = Form(5),
):
    """Generate 3D point cloud and mesh from stereo pair.

    Uses StereoMatcherBM/SGBM for disparity, then applies the
    same projection logic as clean_reconstruction.py.

    Returns:
        JSON with model name and PLY file URLs.
    """
    left = await decode_upload(left_image)
    right = await decode_upload(right_image)
    gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    matcher = _create_matcher(method, num_disparities, block_size)
    disparity = matcher.compute_disparity(gray_l, gray_r)

    model_name = f"upload_{int(time.time())}"
    out_dir = f"outputs/part_b/{model_name}"
    os.makedirs(out_dir, exist_ok=True)

    _reconstruct_clean(left, disparity, out_dir)

    return JSONResponse(content={
        "model_name": model_name,
        "mesh_url": f"/outputs/part_b/{model_name}/mesh.ply",
        "pointcloud_url": f"/outputs/part_b/{model_name}/pointcloud.ply",
    })


def _create_matcher(method, num_disparities, block_size):
    """Create stereo matcher using Part B classes."""
    if method == "bm":
        return StereoMatcherBM(num_disparities, max(block_size, 5))
    return StereoMatcherSGBM(num_disparities, block_size)


def _reconstruct_clean(color, disparity, out_dir):
    """Exact same logic as clean_reconstruction.py.

    Manual pixel projection with Middlebury-compatible parameters,
    Open3D filtering, and Poisson mesh generation.
    """
    import open3d as o3d

    h, w = disparity.shape
    focal, baseline = 500.0, 0.1
    cx, cy = w / 2, h / 2

    points, colors = [], []
    for v in range(h):
        for u in range(w):
            d = disparity[v, u]
            if d > 10:
                z = (focal * baseline) / (d / 4.0)
                if 0.1 < z < 10:
                    x = (u - cx) * z / focal
                    y = (cy - v) * z / focal
                    points.append([x, y, z])
                    bgr = color[v, u]
                    colors.append([bgr[2], bgr[1], bgr[0]])

    points = np.array(points)
    colors = np.array(colors) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=50,
        )
    )
    pcd.orient_normals_towards_camera_location([0, 0, -5])
    o3d.io.write_point_cloud(f"{out_dir}/pointcloud.ply", pcd)

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9,
    )
    mesh.compute_vertex_normals()
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)
    o3d.io.write_triangle_mesh(f"{out_dir}/mesh.ply", mesh)
