"""Good Quality 3D Reconstruction using Ground Truth Disparity.

Creates clean 3D model using Middlebury ground truth disparity.
"""

import os
import cv2
import numpy as np
import open3d as o3d


def reconstruct_with_ground_truth():
    """Reconstruct using ground truth disparity for clean results."""
    output_dir = "outputs/part_b/teddy_clean"
    os.makedirs(output_dir, exist_ok=True)

    # Load images
    print("Loading images...")
    color = cv2.imread("data/stereo_real/left.png")
    disparity = cv2.imread("data/stereo_real/ground_truth_disparity.png", 0)

    h, w = disparity.shape
    print(f"Image size: {w}x{h}")
    print(f"Disparity range: {disparity.min()} - {disparity.max()}")

    # Convert disparity to depth
    focal = 500.0
    baseline = 0.1
    cx, cy = w / 2, h / 2

    # Generate point cloud
    print("Generating point cloud...")
    points = []
    colors = []

    for v in range(h):
        for u in range(w):
            d = disparity[v, u]
            if d > 10:  # Filter invalid disparities
                z = (focal * baseline) / (d / 4.0)  # Middlebury scale factor
                if 0.1 < z < 10:  # Valid depth range
                    x = (u - cx) * z / focal
                    y = (cy - v) * z / focal
                    points.append([x, y, z])
                    bgr = color[v, u]
                    colors.append([bgr[2], bgr[1], bgr[0]])

    points = np.array(points)
    colors = np.array(colors) / 255.0
    print(f"Generated {len(points)} points")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Filter and smooth
    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50)
    )
    pcd.orient_normals_towards_camera_location([0, 0, -5])

    # Save point cloud
    o3d.io.write_point_cloud(f"{output_dir}/pointcloud.ply", pcd)
    print(f"After filtering: {len(pcd.points)} points")

    # Create mesh with Poisson (works better with clean data)
    print("Creating mesh with Poisson reconstruction...")
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    mesh.compute_vertex_normals()

    # Crop mesh to remove background
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    o3d.io.write_triangle_mesh(f"{output_dir}/mesh.ply", mesh)
    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    print(f"\nDone! Saved to {output_dir}/")
    return f"{output_dir}/mesh.ply"


if __name__ == "__main__":
    mesh_path = reconstruct_with_ground_truth()

    print("\nOpening 3D viewer...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries(
        [mesh],
        window_name="Teddy 3D (Ground Truth)",
        width=1280,
        height=720,
    )
