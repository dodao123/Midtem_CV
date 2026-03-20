"""Complete Part B Demo - Generates all visualizations for teddy_clean.

Run this to generate disparity maps, epipolar lines, point cloud, and mesh.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from src.part_b.stereo_matcher import StereoMatcherBM, StereoMatcherSGBM
from src.part_b.epipolar_geometry import EpipolarGeometry


def main():
    """Generate all Part B visualizations."""
    output_dir = "outputs/part_b/teddy_clean"
    os.makedirs(output_dir, exist_ok=True)

    # Load images
    print("=" * 60)
    print("Part B: 3D Reconstruction - Teddy Clean")
    print("=" * 60)

    left_color = cv2.imread("data/stereo_real/left.png")
    right_color = cv2.imread("data/stereo_real/right.png")
    left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)
    ground_truth = cv2.imread("data/stereo_real/ground_truth_disparity.png", 0)

    print(f"Image size: {left_gray.shape}")

    # 1. Disparity BM
    print("\n[1/6] Computing Block Matching disparity...")
    matcher_bm = StereoMatcherBM(num_disparities=64, block_size=15)
    disp_bm = matcher_bm.compute_disparity(left_gray, right_gray)
    _save_disparity(disp_bm, f"{output_dir}/disparity_bm.png", "Block Matching")

    # 2. Disparity SGBM
    print("[2/6] Computing SGBM disparity...")
    matcher_sgbm = StereoMatcherSGBM(num_disparities=64, block_size=5)
    disp_sgbm = matcher_sgbm.compute_disparity(left_gray, right_gray)
    _save_disparity(disp_sgbm, f"{output_dir}/disparity_sgbm.png", "SGBM")

    # 3. Comparison
    print("[3/6] Creating comparison figure...")
    _create_comparison(disp_bm, disp_sgbm, f"{output_dir}/comparison.png")

    # 4. Epipolar lines
    print("[4/6] Computing epipolar geometry...")
    _create_epipolar(left_gray, right_gray, left_color, right_color, output_dir)

    # 5. Point cloud 3D
    print("[5/6] Generating point cloud visualization...")
    pcd, mesh = _create_3d(ground_truth, left_color, output_dir)

    # 6. Mesh render
    print("[6/6] Rendering mesh...")
    _render_mesh(mesh, f"{output_dir}/mesh_3d_render.png")

    print("\n" + "=" * 60)
    print("All visualizations saved to:", output_dir)
    print("=" * 60)


def _save_disparity(disp, path, title):
    """Save disparity as colored image."""
    disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
    cv2.imwrite(path, disp_color)
    print(f"  Saved: {path}")


def _create_comparison(disp_bm, disp_sgbm, path):
    """Create side-by-side disparity comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(disp_bm, cmap='jet')
    axes[0].set_title("Block Matching", fontsize=12)
    axes[0].axis('off')
    axes[1].imshow(disp_sgbm, cmap='jet')
    axes[1].set_title("SGBM", fontsize=12)
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _create_epipolar(left_gray, right_gray, left_color, right_color, output_dir):
    """Create epipolar lines visualization."""
    # Detect features
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(left_gray, None)
    kp2, des2 = sift.detectAndCompute(right_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)[:30]

    pts_l = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts_r = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Compute fundamental matrix
    epipolar = EpipolarGeometry(method=cv2.FM_RANSAC)
    F, mask = epipolar.compute_fundamental_matrix(pts_l, pts_r)

    # Draw epipolar lines
    inliers_l = pts_l[mask.ravel() == 1]
    inliers_r = pts_r[mask.ravel() == 1]

    lines_l = epipolar.compute_epipolar_lines(inliers_r, 2)
    lines_r = epipolar.compute_epipolar_lines(inliers_l, 1)

    img_l = EpipolarGeometry.draw_epipolar_lines(left_color, lines_l, inliers_l)
    img_r = EpipolarGeometry.draw_epipolar_lines(right_color, lines_r, inliers_r)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(cv2.cvtColor(left_color, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Ảnh trái gốc")
    axes[0, 0].axis('off')
    axes[0, 1].imshow(cv2.cvtColor(right_color, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Ảnh phải gốc")
    axes[0, 1].axis('off')
    axes[1, 0].imshow(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Đường epipolar trên ảnh trái")
    axes[1, 0].axis('off')
    axes[1, 1].imshow(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Đường epipolar trên ảnh phải")
    axes[1, 1].axis('off')
    plt.suptitle("Hình học Epipolar", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/epipolar_lines.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/epipolar_lines.png")


def _create_3d(disparity, color, output_dir):
    """Create point cloud and mesh from ground truth disparity."""
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
    pcd.estimate_normals()
    pcd.orient_normals_towards_camera_location([0, 0, -5])

    o3d.io.write_point_cloud(f"{output_dir}/pointcloud.ply", pcd)

    # Create point cloud visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    pts = np.asarray(pcd.points)[::10]
    cols = np.asarray(pcd.colors)[::10]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')
    plt.savefig(f"{output_dir}/point_cloud_3d.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/point_cloud_3d.png")

    # Create mesh
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    mesh.compute_vertex_normals()
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)
    o3d.io.write_triangle_mesh(f"{output_dir}/mesh.ply", mesh)

    return pcd, mesh


def _render_mesh(mesh, path):
    """Render mesh to image."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1024, height=768)
    vis.add_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(path)
    vis.destroy_window()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
