"""Run 3D Reconstruction on any stereo pair.

Usage:
    python run_stereo_3d.py <left_image> <right_image> [ground_truth_disparity]

Examples:
    python run_stereo_3d.py data/stereo_cones/left.png data/stereo_cones/right.png data/stereo_cones/ground_truth.png
    python run_stereo_3d.py data/stereo_real/left.png data/stereo_real/right.png
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from src.part_b.stereo_matcher import StereoMatcherBM, StereoMatcherSGBM
from src.part_b.epipolar_geometry import EpipolarGeometry


def run_reconstruction(left_path, right_path, gt_path=None, output_name="output"):
    """Run complete 3D reconstruction pipeline.

    Args:
        left_path: Path to left stereo image.
        right_path: Path to right stereo image.
        gt_path: Optional path to ground truth disparity.
        output_name: Name for output folder.
    """
    output_dir = f"outputs/part_b/{output_name}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"3D Reconstruction: {output_name}")
    print("=" * 60)

    # Load images
    left_color = cv2.imread(left_path)
    right_color = cv2.imread(right_path)
    left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

    print(f"Image size: {left_gray.shape}")

    # 1. Disparity maps
    print("\n[1/5] Computing disparity maps...")
    bm = StereoMatcherBM(num_disparities=64, block_size=15)
    sgbm = StereoMatcherSGBM(num_disparities=64, block_size=5)
    disp_bm = bm.compute_disparity(left_gray, right_gray)
    disp_sgbm = sgbm.compute_disparity(left_gray, right_gray)

    _save_disparity(disp_bm, f"{output_dir}/disparity_bm.png")
    _save_disparity(disp_sgbm, f"{output_dir}/disparity_sgbm.png")
    _save_comparison(disp_bm, disp_sgbm, f"{output_dir}/comparison.png")

    # 2. Epipolar geometry
    print("[2/5] Computing epipolar geometry...")
    _save_epipolar(left_gray, right_gray, left_color, right_color, output_dir)

    # 3. Point cloud & mesh
    print("[3/5] Generating 3D model...")
    disparity = cv2.imread(gt_path, 0) if gt_path else disp_sgbm
    _create_3d_model(disparity, left_color, output_dir, use_gt=gt_path is not None)

    print("\n" + "=" * 60)
    print(f"Done! Output saved to: {output_dir}/")
    print("=" * 60)

    # Open viewer
    mesh_path = f"{output_dir}/mesh.ply"
    if os.path.exists(mesh_path):
        print("\nOpening 3D viewer...")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh], window_name=f"3D - {output_name}")


def _save_disparity(disp, path):
    """Save disparity as color image."""
    d = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(path, cv2.applyColorMap(d, cv2.COLORMAP_JET))


def _save_comparison(bm, sgbm, path):
    """Save BM vs SGBM comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(bm, cmap='jet')
    axes[0].set_title("Block Matching")
    axes[0].axis('off')
    axes[1].imshow(sgbm, cmap='jet')
    axes[1].set_title("SGBM")
    axes[1].axis('off')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def _save_epipolar(lg, rg, lc, rc, out):
    """Save epipolar lines visualization."""
    sift = cv2.SIFT_create()
    kp1, d1 = sift.detectAndCompute(lg, None)
    kp2, d2 = sift.detectAndCompute(rg, None)
    matches = sorted(cv2.BFMatcher(cv2.NORM_L2, True).match(d1, d2), key=lambda x: x.distance)[:30]
    pts_l = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts_r = np.float32([kp2[m.trainIdx].pt for m in matches])

    epi = EpipolarGeometry()
    F, mask = epi.compute_fundamental_matrix(pts_l, pts_r)
    inl, inr = pts_l[mask.ravel()==1], pts_r[mask.ravel()==1]
    ll = epi.compute_epipolar_lines(inr, 2)
    lr = epi.compute_epipolar_lines(inl, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(cv2.cvtColor(EpipolarGeometry.draw_epipolar_lines(lc, ll, inl), cv2.COLOR_BGR2RGB))
    axes[0].set_title("Left - Epipolar Lines")
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(EpipolarGeometry.draw_epipolar_lines(rc, lr, inr), cv2.COLOR_BGR2RGB))
    axes[1].set_title("Right - Epipolar Lines")
    axes[1].axis('off')
    plt.savefig(f"{out}/epipolar_lines.png", dpi=150)
    plt.close()


def _create_3d_model(disp, color, out, use_gt=False):
    """Create point cloud and mesh."""
    h, w = disp.shape
    fx, fy, cx, cy = 500.0, 500.0, w/2, h/2
    baseline = 0.1

    pts, cols = [], []
    for v in range(h):
        for u in range(w):
            d = disp[v, u]
            if d > 10:
                z = (fx * baseline) / (d / 4.0 if use_gt else (d + 1e-6))
                if 0.05 < z < 20:
                    x, y = (u - cx) * z / fx, (cy - v) * z / fy
                    pts.append([x, y, z])
                    cols.append(color[v, u][::-1] / 255.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pts))
    pcd.colors = o3d.utility.Vector3dVector(np.array(cols))
    pcd = pcd.voxel_down_sample(0.02)
    pcd, _ = pcd.remove_statistical_outlier(30, 1.5)
    pcd.estimate_normals()
    pcd.orient_normals_towards_camera_location([0, 0, -5])
    o3d.io.write_point_cloud(f"{out}/pointcloud.ply", pcd)
    print(f"  Point cloud: {len(pcd.points)} points")

    # Point cloud viz
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    p = np.asarray(pcd.points)[::10]
    c = np.asarray(pcd.colors)[::10]
    ax.scatter(p[:,0], p[:,1], p[:,2], c=c, s=0.5)
    plt.savefig(f"{out}/point_cloud_3d.png", dpi=150)
    plt.close()

    # Mesh
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(f"{out}/mesh.ply", mesh)
    print(f"  Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    # Mesh render
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1024, height=768)
    vis.add_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f"{out}/mesh_3d_render.png")
    vis.destroy_window()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    left = sys.argv[1]
    right = sys.argv[2]
    gt = sys.argv[3] if len(sys.argv) > 3 else None
    name = os.path.basename(os.path.dirname(left))

    run_reconstruction(left, right, gt, name)
