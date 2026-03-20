"""Simple 3D Mesh Viewer - Run this file to view the 3D model."""

import open3d as o3d


def main():
    """Load and display 3D mesh in interactive viewer."""
    mesh_path = "outputs/part_b/teddy_clean/mesh.ply"

    print("Loading mesh...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Triangles: {len(mesh.triangles)}")
    print("\nOpening 3D viewer... (close window to exit)")
    print("Controls: Left-drag=Rotate, Right-drag=Pan, Scroll=Zoom, R=Reset")

    o3d.visualization.draw_geometries(
        [mesh],
        window_name="3D Mesh - Close window to exit",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    main()
