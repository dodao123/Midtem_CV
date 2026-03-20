"""Part B: 3D model serving API routes.

Dynamically scans outputs/part_b/ for PLY files and serves them.
Also provides a reconstruction endpoint to generate new 3D models.
"""

import os

from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter(prefix="/api/stereo", tags=["Part B - Stereo 3D"])

PLY_BASE = "outputs/part_b"


@router.get("/models")
async def list_3d_models():
    """Dynamically scan for all available 3D models.

    Scans every subdirectory under outputs/part_b/ for PLY files.

    Returns:
        JSON list of all datasets that have mesh or pointcloud PLY.
    """
    models = []
    if not os.path.exists(PLY_BASE):
        return JSONResponse(content={"models": []})

    for entry in sorted(os.listdir(PLY_BASE)):
        ds_dir = os.path.join(PLY_BASE, entry)
        if not os.path.isdir(ds_dir):
            continue

        mesh_path = os.path.join(ds_dir, "mesh.ply")
        pc_path = os.path.join(ds_dir, "pointcloud.ply")
        has_mesh = os.path.exists(mesh_path)
        has_pc = os.path.exists(pc_path)

        if not has_mesh and not has_pc:
            continue

        label = entry.replace("_", " ").replace("-", " ").title()
        model = {
            "name": entry,
            "label": label,
            "has_mesh": has_mesh,
            "has_pointcloud": has_pc,
        }

        if has_mesh:
            model["mesh_url"] = f"/outputs/part_b/{entry}/mesh.ply"
            model["mesh_size_mb"] = round(
                os.path.getsize(mesh_path) / (1024 * 1024), 2
            )
        if has_pc:
            model["pointcloud_url"] = f"/outputs/part_b/{entry}/pointcloud.ply"

        models.append(model)

    return JSONResponse(content={"models": models})


@router.get("/model/{dataset}/{file_type}")
async def get_model_file(dataset: str, file_type: str):
    """Serve a specific PLY file.

    Args:
        dataset: Dataset subdirectory name.
        file_type: 'mesh' or 'pointcloud'.

    Returns:
        PLY file for browser download/rendering.
    """
    filename = f"{file_type}.ply"
    filepath = os.path.join(PLY_BASE, dataset, filename)

    if not os.path.exists(filepath):
        return JSONResponse(
            status_code=404,
            content={"error": f"File not found: {filepath}"},
        )

    return FileResponse(
        filepath,
        media_type="application/octet-stream",
        filename=f"{dataset}_{filename}",
    )
