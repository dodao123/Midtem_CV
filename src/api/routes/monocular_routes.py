"""Monocular-to-3D reconstruction API route.

Accepts a single image upload, runs the full pipeline
(depth → stereo → 3D), and returns URLs to all outputs.
"""

import time

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from src.api.image_utils import decode_upload
from src.part_b.monocular_pipeline import MonocularTo3DPipeline

router = APIRouter(
    prefix="/api/monocular",
    tags=["Part B - Monocular 3D"],
)

_pipeline = None


def _get_pipeline() -> MonocularTo3DPipeline:
    """Lazy-initialize the pipeline singleton.

    Returns:
        MonocularTo3DPipeline: Shared pipeline instance.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = MonocularTo3DPipeline(baseline_px=40)
    return _pipeline


@router.post("/reconstruct")
async def monocular_reconstruct(
    image: UploadFile = File(...),
):
    """Reconstruct 3D model from a single image.

    Pipeline steps:
    1. Depth estimation (MiDaS)
    2. Right view synthesis (for visualization)
    3. 3D reconstruction from depth

    Args:
        image: Single image file upload.

    Returns:
        JSON with URLs to all generated outputs.
    """
    input_image = await decode_upload(image)

    model_name = f"mono_{int(time.time())}"
    output_dir = f"outputs/part_b/{model_name}"

    pipeline = _get_pipeline()
    pipeline.run(input_image, output_dir)

    base_url = f"/outputs/part_b/{model_name}"
    return JSONResponse(content={
        "model_name": model_name,
        "depth_url": f"{base_url}/depth_map.png",
        "left_url": f"{base_url}/left.png",
        "right_url": f"{base_url}/right.png",
        "mesh_url": f"{base_url}/mesh.ply",
        "pointcloud_url": f"{base_url}/pointcloud.ply",
    })
