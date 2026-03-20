"""Part C: Panorama stitching and comparison API routes.

Separated from stitching_routes to stay within 100-line limit.
"""

from typing import List

from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import Response, JSONResponse

from src.api.image_utils import (
    decode_upload,
    encode_image_to_base64,
    encode_image_to_bytes,
)
from src.part_c.sift_detector import SiftDetector
from src.part_c.orb_detector import OrbDetector
from src.part_c.panorama_builder import PanoramaBuilder

router = APIRouter(prefix="/api/stitching", tags=["Part C - Stitching"])


@router.post("/stitch")
async def stitch_images(
    images: List[UploadFile] = File(...),
    detector: str = Form("sift"),
):
    """Stitch multiple images into a panorama.

    Args:
        images: List of uploaded overlapping images (2-6).
        detector: 'sift' or 'orb'.

    Returns:
        Stitched panorama as PNG.
    """
    decoded = []
    for upload in images:
        img = await decode_upload(upload)
        decoded.append(img)

    if len(decoded) < 2:
        return JSONResponse(
            status_code=400,
            content={"error": "Need at least 2 images."},
        )

    det = SiftDetector() if detector != "orb" else OrbDetector(n_features=2000)
    builder = PanoramaBuilder(det)
    panorama = builder.stitch(decoded)

    return Response(
        content=encode_image_to_bytes(panorama),
        media_type="image/png",
    )


@router.post("/compare")
async def compare_detectors(
    images: List[UploadFile] = File(...),
):
    """Compare SIFT vs ORB stitching results.

    Args:
        images: List of uploaded overlapping images.

    Returns:
        JSON with both panoramas as base64 and match counts.
    """
    decoded = []
    for upload in images:
        img = await decode_upload(upload)
        decoded.append(img)

    if len(decoded) < 2:
        return JSONResponse(
            status_code=400,
            content={"error": "Need at least 2 images."},
        )

    sift_builder = PanoramaBuilder(SiftDetector())
    orb_builder = PanoramaBuilder(OrbDetector(n_features=2000))

    sift_result = sift_builder.stitch(decoded)
    orb_result = orb_builder.stitch(decoded)

    return JSONResponse(content={
        "sift_panorama_b64": encode_image_to_base64(sift_result),
        "orb_panorama_b64": encode_image_to_base64(orb_result),
    })
