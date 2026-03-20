"""Part B: Stereo 3D Reconstruction API routes.

Endpoints for disparity map computation, epipolar geometry,
and stereo method comparison.
"""

from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import Response, JSONResponse

from src.api.image_utils import (
    decode_upload,
    decode_upload_gray,
    encode_image_to_base64,
    encode_image_to_bytes,
)
from src.part_b.stereo_matcher import StereoMatcherBM, StereoMatcherSGBM

import cv2
import numpy as np

router = APIRouter(prefix="/api/stereo", tags=["Part B - Stereo 3D"])


@router.post("/disparity")
async def compute_disparity(
    left_image: UploadFile = File(...),
    right_image: UploadFile = File(...),
    method: str = Form("sgbm"),
    num_disparities: int = Form(64),
    block_size: int = Form(5),
):
    """Compute disparity map from stereo images.

    Args:
        left_image: Left camera image.
        right_image: Right camera image.
        method: 'bm' or 'sgbm'.
        num_disparities: Max disparity range (multiple of 16).
        block_size: Matching block size (odd number).

    Returns:
        Colorized disparity map as PNG.
    """
    left = await decode_upload_gray(left_image)
    right = await decode_upload_gray(right_image)

    if method == "bm":
        matcher = StereoMatcherBM(num_disparities, max(block_size, 5))
    else:
        matcher = StereoMatcherSGBM(num_disparities, block_size)

    disparity = matcher.compute_disparity(left, right)
    disp_norm = cv2.normalize(
        disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U,
    )
    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

    return Response(
        content=encode_image_to_bytes(disp_color),
        media_type="image/png",
    )


@router.post("/compare")
async def compare_stereo(
    left_image: UploadFile = File(...),
    right_image: UploadFile = File(...),
    num_disparities: int = Form(64),
):
    """Compare BM vs SGBM disparity results.

    Args:
        left_image: Left camera image.
        right_image: Right camera image.
        num_disparities: Max disparity range.

    Returns:
        JSON with both disparity maps as base64.
    """
    left = await decode_upload_gray(left_image)
    right = await decode_upload_gray(right_image)

    bm = StereoMatcherBM(num_disparities, block_size=15)
    sgbm = StereoMatcherSGBM(num_disparities, block_size=5)

    disp_bm = bm.compute_disparity(left, right)
    disp_sgbm = sgbm.compute_disparity(left, right)

    return JSONResponse(content={
        "bm": _colorize_b64(disp_bm),
        "sgbm": _colorize_b64(disp_sgbm),
        "bm_name": bm.name,
        "sgbm_name": sgbm.name,
    })


def _colorize_b64(disparity: np.ndarray) -> str:
    """Convert disparity to colorized base64 PNG.

    Args:
        disparity: Raw disparity map.

    Returns:
        str: Base64-encoded colorized PNG.
    """
    norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return encode_image_to_base64(color)
