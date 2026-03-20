"""Part B: Epipolar geometry API route.

Separated from stereo_routes to stay within 100-line limit.
Computes fundamental matrix and draws epipolar lines.
"""

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import Response

import cv2
import numpy as np

from src.api.image_utils import (
    decode_upload,
    decode_upload_gray,
    encode_image_to_bytes,
)
from src.part_b.epipolar_geometry import EpipolarGeometry

router = APIRouter(prefix="/api/stereo", tags=["Part B - Stereo 3D"])


@router.post("/epipolar")
async def compute_epipolar(
    left_image: UploadFile = File(...),
    right_image: UploadFile = File(...),
):
    """Compute and visualize epipolar lines.

    Args:
        left_image: Left camera image.
        right_image: Right camera image.

    Returns:
        PNG image with epipolar lines drawn on left image.
    """
    left_color = await decode_upload(left_image)
    right_color = await decode_upload(right_image)
    left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

    # Detect features for correspondence
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(left_gray, None)
    kp2, des2 = sift.detectAndCompute(right_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)[:30]

    pts_l = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts_r = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Compute fundamental matrix
    epipolar = EpipolarGeometry(method=cv2.FM_RANSAC)
    fundamental, mask = epipolar.compute_fundamental_matrix(pts_l, pts_r)

    inliers_l = pts_l[mask.ravel() == 1]
    inliers_r = pts_r[mask.ravel() == 1]
    lines_l = epipolar.compute_epipolar_lines(inliers_r, 2)

    result = EpipolarGeometry.draw_epipolar_lines(
        left_color, lines_l, inliers_l,
    )
    return Response(
        content=encode_image_to_bytes(result),
        media_type="image/png",
    )
