"""Part C: Image Stitching API routes.

Endpoints for feature detection, matching, and panorama
stitching via REST API.
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
from src.part_c.feature_matcher import FeatureMatcher
from src.part_c.panorama_builder import PanoramaBuilder
from src.part_c.stitching_visualizer import StitchingVisualizer

import cv2

router = APIRouter(prefix="/api/stitching", tags=["Part C - Stitching"])


def _get_detector(detector_type: str):
    """Create detector instance by type name.

    Args:
        detector_type: 'sift' or 'orb'.

    Returns:
        FeatureDetectorInterface instance.
    """
    if detector_type == "orb":
        return OrbDetector(n_features=2000)
    return SiftDetector(n_features=0)


@router.post("/detect-features")
async def detect_features(
    image: UploadFile = File(...),
    detector: str = Form("sift"),
):
    """Detect and visualize keypoints on an image.

    Args:
        image: Uploaded image file.
        detector: 'sift' or 'orb'.

    Returns:
        PNG image with keypoints drawn.
    """
    img = await decode_upload(image)
    det = _get_detector(detector)
    keypoints, _ = det.detect_and_compute(img)

    result = cv2.drawKeypoints(
        img, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return Response(
        content=encode_image_to_bytes(result),
        media_type="image/png",
    )


@router.post("/match-features")
async def match_features(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    detector: str = Form("sift"),
):
    """Match features between two images.

    Args:
        image1: First uploaded image.
        image2: Second uploaded image.
        detector: 'sift' or 'orb'.

    Returns:
        JSON with match count and visualization as base64.
    """
    img1 = await decode_upload(image1)
    img2 = await decode_upload(image2)
    det = _get_detector(detector)

    kp1, des1 = det.detect_and_compute(img1)
    kp2, des2 = det.detect_and_compute(img2)

    matcher = FeatureMatcher(det.norm_type, ratio_threshold=0.75)
    matches = matcher.match(des1, des2)
    top = sorted(matches, key=lambda m: m.distance)[:50]

    vis = cv2.drawMatches(
        img1, kp1, img2, kp2, top, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return JSONResponse(content={
        "num_matches": len(matches),
        "detector": det.name,
        "visualization_b64": encode_image_to_base64(vis),
    })
