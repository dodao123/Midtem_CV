"""Part A: Image Filtering API routes.

Endpoints for applying filters, adding noise, and running
comparative experiments via REST API.
"""

from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import Response

from src.api.image_utils import (
    decode_upload_gray,
    encode_image_to_base64,
    encode_image_to_bytes,
)
from src.part_a.filters import (
    MeanFilter,
    GaussianFilter,
    MedianFilter,
    LaplacianSharpener,
)
from src.part_a.noise_generator import NoiseGenerator
from src.part_a.metrics import ImageMetrics

router = APIRouter(prefix="/api/filtering", tags=["Part A - Filtering"])


FILTER_MAP = {
    "mean": lambda ks, **_: MeanFilter(kernel_size=ks),
    "gaussian": lambda ks, sigma=1.0, **_: GaussianFilter(
        kernel_size=ks, sigma=sigma,
    ),
    "median": lambda ks, **_: MedianFilter(kernel_size=ks),
    "laplacian": lambda ks, alpha=0.5, **_: LaplacianSharpener(
        kernel_size=ks, alpha=alpha,
    ),
}


@router.post("/apply")
async def apply_filter(
    image: UploadFile = File(...),
    filter_type: str = Form("gaussian"),
    kernel_size: int = Form(5),
    sigma: float = Form(1.0),
    alpha: float = Form(0.5),
):
    """Apply a single filter to an uploaded image.

    Args:
        image: Uploaded image file.
        filter_type: One of 'mean', 'gaussian', 'median', 'laplacian'.
        kernel_size: Filter kernel size (odd number).
        sigma: Gaussian sigma (only for gaussian filter).
        alpha: Sharpening factor (only for laplacian).

    Returns:
        PNG image response.
    """
    img = await decode_upload_gray(image)
    factory = FILTER_MAP.get(filter_type)
    if factory is None:
        return {"error": f"Unknown filter: {filter_type}"}

    filt = factory(kernel_size, sigma=sigma, alpha=alpha)
    result = filt.apply(img)
    return Response(
        content=encode_image_to_bytes(result),
        media_type="image/png",
    )


@router.post("/add-noise")
async def add_noise(
    image: UploadFile = File(...),
    noise_type: str = Form("gaussian"),
    sigma: float = Form(25.0),
    salt_prob: float = Form(0.05),
    pepper_prob: float = Form(0.05),
):
    """Add noise to an uploaded image.

    Args:
        image: Uploaded image file.
        noise_type: 'gaussian' or 'salt_pepper'.
        sigma: Gaussian noise std dev (for gaussian).
        salt_prob: Salt probability (for salt_pepper).
        pepper_prob: Pepper probability (for salt_pepper).

    Returns:
        PNG image with added noise.
    """
    img = await decode_upload_gray(image)
    if noise_type == "salt_pepper":
        noisy = NoiseGenerator.add_salt_and_pepper_noise(
            img, salt_prob, pepper_prob,
        )
    else:
        noisy = NoiseGenerator.add_gaussian_noise(img, sigma=sigma)

    return Response(
        content=encode_image_to_bytes(noisy),
        media_type="image/png",
    )
