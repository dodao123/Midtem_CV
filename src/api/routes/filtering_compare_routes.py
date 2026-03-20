"""Part A: Filter comparison API route.

Separated from filtering_routes to stay within 100-line limit.
Runs all filters and returns comparative metrics.
"""

from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse

from src.api.image_utils import (
    decode_upload_gray,
    encode_image_to_base64,
)
from src.part_a.noise_generator import NoiseGenerator
from src.part_a.experiment_runner import FilterExperiment

router = APIRouter(prefix="/api/filtering", tags=["Part A - Filtering"])


@router.post("/compare")
async def compare_filters(
    image: UploadFile = File(...),
    noise_type: str = Form("gaussian"),
    sigma: float = Form(25.0),
):
    """Run all filters on image and return comparative metrics.

    Args:
        image: Uploaded clean image file.
        noise_type: Noise type to add before filtering.
        sigma: Noise sigma for gaussian noise.

    Returns:
        JSON with metrics table and base64 filtered images.
    """
    original = await decode_upload_gray(image)

    if noise_type == "salt_pepper":
        noisy = NoiseGenerator.add_salt_and_pepper_noise(original)
    else:
        noisy = NoiseGenerator.add_gaussian_noise(original, sigma=sigma)

    experiment = FilterExperiment(original, noisy)
    results = experiment.run()

    response_data = {
        "noise_type": noise_type,
        "original_b64": encode_image_to_base64(original),
        "noisy_b64": encode_image_to_base64(noisy),
        "filters": [],
    }

    for result in results:
        response_data["filters"].append({
            "name": result["filter_name"],
            "psnr": round(result["psnr"], 2),
            "ssim": round(result["ssim"], 4),
            "image_b64": encode_image_to_base64(result["filtered_image"]),
        })

    best_psnr = experiment.get_best_filter_by_psnr()
    best_ssim = experiment.get_best_filter_by_ssim()
    response_data["best_psnr"] = best_psnr["filter_name"]
    response_data["best_ssim"] = best_ssim["filter_name"]

    return JSONResponse(content=response_data)
