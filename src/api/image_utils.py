"""API image utilities for upload/download handling.

Provides helper functions to decode uploaded images and
encode processed results for API responses.
"""

import io
import base64

import cv2
import numpy as np
from fastapi import UploadFile


async def decode_upload(file: UploadFile) -> np.ndarray:
    """Decode an uploaded image file to numpy array.

    Args:
        file: FastAPI UploadFile object.

    Returns:
        np.ndarray: Decoded BGR image.

    Raises:
        ValueError: If image cannot be decoded.
    """
    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {file.filename}")
    return image


async def decode_upload_gray(file: UploadFile) -> np.ndarray:
    """Decode an uploaded image as grayscale.

    Args:
        file: FastAPI UploadFile object.

    Returns:
        np.ndarray: Decoded grayscale image.

    Raises:
        ValueError: If image cannot be decoded.
    """
    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode image: {file.filename}")
    return image


def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode a numpy image to base64 PNG string.

    Args:
        image: BGR or grayscale image.

    Returns:
        str: Base64-encoded PNG string.
    """
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")


def encode_image_to_bytes(image: np.ndarray) -> bytes:
    """Encode a numpy image to PNG bytes.

    Args:
        image: BGR or grayscale image.

    Returns:
        bytes: PNG-encoded image bytes.
    """
    _, buffer = cv2.imencode(".png", image)
    return buffer.tobytes()
