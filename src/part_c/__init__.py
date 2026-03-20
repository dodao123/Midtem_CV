"""Part C: Image Stitching - Public API.

Exports all classes for the stitching pipeline.
"""

from src.part_c.detector_interface import FeatureDetectorInterface
from src.part_c.sift_detector import SiftDetector
from src.part_c.orb_detector import OrbDetector
from src.part_c.feature_matcher import FeatureMatcher
from src.part_c.homography_estimator import HomographyEstimator
from src.part_c.image_warper import ImageWarper
from src.part_c.panorama_builder import PanoramaBuilder
from src.part_c.stitching_visualizer import StitchingVisualizer

__all__ = [
    "FeatureDetectorInterface",
    "SiftDetector",
    "OrbDetector",
    "FeatureMatcher",
    "HomographyEstimator",
    "ImageWarper",
    "PanoramaBuilder",
    "StitchingVisualizer",
]
