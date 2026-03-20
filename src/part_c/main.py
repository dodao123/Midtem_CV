"""Part C: Image Stitching - Main Demo Script.

Runs the full stitching pipeline with SIFT and ORB detectors,
generates visualizations and comparative analysis outputs.
"""

import os
import glob

import cv2

from src.part_c.sift_detector import SiftDetector
from src.part_c.orb_detector import OrbDetector
from src.part_c.feature_matcher import FeatureMatcher
from src.part_c.panorama_builder import PanoramaBuilder
from src.part_c.stitching_visualizer import StitchingVisualizer


def main():
    """Run the complete Part C demonstration."""
    print("=" * 60)
    print("Part C: Image Stitching Demo")
    print("=" * 60)

    output_dir = "outputs/part_c"
    images = _load_images("data/stitching")
    if len(images) < 2:
        print("ERROR: Need at least 2 images in data/stitching/")
        return

    print(f"Loaded {len(images)} images for stitching.")
    visualizer = StitchingVisualizer(output_dir)

    # --- SIFT pipeline ---
    print("\n[1/4] Running SIFT pipeline...")
    sift = SiftDetector(n_features=0)
    _run_pipeline(images, sift, visualizer, "sift")

    # --- ORB pipeline ---
    print("\n[2/4] Running ORB pipeline...")
    orb = OrbDetector(n_features=2000)
    _run_pipeline(images, orb, visualizer, "orb")

    print("\n" + "=" * 60)
    print("Part C Demo Complete!")
    print(f"Outputs saved to: {output_dir}")
    print("=" * 60)


def _load_images(data_dir: str) -> list:
    """Load all images from directory sorted by name.

    Args:
        data_dir: Path to image directory.

    Returns:
        List of BGR images.
    """
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(data_dir, pattern)))

    paths.sort()
    images = []
    for path in paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
            print(f"  Loaded: {os.path.basename(path)} ({img.shape})")
    return images


def _run_pipeline(images, detector, visualizer, prefix):
    """Run stitching pipeline with given detector.

    Args:
        images: List of input images.
        detector: Feature detector instance.
        visualizer: StitchingVisualizer instance.
        prefix: Output filename prefix.
    """
    # Show keypoint matches for first pair
    matcher = FeatureMatcher(
        norm_type=detector.norm_type, ratio_threshold=0.75,
    )
    kp1, des1 = detector.detect_and_compute(images[0])
    kp2, des2 = detector.detect_and_compute(images[1])
    matches = matcher.match(des1, des2)

    print(f"  {detector.name}: {len(matches)} matches found")
    visualizer.draw_matches(
        images[0], kp1, images[1], kp2, matches,
        f"matches_{prefix}",
    )
    visualizer.draw_keypoints(images[0], kp1, f"keypoints_{prefix}")

    # Build panorama
    builder = PanoramaBuilder(detector)
    panorama = builder.stitch(images)
    path = visualizer.save_panorama(panorama, f"panorama_{prefix}")
    print(f"  Panorama saved: {path}")


if __name__ == "__main__":
    main()
