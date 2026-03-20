"""Part A: Image Filtering - Main Demo Script.

This script demonstrates all filtering operations and generates
comparison visualizations for the CV Midterm project.
"""

import os

import cv2
import numpy as np

from src.part_a.filters import (
    MeanFilter,
    GaussianFilter,
    MedianFilter,
    LaplacianSharpener,
)
from src.part_a.noise_generator import NoiseGenerator
from src.part_a.experiment_runner import FilterExperiment
from src.part_a.visualizer import FilterVisualizer
from src.utils.image_loader import ImageLoader


def main():
    """Run the complete Part A demonstration."""
    print("=" * 60)
    print("Part A: Image Filtering Demo")
    print("=" * 60)

    # Setup output directory
    output_dir = "outputs/part_a"
    os.makedirs(output_dir, exist_ok=True)

    # Load or create test image
    test_image_path = "data/filtering/original/Screenshot 2025-11-26 114654.png"

    if os.path.exists(test_image_path):
        original = ImageLoader.load_image(test_image_path, grayscale=True)
        print(f"Loaded image from: {test_image_path}")
    else:
        print("Creating sample test image...")
        original = ImageLoader.create_sample_image(256, 256, "circles")
        os.makedirs("data/filtering/original", exist_ok=True)
        ImageLoader.save_image(original, test_image_path)

    # Generate noisy images
    print("\nGenerating noisy images...")
    gaussian_noisy = NoiseGenerator.add_gaussian_noise(original, sigma=25, seed=42)
    salt_pepper_noisy = NoiseGenerator.add_salt_and_pepper_noise(
        original, salt_probability=0.05, pepper_probability=0.05, seed=42
    )

    # Save noisy images
    ImageLoader.save_image(gaussian_noisy, f"{output_dir}/noisy_gaussian.png")
    ImageLoader.save_image(salt_pepper_noisy, f"{output_dir}/noisy_salt_pepper.png")

    # Run experiments
    print("\nRunning filter experiments on Gaussian noise...")
    experiment = FilterExperiment(original, gaussian_noisy)
    results = experiment.run()

    # Print results
    print("\n" + "-" * 60)
    print("Filter Comparison Results (Gaussian Noise)")
    print("-" * 60)
    print(f"{'Filter Name':<30} {'PSNR (dB)':<12} {'SSIM':<10}")
    print("-" * 60)

    for result in results:
        print(f"{result['filter_name']:<30} {result['psnr']:<12.2f} {result['ssim']:<10.4f}")

    best_psnr = experiment.get_best_filter_by_psnr()
    best_ssim = experiment.get_best_filter_by_ssim()

    print("-" * 60)
    print(f"Best by PSNR: {best_psnr['filter_name']} ({best_psnr['psnr']:.2f} dB)")
    print(f"Best by SSIM: {best_ssim['filter_name']} ({best_ssim['ssim']:.4f})")

    # Create visualizations
    print("\nCreating visualizations...")
    visualizer = FilterVisualizer(output_dir)

    # Comparison grid
    images = [original, gaussian_noisy] + [r["filtered_image"] for r in results]
    titles = ["Original", "Noisy (Gaussian)"] + [r["filter_name"] for r in results]
    grid_path = visualizer.create_comparison_grid(images, titles, "filter_comparison")
    print(f"Saved comparison grid: {grid_path}")

    # Metrics chart
    filter_names = [r["filter_name"] for r in results]
    psnr_values = [r["psnr"] for r in results]
    ssim_values = [r["ssim"] for r in results]

    chart_path = visualizer.create_metrics_bar_chart(
        filter_names, psnr_values, ssim_values, "metrics_comparison"
    )
    print(f"Saved metrics chart: {chart_path}")

    print("\n" + "=" * 60)
    print("Part A Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
