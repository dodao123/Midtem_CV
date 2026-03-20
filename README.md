# Computer Vision Midterm Project (INS3155)

Traditional computer vision techniques for **Image Filtering**, **3D Reconstruction**, and **Image Stitching** — no deep learning required.

**Author:** DAO DUC DO — 23070405  
**Course:** INS3155 — Computer Vision

---

## Overview

| Part | Task | Description |
|------|------|-------------|
| **A** | Image Filtering | Apply Mean, Gaussian, Median, and Laplacian filters to denoise and sharpen images |
| **B** | 3D Reconstruction | Estimate depth from stereo image pairs, generate 3D point clouds and surface meshes |
| **C** | Image Stitching | Combine overlapping photos into panoramas using SIFT and ORB feature detectors |

---

## Project Structure

```
CV_Midterm/
├── src/                          # Source code
│   ├── part_a/                   # Image Filtering
│   │   ├── main.py               # Entry point
│   │   ├── filter_interface.py   # Abstract base class for all filters
│   │   ├── mean_filter.py        # Mean (Box) filter
│   │   ├── gaussian_filter.py    # Gaussian filter
│   │   ├── median_filter.py      # Median filter
│   │   ├── laplacian_sharpener.py# Laplacian edge sharpener
│   │   ├── noise_generator.py    # Add Gaussian / salt-and-pepper noise
│   │   ├── metrics.py            # PSNR and SSIM evaluation
│   │   ├── visualizer.py         # Plot and save comparison images
│   │   └── experiment_runner.py  # Run all filter experiments
│   │
│   ├── part_b/                   # 3D Reconstruction
│   │   ├── main.py               # Entry point
│   │   ├── stereo_interface.py   # Abstract base class for stereo matchers
│   │   ├── stereo_matcher.py     # Block Matching (BM) and SGBM algorithms
│   │   ├── epipolar_geometry.py  # Fundamental matrix and epipolar lines
│   │   ├── depth_estimator.py    # Disparity-to-depth conversion
│   │   ├── point_cloud_generator.py # 3D point cloud from disparity
│   │   ├── mesh_builder.py       # Poisson surface reconstruction
│   │   ├── clean_reconstruction.py  # Full pipeline using ground truth
│   │   ├── monocular_pipeline.py # MiDaS-based monocular depth
│   │   ├── stereo_synthesizer.py # Synthesize stereo pairs
│   │   └── visualizer.py         # 3D visualization helpers
│   │
│   ├── part_c/                   # Image Stitching
│   │   ├── main.py               # Entry point
│   │   ├── detector_interface.py # Abstract base class for detectors
│   │   ├── sift_detector.py      # SIFT feature detector
│   │   ├── orb_detector.py       # ORB feature detector
│   │   ├── feature_matcher.py    # BFMatcher + Lowe's ratio test
│   │   ├── homography_estimator.py # RANSAC homography estimation
│   │   ├── image_warper.py       # Perspective warping + alpha blending
│   │   ├── panorama_builder.py   # Full stitching pipeline
│   │   └── stitching_visualizer.py # Visualize matches and panoramas
│   │
│   ├── api/                      # FastAPI web server (optional)
│   │   ├── app.py                # API application setup
│   │   ├── image_utils.py        # Image encoding utilities
│   │   └── routes/               # API endpoints
│   │
│   └── utils/                    # Shared utilities
│       └── image_loader.py       # Image loading helpers
│
├── data/                         # Input images (see below)
├── outputs/                      # Generated results (auto-created)
├── main.tex                      # LaTeX report
├── requirements.txt              # Python dependencies
├── run_api.py                    # Start FastAPI server
├── run_stereo_3d.py              # Standalone stereo 3D script
├── view_3d_mesh.py               # Interactive 3D mesh viewer
└── view_3d_mesh.bat              # Windows shortcut for mesh viewer
```

---

## Requirements

- **Python** 3.9+
- **OS:** Windows / macOS / Linux

### Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python`, `opencv-contrib-python` | Image processing, SIFT, ORB, stereo matching |
| `numpy` | Array operations |
| `matplotlib` | Plotting and visualization |
| `open3d` | 3D point clouds and mesh reconstruction |
| `scikit-image` | SSIM metric |
| `torch`, `torchvision`, `timm` | MiDaS monocular depth (optional) |
| `fastapi`, `uvicorn` | Web API server (optional) |

---

## Installation

```bash
# 1. Clone or download the project
cd CV_Midterm

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Part A — Image Filtering

```bash
python -m src.part_a.main
```

- **Input:** `data/filtering/original/` — original images (`.jpg`, `.png`)
- **Output:** `outputs/part_a/`
  - `filter_comparison.png` — side-by-side visual comparison
  - `metrics_comparison.png` — PSNR and SSIM bar charts
  - `noisy_*.png` — noisy versions of input images

### Part B — 3D Reconstruction

```bash
python -m src.part_b.main
```

- **Input:** `data/stereo_*/` — stereo image pairs  
  Each folder contains: `left.png`, `right.png`, `ground_truth.png`
- **Output:** `outputs/part_b/<dataset_name>/`
  - `comparison.png` — BM vs SGBM disparity maps
  - `epipolar_lines.png` — epipolar geometry visualization
  - `point_cloud_3d.png` — 3D point cloud render
  - `mesh_3d_render.png` — reconstructed mesh render
  - `pointcloud.ply` — point cloud file (viewable in MeshLab / Open3D)
  - `mesh.ply` — triangle mesh file

#### View 3D Mesh Interactively

```bash
python view_3d_mesh.py                    # Opens interactive 3D viewer
# or on Windows:
view_3d_mesh.bat
```

### Part C — Image Stitching

```bash
python -m src.part_c.main
```

- **Input:** `data/stitching/` — overlapping images (`image_1.png`, `image_2.png`, ...)
- **Output:** `outputs/part_c/`
  - `matches_sift.png` — SIFT feature matches
  - `matches_orb.png` — ORB feature matches
  - `keypoints_sift.png` — SIFT keypoints
  - `keypoints_orb.png` — ORB keypoints
  - `panorama_sift.png` — final panorama (SIFT)
  - `panorama_orb.png` — final panorama (ORB)

### Web API (Optional)

```bash
python run_api.py
```

Starts a FastAPI server for uploading images and running the pipelines via HTTP.

---

## Input Data Format

### Part A — Filtering
Place images in `data/filtering/original/`. Supported formats: `.jpg`, `.png`.

### Part B — Stereo
Create a folder under `data/` with the prefix `stereo_` (e.g., `data/stereo_teddy/`) containing:

| File | Description |
|------|-------------|
| `left.png` | Left camera image |
| `right.png` | Right camera image |
| `ground_truth.png` | (Optional) Ground truth disparity map |

Pre-included datasets: `stereo_art`, `stereo_books`, `stereo_cones`, `stereo_dolls`, `stereo_moebius`, `stereo_reindeer`, `stereo_real`.

### Part C — Stitching
Place 2+ overlapping images in `data/stitching/` named sequentially: `image_1.png`, `image_2.png`, etc.

---

## Output Examples

After running all parts, the `outputs/` folder will look like:

```
outputs/
├── part_a/
│   ├── filter_comparison.png       # All filters side-by-side
│   └── metrics_comparison.png      # PSNR / SSIM charts
├── part_b/
│   └── teddy_clean/
│       ├── comparison.png          # BM vs SGBM disparity
│       ├── epipolar_lines.png      # Epipolar line validation
│       ├── point_cloud_3d.png      # 3D point cloud
│       ├── mesh_3d_render.png      # Surface mesh
│       ├── pointcloud.ply          # 3D file
│       └── mesh.ply                # 3D file
└── part_c/
    ├── panorama_sift.png           # SIFT panorama
    └── panorama_orb.png            # ORB panorama
```

---

## LaTeX Report

The full report is in `main.tex`. To compile:

```bash
pdflatex main.tex
```

> Make sure `outputs/` contains the generated figures before compiling.

---

## Tech Stack

- **Language:** Python 3.9+
- **Core Libraries:** OpenCV, NumPy, Open3D, Matplotlib, scikit-image
- **Architecture:** OOP with abstract interfaces (Strategy pattern)
- **Optional:** FastAPI for web-based demos, PyTorch for MiDaS depth estimation
