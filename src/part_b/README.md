# Part B – 3D Reconstruction from Stereo Images

Tái tạo 3D từ cặp ảnh stereo: tính disparity, epipolar geometry, point cloud, và mesh 3D.

## 1. Tổng quan kiến trúc

```
part_b/
├── stereo_interface.py       # Abstract Base Class cho stereo matcher
├── stereo_matcher.py         # Block Matching + SGBM
├── depth_estimator.py        # Ước lượng depth 1 ảnh (MiDaS)
├── epipolar_geometry.py      # Tính Fundamental Matrix + epipolar lines
├── point_cloud_generator.py  # Tạo point cloud từ disparity
├── mesh_builder.py           # Tạo mesh 3D từ depth map
├── stereo_synthesizer.py     # Tạo ảnh phải giả từ 1 ảnh + depth
├── monocular_pipeline.py     # Pipeline: 1 ảnh → depth → stereo → 3D
├── clean_reconstruction.py   # Tái tạo 3D từ ground truth disparity
├── visualizer.py             # Trực quan hóa disparity, epipolar
├── main.py                   # Entry point chạy đầy đủ
└── __init__.py               # Public API
```

## 2. Thư viện sử dụng

| Thư viện | Mục đích |
|---|---|
| `cv2` (OpenCV) | Stereo matching (`StereoBM`, `StereoSGBM`), SIFT features, `findFundamentalMat`, `reprojectImageTo3D`, `remap`, `inpaint` |
| `numpy` | Xử lí mảng pixel, tính toán ma trận, tạo lưới tọa độ |
| `torch` (PyTorch) | Chạy model MiDaS (deep learning) để ước lượng depth từ 1 ảnh |
| `open3d` | Tạo/lưu point cloud, mesh 3D (PLY), Poisson reconstruction, visualize |
| `matplotlib` | Vẽ biểu đồ so sánh disparity, 3D scatter plot |
| `abc` | Abstract base class cho stereo matcher interface |

## 3. Chi tiết từng module

### 3.1. `stereo_interface.py` — Interface trừu tượng

**Class**: `StereoMatcherInterface(ABC)`

| Abstract member | Kiểu trả về | Mô tả |
|---|---|---|
| `name` (property) | `str` | Tên thuật toán |
| `compute_disparity(left, right)` | `np.ndarray` | Tính disparity map từ cặp ảnh stereo |

**Static method bổ sung**: `disparity_to_depth(disparity, focal_length, baseline)`:
```
Z = (f × B) / d
```
- `f`: tiêu cự camera (pixel), `B`: khoảng cách 2 camera, `d`: disparity
- Disparity ≤ 0 → depth = 0 (invalid)

---

### 3.2. `stereo_matcher.py` — 2 thuật toán Stereo Matching

#### `StereoMatcherBM` — Block Matching

**Tham số**: `num_disparities=64` (chia hết cho 16), `block_size=15` (lẻ, ≥5)

```python
self._stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity = self._stereo.compute(left, right).astype(np.float32) / 16.0
```

**Logic `StereoBM`**:
1. Với mỗi pixel ảnh trái, tìm pixel tương ứng trên ảnh phải
2. Dùng **SAD** (Sum of Absolute Differences) so sánh block lân cận
3. Disparity = khoảng cách ngang giữa 2 pixel tương ứng
4. Chia 16.0 vì OpenCV trả giá trị fixed-point (nhân 16)

#### `StereoMatcherSGBM` — Semi-Global Block Matching

**Tham số**: `num_disparities=80`, `block_size=5`, `mode=SGBM_3WAY`

```python
self._stereo = cv2.StereoSGBM_create(
    P1=8*3*block_size², P2=32*3*block_size²,  # smoothness
    disp12MaxDiff=1, uniquenessRatio=10,
    speckleWindowSize=100, speckleRange=32,
)
```

**Logic `StereoSGBM`**:
1. Giống BM nhưng **tối ưu hóa toàn cục** theo nhiều hướng (8 hoặc 5 đường)
2. `P1`, `P2`: phạt cho sự thay đổi disparity giữa pixel lân cận → smooth hơn
3. `speckleWindowSize/Range`: loại bỏ vùng noise nhỏ
4. Chính xác hơn BM nhưng **chậm hơn**

---

### 3.3. `depth_estimator.py` — Depth từ 1 ảnh (MiDaS)

**Class**: `MonocularDepthEstimator` (Singleton pattern)

**Tham số**: không có (model tự load)

```python
self._model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
```

**Hàm `estimate(image)`** — Logic:

| Bước | Code | Giải thích |
|---|---|---|
| 1 | `cv2.cvtColor(image, BGR2RGB)` | MiDaS nhận ảnh RGB |
| 2 | `self._transform(rgb)` | Resize + normalize ảnh cho model |
| 3 | `self._model(input_batch)` | Chạy neural network, trả **inverse depth** |
| 4 | `cv2.resize(raw_depth, ...)` | Scale về kích thước ảnh gốc |
| 5 | `_normalize_depth(raw_depth)` | Chuẩn hóa về `[0, 1]` (0=xa, 1=gần) |

**`_normalize_depth()`**: Min-max normalize + auto-detect hướng depth bằng cách so sánh vùng trung tâm vs viền ảnh. Nếu trung tâm < viền → đảo ngược.

---

### 3.4. `epipolar_geometry.py` — Hình học Epipolar

**Class**: `EpipolarGeometry`

**Tham số**: `method=cv2.FM_RANSAC`, `threshold=3.0`

| Hàm | Tham số | Logic |
|---|---|---|
| `compute_fundamental_matrix(pts_l, pts_r)` | Nx2 point arrays | `cv2.findFundamentalMat()` với RANSAC → ma trận F (3×3) thỏa `x'ᵀ F x = 0` |
| `compute_epipolar_lines(points, which_image)` | Nx2 points, 1 or 2 | `cv2.computeCorrespondEpilines()` → Nx3 hệ số đường `[a, b, c]` |
| `draw_epipolar_lines(image, lines, points)` | ảnh, lines, points | Vẽ đường `ax + by + c = 0` + chấm tròn lên ảnh |

**Property**: `inlier_ratio` — tỉ lệ điểm inlier sau RANSAC

---

### 3.5. `point_cloud_generator.py` — Tạo Point Cloud

**Class**: `PointCloudGenerator`

**Tham số** (mặc định KITTI): `focal_length=718.856`, `baseline=0.54`, `cx=607.19`, `cy=185.22`

**Hàm `generate_point_cloud(disparity, color_image)`**:

| Bước | Code | Giải thích |
|---|---|---|
| 1 | `_create_q_matrix()` | Tạo ma trận Q 4×4 cho reprojection |
| 2 | `cv2.reprojectImageTo3D(disparity, Q)` | Chiếu ngược 2D → 3D: mỗi pixel → tọa độ `(X, Y, Z)` |
| 3 | Tạo mask | Lọc: `disparity > 0`, `Z hữu hạn`, `0 < Z < max_depth` |
| 4 | `colors = image[mask][:, ::-1]` | Lấy màu + đổi BGR → RGB |

**Hàm `save_ply()`**: Ghi file PLY ASCII  (header + `x y z r g b` mỗi dòng)

---

### 3.6. `mesh_builder.py` — Tạo Mesh 3D lưới

**Hàm `build_grid_mesh(color, depth, step=2)`**:

| Bước | Hàm | Giải thích |
|---|---|---|
| 1 | `_compute_foreground_mask()` | Loại pixel depth < 0.15 (background) + cắt viền 5% |
| 2 | `_create_vertices()` | Mỗi pixel → vertex 3D: `z = z_far - d*(z_far-z_near)`, `x = (u-cx)*z/f`, `y = (cy-v)*z/f` |
| 3 | `_create_triangles()` | Nối 4 pixel lân cận thành 2 tam giác. Bỏ qua nếu chênh depth > 0.12 (tránh cạnh răng cưa) |
| 4 | `o3d.geometry.TriangleMesh()` | Tạo mesh Open3D + tính normals |

---

### 3.7. `stereo_synthesizer.py` — Tạo ảnh phải giả

**Class**: `StereoViewSynthesizer`

**Tham số**: `baseline_px=40` (độ dịch pixel tối đa)

**Hàm `synthesize_right_view(left_image, depth_map)`**:

| Bước | Hàm | Giải thích |
|---|---|---|
| 1 | `_depth_to_disparity()` | `disparity = depth × baseline_px` (gần → dịch nhiều) |
| 2 | `_forward_warp()` | `cv2.remap()`: pixel phải `(u,v)` ← lấy từ pixel trái `(u + disparity, v)` |
| 3 | `_fill_holes()` | Tìm vùng đen (< 5) → `cv2.inpaint(TELEA)` lấp lỗ |

---

### 3.8. `monocular_pipeline.py` — Pipeline 1 ảnh → 3D

**Class**: `MonocularTo3DPipeline`

**Hàm `run(image, output_dir)`** — 3 bước:
1. `_estimate_depth()` → MiDaS depth + colormap INFERNO
2. `_synthesize_stereo()` → tạo cặp ảnh left/right
3. `_reconstruct_from_depth()` → `build_grid_mesh()` + sample point cloud

---

### 3.9. `clean_reconstruction.py` — Tái tạo từ Ground Truth

**Hàm `reconstruct_with_ground_truth()`**:
1. Load ảnh + ground truth disparity
2. `Z = (f × B) / (d / 4.0)` — chia 4.0 là hệ số Middlebury dataset
3. Lọc `0.1 < Z < 10`
4. Tạo point cloud → `voxel_down_sample(0.02)` + `remove_statistical_outlier()`
5. **Poisson reconstruction** (`depth=9`) → crop bằng bounding box

---

### 3.10. `visualizer.py` — Trực quan hóa

**Class**: `StereoVisualizer`

| Hàm | Output |
|---|---|
| `visualize_disparity()` | Disparity → normalize → `applyColorMap(JET)` → PNG |
| `compare_disparities()` | So sánh nhiều disparity cạnh nhau (matplotlib) |
| `visualize_epipolar()` | Lưới 2×2: ảnh gốc + ảnh có epipolar lines |

---

### 3.11. `main.py` — Entry point

**Flow 6 bước**:
1. Load ảnh stereo left/right + ground truth
2. Tính disparity BM (`StereoMatcherBM`)
3. Tính disparity SGBM (`StereoMatcherSGBM`)
4. Tính epipolar lines (SIFT → BFMatcher → `EpipolarGeometry`)
5. Tạo point cloud + mesh từ ground truth (`_create_3d()`)
6. Render mesh → PNG (`open3d Visualizer`)

---

## 4. Các phần tử tương tác với nhau như thế nào?

### 4.1. Sơ đồ tương tác tổng thể

```
┌─────────────┐      ┌───────────────┐      ┌──────────────────┐
│   main.py   │─────▶│ StereoMatcher │─────▶│ disparity map    │
│ (điều phối) │      │  BM / SGBM    │      │ (np.ndarray)     │
└──────┬──────┘      └───────────────┘      └────────┬─────────┘
       │                                             │
       │                                             ▼
       │             ┌───────────────┐      ┌──────────────────┐
       ├────────────▶│  Epipolar     │─────▶│ F matrix + lines │
       │             │  Geometry     │      │ (visualization)  │
       │             └───────────────┘      └──────────────────┘
       │                                             │
       │                                             ▼
       │             ┌───────────────┐      ┌──────────────────┐
       └────────────▶│  _create_3d() │─────▶│ Point Cloud +    │
                     │  (disparity→  │      │ Mesh (PLY)       │
                     │   3D coords)  │      └──────────────────┘
                     └───────────────┘
```

### 4.2. Luồng dữ liệu chi tiết — Pipeline Stereo (main.py)

```
Ảnh trái (BGR) ──┐
                  ├──▶ cv2.cvtColor() ──▶ Ảnh grayscale
Ảnh phải (BGR) ──┘         │
                           ▼
                 StereoMatcherBM.compute_disparity()
                 StereoMatcherSGBM.compute_disparity()
                           │
                    disparity map (float32)
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        _save_disparity  _create_     _create_epipolar
        (colormap PNG)   comparison   (SIFT + F matrix)
                         (matplotlib)
```

### 4.3. Luồng dữ liệu — Pipeline Monocular (monocular_pipeline.py)

```
1 ảnh BGR ──▶ MonocularDepthEstimator.estimate()
                      │
                      ▼
              depth map (0.0-1.0 float32)
                      │
         ┌────────────┼────────────┐
         ▼                         ▼
 StereoViewSynthesizer      build_grid_mesh()
 .synthesize_right_view()         │
         │                        ▼
         ▼                  TriangleMesh (Open3D)
  Ảnh phải giả (BGR)              │
         │                        ▼
         ▼                  mesh.ply + pointcloud.ply
  left.png + right.png
```

### 4.4. Tương tác giữa các class

| Caller (gọi) | Callee (được gọi) | Dữ liệu truyền | Dữ liệu nhận |
|---|---|---|---|
| `main.py` | `StereoMatcherBM` | `left_gray`, `right_gray` | `disparity` (float32) |
| `main.py` | `StereoMatcherSGBM` | `left_gray`, `right_gray` | `disparity` (float32) |
| `main.py` | `EpipolarGeometry` | `pts_left`, `pts_right` (từ SIFT) | `F` matrix, `inlier_mask` |
| `EpipolarGeometry` | `cv2.findFundamentalMat` | float32 points, RANSAC | F (3×3), mask |
| `EpipolarGeometry` | `cv2.computeCorrespondEpilines` | points, F | lines `[a,b,c]` |
| `main._create_3d()` | `o3d.TriangleMesh.create_from_point_cloud_poisson` | point cloud | mesh |
| `MonocularTo3DPipeline` | `MonocularDepthEstimator` | BGR image | depth map `[0,1]` |
| `MonocularTo3DPipeline` | `StereoViewSynthesizer` | left image + depth | right image (giả) |
| `MonocularTo3DPipeline` | `build_grid_mesh()` | color + depth | `TriangleMesh` |
| `StereoViewSynthesizer` | `cv2.remap()` | image + disparity map | warped image |
| `StereoViewSynthesizer` | `cv2.inpaint()` | warped image + hole mask | filled image |

### 4.5. Quan hệ kế thừa

```
StereoMatcherInterface (ABC)
├── StereoMatcherBM      ← cv2.StereoBM_create()
└── StereoMatcherSGBM    ← cv2.StereoSGBM_create()
```

### 4.6. Design Patterns sử dụng

| Pattern | Nơi áp dụng | Mục đích |
|---|---|---|
| **Strategy** | `StereoMatcherInterface` → BM / SGBM | Đổi thuật toán stereo mà không sửa code gọi |
| **Singleton** | `MonocularDepthEstimator._instance` | Model MiDaS nặng, chỉ load 1 lần |
| **Pipeline** | `MonocularTo3DPipeline.run()` | Chuỗi xử lí: depth → stereo → 3D |
| **Lazy Loading** | `_load_model()` trong depth estimator | Load model khi cần, không phải lúc khởi tạo |

## 5. Mapping yêu cầu → Code

| Yêu cầu | File | Hàm/Class chính |
|---|---|---|
| Stereo matching (disparity) | `stereo_matcher.py` | `StereoMatcherBM`, `StereoMatcherSGBM` |
| Epipolar geometry | `epipolar_geometry.py` | `EpipolarGeometry` |
| Depth estimation | `depth_estimator.py` | `MonocularDepthEstimator` |
| Point cloud 3D | `point_cloud_generator.py` | `PointCloudGenerator` |
| Mesh reconstruction | `mesh_builder.py`, `clean_reconstruction.py` | `build_grid_mesh()`, Poisson |
| Stereo synthesis | `stereo_synthesizer.py` | `StereoViewSynthesizer` |
| Visualization | `visualizer.py`, `main.py` | `StereoVisualizer`, `_render_mesh()` |
