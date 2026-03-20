# Part B: 3D Reconstruction from Stereo Images

## Tổng quan

Module này thực hiện tái tạo 3D từ cặp ảnh stereo sử dụng các phương pháp truyền thống:

1. **Stereo Matching** - Tính disparity map (BM và SGBM)
2. **Epipolar Geometry** - Tính ma trận cơ bản F và vẽ đường epipolar
3. **3D Reconstruction** - Tạo point cloud và mesh 3D

---

## Cấu trúc thư mục

```
src/part_b/
├── __init__.py               # Package exports
├── stereo_interface.py       # Abstract interface
├── stereo_matcher.py         # BM & SGBM algorithms
├── point_cloud_generator.py  # Disparity → 3D points
├── epipolar_geometry.py      # Fundamental matrix
├── visualizer.py             # Visualization tools
├── main.py                   # Demo cho dataset mặc định
└── clean_reconstruction.py   # High-quality reconstruction
```

---

## Cách sử dụng

### 1. Chạy nhanh với dataset có sẵn

```bash
# Chạy Teddy dataset
py -3.10 -m src.part_b.main

# Chạy Cones dataset
py -3.10 run_stereo_3d.py data/stereo_cones/left.png data/stereo_cones/right.png data/stereo_cones/ground_truth.png
```

### 2. Chạy với ảnh stereo của riêng bạn

```bash
py -3.10 run_stereo_3d.py <ảnh_trái> <ảnh_phải> [ground_truth_disparity]
```

**Ví dụ:**
```bash
# Có ground truth disparity (chất lượng cao)
py -3.10 run_stereo_3d.py my_images/left.png my_images/right.png my_images/disparity.png

# Không có ground truth (dùng SGBM tự tính)
py -3.10 run_stereo_3d.py my_images/left.png my_images/right.png
```

### 3. Xem mô hình 3D

```bash
# Sửa đường dẫn trong file trước khi chạy
py -3.10 view_3d_mesh.py
```

Hoặc mở file `.ply` bằng:
- **MeshLab** (miễn phí): https://www.meshlab.net/
- **Blender** (miễn phí): https://www.blender.org/
- **Online**: https://www.creators3d.com/online-viewer

---

## Yêu cầu ảnh stereo

Để có kết quả tốt, ảnh stereo cần thỏa mãn:

| Yêu cầu | Mô tả |
|---------|-------|
| **Rectified** | Hai ảnh phải nằm trên cùng một đường scan ngang |
| **Cùng kích thước** | Hai ảnh có cùng resolution |
| **Baseline hợp lý** | Camera dịch ngang 5-15cm |
| **Texture** | Vật thể có texture rõ ràng (không trơn láng) |

### Nguồn ảnh stereo:
- [Middlebury 2003](https://vision.middlebury.edu/stereo/data/scenes2003/)
- [Middlebury 2014](https://vision.middlebury.edu/stereo/data/scenes2014/)
- [KITTI Dataset](https://www.cvlibs.net/datasets/kitti/eval_stereo.php)

---

## Output files

Sau khi chạy, thư mục output sẽ chứa:

| File | Mô tả |
|------|-------|
| `disparity_bm.png` | Disparity map - Block Matching |
| `disparity_sgbm.png` | Disparity map - SGBM |
| `comparison.png` | So sánh BM vs SGBM |
| `epipolar_lines.png` | Visualization đường epipolar |
| `point_cloud_3d.png` | Point cloud visualization |
| `mesh_3d_render.png` | 3D mesh render |
| `pointcloud.ply` | Point cloud file |
| `mesh.ply` | 3D mesh file |

---

## Giải thích thuật toán

### Block Matching (BM)
- Tìm điểm matching bằng cách so sánh block pixels
- **Ưu điểm**: Nhanh
- **Nhược điểm**: Nhiều noise, không xử lý tốt vùng textureless

### Semi-Global Block Matching (SGBM)
- Kết hợp matching cost với smoothness constraint
- **Ưu điểm**: Chất lượng cao, ít lỗ hổng
- **Nhược điểm**: Chậm hơn BM

### Disparity → 3D
```
Z = (focal_length × baseline) / disparity
X = (u - cx) × Z / focal_length  
Y = (v - cy) × Z / focal_length
```

---

## Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| Mesh bị rỗng | Kiểm tra disparity có > 0 không |
| Mesh bị phi tiêu | Dùng Ball Pivoting thay Poisson |
| Point cloud ít điểm | Giảm threshold depth |
| Lỗi OpenCV stereo | Giảm `num_disparities` < image_width |

---

## Dependencies

```bash
pip install opencv-python numpy matplotlib open3d
```
