# Part A – Image Filtering

Apply and compare different traditional filters to remove noise and enhance images.

## 1. Tổng quan kiến trúc

```
part_a/
├── filter_interface.py      # Abstract Base Class cho mọi filter
├── mean_filter.py           # Bộ lọc trung bình
├── gaussian_filter.py       # Bộ lọc Gaussian
├── median_filter.py         # Bộ lọc trung vị
├── laplacian_sharpener.py   # Làm sắc nét Laplacian
├── filters.py               # Re-export tất cả filter
├── noise_generator.py       # Tạo ảnh nhiễu (Gaussian, S&P, Speckle)
├── metrics.py               # Đo chất lượng ảnh (PSNR, SSIM)
├── experiment_runner.py     # Chạy thí nghiệm so sánh
├── visualizer.py            # Tạo lưới ảnh và biểu đồ
├── main.py                  # Entry point
└── __init__.py              # Public API
```

## 2. Thư viện sử dụng

| Thư viện | Mục đích |
|---|---|
| `cv2` (OpenCV) | Đọc ảnh, áp dụng filter (`blur`, `GaussianBlur`, `medianBlur`, `Laplacian`) |
| `numpy` | Xử lí mảng pixel, tính toán ma trận, tạo noise |
| `matplotlib` | Vẽ biểu đồ so sánh, tạo lưới ảnh side-by-side |
| `skimage` | Tính chỉ số SSIM (Structural Similarity Index) |
| `abc` | Tạo abstract base class cho filter interface |

## 3. Chi tiết từng module

### 3.1. `filter_interface.py` — Interface trừu tượng

**Class**: `ImageFilterInterface(ABC)`

Định nghĩa "hợp đồng" mà mọi filter phải implement:

| Abstract member | Kiểu trả về | Mô tả |
|---|---|---|
| `name` (property) | `str` | Tên filter để hiển thị |
| `kernel_size` (property) | `Tuple[int, int]` | Kích thước kernel (width, height) |
| `apply(image)` | `np.ndarray` | Áp filter lên ảnh, trả về ảnh đã lọc |

### 3.2. `mean_filter.py` — Bộ lọc trung bình

**Class**: `MeanFilter` — **Tham số**: `kernel_size: int = 5`

```python
def apply(self, image):
    return cv2.blur(image, self.kernel_size)  # VD: (5, 5)
```

**Logic bên trong `cv2.blur()`**:
1. Tạo kernel `K` kích thước `k × k`, mọi phần tử = `1 / (k × k)`
2. Trượt kernel qua từng pixel → tính **trung bình cộng** các pixel lân cận
3. Pixel mới = tổng(pixel trong cửa sổ) / (k × k)

> Giảm noise nhưng **làm mờ cạnh** vì trọng số pixel giống nhau.

### 3.3. `gaussian_filter.py` — Bộ lọc Gaussian

**Class**: `GaussianFilter` — **Tham số**: `kernel_size: int = 5`, `sigma: float = 1.0`

```python
def apply(self, image):
    return cv2.GaussianBlur(image, self.kernel_size, self._sigma)
```

**Logic bên trong `cv2.GaussianBlur()`**:
1. Tạo kernel Gaussian 2D: `G(x,y) = (1/2πσ²) × exp(-(x²+y²) / 2σ²)`
2. Pixel ở **trung tâm** kernel có trọng số cao nhất, giảm dần theo phân phối chuông
3. `sigma` lớn → mờ nhiều hơn

> Bảo tồn cạnh **tốt hơn Mean** vì pixel gần trung tâm đóng góp nhiều hơn.

### 3.4. `median_filter.py` — Bộ lọc trung vị

**Class**: `MedianFilter` — **Tham số**: `kernel_size: int = 5`

```python
def apply(self, image):
    return cv2.medianBlur(image, self._kernel_size)
```

**Logic bên trong `cv2.medianBlur()`**:
1. Trượt cửa sổ `k × k` qua mỗi pixel
2. Thu thập tất cả pixel lân cận → **sắp xếp** danh sách
3. Pixel mới = giá trị **trung vị** (vị trí giữa sau khi sort)

> **Rất hiệu quả với noise salt-and-pepper** vì trung vị loại bỏ giá trị cực đoan.

### 3.5. `laplacian_sharpener.py` — Làm sắc nét Laplacian

**Class**: `LaplacianSharpener` — **Tham số**: `kernel_size: int = 3`, `alpha: float = 1.0`

```python
def apply(self, image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=self._kernel_size)
    sharpened = image.astype(np.float64) - self._alpha * laplacian
    return np.clip(sharpened, 0, 255).astype(np.uint8)
```

**Logic từng bước**:

| Bước | Giải thích |
|---|---|
| `cv2.Laplacian(image, cv2.CV_64F)` | Tính **đạo hàm bậc 2** → phát hiện biên. `CV_64F` cho phép giá trị âm |
| `image - α × laplacian` | Trừ Laplacian khỏi ảnh gốc → **tăng cường biên**. Vùng biên có laplacian ≠ 0 |
| `np.clip(sharpened, 0, 255)` | Cắt giá trị về `[0, 255]`, chuyển về `uint8` |

> Công thức: `sharpened = original - α × ∇²(original)`. α lớn → cạnh rõ hơn.

### 3.6. `noise_generator.py` — Tạo ảnh nhiễu

**Class**: `NoiseGenerator` (static methods)

| Hàm | Tham số chính | Logic |
|---|---|---|
| `add_gaussian_noise()` | `sigma=25.0` | `noise = np.random.normal(mean, sigma)` → cộng vào ảnh |
| `add_salt_and_pepper_noise()` | `salt_prob=0.02`, `pepper_prob=0.02` | Random chọn tọa độ → gán `255` (salt) / `0` (pepper) |
| `add_speckle_noise()` | `variance=0.04` | `noisy = image × (1 + noise)` — nhiễu nhân (multiplicative) |

Tham số `seed` dùng `np.random.seed()` để đảm bảo kết quả **tái lặp** được.

### 3.7. `metrics.py` — Đo chất lượng ảnh

**Class**: `ImageMetrics` (static methods)

| Hàm | Công thức | Ý nghĩa |
|---|---|---|
| `calculate_mse()` | `MSE = mean((original - processed)²)` | Sai số bình phương. **Thấp = tốt** |
| `calculate_psnr()` | `PSNR = 10 × log₁₀(255² / MSE)` | Tỉ số tín hiệu/nhiễu (dB). **Cao = tốt** |
| `calculate_ssim()` | `skimage.metrics.structural_similarity()` | So sánh cấu trúc. `[-1, 1]`, **1 = giống hệt** |

### 3.8. `experiment_runner.py` — Chạy thí nghiệm

**Class**: `FilterExperiment`

- **Input**: `original_image`, `noisy_image`, danh sách `filters` (mặc định 7 filter)
- **Hàm `run()`**: Với mỗi filter → `apply()` → tính PSNR, SSIM → lưu kết quả
- **Tiện ích**: `get_best_filter_by_psnr()`, `get_best_filter_by_ssim()`

### 3.9. `visualizer.py` — Trực quan hóa

**Class**: `FilterVisualizer`

| Hàm | Output |
|---|---|
| `create_comparison_grid()` | Lưới ảnh side-by-side (Original → Noisy → Filtered). Lưu PNG 150 DPI |
| `create_metrics_bar_chart()` | 2 biểu đồ cột: PSNR và SSIM cho từng filter |

### 3.10. `main.py` — Entry point

**Flow chính**:
1. Load ảnh gốc (hoặc tạo sample)
2. Tạo 2 ảnh nhiễu: Gaussian (`σ=25`) và Salt & Pepper (`prob=0.05`)
3. Chạy `FilterExperiment` với 7 filter mặc định
4. In bảng kết quả PSNR/SSIM ra console
5. Tạo lưới ảnh so sánh và biểu đồ metrics → lưu vào `outputs/part_a/`

## 4. Mapping yêu cầu → Code

| Yêu cầu đề bài | File | Hàm chính |
|---|---|---|
| Mean filter | `mean_filter.py` | `MeanFilter.apply()` → `cv2.blur()` |
| Gaussian filter | `gaussian_filter.py` | `GaussianFilter.apply()` → `cv2.GaussianBlur()` |
| Median filter | `median_filter.py` | `MedianFilter.apply()` → `cv2.medianBlur()` |
| Laplacian sharpening | `laplacian_sharpener.py` | `LaplacianSharpener.apply()` → `cv2.Laplacian()` |
| Side-by-side images | `visualizer.py` | `FilterVisualizer.create_comparison_grid()` |
| Filter comparison | `metrics.py` + `visualizer.py` | PSNR/SSIM + `create_metrics_bar_chart()` |
