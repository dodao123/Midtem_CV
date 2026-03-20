# Phần A: Lọc Ảnh - Kết Quả Thực Nghiệm và Phân Tích

## INS3155 - Đồ Án Giữa Kỳ Thị Giác Máy Tính
**Mã số sinh viên:** [Điền MSSV]  
**Ngày:** 31 tháng 1, 2025

---

## 1. Giới Thiệu

### 1.1 Mục Tiêu
Mục tiêu của thực nghiệm này là áp dụng và so sánh các kỹ thuật lọc không gian truyền thống để khử nhiễu và cải thiện chất lượng ảnh. Chúng tôi đánh giá hiệu suất của bốn bộ lọc cơ bản:

1. **Bộ lọc Trung bình (Mean Filter)** - Lọc tính trung bình đơn giản
2. **Bộ lọc Gaussian** - Trung bình có trọng số theo phân phối Gaussian
3. **Bộ lọc Trung vị (Median Filter)** - Bộ lọc phi tuyến tính
4. **Làm sắc nét Laplacian** - Tăng cường cạnh bằng đạo hàm bậc hai

### 1.2 Phương Pháp Thực Nghiệm
Quy trình thực nghiệm bao gồm:
1. Tải ảnh gốc sạch làm tham chiếu
2. Thêm nhiễu Gaussian tổng hợp (σ = 25)
3. Áp dụng từng bộ lọc với các cấu hình tham số khác nhau
4. Tính toán các chỉ số định lượng (PSNR, SSIM)
5. So sánh chất lượng hình ảnh trực quan

---

## 2. Cơ Sở Lý Thuyết

### 2.1 Bộ Lọc Trung Bình (Mean Filter)
📁 **Mã nguồn:** [`src/part_a/mean_filter.py`](../../../src/part_a/mean_filter.py)

Bộ lọc trung bình thay thế mỗi pixel bằng giá trị trung bình của các pixel lân cận trong vùng kernel:

$$g(x,y) = \frac{1}{mn} \sum_{s=-a}^{a} \sum_{t=-b}^{b} f(x+s, y+t)$$

**Giải thích công thức:**
| Ký hiệu | Ý nghĩa |
|---------|---------|
| $g(x,y)$ | Giá trị pixel đầu ra tại vị trí (x, y) |
| $f(x+s, y+t)$ | Giá trị pixel đầu vào tại vị trí lân cận |
| $m \times n$ | Kích thước kernel (ví dụ: 3×3, 5×5) |
| $a = (m-1)/2$ | Bán kính theo chiều ngang |
| $b = (n-1)/2$ | Bán kính theo chiều dọc |
| $\sum$ | Tổng tất cả pixel trong vùng kernel |

**Cách hoạt động:**
- Lấy tất cả pixel trong vùng cửa sổ
- Cộng tổng các giá trị lại
- Chia cho số lượng pixel (m × n)
- Kết quả = giá trị trung bình

**Triển khai OpenCV:**
```python
# Trong mean_filter.py, dòng 54-55
def apply(self, image: np.ndarray) -> np.ndarray:
    return cv2.blur(image, self.kernel_size)  # cv2.blur tính trung bình
```

**Đặc điểm:**
- ✅ Tính toán nhanh, đơn giản
- ✅ Hiệu quả với nhiễu đều
- ❌ Làm mờ cạnh

---

### 2.2 Bộ Lọc Gaussian
📁 **Mã nguồn:** [`src/part_a/gaussian_filter.py`](../../../src/part_a/gaussian_filter.py)

Bộ lọc Gaussian sử dụng kernel có trọng số theo phân phối Gaussian:

$$G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$

**Giải thích công thức:**
| Ký hiệu | Ý nghĩa |
|---------|---------|
| $G(x,y)$ | Trọng số của pixel tại vị trí (x, y) trong kernel |
| $\sigma$ (sigma) | Độ lệch chuẩn - điều khiển độ mờ (σ lớn = mờ nhiều) |
| $e^{...}$ | Hàm mũ tự nhiên (exponential) |
| $x^2 + y^2$ | Khoảng cách bình phương từ tâm kernel |
| $2\pi\sigma^2$ | Hệ số chuẩn hóa để tổng trọng số = 1 |

**Ví dụ Kernel 3×3 với σ=1:**
```
┌─────────────────────────┐
│ 0.075  0.124  0.075 │
│ 0.124  0.204  0.124 │  ← Tâm có trọng số cao nhất
│ 0.075  0.124  0.075 │
└─────────────────────────┘
```
*Pixel càng xa tâm → trọng số càng nhỏ*

**Triển khai OpenCV:**
```python
# Trong gaussian_filter.py, dòng 67-68
def apply(self, image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, self.kernel_size, self._sigma)
```

**Đặc điểm:**
- ✅ Bảo toàn cạnh tốt hơn bộ lọc trung bình
- ✅ Làm mượt tự nhiên
- ❌ Có thể làm mờ quá nếu σ lớn

---

### 2.3 Bộ Lọc Trung Vị (Median Filter)
📁 **Mã nguồn:** [`src/part_a/median_filter.py`](../../../src/part_a/median_filter.py)

Bộ lọc trung vị thay thế mỗi pixel bằng **giá trị trung vị** (median) của các pixel lân cận:

$$g(x,y) = \text{median}\{f(x+s, y+t) : (s,t) \in W\}$$

**Giải thích công thức:**
| Ký hiệu | Ý nghĩa |
|---------|---------|
| $g(x,y)$ | Giá trị pixel đầu ra |
| $\text{median}\{\}$ | Hàm lấy giá trị trung vị (sắp xếp rồi lấy phần tử giữa) |
| $W$ | Vùng cửa sổ (window) xung quanh pixel |
| $(s,t) \in W$ | Tất cả các vị trí offset trong window |

**Ví dụ với kernel 3×3:**
```
Giá trị pixel:       Sau khi sắp xếp:      Kết quả:
┌───┬───┬───┐    
│ 10│150│ 20│    [10, 15, 20, 25, 30, 35, 150, 200, 255]
├───┼───┼───┤                    ↓
│ 25│ 30│ 35│ →  Trung vị = 30 (phần tử thứ 5 trong 9)
├───┼───┼───┤    
│ 15│200│255│    → g(x,y) = 30
└───┴───┴───┘
```
*Nhiễu 150, 200, 255 bị loại bỏ vì không nằm ở giữa dãy số!*

**Triển khai OpenCV:**
```python
# Trong median_filter.py, dòng 54-55
def apply(self, image: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(image, self._kernel_size)
```

**Đặc điểm:**
- ✅ Rất hiệu quả với nhiễu muối tiêu (salt-and-pepper)
- ✅ Bảo toàn cạnh tốt
- ❌ Tính toán chậm hơn (phải sắp xếp)

---

### 2.4 Làm Sắc Nét Laplacian
📁 **Mã nguồn:** [`src/part_a/laplacian_sharpener.py`](../../../src/part_a/laplacian_sharpener.py)

Laplacian sử dụng đạo hàm bậc 2 để phát hiện và tăng cường cạnh:

$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$$

**Công thức làm sắc nét:**
$$g(x,y) = f(x,y) - \alpha \cdot \nabla^2 f(x,y)$$

**Giải thích công thức:**
| Ký hiệu | Ý nghĩa |
|---------|---------|
| $\nabla^2 f$ | Toán tử Laplacian (đạo hàm bậc 2) |
| $\frac{\partial^2 f}{\partial x^2}$ | Đạo hàm bậc 2 theo chiều x - đo sự thay đổi cường độ |
| $f(x,y)$ | Ảnh gốc |
| $\alpha$ | Hệ số làm sắc (α lớn = sắc nét hơn) |
| $f - \alpha \cdot \nabla^2 f$ | Trừ đi Laplacian → làm nổi cạnh |

**Kernel Laplacian 3×3:**
```
┌────┬────┬────┐
│  0 │ -1 │  0 │
├────┼────┼────┤
│ -1 │  4 │ -1 │  ← Phát hiện thay đổi đột ngột (cạnh)
├────┼────┼────┤
│  0 │ -1 │  0 │
└────┴────┴────┘
```

**Triển khai OpenCV:**
```python
# Trong laplacian_sharpener.py, dòng 71-74
def apply(self, image: np.ndarray) -> np.ndarray:
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=self._kernel_size)
    sharpened = image.astype(np.float64) - self._alpha * laplacian
    return np.clip(sharpened, 0, 255).astype(np.uint8)
```

**Đặc điểm:**
- ✅ Tăng cường cạnh và chi tiết
- ❌ **Khuếch đại nhiễu** - không phù hợp để khử nhiễu
- ⚠️ Chỉ dùng với ảnh sạch

---

## 3. Thiết Lập Thực Nghiệm

### 3.1 Ảnh Kiểm Tra
- **Nguồn:** Ảnh tự chọn có hoa văn vòng nguyệt quế
- **Độ phân giải:** Ảnh xám (grayscale)
- **Đặc điểm:** Có vùng mượt, cạnh sắc nét, và chi tiết nhỏ

### 3.2 Mô Hình Nhiễu
- **Loại:** Nhiễu Gaussian cộng tính (Additive Gaussian Noise)
- **Trung bình (μ):** 0
- **Độ lệch chuẩn (σ):** 25
- **Seed:** 42 (để tái tạo kết quả)

### 3.3 Cấu Hình Bộ Lọc

| Bộ lọc | Tham số 1 | Tham số 2 |
|--------|-----------|-----------|
| Trung bình | kernel = 3×3 | - |
| Trung bình | kernel = 5×5 | - |
| Gaussian | kernel = 5×5 | σ = 1.0 |
| Gaussian | kernel = 5×5 | σ = 2.0 |
| Trung vị | kernel = 3×3 | - |
| Trung vị | kernel = 5×5 | - |
| Laplacian | kernel = 3×3 | α = 0.5 |

### 3.4 Chỉ Số Đánh Giá

**Tỷ lệ Tín hiệu trên Nhiễu Đỉnh (PSNR):**
$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{MAX^2}{MSE}\right) \text{ dB}$$

📁 **Mã nguồn:** [`src/part_a/metrics.py`](../../../src/part_a/metrics.py)

PSNR cao hơn = chất lượng khôi phục tốt hơn.

**Chỉ số Tương đồng Cấu trúc (SSIM):**
$$\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

SSIM nằm trong khoảng -1 đến 1, trong đó 1 = hoàn toàn giống nhau.

---

## 4. Kết Quả

### 4.1 So Sánh Trực Quan

![Lưới so sánh bộ lọc](outputs/part_a/filter_comparison.png)
*Hình 1: So sánh trực quan giữa ảnh gốc, ảnh nhiễu, và các ảnh sau khi lọc.*

### 4.2 Kết Quả Định Lượng

| Bộ lọc | Kích thước Kernel | PSNR (dB) | SSIM | Xếp hạng PSNR | Xếp hạng SSIM |
|--------|-------------------|-----------|------|---------------|---------------|
| Trung bình | 3×3 | 22.09 | 0.6728 | 4 | 5 |
| Trung bình | 5×5 | 19.57 | 0.7105 | 6 | 4 |
| Gaussian | σ=1.0 | 22.41 | 0.7184 | 3 | 3 |
| **Gaussian** | **σ=2.0** | 20.33 | **0.7343** | 5 | **1** |
| **Trung vị** | **3×3** | **23.45** | 0.6254 | **1** | 6 |
| Trung vị | 5×5 | 21.49 | 0.7177 | 2 | 2 |
| Laplacian | α=0.5 | 9.77 | 0.2510 | 7 | 7 |

### 4.3 Biểu Đồ Chỉ Số

![So sánh PSNR và SSIM](outputs/part_a/metrics_comparison.png)
*Hình 2: Biểu đồ cột so sánh PSNR và SSIM của các cấu hình bộ lọc.*

---

## 5. Phân Tích và Thảo Luận

### 5.1 Bộ Lọc Tốt Nhất

| Chỉ số | Bộ lọc tốt nhất | Giá trị |
|--------|-----------------|---------|
| **PSNR** | Trung vị (3×3) | 23.45 dB |
| **SSIM** | Gaussian (σ=2.0) | 0.7343 |

### 5.2 Phân Tích So Sánh

#### Bộ lọc Trung bình
- **Ưu điểm:** Tính toán nhanh, triển khai đơn giản
- **Nhược điểm:** Làm mờ cạnh đáng kể, khử nhiễu vừa phải
- **Nhận xét:** Kernel lớn (5×5) gây mờ nhiều hơn nhưng bảo toàn cấu trúc tốt hơn (SSIM cao hơn)

#### Bộ lọc Gaussian
- **Ưu điểm:** Bảo toàn cạnh tốt hơn bộ lọc trung bình, làm mượt tự nhiên
- **Nhược điểm:** Có thể làm mờ quá nếu σ lớn
- **Nhận xét:** σ=2.0 đạt **SSIM cao nhất (0.7343)**, cho thấy khả năng bảo toàn cấu trúc vượt trội

#### Bộ lọc Trung vị
- **Ưu điểm:** Khử nhiễu xuất sắc, bảo toàn cạnh tốt nhất trong các bộ lọc khử nhiễu
- **Nhược điểm:** Chi phí tính toán cao hơn
- **Nhận xét:** Đạt **PSNR cao nhất (23.45 dB)** với kernel 3×3, xác nhận hiệu quả với nhiễu Gaussian

#### Làm sắc nét Laplacian
- **Ưu điểm:** Tăng cường cạnh và chi tiết
- **Nhược điểm:** **Khuếch đại nhiễu** - không phù hợp cho tác vụ khử nhiễu
- **Nhận xét:** Hiệu suất kém nhất (PSNR=9.77 dB) như dự kiến; được thiết kế để làm sắc nét, không phải khử nhiễu

### 5.3 Phân Tích Đánh Đổi (Trade-off)

Có sự **đánh đổi đáng chú ý giữa PSNR-SSIM**:
- Trung vị (3×3) tối đa hóa PSNR nhưng có SSIM thấp hơn
- Gaussian (σ=2.0) tối đa hóa SSIM nhưng có PSNR vừa phải

Điều này cho thấy:
- **PSNR** ưu tiên độ chính xác từng pixel (ít nhiễu hơn)
- **SSIM** ưu tiên chất lượng cảm nhận (bảo toàn cấu trúc)

### 5.4 Ảnh Hưởng Của Kích Thước Kernel

| Bộ lọc | Kernel nhỏ (3×3) | Kernel lớn (5×5) |
|--------|------------------|------------------|
| Trung bình | PSNR cao, SSIM thấp | PSNR thấp, SSIM cao |
| Trung vị | PSNR cao, SSIM thấp | PSNR thấp, SSIM cao |

Kernel lớn hơn cho kết quả mượt hơn với độ tương đồng cấu trúc tốt hơn nhưng hy sinh chi tiết nhỏ.

---

## 6. Kết Luận

### 6.1 Phát Hiện Chính

1. **Trung vị (3×3)** là lựa chọn tốt nhất để **khử nhiễu tối đa** (PSNR = 23.45 dB)
2. **Gaussian (σ=2.0)** là lựa chọn tốt nhất cho **chất lượng cảm nhận** (SSIM = 0.7343)
3. **Laplacian không phù hợp** cho khử nhiễu; nó khuếch đại nhiễu hiện có
4. **Đánh đổi kích thước kernel:** Kernel lớn hơn cải thiện tương đồng cấu trúc nhưng giảm bảo toàn chi tiết

### 6.2 Khuyến Nghị

| Trường hợp sử dụng | Bộ lọc khuyến nghị |
|--------------------|-------------------|
| Khử nhiễu tổng quát | Trung vị (3×3) |
| Ưu tiên chất lượng cảm nhận | Gaussian (σ=2.0) |
| Nhiễu muối tiêu | Trung vị (5×5) |
| Tăng cường cạnh (ảnh sạch) | Laplacian |

### 6.3 Hướng Phát Triển

1. Thử nghiệm trên **nhiễu muối tiêu** để xác nhận ưu thế của bộ lọc trung vị
2. Triển khai **bộ lọc song phương (bilateral filter)** cho làm mượt bảo toàn cạnh
3. Kết hợp bộ lọc: Khử nhiễu trước, sau đó làm sắc nét Laplacian
4. Thử nghiệm với **kích thước kernel thích ứng**

---

## 7. Phụ Lục

### A. Mẫu Ảnh Nhiễu

| Nhiễu Gaussian (σ=25) | Nhiễu Muối Tiêu |
|----------------------|-----------------|
| ![Nhiễu Gaussian](outputs/part_a/noisy_gaussian.png) | ![Nhiễu Muối Tiêu](outputs/part_a/noisy_salt_pepper.png) |

### B. Cấu Trúc Mã Nguồn

```
src/part_a/
├── filter_interface.py    # Lớp cơ sở trừu tượng
├── mean_filter.py         # Bộ lọc trung bình
├── gaussian_filter.py     # Bộ lọc Gaussian
├── median_filter.py       # Bộ lọc trung vị
├── laplacian_sharpener.py # Làm sắc nét Laplacian
├── noise_generator.py     # Tiện ích tạo nhiễu
├── metrics.py             # Tính toán PSNR, SSIM
├── visualizer.py          # Công cụ trực quan hóa
└── main.py                # Điểm khởi chạy demo
```

### C. Chạy Thực Nghiệm

```bash
py -3.10 -m src.part_a.main
```

---

## Tài Liệu Tham Khảo

1. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (Ấn bản 4). Pearson.
2. Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. *IEEE TIP*, 13(4), 600-612.
3. Tài liệu OpenCV. https://docs.opencv.org/
