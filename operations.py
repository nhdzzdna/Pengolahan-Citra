import numpy as np
import cv2

# (kunci, label, minimum, maksimum, nilai awal, langkah)
PARAMS = {
    # Penyesuaian Cahaya & Warna
    "Brightness": [("brightness", "Kecerahan (negatif untuk menggelapkan)", -100, 100, 0, 5)],
    "Contrast": [("contrast", "Faktor kontras (di bawah 1 untuk menurunkan)", 0.0, 3.0, 1.0, 0.1)],
    "Grayscale": [],
    "Negative": [],
    "Histogram Equalization": [],
    
    # Geometri
    "Rotation": [("angle", "Sudut rotasi (derajat)", -180, 180, 0, 5)],
    "Flip Horizontal": [],
    "Flip Vertical": [],
    
    # Filter Dasar & Penajaman
    "Gaussian Blur": [("blur_radius", "Radius blur", 1, 15, 2, 1)],
    "Sharpening": [("sharpness", "Kekuatan penajaman", 0.0, 3.0, 1.0, 0.1)],
    
    # Deteksi Tepi & Segmentasi
    "Thresholding": [("threshold", "Nilai ambang", 0, 255, 127, 1)],
    "Edge Detection": [("canny_low", "Ambang bawah Canny", 0, 255, 100, 5),
                       ("canny_high", "Ambang atas Canny", 0, 255, 200, 5)],
    
    # Simulasi Noise & Restorasi
    "Noise": [("intensity", "Intensitas noise", 0, 100, 25, 5)],
    "Motion Blur": [("ksize", "Ukuran blur (harus ganjil)", 3, 31, 9, 2)],
}


def limit_size(arr, max_w=900):
    h, w = arr.shape[:2]
    if w <= max_w:
        return arr
    scale = max_w / w
    return cv2.resize(arr, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)


def _gray3(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def apply_operation(arr, op, p):
    """arr: array RGB uint8. p: kamus parameter. Mengembalikan array RGB uint8."""
    
    # Penyesuaian Cahaya & Warna
    if op == "Brightness":
        return np.clip(arr.astype(np.int16) + int(p.get("brightness", 0)), 0, 255).astype(np.uint8)

    if op == "Contrast":
        f = p.get("contrast", 1.0)
        return np.clip((arr.astype(np.float32) - 128) * f + 128, 0, 255).astype(np.uint8)

    if op == "Grayscale":
        return _gray3(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY))

    if op == "Negative":
        return 255 - arr

    if op == "Histogram Equalization":
        ycrcb = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    # Geometri
    if op == "Rotation":
        h, w = arr.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), p.get("angle", 0), 1.0)
        return cv2.warpAffine(arr, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    if op == "Flip Horizontal":
        return np.ascontiguousarray(arr[:, ::-1])

    if op == "Flip Vertical":
        return np.ascontiguousarray(arr[::-1])

    # Filter Dasar & Penajaman
    if op == "Gaussian Blur":
        k = 2 * int(p.get("blur_radius", 2)) + 1
        return cv2.GaussianBlur(arr, (k, k), 0)

    if op == "Sharpening":
        s = p.get("sharpness", 1.0)
        blur = cv2.GaussianBlur(arr, (0, 0), 1.5)
        return cv2.addWeighted(arr, 1 + s, blur, -s, 0)

    # Deteksi Tepi & Segmentasi
    if op == "Thresholding":
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        _, b = cv2.threshold(gray, int(p.get("threshold", 127)), 255, cv2.THRESH_BINARY)
        return _gray3(b)

    if op == "Edge Detection":
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        e = cv2.Canny(gray, int(p.get("canny_low", 100)), int(p.get("canny_high", 200)))
        return _gray3(e)

    # Simulasi Noise & Restorasi
    if op == "Noise":
        intensity = p.get("intensity", 25)
        noise = np.random.normal(0, intensity, arr.shape).astype(np.float32)
        return np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if op == "Motion Blur":
        size = int(p.get("ksize", 9))
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = kernel / size
        return cv2.filter2D(arr, -1, kernel)

    return arr.copy()


def apply_pipeline(arr, steps):
    out = arr
    for s in steps:
        out = apply_operation(out, s["op"], s["params"])
    return out


def steps_label(steps):
    return " + ".join(s["op"] for s in steps) if steps else "Tanpa operasi"