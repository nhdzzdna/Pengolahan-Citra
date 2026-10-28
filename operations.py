import numpy as np
import cv2

GROUPS = {
    "Penyesuaian cahaya": ["Brightness", "Contrast", "Gamma Correction",
                           "Histogram Equalization", "Negative"],
    "Warna": ["Grayscale", "Saturation"],
    "Geometri": ["Flip Horizontal", "Flip Vertical", "Rotation"],
    "Filter": ["Sharpening", "Gaussian Blur", "Median Filter"],
    "Segmentasi dan tepi": ["Thresholding", "Otsu Thresholding", "Edge Detection"],
}

# (kunci, label, minimum, maksimum, nilai awal, langkah)
PARAMS = {
    "Brightness": [("brightness", "Kecerahan (negatif untuk menggelapkan)", -100, 100, 0, 5)],
    "Contrast": [("contrast", "Faktor kontras (di bawah 1 untuk menurunkan)", 0.0, 3.0, 1.0, 0.1)],
    "Gamma Correction": [("gamma", "Nilai gamma", 0.1, 3.0, 1.0, 0.1)],
    "Saturation": [("saturation", "Faktor saturasi", 0.0, 3.0, 1.0, 0.1)],
    "Rotation": [("angle", "Sudut rotasi (derajat)", -180, 180, 0, 5)],
    "Sharpening": [("sharpness", "Kekuatan penajaman", 0.0, 3.0, 1.0, 0.1)],
    "Gaussian Blur": [("blur_radius", "Radius blur", 1, 15, 2, 1)],
    "Median Filter": [("ksize", "Ukuran filter", 1, 7, 1, 1)],
    "Thresholding": [("threshold", "Nilai ambang", 0, 255, 127, 1)],
    "Edge Detection": [("canny_low", "Ambang bawah Canny", 0, 255, 100, 5),
                       ("canny_high", "Ambang atas Canny", 0, 255, 200, 5)],
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
    if op == "Brightness":
        return np.clip(arr.astype(np.int16) + int(p.get("brightness", 0)), 0, 255).astype(np.uint8)

    if op == "Contrast":
        f = p.get("contrast", 1.0)
        return np.clip((arr.astype(np.float32) - 128) * f + 128, 0, 255).astype(np.uint8)

    if op == "Gamma Correction":
        g = max(p.get("gamma", 1.0), 0.01)
        lut = np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)])
        return cv2.LUT(arr, lut.clip(0, 255).astype(np.uint8))

    if op == "Histogram Equalization":
        ycrcb = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    if op == "Negative":
        return 255 - arr

    if op == "Grayscale":
        return _gray3(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY))

    if op == "Saturation":
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * p.get("saturation", 1.0), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if op == "Flip Horizontal":
        return np.ascontiguousarray(arr[:, ::-1])

    if op == "Flip Vertical":
        return np.ascontiguousarray(arr[::-1])

    if op == "Rotation":
        h, w = arr.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), p.get("angle", 0), 1.0)
        return cv2.warpAffine(arr, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    if op == "Sharpening":
        s = p.get("sharpness", 1.0)
        blur = cv2.GaussianBlur(arr, (0, 0), 1.5)
        return cv2.addWeighted(arr, 1 + s, blur, -s, 0)

    if op == "Gaussian Blur":
        k = 2 * int(p.get("blur_radius", 2)) + 1
        return cv2.GaussianBlur(arr, (k, k), 0)

    if op == "Median Filter":
        k = 2 * int(p.get("ksize", 1)) + 1
        return cv2.medianBlur(arr, k)

    if op == "Thresholding":
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        _, b = cv2.threshold(gray, int(p.get("threshold", 127)), 255, cv2.THRESH_BINARY)
        return _gray3(b)

    if op == "Otsu Thresholding":
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return _gray3(b)

    if op == "Edge Detection":
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        e = cv2.Canny(gray, int(p.get("canny_low", 100)), int(p.get("canny_high", 200)))
        return _gray3(e)

    return arr.copy()


def apply_pipeline(arr, steps):
    out = arr
    for s in steps:
        out = apply_operation(out, s["op"], s["params"])
    return out


def steps_label(steps):
    return " + ".join(s["op"] for s in steps) if steps else "Tanpa operasi"