import numpy as np  # untuk menghitung angka pada matriks piksel
import cv2          # OpenCV: alat untuk mengolah gambar (ubah warna, blur, putar, cari tepi)


# (kunci, label, minimum, maksimum, nilai awal, langkah)
# PARAMS adalah daftar semua operasi yang muncul di menu samping (sidebar).
# Nama operasi menjadi kunci. Isinya adalah daftar slider untuk operasi itu.
# Kalau daftarnya kosong, berarti operasi itu tidak punya slider.
PARAMS = {
    # Penyesuaian Cahaya & Warna
    # Brightness: slider berisi angka yang DITAMBAHKAN ke setiap piksel (bukan nilai pikselnya).
    # Angka 0 berarti gambar tidak berubah.
    "Brightness": [("brightness", "Kecerahan (negatif untuk menggelapkan)", -100, 100, 0, 5)],
    # Contrast: slider berisi angka PENGALI. Angka 1.0 berarti gambar tidak berubah.
    # Di bawah 1 gambar jadi pudar, di atas 1 gambar jadi lebih tegas.
    "Contrast": [("contrast", "Faktor kontras (di bawah 1 untuk menurunkan)", 0.0, 3.0, 1.0, 0.1)],
    # Grayscale dan Negative tidak punya slider.
    "Grayscale": [],
    "Negative": [],
    # Histogram Equalization juga tanpa slider karena prosesnya otomatis.
    "Histogram Equalization": [],
    
    # Geometri
    # Rotation: sudut putar dalam derajat. 0 berarti tidak diputar.
    "Rotation": [("angle", "Sudut rotasi (derajat)", -180, 180, 0, 5)],
    "Flip Horizontal": [],
    "Flip Vertical": [],
    
    # Filter Dasar & Penajaman
    # Gaussian Smoothing: makin besar radius, makin halus (buram) gambarnya.
    "Gaussian Smoothing": [("radius", "Radius penghalusan", 1, 15, 2, 1)],
    # Sharpening: makin besar angkanya, makin tajam gambarnya.
    "Sharpening": [("sharpness", "Kekuatan penajaman", 0.0, 3.0, 1.0, 0.1)],
    
    # Deteksi Tepi & Segmentasi
    # Thresholding: piksel yang lebih terang dari batas jadi putih, sisanya hitam.
    "Thresholding": [("threshold", "Nilai ambang", 0, 255, 127, 1)],
    # Edge Detection (Canny): memakai dua batas, yaitu batas bawah dan batas atas.
    "Edge Detection": [("canny_low", "Ambang bawah Canny", 0, 255, 100, 5),
                       ("canny_high", "Ambang atas Canny", 0, 255, 200, 5)],
    
    # Simulasi Noise & Restorasi
    # Noise: makin besar angkanya, makin kasar bintik acaknya.
    "Noise": [("intensity", "Intensitas noise", 0, 100, 25, 5)],
    # Motion Blur: ukuran harus ganjil (mulai 3, naik 2 demi 2).
    "Motion Blur": [("ksize", "Ukuran blur (harus ganjil)", 3, 31, 9, 2)],
}


# Memperkecil gambar supaya prosesnya cepat.
# Lebar dibatasi sampai max_w piksel, tinggi ikut mengecil dengan perbandingan yang sama.
# Kalau gambarnya sudah kecil, dikembalikan apa adanya (tidak diperbesar).
def limit_size(arr, max_w=900):
    h, w = arr.shape[:2]            # tinggi dan lebar gambar
    if w <= max_w:
        return arr
    scale = max_w / w               # seberapa kecil gambar akan dibuat
    # INTER_AREA adalah cara mengecilkan gambar yang hasilnya paling bagus
    return cv2.resize(arr, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)


# Mengubah gambar abu-abu (1 lapis) menjadi 3 lapis dengan nilai R, G, B yang sama.
# Tampilannya tetap abu-abu, tetapi bentuk datanya jadi sama dengan gambar berwarna.
# Ini perlu supaya operasi lain, penampil gambar, dan histogram tidak error.
def _gray3(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def apply_operation(arr, op, p):
    """arr: array RGB uint8. p: kamus parameter. Mengembalikan array RGB uint8."""
    
    # Penyesuaian Cahaya & Warna

    # BRIGHTNESS: menambahkan angka slider ke semua piksel.
    # Rumus: hasil = piksel + angka slider.
    # Gambar diubah ke int16 dulu supaya angkanya muat. Kalau tetap uint8 (maksimal 255),
    # hasil 250 + 20 akan muter balik jadi 14, bukan 255.
    # np.clip memotong hasil supaya tetap 0 sampai 255. Di bawah 0 jadi 0, di atas 255 jadi 255.
    # astype(np.uint8) mengembalikan tipe data ke bentuk gambar semula.
    if op == "Brightness":
        return np.clip(arr.astype(np.int16) + int(p.get("brightness", 0)), 0, 255).astype(np.uint8)

    # CONTRAST: membuat bagian terang makin terang dan bagian gelap makin gelap (atau sebaliknya).
    # Rumus: hasil = (piksel - 128) x faktor + 128.
    # 128 adalah titik tengah dari 0 sampai 255. Piksel diukur jaraknya dari titik tengah,
    # jaraknya dikali faktor, lalu dikembalikan ke skala semula.
    # Faktor di atas 1 memperlebar jarak (kontras naik). Faktor di bawah 1 mempersempit (kontras turun).
    # float32 dipakai karena faktornya bisa berupa desimal, misalnya 0.8.
    if op == "Contrast":
        f = p.get("contrast", 1.0)
        return np.clip((arr.astype(np.float32) - 128) * f + 128, 0, 255).astype(np.uint8)

    # GRAYSCALE: mengubah gambar berwarna jadi abu-abu.
    # Rumus: abu-abu = 0.299 x R + 0.587 x G + 0.114 x B.
    # Angka pengalinya tidak sama karena mata manusia paling peka pada warna hijau.
    # Hasilnya disalin lagi jadi 3 lapis lewat _gray3.
    if op == "Grayscale":
        return _gray3(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY))

    # NEGATIVE: membalik terang jadi gelap dan gelap jadi terang.
    # Rumus: hasil = 255 - piksel, berlaku untuk semua warna (R, G, B) sekaligus.
    # Tidak perlu np.clip karena hasilnya pasti tetap 0 sampai 255.
    if op == "Negative":
        return 255 - arr

    # HISTOGRAM EQUALIZATION: meratakan sebaran terang gelap ke seluruh rentang 0 sampai 255.
    # Gambar diubah ke bentuk YCrCb: Y adalah kecerahan, Cr dan Cb adalah warna.
    # Yang diratakan hanya Y, jadi warnanya tidak ikut berubah.
    # Kalau R, G, B diratakan satu per satu, warna gambar bisa jadi aneh.
    if op == "Histogram Equalization":
        ycrcb = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    # Geometri

    # ROTATION: memutar gambar dengan titik tengah gambar sebagai pusat putaran.
    # getRotationMatrix2D membuat rumus putarannya (pusat, sudut, skala 1.0 artinya ukuran tetap).
    # warpAffine menjalankan putaran itu. Ukuran gambar tetap, jadi bagian yang keluar dari
    # bingkai terpotong, dan sudut yang kosong diisi warna hitam.
    if op == "Rotation":
        h, w = arr.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), p.get("angle", 0), 1.0)
        return cv2.warpAffine(arr, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # FLIP HORIZONTAL: membalik gambar kiri dan kanan (seperti cermin).
    # [:, ::-1] artinya urutan kolom dibalik.
    # ascontiguousarray merapikan data di memori supaya tidak error di OpenCV dan Streamlit.
    if op == "Flip Horizontal":
        return np.ascontiguousarray(arr[:, ::-1])

    # FLIP VERTICAL: membalik gambar atas dan bawah.
    # [::-1] artinya urutan baris dibalik.
    if op == "Flip Vertical":
        return np.ascontiguousarray(arr[::-1])

    # Filter Dasar & Penajaman

    # GAUSSIAN SMOOTHING: setiap piksel diganti dengan rata-rata piksel di sekitarnya.
    # Piksel yang paling dekat pengaruhnya paling besar. Hasilnya gambar lebih halus
    # dan bintik-bintik kecil (noise) berkurang.
    if op == "Gaussian Smoothing":
        # Mengubah nilai radius dari slider menjadi ukuran kernel ganjil (agar ada titik pusat yang pas)
        # Contoh: radius 2 menjadi kotak ukuran 5 x 5.
        k = 2 * int(p.get("radius", 2)) + 1
        # Menerapkan fungsi konvolusi Gaussian bawaan OpenCV
        # Angka 0 di akhir artinya OpenCV menentukan sendiri tingkat kehalusannya.
        return cv2.GaussianBlur(arr, (k, k), 0)

    # SHARPENING: membuat gambar lebih tajam.
    # Caranya: buat versi buram dari gambar, cari selisihnya dengan gambar asli
    # (selisih itu adalah bagian detail), lalu detailnya diperkuat.
    # Rumus: hasil = (1 + s) x asli - s x buram.
    if op == "Sharpening":
       # Mengambil nilai kekuatan penajaman dari slider 
        s = p.get("sharpness", 1.0)
        # Menerapkan Gaussian Blur untuk mendapatkan versi buram dari citra
        # Ukuran (0, 0) artinya OpenCV menghitung sendiri dari angka 1.5.
        blur = cv2.GaussianBlur(arr, (0, 0), 1.5)
        # Menerapkan operasi penajaman dengan menggabungkan citra asli dan citra buram
        # addWeighted menghitung: asli x (1 + s) + buram x (-s) + 0
        return cv2.addWeighted(arr, 1 + s, blur, -s, 0)

    # Deteksi Tepi & Segmentasi

    # THRESHOLDING: piksel yang lebih terang dari batas jadi putih (255), sisanya hitam (0).
    # Hasilnya gambar hitam putih saja. Garis putus-putus di histogram menandai posisi batas ini
    # (lihat variabel marker di app.py).
    if op == "Thresholding":
        # Mengubah citra warna menjadi citra Grayscale (derajat keabuan)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        # Menerapkan fungsi thresholding OpenCV dengan nilai ambang (T) dari slider
        # THRESH_BINARY artinya: di atas batas jadi 255, selain itu jadi 0.
        _, b = cv2.threshold(gray, int(p.get("threshold", 127)), 255, cv2.THRESH_BINARY)
        return _gray3(b)

    # EDGE DETECTION (Canny): mencari garis tepi benda pada gambar abu-abu.
    # Perubahan terang gelap yang tajam (di atas batas atas) pasti dianggap tepi.
    # Yang lemah (di bawah batas bawah) dibuang.
    # Yang berada di antaranya hanya dipakai kalau tersambung dengan tepi yang kuat.
    # Hasilnya gambar hitam dengan garis putih.
    if op == "Edge Detection":
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        e = cv2.Canny(gray, int(p.get("canny_low", 100)), int(p.get("canny_high", 200)))
        return _gray3(e)

    # Simulasi Noise & Restorasi

    # NOISE: menambahkan bintik acak ke gambar.
    # Setiap nilai piksel ditambah angka acak. Makin besar intensitas, makin kasar bintiknya.
    # float32 dipakai supaya angka negatif dan desimal tidak hilang sebelum dipotong ke 0 sampai 255.
    # Karena angkanya acak, hasilnya bisa berbeda setiap kali gambar diproses ulang.
    if op == "Noise":
        intensity = p.get("intensity", 25)
        noise = np.random.normal(0, intensity, arr.shape).astype(np.float32)
        return np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # MOTION BLUR: meniru efek gambar yang bergerak ke samping.
    # kernel adalah kotak kecil berisi angka pengali. Isinya nol semua,
    # kecuali baris tengah yang berisi satu.
    # Kotak itu dibagi size supaya jumlahnya 1 dan kecerahan gambar tidak berubah.
    # filter2D menggeser kotak itu ke seluruh gambar, sehingga setiap piksel diganti
    # rata-rata piksel di kiri dan kanannya.
    if op == "Motion Blur":
        size = int(p.get("ksize", 9))
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = kernel / size
        return cv2.filter2D(arr, -1, kernel)

    # Kalau nama operasi tidak dikenali, kembalikan salinan gambar tanpa perubahan.
    return arr.copy()

# Menjalankan semua operasi di pipeline satu per satu sesuai urutan.
# Hasil dari satu operasi menjadi bahan untuk operasi berikutnya,
# jadi urutan operasi bisa mengubah hasil akhir.
def apply_pipeline(arr, steps):
    out = arr
    for s in steps:
        out = apply_operation(out, s["op"], s["params"])
    return out


# Membuat tulisan untuk keterangan gambar hasil, misalnya "Brightness + Negative".
# Kalau belum ada operasi, tulisannya "Tanpa operasi".
def steps_label(steps):
    return " + ".join(s["op"] for s in steps) if steps else "Tanpa operasi"