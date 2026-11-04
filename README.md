# ✨ Pikselia
### Aplikasi Pengolahan Citra Digital dengan Histogram, Video, dan Kamera

---

## 👥 Identitas Kelompok

| Keterangan | Detail |
|---|---|
| Mata Kuliah | Pengolahan Citra |
| Kelas | IF-C |
| Kelompok | 1 (Satu) |
| Nama Proyek | ImageLab |

**Anggota Kelompok:**

1. Cantika Zahra Putri Maharani — 123240091
2. Friska Rima Indriani — 123240111
3. Safna Recyfa Naqiya — 123240115
4. Nahdia Izzadina — 123240251

---

## 📌 Deskripsi Proyek

ImageLab dikembangkan untuk mempermudah pengguna dalam memahami dan menerapkan berbagai teknik pengolahan citra digital melalui antarmuka yang interaktif.

Aplikasi ini memiliki tiga mode utama, yaitu pengolahan gambar, video, dan kamera. Pengguna dapat memilih operasi pengolahan citra, mengatur parameter yang tersedia, serta menggabungkan beberapa operasi ke dalam satu pipeline. Setiap hasil pemrosesan dapat dibandingkan dengan citra asli melalui tampilan visual dan analisis histogram.

Selain menghasilkan citra yang telah diproses, ImageLab juga menyediakan informasi statistik seperti nilai minimum, maksimum, rata-rata, standar deviasi, Mean Squared Error (MSE), dan Peak Signal-to-Noise Ratio (PSNR) untuk membantu mengevaluasi perubahan yang terjadi pada citra.

---

## ✨ Fitur Utama

### 1. Pengolahan Gambar
- Mengunggah gambar dengan format PNG, JPG, JPEG, BMP, dan WEBP.
- Menampilkan perbandingan gambar asli dan hasil pemrosesan.
- Mengatur parameter operasi menggunakan slider.
- Menggabungkan beberapa operasi dalam satu pipeline.
- Menghapus operasi tertentu atau mereset seluruh pipeline.
- Mengunduh hasil pemrosesan dalam format PNG dengan resolusi penuh.

### 2. Pengolahan Video
- Mengunggah video dengan format MP4, AVI, MOV, dan MKV.
- Memproses seluruh frame video menggunakan operasi yang dipilih.
- Menampilkan perbandingan video asli dan hasil pemrosesan.
- Menampilkan histogram yang bergerak mengikuti perubahan frame.
- Menyediakan histogram agregat dari seluruh video berdasarkan sampel frame.
- Menampilkan grafik perubahan rata-rata intensitas per frame.
- Mengunduh video hasil pemrosesan tanpa histogram.

### 3. Pengolahan Kamera
Tersedia dua mode penggunaan kamera:

- **Ambil Foto:** Mengambil gambar menggunakan kamera perangkat, kemudian menerapkan operasi pengolahan citra dan menampilkan hasilnya.
- **Langsung (Real-time):** Memproses video dari kamera secara langsung serta menampilkan perbandingan citra asli, hasil pemrosesan, dan histogram.

### 4. Analisis Histogram
- Menampilkan histogram citra asli dan hasil pemrosesan.
- Menampilkan perbandingan histogram dalam bentuk overlay.
- Mendukung histogram RGB dan Grayscale.
- Menyediakan skala logaritmik.
- Menampilkan Cumulative Distribution Function (CDF).
- Menampilkan garis penanda nilai ambang pada operasi Thresholding.

### 5. Analisis Statistik Citra
Menyediakan perbandingan statistik antara citra asli dan hasil pemrosesan, meliputi:
- Minimum: Intensitas piksel terkecil.
- Maksimum: Intensitas piksel terbesar.
- Rata-rata: Nilai rata-rata intensitas piksel.
- Standar deviasi: Tingkat penyebaran intensitas piksel.
- MSE: Mengukur rata-rata kuadrat perbedaan nilai piksel.
- PSNR: Mengukur kualitas perbedaan citra berdasarkan nilai MSE.

Aplikasi juga memberikan deskripsi otomatis mengenai perubahan rata-rata intensitas, pergeseran histogram, standar deviasi, dan persentase clipping.

---

## 🛠️ Jenis Operasi Pengolahan Citra

ImageLab menyediakan berbagai operasi pengolahan citra yang dikelompokkan berdasarkan fungsinya.

### 1. Penyesuaian Cahaya dan Warna

| Operasi | Deskripsi |
|---|---|
| Brightness | Mengatur tingkat kecerahan citra. Nilai positif mencerahkan dan nilai negatif menggelapkan citra. |
| Contrast | Mengatur perbedaan intensitas antara bagian terang dan gelap. |
| Grayscale | Mengubah citra RGB menjadi citra keabuan. |
| Negative | Membalik nilai intensitas piksel sehingga menghasilkan efek negatif. |
| Histogram Equalization | Meningkatkan kontras citra melalui pemerataan histogram pada kanal luminance. |

### 2. Transformasi Geometri

| Operasi | Deskripsi |
|---|---|
| Rotation | Memutar citra berdasarkan sudut yang ditentukan. |
| Flip Horizontal | Membalik citra secara horizontal. |
| Flip Vertical | Membalik citra secara vertikal. |

### 3. Filter Dasar dan Penajaman

| Operasi | Deskripsi |
|---|---|
| Gaussian Smoothing | Menghaluskan citra dengan mengurangi detail dan noise menggunakan filter Gaussian. |
| Sharpening | Mempertajam detail dan tepi citra menggunakan teknik unsharp masking. |

### 4. Deteksi Tepi dan Segmentasi

| Operasi | Deskripsi |
|---|---|
| Thresholding | Mengubah citra grayscale menjadi citra biner berdasarkan nilai ambang tertentu. |
| Edge Detection | Mendeteksi tepi objek dalam citra menggunakan algoritma Canny. |

### 5. Simulasi Noise dan Restorasi

| Operasi | Deskripsi |
|---|---|
| Noise | Menambahkan Gaussian noise pada citra dengan intensitas yang dapat diatur. |
| Motion Blur | Memberikan efek blur akibat gerakan menggunakan kernel horizontal. |

---

## 📊 Teknologi yang Digunakan

| Teknologi | Kegunaan |
|---|---|
| Python | Bahasa pemrograman utama. |
| Streamlit | Membangun antarmuka aplikasi berbasis web. |
| OpenCV | Melakukan operasi dan pemrosesan citra digital. |
| NumPy | Mengelola array dan perhitungan numerik pada citra. |
| Pandas | Menyusun dan mengelola tabel statistik. |
| Matplotlib | Membuat visualisasi histogram dan grafik. |
| Pillow (PIL) | Membaca, mengonversi, dan menyimpan gambar. |
| Streamlit-WebRTC | Mendukung pemrosesan kamera secara real-time. |
| PyAV | Mengelola frame video dan streaming kamera. |
| ImageIO | Membaca dan menghasilkan file video. |

---

## 📂 Struktur Proyek

```text
ImageLab/
│
├── app.py              # Aplikasi utama Streamlit
├── operations.py       # Implementasi operasi pengolahan citra
├── histogram.py        # Histogram, statistik, dan visualisasi
├── requirements.txt    # Daftar dependensi proyek
└── README.md           # Dokumentasi proyek
```

### Penjelasan File

- **app.py:** Mengatur antarmuka aplikasi, interaksi pengguna, pengunggahan gambar dan video, pengolahan kamera, serta integrasi seluruh fitur.
- **operations.py:** Berisi daftar parameter dan implementasi fungsi operasi pengolahan citra, termasuk pipeline pemrosesan.
- **histogram.py:** Mengelola perhitungan histogram RGB dan grayscale, visualisasi histogram, analisis perubahan intensitas, serta tabel statistik citra.
- **requirements.txt:** Berisi daftar library Python yang diperlukan untuk menjalankan aplikasi.

---

## ⚙️ Instalasi dan Persiapan

### 1. Clone Repository

```bash
git clone https://github.com/username/ImageLab.git
```

Masuk ke direktori proyek:

```bash
cd ImageLab
```

### 2. Buat Virtual Environment (Opsional)

```bash
python -m venv venv
```

Aktifkan virtual environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Instal Dependensi

```bash
pip install -r requirements.txt
```

Apabila belum tersedia file `requirements.txt`, instal library yang dibutuhkan menggunakan:

```bash
pip install streamlit numpy opencv-python pandas matplotlib pillow streamlit-webrtc av imageio imageio-ffmpeg
```

### 4. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan berjalan melalui browser pada alamat lokal yang ditampilkan oleh Streamlit, biasanya:

```text
http://localhost:8501
```

---

## 🚀 Panduan Penggunaan

### 1. Pengolahan Gambar
1. Buka aplikasi ImageLab.
2. Pilih tab **Gambar**.
3. Unggah gambar yang ingin diproses.
4. Pilih operasi pada sidebar.
5. Atur parameter sesuai kebutuhan.
6. Klik tombol **Tambah** untuk memasukkan operasi ke dalam pipeline.
7. Aktifkan atau nonaktifkan pilihan penerapan operasi yang sedang dipilih sesuai kebutuhan.
8. Amati hasil pemrosesan dan analisis histogram.
9. Klik **Siapkan unduhan resolusi penuh** untuk menyiapkan hasil gambar.
10. Klik **Unduh PNG** untuk menyimpan hasilnya.

### 2. Pengolahan Video
1. Pilih tab **Video**.
2. Unggah video yang ingin diproses.
3. Tentukan operasi pengolahan citra melalui sidebar.
4. Klik **Proses seluruh video**.
5. Tunggu hingga proses selesai.
6. Amati perbandingan video asli dan hasil pemrosesan beserta histogram bergerak.
7. Unduh video hasil melalui tombol unduhan.

### 3. Pengolahan Kamera
1. Pilih tab **Kamera**.
2. Tentukan mode kamera:
   - Ambil Foto.
   - Langsung (real-time).
3. Berikan izin akses kamera apabila diminta oleh browser.
4. Pilih operasi pengolahan citra melalui sidebar.
5. Amati hasil pemrosesan dan histogram.

**Catatan:** Mode kamera real-time memerlukan akses melalui localhost atau HTTPS. Pastikan tidak ada aplikasi lain yang sedang menggunakan kamera.

---

## 🔬 Metode Analisis

ImageLab menggunakan beberapa pendekatan untuk menganalisis perubahan citra setelah proses pengolahan.

### Histogram
Histogram digunakan untuk menggambarkan distribusi intensitas piksel pada citra. Aplikasi menghitung distribusi kanal Red, Green, Blue, dan Grayscale dengan rentang intensitas 0–255.

### Cumulative Distribution Function (CDF)
CDF digunakan untuk menunjukkan distribusi kumulatif intensitas piksel. Visualisasi ini dapat membantu memahami penyebaran intensitas pada citra.

### Mean Squared Error (MSE)
MSE digunakan untuk menghitung rata-rata kuadrat selisih antara piksel citra asli dan hasil pemrosesan.

### Peak Signal-to-Noise Ratio (PSNR)
PSNR digunakan untuk mengukur perbedaan kualitas antara citra asli dan hasil pemrosesan berdasarkan nilai MSE. Nilai PSNR yang lebih tinggi menunjukkan perbedaan piksel yang lebih kecil.

---

## 🎯 Tujuan Pengembangan

Pengembangan ImageLab bertujuan untuk:

1. Mengimplementasikan berbagai teknik pengolahan citra digital dalam satu aplikasi.
2. Memberikan fasilitas pengolahan citra pada gambar, video, dan kamera secara langsung.
3. Memahami pengaruh setiap operasi pengolahan citra terhadap karakteristik visual dan distribusi intensitas piksel.
4. Mengimplementasikan visualisasi histogram sebagai sarana analisis perubahan citra.
5. Menerapkan evaluasi hasil pemrosesan menggunakan parameter statistik dan metrik evaluasi citra.

---

## 👩‍💻 Pengembang

ImageLab dikembangkan oleh mahasiswa Program Studi Informatika, kelas IF-C, Kelompok 1, sebagai pemenuhan proyek akhir mata kuliah Pengolahan Citra.

**Kelompok 1 — IF-C**

- Cantika Zahra Putri Maharani
- Friska Rima Indriani
- Safna Recyfa Naqiya
- Nahdia Izzadina

---

<p align="center">
  <b>ImageLab — Explore, Process, and Analyze Images.</b>
  <br>
  <i>Proyek Akhir Mata Kuliah Pengolahan Citra</i>
</p>
