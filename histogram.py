import numpy as np
import pandas as pd                  # untuk membuat tabel statistik
import cv2
import matplotlib.pyplot as plt      # untuk menggambar grafik histogram

# ROWS: nomor baris untuk tiap warna di tabel hitungan.
# COLORS: warna garis di grafik. Baris ke-3 (Gray) adalah versi abu-abu dari gambar.
ROWS = {"Red": 0, "Green": 1, "Blue": 2, "Gray": 3}
COLORS = {"Red": "red", "Green": "green", "Blue": "blue", "Gray": "black"}


def counts(arr):
    """Array (4, 256): jumlah piksel kanal R, G, B, dan Gray."""
    # Gambar diubah ke abu-abu untuk baris keempat.
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # np.bincount menghitung berapa kali setiap angka 0 sampai 255 muncul di satu warna.
    # Inilah cara menghitung histogram: berapa banyak piksel yang punya nilai tertentu.
    rows = [np.bincount(arr[:, :, i].ravel(), minlength=256) for i in range(3)]
    rows.append(np.bincount(gray.ravel(), minlength=256))
    return np.array(rows, dtype=np.float64)


# Menggambar tiga grafik berdampingan: histogram asli, histogram hasil, dan gabungan keduanya.
# cb dan ca adalah hasil hitungan counts() untuk gambar asli (before) dan hasil (after).
# Pengaturan dari sidebar:
# mode   : "RGB" (tiga garis warna) atau "Grayscale" (satu garis)
# log    : True kalau sumbu tegak memakai skala logaritmik
# cdf    : True kalau garis CDF ikut ditampilkan
# marker : garis putus-putus penanda batas (dipakai untuk Thresholding)
def plot_from_counts(cb, ca, mode="RGB", log=False, cdf=False, marker=None):
    names = ["Gray"] if mode == "Grayscale" else ["Red", "Green", "Blue"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    titles = ["Histogram asli", "Histogram hasil", "Overlay asli dan hasil"]
    # Pengaturan yang sama untuk ketiga grafik: judul, nama sumbu, rentang, dan garis bantu.
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel("Intensitas piksel (0 sampai 255)")
        ax.set_ylabel("Jumlah piksel")
        ax.set_xlim(0, 255)
        ax.grid(True, alpha=0.2)
        if log:
            ax.set_yscale("log")  # skala logaritmik: angka kecil tetap kelihatan
        if marker is not None:
            ax.axvline(marker, color="gray", linestyle=":", linewidth=1)

    # Menggambar garisnya. Grafik 1 untuk gambar asli, grafik 2 untuk hasil.
    # Grafik 3 menumpuk keduanya: garis putus-putus untuk asli, garis penuh untuk hasil.
    for n in names:
        c, i = COLORS[n], ROWS[n]
        axes[0].plot(cb[i], color=c, linewidth=1, label=n)
        axes[1].plot(ca[i], color=c, linewidth=1, label=n)
        axes[2].plot(cb[i], color=c, linewidth=1, linestyle="--", label=f"{n} asli")
        axes[2].plot(ca[i], color=c, linewidth=1, label=f"{n} hasil")

    # CDF: jumlah piksel ditambahkan sedikit demi sedikit dari nilai 0, lalu dibagi total piksel.
    # Artinya: berapa bagian piksel yang nilainya sampai di titik itu. Naik dari 0 sampai 1.
    # Dibuat dengan sumbu kedua di sisi kanan supaya tidak tercampur dengan jumlah piksel.
    if cdf:
        for ax, data in ((axes[0], cb), (axes[1], ca)):
            ax2 = ax.twinx()
            for n in names:
                h = data[ROWS[n]]
                ax2.plot(np.cumsum(h) / max(h.sum(), 1), color=COLORS[n], alpha=0.35, linewidth=1)
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("CDF")

    axes[0].legend()
    axes[2].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


# Cara singkat untuk satu pasang gambar: hitung dulu, lalu langsung digambar.
def plot_histograms(before, after, mode="RGB", log=False, cdf=False, marker=None):
    return plot_from_counts(counts(before), counts(after), mode, log, cdf, marker)


# Menyusun kalimat keterangan otomatis yang muncul di kotak biru.
# Perbandingannya memakai versi abu-abu dari kedua gambar.
def describe(before, after):
    gb = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ga = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mb, ma = gb.mean(), ga.mean()
    # Histogram dianggap bergeser kalau rata-ratanya berubah lebih dari 2.
    # Kalimat ini hanya melihat rata-rata. Pada Negative, rata-ratanya hampir sama,
    # jadi tertulis "tidak bergeser" padahal histogramnya sebenarnya terbalik.
    if ma - mb > 2:
        arah = "bergeser ke kanan (lebih terang)"
    elif mb - ma > 2:
        arah = "bergeser ke kiri (lebih gelap)"
    else:
        arah = "relatif tidak bergeser"
    # Persentase clipping: berapa persen nilai piksel (di semua warna) yang tepat 0 atau 255.
    clip_b = np.mean((before == 0) | (before == 255)) * 100
    clip_a = np.mean((after == 0) | (after == 255)) * 100
    return (f"Rata-rata intensitas berubah dari {mb:.1f} menjadi {ma:.1f}, histogram {arah}. "
            f"Standar deviasi berubah dari {gb.std():.1f} menjadi {ga.std():.1f}. "
            f"Piksel bernilai 0 atau 255 (clipping) berubah dari {clip_b:.2f}% menjadi {clip_a:.2f}%.")


# Membuat tabel statistik: nilai terkecil, terbesar, rata-rata, dan standar deviasi
# (dihitung dari versi abu-abu), ditambah MSE dan PSNR kalau ukuran kedua gambar sama.
def stats_table(before, after):
    gb = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY).astype(np.float64)
    ga = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY).astype(np.float64)
    rows = {
        "Minimum": (gb.min(), ga.min()),
        "Maksimum": (gb.max(), ga.max()),
        "Rata-rata": (gb.mean(), ga.mean()),
        "Standar deviasi": (gb.std(), ga.std()),
    }
    data = [{"Statistik": k, "Asli": round(a, 2), "Hasil": round(b, 2), "Selisih": round(b - a, 2)}
            for k, (a, b) in rows.items()]
    if before.shape == after.shape:
        # MSE: rata-rata dari selisih nilai piksel (asli dan hasil) yang dikuadratkan.
        # Makin besar angkanya, makin beda kedua gambar.
        mse = float(np.mean((before.astype(np.float64) - after.astype(np.float64)) ** 2))
        # PSNR (satuan dB): 10 x log10(255 x 255 / MSE).
        # Makin kecil angkanya, makin beda kedua gambar. Kalau gambarnya sama persis, hasilnya tak hingga.
        psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
        data.append({"Statistik": "MSE (asli vs hasil)", "Asli": "", "Hasil": round(mse, 2), "Selisih": ""})
        data.append({"Statistik": "PSNR dB (asli vs hasil)", "Asli": "",
                     "Hasil": "tak hingga" if psnr == float("inf") else round(psnr, 2), "Selisih": ""})
    # Semua nilai diubah jadi teks supaya kolom yang campur angka dan kosong tetap rapi di tabel.
    return pd.DataFrame(data).astype(str)


# Panel histogram sederhana yang digambar dengan OpenCV (lebih ringan dari matplotlib).
# Dipakai untuk video dan kamera langsung. Isinya garis R, G, B di atas latar putih.
def _panel(arr, w, h, title):
    panel = np.full((h, w, 3), 255, np.uint8)  # latar putih berukuran h x w
    # Menghitung histogram tiap warna (256 batang).
    hs = [cv2.calcHist([arr], [i], None, [256], [0, 256]).ravel() for i in range(3)]
    # Nilai tertinggi dipakai untuk mengatur tinggi garis supaya pas di dalam panel.
    peak = max(x.max() for x in hs) or 1
    # Posisi kiri kanan: 256 nilai dibagi rata ke seluruh lebar panel.
    xs = np.linspace(0, w - 1, 256).astype(np.int32)
    for hh, col in zip(hs, [(220, 0, 0), (0, 160, 0), (0, 0, 220)]):
        # Posisi atas bawah dibalik karena pada gambar, angka y makin besar berarti makin ke bawah.
        ys = (h - 20 - (hh / peak) * (h - 40)).astype(np.int32)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        cv2.polylines(panel, [pts], False, col, 1, cv2.LINE_AA)
    # Menulis judul, serta angka 0 dan 255 di bawah grafik.
    cv2.putText(panel, title, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(panel, "0", (2, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(panel, "255", (w - 28, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return panel


def render_live(before, after, w=420, h=220):
    """Histogram RGB asli dan hasil, digambar dengan OpenCV agar ringan untuk kamera langsung."""
    # Catatan: fungsi ini di-import di app.py, tetapi saat ini belum dipanggil.
    return np.hstack([_panel(before, w, h, "Histogram asli"),
                      _panel(after, w, h, "Histogram hasil")])