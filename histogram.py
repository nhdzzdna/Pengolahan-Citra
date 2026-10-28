import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

ROWS = {"Red": 0, "Green": 1, "Blue": 2, "Gray": 3}
COLORS = {"Red": "red", "Green": "green", "Blue": "blue", "Gray": "black"}


def counts(arr):
    """Array (4, 256): jumlah piksel kanal R, G, B, dan Gray."""
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    rows = [np.bincount(arr[:, :, i].ravel(), minlength=256) for i in range(3)]
    rows.append(np.bincount(gray.ravel(), minlength=256))
    return np.array(rows, dtype=np.float64)


def plot_from_counts(cb, ca, mode="RGB", log=False, cdf=False, marker=None):
    names = ["Gray"] if mode == "Grayscale" else ["Red", "Green", "Blue"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    titles = ["Histogram asli", "Histogram hasil", "Overlay asli dan hasil"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel("Intensitas piksel (0 sampai 255)")
        ax.set_ylabel("Jumlah piksel")
        ax.set_xlim(0, 255)
        ax.grid(True, alpha=0.2)
        if log:
            ax.set_yscale("log")
        if marker is not None:
            ax.axvline(marker, color="gray", linestyle=":", linewidth=1)

    for n in names:
        c, i = COLORS[n], ROWS[n]
        axes[0].plot(cb[i], color=c, linewidth=1, label=n)
        axes[1].plot(ca[i], color=c, linewidth=1, label=n)
        axes[2].plot(cb[i], color=c, linewidth=1, linestyle="--", label=f"{n} asli")
        axes[2].plot(ca[i], color=c, linewidth=1, label=f"{n} hasil")

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


def plot_histograms(before, after, mode="RGB", log=False, cdf=False, marker=None):
    return plot_from_counts(counts(before), counts(after), mode, log, cdf, marker)


def describe(before, after):
    gb = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ga = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mb, ma = gb.mean(), ga.mean()
    if ma - mb > 2:
        arah = "bergeser ke kanan (lebih terang)"
    elif mb - ma > 2:
        arah = "bergeser ke kiri (lebih gelap)"
    else:
        arah = "relatif tidak bergeser"
    clip_b = np.mean((before == 0) | (before == 255)) * 100
    clip_a = np.mean((after == 0) | (after == 255)) * 100
    return (f"Rata-rata intensitas berubah dari {mb:.1f} menjadi {ma:.1f}, histogram {arah}. "
            f"Standar deviasi berubah dari {gb.std():.1f} menjadi {ga.std():.1f}. "
            f"Piksel bernilai 0 atau 255 (clipping) berubah dari {clip_b:.2f}% menjadi {clip_a:.2f}%.")


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
        mse = float(np.mean((before.astype(np.float64) - after.astype(np.float64)) ** 2))
        psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
        data.append({"Statistik": "MSE (asli vs hasil)", "Asli": "", "Hasil": round(mse, 2), "Selisih": ""})
        data.append({"Statistik": "PSNR dB (asli vs hasil)", "Asli": "",
                     "Hasil": "tak hingga" if psnr == float("inf") else round(psnr, 2), "Selisih": ""})
    return pd.DataFrame(data).astype(str)


def _panel(arr, w, h, title):
    panel = np.full((h, w, 3), 255, np.uint8)
    hs = [cv2.calcHist([arr], [i], None, [256], [0, 256]).ravel() for i in range(3)]
    peak = max(x.max() for x in hs) or 1
    xs = np.linspace(0, w - 1, 256).astype(np.int32)
    for hh, col in zip(hs, [(220, 0, 0), (0, 160, 0), (0, 0, 220)]):
        ys = (h - 20 - (hh / peak) * (h - 40)).astype(np.int32)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        cv2.polylines(panel, [pts], False, col, 1, cv2.LINE_AA)
    cv2.putText(panel, title, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(panel, "0", (2, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(panel, "255", (w - 28, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return panel


def render_live(before, after, w=420, h=220):
    """Histogram RGB asli dan hasil, digambar dengan OpenCV agar ringan untuk kamera langsung."""
    return np.hstack([_panel(before, w, h, "Histogram asli"),
                      _panel(after, w, h, "Histogram hasil")])