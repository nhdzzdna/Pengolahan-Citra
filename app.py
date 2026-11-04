import io
import time              
import tempfile          # untuk membuat file sementara (dipakai untuk video)

import av                # untuk membuat frame video pada kamera langsung
import cv2
import imageio           # untuk menyimpan file video hasil
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Fungsi pengolahan gambar dari operations.py, dan fungsi histogram dari histogram.py
from operations import PARAMS, apply_pipeline, limit_size, steps_label
from histogram import (plot_histograms, plot_from_counts, counts, describe,
                       stats_table, _panel, render_live)

# Pengaturan halaman: judul tab browser (nama aplikasi Pikselia), ikon, dan tampilan lebar.
st.set_page_config(page_title="Pikselia", page_icon="✨", layout="wide")

# CSS di bawah hanya mengatur tampilan (warna hijau muda, huruf, tombol, tab, kotak info, dan lain-lain).
# CSS tidak memengaruhi cara gambar diolah.
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp :is(p, h1, h2, h3, h4, h5, h6, label, button, input, textarea, li) {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
[data-testid="stIconMaterial"], .material-symbols-rounded, span[class*="material"] {
    font-family: 'Material Symbols Rounded' !important;
    font-feature-settings: 'liga' !important;
}
.stApp { background: #F6FBF8 !important; color-scheme: light; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
header[data-testid="stHeader"] { background: transparent !important; }
.stApp :is(p, span, label, li, small, h1, h2, h3, h4, h5, h6) { color: #23343A !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: #55696F !important; }
.hero { padding: 1.4rem 1.6rem; border-radius: 18px; margin-bottom: 1.2rem;
        background: linear-gradient(135deg, #E3F5EC 0%, #FDF1E7 100%); border: 1px solid #CFE5DB; }
.hero h1 { margin: 0; font-size: 1.9rem; font-weight: 700; color: #1F4B43 !important; }
.hero p { margin: .3rem 0 0 0; font-size: .95rem; color: #3F5B57 !important; }
[data-testid="stSidebar"] { background: #FFFFFF !important; border-right: 1px solid #DCEDE4; }
[data-testid="stSidebar"] h2 { font-size: 1.05rem; font-weight: 700; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"],
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #23343A !important; }
[data-testid="stSlider"] [role="slider"] { background: #2E8B78 !important; box-shadow: none !important; }
.stButton > button, .stDownloadButton > button {
    border-radius: 10px; border: 1px solid #A9D3C3; background: #FFFFFF !important;
    transition: all .15s ease; }
.stButton > button p, .stDownloadButton > button p { color: #23343A !important; font-weight: 500; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: #2E8B78; background: #E8F4EE !important; }
.stButton > button[kind="primary"] { background: #2E8B78 !important; border-color: #2E8B78; }
.stButton > button[kind="primary"] p { color: #FFFFFF !important; }
.stButton > button[kind="primary"]:hover { background: #247060 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #CFE5DB; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 8px 18px; background: transparent; }
.stTabs [data-baseweb="tab"] p { color: #3F5B57 !important; font-weight: 500; }
.stTabs [aria-selected="true"] { background: #E3F5EC; }
.stTabs [aria-selected="true"] p { color: #1F6E5C !important; font-weight: 700; }
div[data-baseweb="select"], div[data-baseweb="select"] div {
    background-color: #FFFFFF !important; color: #23343A !important; }
div[data-baseweb="select"] > div { border: 1px solid #A9D3C3 !important; border-radius: 10px !important; }
div[data-baseweb="select"] svg { fill: #55696F !important; }
[data-baseweb="popover"], [data-baseweb="popover"] ul, [data-baseweb="popover"] li, [data-baseweb="menu"] {
    background: #FFFFFF !important; color: #23343A !important; }
[data-baseweb="popover"] li * { color: #23343A !important; }
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] { background: #E8F4EE !important; }
[data-testid="stAlert"], [data-testid="stAlert"] > div { background: #E6F4F8 !important; border-radius: 12px; }
[data-testid="stAlert"] { border: 1px solid #BFDDE6; }
[data-testid="stAlert"] * { color: #1F4650 !important; }
[data-testid="stExpander"] { background: #FFFFFF; border: 1px solid #CFE5DB; border-radius: 12px; }
[data-testid="stExpander"] summary { background: #FFFFFF; }
[data-testid="stExpander"] summary * { color: #23343A !important; }
[data-testid="stFileUploaderDropzone"] { background: #FFFFFF !important; border: 2px dashed #A9D3C3; border-radius: 14px; }
[data-testid="stFileUploaderDropzone"] * { color: #3F5B57 !important; }
[data-testid="stFileUploaderDropzone"] button { background: #FFFFFF !important; border: 1px solid #A9D3C3; }
[data-testid="stCameraInput"] * { color: #23343A !important; }
[data-testid="stCameraInput"] button { background: #FFFFFF !important; }
[data-testid="stImage"] img { border-radius: 12px; border: 1px solid #DCEDE4; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid #DCEDE4; background: #FFFFFF; }
[data-baseweb="radio"] *, [data-baseweb="checkbox"] * { color: #23343A !important; }
[data-testid="stProgress"] * { color: #23343A !important; }
</style>
"""
# Memasang CSS ke halaman, lalu menampilkan banner judul (Pikselia) di bagian atas.
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="hero"><h1>Pikselia</h1>'
    '<p>Aplikasi pengolahan citra digital dengan histogram, video dan kamera</p></div>',
    unsafe_allow_html=True,
)

# session_state adalah "tempat menyimpan" data supaya tidak hilang
# (Streamlit menjalankan ulang seluruh kode setiap kali ada klik atau geser slider).
# pipeline      : daftar operasi yang sudah ditambahkan pengguna
# apply_current : status centang "Terapkan operasi yang sedang dipilih"
if "pipeline" not in st.session_state:
    st.session_state.pipeline = []
if "apply_current" not in st.session_state:
    st.session_state.apply_current = True


# Dijalankan saat tombol Tambah ditekan: operasi beserta pengaturannya masuk ke pipeline.
# Centang "apply_current" dimatikan supaya operasi yang baru ditambahkan
# tidak dijalankan dua kali (sekali dari pipeline, sekali dari centang).
def add_step(op, params):
    st.session_state.pipeline.append({"op": op, "params": dict(params)})
    st.session_state.apply_current = False


# Menghapus satu operasi dari pipeline sesuai nomornya.
def remove_step(i):
    st.session_state.pipeline.pop(i)


# Mengosongkan seluruh pipeline (tombol Reset).
def reset_pipeline():
    st.session_state.pipeline = []


# Membaca file gambar dan mengubahnya menjadi data piksel berwarna RGB.
# exif_transpose meluruskan foto dari HP yang suka miring.
# convert("RGB") memastikan gambar selalu punya 3 warna (PNG transparan dan gambar abu-abu ikut disamakan).
def load_image(file):
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    return np.asarray(img)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Pilih operasi")
    # group = st.selectbox("Kelompok", list(GROUPS.keys()))
    # Daftar pilihan operasi diambil dari kunci PARAMS di operations.py.
    op = st.selectbox("Operasi", list(PARAMS.keys()))

    st.header("2. Atur parameter")
    params = {}
    specs = PARAMS.get(op, [])
    # Membuat satu slider untuk setiap pengaturan milik operasi yang dipilih.
    # Tiap pengaturan berisi (kunci, label, minimum, maksimum, nilai awal, langkah).
    # key=f"{op}_{key}" memberi nama unik pada slider supaya nilainya tidak tertukar
    # saat berpindah operasi.
    for key, label, lo, hi, default, step in specs:
        params[key] = st.slider(label, lo, hi, default, step, key=f"{op}_{key}")
    if not specs:
        st.caption("Operasi ini tidak memiliki parameter.")

    # Kalau dicentang, operasi yang sedang dipilih langsung dicoba (pratinjau)
    # tanpa harus ditambahkan ke pipeline dulu.
    st.checkbox("Terapkan operasi yang sedang dipilih", key="apply_current")
    c1, c2 = st.columns(2)
    # on_click: fungsi yang dijalankan saat tombol ditekan. args: bahan untuk fungsi itu.
    c1.button("Tambah", on_click=add_step, args=(op, params),
              type="primary", width="stretch")
    c2.button("Reset", on_click=reset_pipeline, width="stretch")

    st.header("3. Pipeline")
    if not st.session_state.pipeline:
        st.caption("Pipeline masih kosong. Klik Tambah untuk menyusun urutan operasi.")
    # Menampilkan tiap operasi di pipeline, lengkap dengan tombol hapus (✕).
    for i, s in enumerate(st.session_state.pipeline):
        a, b = st.columns([4, 1])
        a.write(f"{i + 1}. {s['op']}")
        b.button("✕", key=f"del{i}", on_click=remove_step, args=(i,))

    st.divider()
    # Pengaturan tampilan histogram (tidak mengubah gambar):
    # mode RGB atau Grayscale, skala logaritmik, dan garis CDF.
    with st.expander("Pengaturan histogram"):
        hist_mode = st.radio("Mode", ["RGB", "Grayscale"], horizontal=True)
        hist_log = st.checkbox("Skala logaritmik", value=False)
        hist_cdf = st.checkbox("Tampilkan CDF", value=False)

# Menyusun daftar operasi yang benar-benar dijalankan:
# isi pipeline, ditambah operasi yang sedang dipilih kalau kotaknya dicentang.
steps = list(st.session_state.pipeline)
if st.session_state.apply_current:
    steps.append({"op": op, "params": dict(params)})
label = steps_label(steps)
# Kalau operasi terakhir adalah Thresholding, nilai batasnya dipakai
# sebagai garis putus-putus di histogram.
marker = None
if steps and steps[-1]["op"] == "Thresholding":
    marker = steps[-1]["params"].get("threshold")


def show_analysis(before, after):
    """Histogram, keterangan otomatis, dan tabel statistik untuk satu pasang gambar."""
    st.pyplot(plot_histograms(before, after, hist_mode, hist_log, hist_cdf, marker), clear_figure=True)
    st.info(describe(before, after))
    st.dataframe(stats_table(before, after), hide_index=True, width="stretch")


# Tiga tab utama aplikasi. original dan processed disiapkan untuk menyimpan gambar asli dan hasil.
tab_img, tab_vid, tab_cam = st.tabs(["🖼️ Gambar", "🎞️ Video", "📷 Kamera"])
original = processed = None

# ---------------------------------------------------------------- tab gambar
with tab_img:
    f = st.file_uploader("Unggah gambar", type=["png", "jpg", "jpeg", "bmp", "webp"], key="img")
    if f is None:
        st.info("Unggah gambar untuk memulai. Pilih operasi pada panel kiri, lalu amati perubahan "
                "gambar dan histogramnya.")
    else:
        try:
            full = load_image(f)
        except Exception as exc:
            st.error(f"Gambar tidak dapat dibuka: {exc}")
            full = None
        if full is not None:
            # Tampilan di layar memakai gambar yang dikecilkan supaya cepat.
            # Gambar yang diunduh memakai ukuran penuh (full).
            original = limit_size(full)
            processed = apply_pipeline(original, steps)
            c1, c2 = st.columns(2)
            c1.image(original, caption=f"Gambar asli ({full.shape[1]} x {full.shape[0]} piksel)",
                     width="stretch")
            c2.image(processed, caption=f"Hasil: {label}", width="stretch")
            st.subheader("Histogram dan statistik")
            show_analysis(original, processed)

            st.caption("Pratinjau diperkecil agar cepat. Hasil unduhan memakai resolusi penuh.")
            if st.button("Siapkan unduhan resolusi penuh"):
                result = apply_pipeline(full, steps)
                buf = io.BytesIO()   # tempat menyimpan sementara di memori, tanpa membuat file
                Image.fromarray(result).save(buf, format="PNG")
                st.download_button("Unduh PNG", buf.getvalue(), "imagelab_hasil.png", "image/png")

# ---------------------------------------------------------------- tab video
def _label_frame(frame, text):
    """Tambahkan label teks di pojok kiri atas frame."""
    out = frame.copy()
    # getTextSize mengukur ukuran tulisan supaya kotak hitam di belakangnya pas.
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(out, (0, 0), (tw + 16, th + 16), (0, 0, 0), -1)
    cv2.putText(out, text, (8, th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def process_video(path, steps, progress, max_w=640):
    """Proses video dan buat tiga output: video hasil, video gabungan
    (asli+hasil+histogram bergerak), serta statistik agregat."""
    from histogram import _panel  # panel histogram ringan berbasis OpenCV

    cap = cv2.VideoCapture(path)                    # membuka video asli
    fps = cap.get(cv2.CAP_PROP_FPS) or 25           # kalau FPS tidak terbaca, pakai 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1  # jumlah frame, untuk bar kemajuan

    # File sementara untuk dua video hasil.
    result_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    combined_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    result_writer = imageio.get_writer(result_path, fps=fps, codec="libx264",
                                       pixelformat="yuv420p", macro_block_size=1)
    combined_writer = None  # dibuat setelah ukuran frame pertama diketahui

    # cb dan ca menampung histogram gabungan dari semua frame yang diambil.
    # mean_b, mean_a, idxs menyimpan rata-rata terang gelap tiap frame untuk grafik garis.
    cb, ca = np.zeros((4, 256)), np.zeros((4, 256))
    mean_b, mean_a, idxs = [], [], []
    i = 0
    sample_step = 5   # statistik diambil tiap 5 frame supaya prosesnya tidak terlalu lama
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # OpenCV membaca warna dengan urutan BGR, jadi diubah ke RGB, lalu dikecilkan.
        rgb = limit_size(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), max_w)
        h, w = rgb.shape[:2]
        # Video butuh lebar dan tinggi genap, jadi kelebihan 1 piksel dibuang.
        rgb = np.ascontiguousarray(rgb[:h - h % 2, :w - w % 2])
        out = apply_pipeline(rgb, steps)
        result_writer.append_data(out)

        # --- bangun frame gabungan: [asli | hasil] di atas, [histogram] di bawah ---
        fh, fw = rgb.shape[:2]
        labeled_orig = _label_frame(rgb, "Asli")
        labeled_result = _label_frame(out, "Hasil")
        top_row = np.hstack([labeled_orig, labeled_result])

        hist_h = max(fh // 3, 120)
        hist_panel = np.hstack([
            _panel(rgb, fw, hist_h, "Histogram asli"),
            _panel(out, fw, hist_h, "Histogram hasil"),
        ])

        combined_frame = np.vstack([top_row, hist_panel])
        ch, cw = combined_frame.shape[:2]
        combined_frame = np.ascontiguousarray(
            combined_frame[:ch - ch % 2, :cw - cw % 2]
        )

        if combined_writer is None:
            combined_writer = imageio.get_writer(
                combined_path, fps=fps, codec="libx264",
                pixelformat="yuv420p", macro_block_size=1,
            )
        combined_writer.append_data(combined_frame)

        # Mengambil contoh statistik: menjumlahkan histogram dan mencatat rata-rata terang gelap.
        if i % sample_step == 0:
            cb += counts(rgb)
            ca += counts(out)
            mean_b.append(float(rgb.mean()))
            mean_a.append(float(out.mean()))
            idxs.append(i)
        i += 1
        progress.progress(min(i / total, 1.0))   # menggerakkan bar kemajuan

    cap.release()
    result_writer.close()
    if combined_writer is not None:
        combined_writer.close()
    return result_path, combined_path, cb, ca, idxs, mean_b, mean_a


with tab_vid:
    vf = st.file_uploader("Unggah video", type=["mp4", "avi", "mov", "mkv"], key="vid")
    if vf is None:
        st.info("Unggah video untuk memulai. Pilih operasi pada panel kiri, lalu klik **Proses seluruh video** "
                "untuk melihat perbandingan video asli dan hasil beserta histogram bergeraknya.")
    else:
        # Kalau yang diunggah video baru, simpan ke file sementara dan hapus hasil proses video lama.
        if st.session_state.get("video_name") != vf.name:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(vf.read())
            tmp.close()
            st.session_state["video_path"] = tmp.name
            st.session_state["video_name"] = vf.name
            # reset hasil proses jika video baru di-upload
            for k in ["vid_combined_data", "vid_result_data", "vid_cb", "vid_ca",
                       "vid_idxs", "vid_mean_b", "vid_mean_a"]:
                st.session_state.pop(k, None)
        vpath = st.session_state["video_path"]

        # Membaca info dasar video (jumlah frame dan FPS) untuk ditampilkan.
        cap = cv2.VideoCapture(vpath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        st.caption(f"Jumlah frame: {total_frames}, FPS: {fps:.1f}")

        st.caption("Pemrosesan memperkecil video ke lebar maksimum 640 piksel agar tidak terlalu lama.")
        if st.button("Proses seluruh video"):
            bar = st.progress(0.0)
            result_path, combined_path, cb, ca, idxs, mean_b, mean_a = \
                process_video(vpath, steps, bar)
            # Hasil disimpan di session_state supaya tidak hilang saat halaman dijalankan ulang.
            with open(combined_path, "rb") as fh:
                st.session_state["vid_combined_data"] = fh.read()
            with open(result_path, "rb") as fh:
                st.session_state["vid_result_data"] = fh.read()
            st.session_state["vid_cb"] = cb
            st.session_state["vid_ca"] = ca
            st.session_state["vid_idxs"] = idxs
            st.session_state["vid_mean_b"] = mean_b
            st.session_state["vid_mean_a"] = mean_a

        # ---- tampilkan hasil jika sudah diproses ----
        if "vid_combined_data" in st.session_state:
            st.subheader("Perbandingan video dengan histogram bergerak")
            st.caption("Tekan play — video asli dan hasil diputar bersamaan, "
                       "histogram di bawahnya bergerak mengikuti setiap frame.")
            st.video(st.session_state["vid_combined_data"])

            st.download_button("Unduh video hasil (tanpa histogram)",
                               st.session_state["vid_result_data"],
                               "imagelab_video.mp4", "video/mp4")

            st.divider()

            # ---- histogram agregat ----
            st.subheader("Histogram agregat seluruh video")
            st.pyplot(plot_from_counts(
                st.session_state["vid_cb"], st.session_state["vid_ca"],
                hist_mode, hist_log, hist_cdf, marker
            ), clear_figure=True)

            # ---- rata-rata intensitas per frame ----
            st.subheader("Rata-rata intensitas per frame")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(st.session_state["vid_idxs"], st.session_state["vid_mean_b"],
                    "--", label="Video asli")
            ax.plot(st.session_state["vid_idxs"], st.session_state["vid_mean_a"],
                    label="Video hasil")
            ax.set_xlabel("Nomor frame")
            ax.set_ylabel("Rata-rata intensitas")
            ax.grid(True, alpha=0.2)
            ax.legend()
            st.pyplot(fig, clear_figure=True)

# ---------------------------------------------------------------- tab kamera
# Kelas untuk mengolah video kamera langsung. Fungsi recv dipanggil untuk setiap frame dari kamera.
class LiveProcessor(VideoProcessorBase):
    def __init__(self):
        self.steps = []      # daftar operasi yang dijalankan (diperbarui dari bagian paling bawah)
        self.before = None   # frame asli terakhir
        self.after = None    # frame hasil terakhir
        self.n = 0           # penghitung frame
        self.panels = None   # panel histogram yang disimpan (supaya tidak dihitung ulang tiap frame)

    def recv(self, frame):
        rgb = frame.to_ndarray(format="rgb24")
        out = apply_pipeline(rgb, self.steps)
        out = np.clip(out, 0, 255).astype(np.uint8) 
        # Kalau hasilnya hanya 1 lapis, disalin jadi 3 lapis supaya bisa digabung dengan frame asli.
        if out.ndim == 2:
            out = np.stack([out] * 3, axis=-1)
        h, w = rgb.shape[:2]
        # Kalau ukuran hasil beda dengan frame asli, ukurannya disamakan supaya bisa ditumpuk.
        if out.shape[:2] != (h, w):
            out = cv2.resize(out, (w, h))
        self.before, self.after = rgb, out

        top = np.hstack([_label_frame(rgb, "Asli"), _label_frame(out, "Hasil")])

        # histogram dihitung tiap 3 frame agar stream tetap lancar
        # Catatan: kode di bawah memakai angka 6, jadi histogram diperbarui tiap 6 frame.
        if self.panels is None or self.panels.shape[1] != 2 * w or self.n % 6 == 0:
            self.panels = np.hstack([
                _panel(rgb, w, 220, "Histogram asli"),
                _panel(out, w, 220, "Histogram hasil"),
            ])
        self.n += 1

        # Susunan akhir: bagian atas berisi frame asli dan hasil, bagian bawah berisi histogram.
        # Ukurannya dibuat genap supaya bisa diproses sebagai video.
        combined = np.vstack([top, self.panels])
        ch, cw = combined.shape[:2]
        combined = np.ascontiguousarray(combined[:ch - ch % 2, :cw - cw % 2])
        return av.VideoFrame.from_ndarray(combined, format="rgb24")


# Bagian kamera sengaja diletakkan paling akhir karena mode langsung memakai loop.
with tab_cam:
    mode = st.radio("Mode kamera", ["Ambil foto", "Langsung (real-time)"], horizontal=True)

    if mode == "Ambil foto":
        # Mode foto: ambil satu foto dari kamera, lalu diolah seperti di tab Gambar.
        shot = st.camera_input("Ambil foto dari kamera")
        if shot is not None:
            arr = load_image(shot)
            out = apply_pipeline(arr, steps)
            c1, c2 = st.columns(2)
            c1.image(arr, caption="Asli", width="stretch")
            c2.image(out, caption=f"Hasil: {label}", width="stretch")
            show_analysis(arr, out)
    else:
        st.caption("Kamera hanya dapat dibuka melalui localhost atau HTTPS. Tutup aplikasi lain "
                   "yang sedang memakai kamera, misalnya Zoom atau Google Meet.")
        # Mode langsung: webrtc_streamer menjalankan LiveProcessor untuk setiap frame kamera.
        # Server STUN Google membantu menyambungkan browser dengan aplikasi.
        ctx = webrtc_streamer(
            key="live",
            video_processor_factory=LiveProcessor,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
        # Setiap halaman dijalankan ulang, daftar operasi terbaru dikirim ke pengolah kamera.
        if ctx.video_processor:
            ctx.video_processor.steps = steps