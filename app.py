import io
import time
import tempfile

import av
import cv2
import imageio
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from operations import PARAMS, apply_pipeline, limit_size, steps_label
from histogram import (plot_histograms, plot_from_counts, counts, describe,
                       stats_table, render_live)

st.set_page_config(page_title="ImageLab", page_icon="🖼️", layout="wide")

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
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="hero"><h1>ImageLab</h1>'
    '<p>Aplikasi pengolahan citra digital dengan histogram, video, kamera, dan analisis CNN</p></div>',
    unsafe_allow_html=True,
)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = []
if "apply_current" not in st.session_state:
    st.session_state.apply_current = True


def add_step(op, params):
    st.session_state.pipeline.append({"op": op, "params": dict(params)})
    st.session_state.apply_current = False


def remove_step(i):
    st.session_state.pipeline.pop(i)


def reset_pipeline():
    st.session_state.pipeline = []


def load_image(file):
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    return np.asarray(img)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Pilih operasi")
    # group = st.selectbox("Kelompok", list(GROUPS.keys()))
    op = st.selectbox("Operasi", list(PARAMS.keys()))

    st.header("2. Atur parameter")
    params = {}
    specs = PARAMS.get(op, [])
    for key, label, lo, hi, default, step in specs:
        params[key] = st.slider(label, lo, hi, default, step, key=f"{op}_{key}")
    if not specs:
        st.caption("Operasi ini tidak memiliki parameter.")

    st.checkbox("Terapkan operasi yang sedang dipilih", key="apply_current")
    c1, c2 = st.columns(2)
    c1.button("Tambah", on_click=add_step, args=(op, params),
              type="primary", width="stretch")
    c2.button("Reset", on_click=reset_pipeline, width="stretch")

    st.header("3. Pipeline")
    if not st.session_state.pipeline:
        st.caption("Pipeline masih kosong. Klik Tambah untuk menyusun urutan operasi.")
    for i, s in enumerate(st.session_state.pipeline):
        a, b = st.columns([4, 1])
        a.write(f"{i + 1}. {s['op']}")
        b.button("✕", key=f"del{i}", on_click=remove_step, args=(i,))

    st.divider()
    with st.expander("Pengaturan histogram"):
        hist_mode = st.radio("Mode", ["RGB", "Grayscale"], horizontal=True)
        hist_log = st.checkbox("Skala logaritmik", value=False)
        hist_cdf = st.checkbox("Tampilkan CDF", value=False)

steps = list(st.session_state.pipeline)
if st.session_state.apply_current:
    steps.append({"op": op, "params": dict(params)})
label = steps_label(steps)
marker = None
if steps and steps[-1]["op"] == "Thresholding":
    marker = steps[-1]["params"].get("threshold")


def show_analysis(before, after):
    """Histogram, keterangan otomatis, dan tabel statistik untuk satu pasang gambar."""
    st.pyplot(plot_histograms(before, after, hist_mode, hist_log, hist_cdf, marker), clear_figure=True)
    st.info(describe(before, after))
    st.dataframe(stats_table(before, after), hide_index=True, width="stretch")


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
                buf = io.BytesIO()
                Image.fromarray(result).save(buf, format="PNG")
                st.download_button("Unduh PNG", buf.getvalue(), "imagelab_hasil.png", "image/png")

# ---------------------------------------------------------------- tab video
def process_video(path, steps, progress, step=5, max_w=640):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                pixelformat="yuv420p", macro_block_size=1)
    cb, ca = np.zeros((4, 256)), np.zeros((4, 256))
    mean_b, mean_a, idxs = [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = limit_size(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), max_w)
        h, w = rgb.shape[:2]
        rgb = np.ascontiguousarray(rgb[:h - h % 2, :w - w % 2])
        out = apply_pipeline(rgb, steps)
        writer.append_data(out)
        if i % step == 0:
            cb += counts(rgb)
            ca += counts(out)
            mean_b.append(float(rgb.mean()))
            mean_a.append(float(out.mean()))
            idxs.append(i)
        i += 1
        progress.progress(min(i / total, 1.0))
    cap.release()
    writer.close()
    return out_path, cb, ca, idxs, mean_b, mean_a


with tab_vid:
    vf = st.file_uploader("Unggah video", type=["mp4", "avi", "mov", "mkv"], key="vid")
    if vf is None:
        st.info("Unggah video untuk melihat pratinjau per frame beserta histogramnya.")
    else:
        if st.session_state.get("video_name") != vf.name:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(vf.read())
            tmp.close()
            st.session_state["video_path"] = tmp.name
            st.session_state["video_name"] = vf.name
        vpath = st.session_state["video_path"]

        cap = cv2.VideoCapture(vpath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        st.caption(f"Jumlah frame: {total_frames}, FPS: {fps:.1f}")
        idx = st.slider("Pilih frame untuk pratinjau", 0, max(total_frames - 1, 0), 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()

        if ok:
            rgb = limit_size(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            out = apply_pipeline(rgb, steps)
            c1, c2 = st.columns(2)
            c1.image(rgb, caption="Frame asli", width="stretch")
            c2.image(out, caption=f"Frame hasil: {label}", width="stretch")
            show_analysis(rgb, out)
        else:
            st.error("Frame tidak dapat dibaca. Coba format video lain.")

        st.divider()
        st.caption("Pemrosesan penuh memperkecil video ke lebar maksimum 640 piksel agar tidak terlalu lama.")
        if st.button("Proses seluruh video"):
            bar = st.progress(0.0)
            result_path, cb, ca, idxs, mean_b, mean_a = process_video(vpath, steps, bar)
            with open(result_path, "rb") as fh:
                data = fh.read()
            st.video(data)
            st.download_button("Unduh video hasil", data, "imagelab_video.mp4", "video/mp4")

            st.subheader("Histogram agregat seluruh video")
            st.pyplot(plot_from_counts(cb, ca, hist_mode, hist_log, hist_cdf, marker),
                      clear_figure=True)

            st.subheader("Rata-rata intensitas per frame")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(idxs, mean_b, "--", label="Video asli")
            ax.plot(idxs, mean_a, label="Video hasil")
            ax.set_xlabel("Nomor frame")
            ax.set_ylabel("Rata-rata intensitas")
            ax.grid(True, alpha=0.2)
            ax.legend()
            st.pyplot(fig, clear_figure=True)

# ---------------------------------------------------------------- tab kamera
class LiveProcessor(VideoProcessorBase):
    def __init__(self):
        self.steps = []
        self.before = None
        self.after = None

    def recv(self, frame):
        rgb = frame.to_ndarray(format="rgb24")
        out = apply_pipeline(rgb, self.steps)
        self.before, self.after = rgb, out
        return av.VideoFrame.from_ndarray(out, format="rgb24")


# Bagian kamera sengaja diletakkan paling akhir karena mode langsung memakai loop.
with tab_cam:
    mode = st.radio("Mode kamera", ["Ambil foto", "Langsung (real-time)"], horizontal=True)

    if mode == "Ambil foto":
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
        ctx = webrtc_streamer(
            key="live",
            video_processor_factory=LiveProcessor,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
        placeholder = st.empty()
        while ctx.state.playing and ctx.video_processor:
            vp = ctx.video_processor
            vp.steps = steps
            if vp.before is not None and vp.after is not None:
                placeholder.image(render_live(vp.before, vp.after), width="stretch")
            time.sleep(0.3)