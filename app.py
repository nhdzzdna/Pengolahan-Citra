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

import cnn
from operations import GROUPS, PARAMS, apply_pipeline, limit_size, steps_label
from histogram import (plot_histograms, plot_from_counts, counts, describe,
                       stats_table, render_live)

st.set_page_config(page_title="ImageLab", layout="wide")
st.title("ImageLab")
st.caption("Aplikasi pengolahan citra digital dengan histogram, video, kamera, dan analisis CNN")

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
    st.header("Operasi")
    group = st.selectbox("Kelompok", list(GROUPS.keys()))
    op = st.selectbox("Operasi", GROUPS[group])

    params = {}
    specs = PARAMS.get(op, [])
    for key, label, lo, hi, default, step in specs:
        params[key] = st.slider(label, lo, hi, default, step, key=f"{op}_{key}")
    if not specs:
        st.caption("Operasi ini tidak memiliki parameter.")

    st.checkbox("Terapkan operasi yang sedang dipilih", key="apply_current")
    c1, c2 = st.columns(2)
    c1.button("Tambah ke pipeline", on_click=add_step, args=(op, params))
    c2.button("Reset pipeline", on_click=reset_pipeline)

    st.subheader("Pipeline")
    if not st.session_state.pipeline:
        st.caption("Pipeline masih kosong.")
    for i, s in enumerate(st.session_state.pipeline):
        a, b = st.columns([4, 1])
        a.write(f"{i + 1}. {s['op']}")
        b.button("Hapus", key=f"del{i}", on_click=remove_step, args=(i,))

    st.divider()
    st.header("Histogram")
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


tab_img, tab_vid, tab_cnn, tab_cam = st.tabs(["Gambar", "Video", "CNN", "Kamera"])
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

# ---------------------------------------------------------------- tab CNN
with tab_cnn:
    st.write("Bagian ini menjalankan klasifikasi MobileNetV2 (ImageNet) pada gambar dari tab Gambar, "
             "sebelum dan sesudah pengolahan, untuk melihat apakah perubahan distribusi piksel "
             "memengaruhi prediksi.")
    if original is None:
        st.info("Unggah gambar pada tab Gambar terlebih dahulu.")
    else:
        if st.button("Bandingkan klasifikasi CNN: asli dan hasil"):
            try:
                pa, pb = cnn.classify(original), cnn.classify(processed)
                c1, c2 = st.columns(2)
                c1.markdown("**Prediksi gambar asli**")
                c2.markdown(f"**Prediksi gambar hasil ({label})**")
                for (la, sa), (lb, sb) in zip(pa, pb):
                    c1.write(f"{la}: {sa * 100:.2f}%")
                    c2.write(f"{lb}: {sb * 100:.2f}%")
                st.subheader("Histogram gambar yang masuk ke CNN")
                show_analysis(original, processed)
            except ImportError:
                st.error("TensorFlow belum terpasang. Jalankan: python -m pip install tensorflow")
            except Exception as exc:
                st.error(f"CNN belum berhasil dijalankan. Periksa koneksi internet untuk unduhan "
                         f"bobot model. Detail: {exc}")

        if st.button("Tampilkan Grad-CAM"):
            try:
                c1, c2 = st.columns(2)
                c1.image(cnn.gradcam(original), caption="Grad-CAM gambar asli", width="stretch")
                c2.image(cnn.gradcam(processed), caption=f"Grad-CAM gambar hasil ({label})",
                         width="stretch")
                st.caption("Area merah dan kuning adalah bagian gambar yang paling memengaruhi "
                           "prediksi kelas teratas.")
            except ImportError:
                st.error("TensorFlow belum terpasang. Jalankan: python -m pip install tensorflow")
            except Exception as exc:
                st.error(f"Grad-CAM belum berhasil dijalankan. Detail: {exc}")

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