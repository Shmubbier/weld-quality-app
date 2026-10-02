"""Weld Quality & Defect Classifier - Streamlit app.

Upload a weld image (or pick a sample); a YOLO26 segmentation model detects and
outlines bad-weld / defect / good-weld regions.

Iterate after the deadline: drop a better `best.pt` in this folder (or host it and
set `url` in model_version.json), edit the name/date in model_version.json, push.
See README.md.
"""
import io
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

HERE = Path(__file__).parent
CFG = json.loads((HERE / "model_version.json").read_text())
WEIGHTS = HERE / CFG.get("weights", "best.pt")
CLASS_COLORS = {"bad-weld": "#d62728", "defect": "#ff9a3c", "good-weld": "#2ca02c"}


@st.cache_resource(show_spinner="Loading model...")
def load_model(_cache_key: str):
    # _cache_key (name+date from json) busts the cache when the model is swapped.
    if not WEIGHTS.exists():
        url = CFG.get("url")
        if not url:
            st.error(f"Model file '{WEIGHTS.name}' not found and no 'url' in model_version.json.")
            st.stop()
        with st.spinner(f"Downloading model from {url} ..."):
            try:
                urllib.request.urlretrieve(url, WEIGHTS)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not download model from {url}\n\n{e}")
                st.stop()
    return YOLO(str(WEIGHTS))


def mask_area_pcts(result):
    """Per-instance mask area as % of image, aligned to result.boxes order."""
    if result.masks is None:
        return [None] * len(result.boxes)
    data = result.masks.data  # (N, H, W) in [0,1]
    hw = data.shape[1] * data.shape[2]
    return [float(data[i].sum()) / hw * 100.0 for i in range(data.shape[0])]


def annotate(result):
    """Return the annotated image as an RGB PIL Image."""
    bgr = result.plot()  # ndarray, BGR
    return Image.fromarray(bgr[:, :, ::-1])


def run(model, pil_img, conf, iou):
    arr = np.array(pil_img.convert("RGB"))
    result = model.predict(arr, conf=conf, iou=iou, verbose=False)[0]
    return result


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Weld Quality Classifier", page_icon="🔩", layout="wide")
st.title("🔩 Weld Quality & Defect Classifier")
st.caption(
    "Deep-learning image segmentation that classifies weld quality into "
    "**bad-weld**, **defect**, and **good-weld** regions."
)

model = load_model(f"{CFG.get('name')}|{CFG.get('date')}")

with st.sidebar:
    st.header("Model")
    st.write(f"**{CFG.get('name', WEIGHTS.name)}**")
    st.caption(f"Trained: {CFG.get('date', 'n/a')} · task: {model.task}")
    st.divider()
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                     help="Higher = fewer, surer detections (raises precision).")
    iou = st.slider("IoU (overlap) threshold", 0.10, 0.90, 0.45, 0.05)
    st.divider()
    st.header("Classes")
    for name, color in CLASS_COLORS.items():
        st.markdown(
            f"<span style='display:inline-block;width:12px;height:12px;"
            f"background:{color};border-radius:2px;margin-right:6px'></span>{name}",
            unsafe_allow_html=True,
        )

# pick input: upload or sample
sample_dir = HERE / "sample_images"
samples = sorted(sample_dir.glob("*")) if sample_dir.exists() else []
tab_upload, tab_sample = st.tabs(["📤 Upload image", "🖼️ Try a sample"])

pil_img, src_name = None, None
with tab_upload:
    up = st.file_uploader("Weld image", type=["jpg", "jpeg", "png", "bmp"])
    if up:
        pil_img, src_name = Image.open(up), up.name
with tab_sample:
    if samples:
        choice = st.selectbox("Sample weld image", [p.name for p in samples])
        if choice:
            pil_img, src_name = Image.open(sample_dir / choice), choice
    else:
        st.info("No sample images bundled.")

if pil_img is None:
    st.info("⬆️ Upload a weld image or pick a sample to run the model.")
    st.stop()

result = run(model, pil_img, conf, iou)
annotated = annotate(result)

c1, c2 = st.columns(2)
c1.subheader("Original")
c1.image(pil_img, use_container_width=True)
c2.subheader("Detected")
c2.image(annotated, use_container_width=True)

# detection table
names = model.names
areas = mask_area_pcts(result)
rows = []
if result.boxes is not None:
    for i in range(len(result.boxes)):
        cls = names[int(result.boxes.cls[i])]
        rows.append({
            "class": cls,
            "confidence": round(float(result.boxes.conf[i]), 3),
            "mask area %": None if areas[i] is None else round(areas[i], 2),
        })

st.subheader(f"Detections ({len(rows)})")
if rows:
    st.dataframe(rows, use_container_width=True, hide_index=True)
    counts = {}
    for r in rows:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    st.write("**Summary:** " + " · ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
else:
    st.warning("No regions detected above the confidence threshold. Try lowering it.")

buf = io.BytesIO()
annotated.save(buf, format="PNG")
st.download_button("⬇️ Download annotated image", buf.getvalue(),
                   file_name=f"annotated_{src_name or 'weld'}.png", mime="image/png")


def demo():
    """Self-check: model loads and exposes the 3 weld classes. Run: python app.py"""
    m = YOLO(str(WEIGHTS))
    assert set(m.names.values()) == {"bad-weld", "defect", "good-weld"}, m.names
    assert m.task == "segment", m.task
    print("demo OK:", m.names)


if __name__ == "__main__":
    demo()
