"""Weld Quality and Defect Classifier (Streamlit).

Upload a weld image or pick a sample, choose a trained model (or compare two), and the
app segments and outlines bad-weld, defect, and good-weld regions.

Styling follows DESIGN.md (dark editorial, teal and orange accents).
Iterate after the deadline: add or edit entries in model_version.json (name, date, url).
See README.md.
"""
import io
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

HERE = Path(__file__).parent
CFG = json.loads((HERE / "model_version.json").read_text())

# Backward compatible: accept either a single-model json or a {"models": [...]} list.
if "models" in CFG:
    MODELS = CFG["models"]
else:
    MODELS = [CFG]
MODEL_BY_ID = {m["id"]: m for m in MODELS} if all("id" in m for m in MODELS) else {
    m.get("name", str(i)): {**m, "id": m.get("name", str(i))} for i, m in enumerate(MODELS)
}
MODELS = list(MODEL_BY_ID.values())
DEFAULT_ID = next((m["id"] for m in MODELS if m.get("default")), MODELS[0]["id"])

CLASS_COLORS = {"bad-weld": "#F87171", "defect": "#FB923C", "good-weld": "#5EEAD4"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400..700&family=Instrument+Serif:ital@1&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --bg:#0a0b0e; --surface-1:#111217; --surface-2:#171921; --surface-3:#1f222d; --surface-hover:#242836;
  --border:rgba(67,70,81,.5); --border-strong:rgba(67,70,81,.9); --border-accent:rgba(94,234,212,.3);
  --text-1:#ebecef; --text-2:#c6c9d2; --text-3:#8d909c; --text-4:#60636f;
  --accent-cool:#5EEAD4; --accent-cool-hover:#8CF5E3; --accent-cool-soft:rgba(94,234,212,.12);
  --accent-warm:#FB923C; --gradient-key:linear-gradient(135deg,#5EEAD4 0%,#FB923C 100%);
  --font-ui:'DM Sans',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  --font-serif:'Instrument Serif',Georgia,serif;
  --font-mono:'JetBrains Mono',ui-monospace,Menlo,monospace;
}
.stApp,body{background:var(--bg);color:var(--text-2);font-family:var(--font-ui);}
[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:1100px;padding-top:2.2rem;animation:fadeUp .4s ease both;}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
h1,h2,h3{font-family:var(--font-ui);color:var(--text-1);letter-spacing:-0.015em;}
h2{font-size:22px;font-weight:600;} h3{font-size:18px;font-weight:600;}
.app-title{font-size:clamp(30px,4vw,46px);font-weight:700;line-height:1.08;letter-spacing:-0.02em;color:var(--text-1);margin:0;}
.app-title .key{background:var(--gradient-key);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;}
.app-title .ser{font-family:var(--font-serif);font-style:italic;font-weight:400;}
.subtitle{color:var(--text-3);font-size:15px;margin:.4rem 0 0;}
[data-testid="stSidebar"]{background:var(--surface-1);border-right:1px solid var(--border);}
[data-testid="stSidebar"] .eyebrow{font-size:12px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--text-4);}
.stButton>button,.stDownloadButton>button{
  background:var(--accent-cool);color:var(--bg);border:1px solid var(--accent-cool);
  border-radius:9999px;font-weight:600;font-family:var(--font-ui);padding:.5rem 1.1rem;
  transition:transform .18s cubic-bezier(.2,0,0,1),background .12s,box-shadow .18s;}
.stButton>button:hover,.stDownloadButton>button:hover{background:var(--accent-cool-hover);transform:translateY(-1px);}
.stButton>button:focus,.stDownloadButton>button:focus{box-shadow:0 0 0 3px var(--accent-cool-soft);outline:none;}
.stButton>button:active,.stDownloadButton>button:active{transform:none;}
.stSlider [data-baseweb="slider"] div[role="slider"]{background:var(--accent-cool);}
.stTabs [data-baseweb="tab-list"]{gap:1.2rem;border-bottom:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{color:var(--text-3);font-weight:500;}
.stTabs [aria-selected="true"]{color:var(--text-1);}
[data-testid="stMetricValue"]{font-family:var(--font-mono);color:var(--text-1);}
.panel{background:var(--surface-1);border:1px solid var(--border);border-radius:14px;padding:14px;}
.chip{display:inline-flex;align-items:center;gap:7px;background:var(--surface-2);border:1px solid var(--border);
  border-radius:9999px;padding:4px 11px;font-size:13px;color:var(--text-2);margin:2px 6px 2px 0;}
.dot{width:11px;height:11px;border-radius:3px;display:inline-block;}
table.det{width:100%;border-collapse:collapse;font-size:14px;}
table.det th{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--text-4);
  text-align:left;padding:6px 10px;border-bottom:1px solid var(--border);}
table.det td{padding:7px 10px;border-bottom:1px solid var(--border);color:var(--text-2);}
table.det td.num{font-family:var(--font-mono);color:var(--text-1);}
.clslabel{font-weight:600;}
@media (prefers-reduced-motion: reduce){.block-container{animation:none;}
  .stButton>button,.stDownloadButton>button{transition:background .12s;}}
</style>
"""


@st.cache_resource(show_spinner=False)
def load_model(model_id: str, url: str, weights: str):
    path = HERE / weights
    if not path.exists():
        if not url:
            st.error(f"Model file '{weights}' not found and no url given in model_version.json.")
            st.stop()
        with st.spinner(f"Downloading {model_id} ..."):
            try:
                urlretrieve(url, path)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not download model from {url}: {e}")
                st.stop()
    return YOLO(str(path))


def run(model, pil_img, conf, iou):
    arr = np.array(pil_img.convert("RGB"))
    return model.predict(arr, conf=conf, iou=iou, verbose=False)[0]


def annotated_image(result):
    bgr = result.plot()
    return Image.fromarray(bgr[:, :, ::-1])


def detections(result, names):
    """List of (class, confidence, mask_area_pct)."""
    out = []
    if result.boxes is None:
        return out
    areas = None
    if result.masks is not None:
        d = result.masks.data
        hw = d.shape[1] * d.shape[2]
        areas = [float(d[i].sum()) / hw * 100.0 for i in range(d.shape[0])]
    for i in range(len(result.boxes)):
        cls = names[int(result.boxes.cls[i])]
        conf = float(result.boxes.conf[i])
        area = None if areas is None else areas[i]
        out.append((cls, conf, area))
    return out


def det_table_html(dets):
    if not dets:
        return "<p style='color:var(--text-3)'>No regions detected above the confidence threshold.</p>"
    rows = []
    for cls, conf, area in dets:
        color = CLASS_COLORS.get(cls, "#8d909c")
        area_txt = "n/a" if area is None else f"{area:.2f}"
        rows.append(
            f"<tr><td><span class='clslabel' style='color:{color}'>{cls}</span></td>"
            f"<td class='num'>{conf:.3f}</td><td class='num'>{area_txt}</td></tr>"
        )
    return ("<table class='det'><tr><th>Class</th><th>Confidence</th><th>Mask area %</th></tr>"
            + "".join(rows) + "</table>")


def summary_chips(dets):
    counts = {}
    for cls, _, _ in dets:
        counts[cls] = counts.get(cls, 0) + 1
    if not counts:
        return ""
    chips = []
    for cls in sorted(counts):
        color = CLASS_COLORS.get(cls, "#8d909c")
        chips.append(f"<span class='chip'><span class='dot' style='background:{color}'></span>"
                     f"{cls}: {counts[cls]}</span>")
    return "".join(chips)


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Weld Quality Classifier", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    "<h1 class='app-title'>Weld Quality and <span class='key'>Defect</span> Classifier</h1>"
    "<p class='subtitle'>Deep-learning image segmentation that classifies weld quality into "
    "bad-weld, <span class='ser'>defect</span>, and good-weld regions.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("<p class='eyebrow'>Model</p>", unsafe_allow_html=True)

    def label_for(mid):
        m = MODEL_BY_ID[mid]
        score = m.get("box_map50")
        tail = f"  (box mAP50 {score:.2f})" if isinstance(score, (int, float)) else ""
        return f"{m.get('name', mid)}{tail}"

    ids = list(MODEL_BY_ID.keys())
    primary_id = st.selectbox("Active model", ids, index=ids.index(DEFAULT_ID), format_func=label_for)
    compare = st.checkbox("Compare with a second model", value=False)
    second_id = None
    if compare and len(ids) > 1:
        rest = [i for i in ids if i != primary_id]
        second_id = st.selectbox("Second model", rest, format_func=label_for)

    st.divider()
    st.markdown("<p class='eyebrow'>Settings</p>", unsafe_allow_html=True)
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                     help="Higher values keep fewer, surer detections (raises precision).")
    iou = st.slider("Overlap (IoU) threshold", 0.10, 0.90, 0.45, 0.05)

    st.divider()
    st.markdown("<p class='eyebrow'>Class key</p>", unsafe_allow_html=True)
    key_html = "".join(
        f"<span class='chip'><span class='dot' style='background:{c}'></span>{n}</span>"
        for n, c in CLASS_COLORS.items()
    )
    st.markdown(key_html, unsafe_allow_html=True)
    st.caption("Overlay colors come from the model; this key labels the table and summary.")

# input selection
sample_dir = HERE / "sample_images"
samples = sorted(sample_dir.glob("*")) if sample_dir.exists() else []
tab_upload, tab_sample = st.tabs(["Upload image", "Try a sample"])
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
    st.info("Upload a weld image or pick a sample to run the model.")
    st.stop()

chosen = [primary_id] + ([second_id] if (compare and second_id) else [])


def render_model(mid, container):
    m = MODEL_BY_ID[mid]
    model = load_model(mid, m.get("url", ""), m.get("weights", "best.pt"))
    result = run(model, pil_img, conf, iou)
    dets = detections(result, model.names)
    with container:
        st.markdown(f"<h3>{m.get('name', mid)}</h3>", unsafe_allow_html=True)
        meta = []
        if m.get("arch"):
            meta.append(m["arch"])
        if isinstance(m.get("box_map50"), (int, float)):
            meta.append(f"box mAP50 {m['box_map50']:.2f}")
        if m.get("epochs"):
            meta.append(f"{m['epochs']} epochs")
        if meta:
            st.caption("  |  ".join(meta))
        st.image(annotated_image(result), use_container_width=True)
        chips = summary_chips(dets)
        if chips:
            st.markdown(chips, unsafe_allow_html=True)
        st.markdown(det_table_html(dets), unsafe_allow_html=True)
    return result


st.markdown("<h2>Original</h2>", unsafe_allow_html=True)
st.image(pil_img, use_container_width=True)

st.markdown("<h2>Detections</h2>", unsafe_allow_html=True)
cols = st.columns(len(chosen))
results_by_id = {}
for mid, col in zip(chosen, cols):
    results_by_id[mid] = render_model(mid, col)

if len(chosen) > 1:
    a, b = MODEL_BY_ID[chosen[0]], MODEL_BY_ID[chosen[1]]
    if isinstance(a.get("box_map50"), (int, float)) and isinstance(b.get("box_map50"), (int, float)):
        diff = (a["box_map50"] - b["box_map50"]) * 100
        st.caption(f"Reported validation gap: {a.get('name')} vs {b.get('name')} "
                   f"is {diff:+.1f} box mAP50 points.")

# download the primary model's annotated image (reuse the result already computed)
buf = io.BytesIO()
annotated_image(results_by_id[primary_id]).save(buf, format="PNG")
st.download_button("Download annotated image", buf.getvalue(),
                   file_name=f"annotated_{src_name or 'weld'}.png", mime="image/png")


def demo():
    """Self-check: the default model loads and exposes the three weld classes."""
    m = MODEL_BY_ID[DEFAULT_ID]
    y = YOLO(str(HERE / m.get("weights", "best.pt")))
    assert set(y.names.values()) == {"bad-weld", "defect", "good-weld"}, y.names
    assert y.task == "segment", y.task
    print("demo OK:", m["id"], y.names)


if __name__ == "__main__":
    demo()
