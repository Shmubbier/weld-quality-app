# Weld Quality & Defect Classifier

Deep-learning image segmentation that classifies weld quality into **bad-weld**,
**defect**, and **good-weld** regions. Upload a weld image and the model outlines
each region with a confidence score and mask area.

**Model:** YOLO26 medium segmentation (`yolo26m-seg`, imgsz 1024), trained on the
merged 4-member weld dataset.

> Live app: _add your Streamlit Cloud URL here after deploying_

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Self-check the model loads correctly (no Streamlit needed):

```bash
python app.py        # prints "demo OK: {0: 'bad-weld', 1: 'defect', 2: 'good-weld'}"
```

---

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo (see below).
2. Go to <https://share.streamlit.io> → sign in with GitHub → **New app**.
3. Pick the repo, branch `main`, main file `app.py` → **Deploy**.

That's it — Streamlit installs `requirements.txt` and serves `app.py`. Every push
to `main` auto-redeploys.

---

## Iterate after the deadline (ship a better model)

The model is **one file** (`best.pt`) plus `model_version.json` for its display
name/date. When a future training run scores better:

**If the new `best.pt` is ≤ 100 MB** (GitHub's per-file limit — a `yolo26m` model
is ~55 MB, so this is the normal case):

```bash
# 1. Strip the new weights to shrink them (one-time, run in your training env):
python -c "from ultralytics.utils.torch_utils import strip_optimizer; strip_optimizer('best.pt')"

# 2. Replace the file and update its label:
cp /path/to/new/best.pt best.pt
#    edit model_version.json -> bump "name" and "date"

# 3. Push:
git add best.pt model_version.json && git commit -m "Update model" && git push
```

Streamlit redeploys in ~1 minute and serves the new model. The app keys its cache
on `name`+`date`, so the new weights take effect automatically — no code change.

**If the new `best.pt` is > 100 MB** (e.g. a larger `yolo26l`): don't commit it.
Upload it as a GitHub **Release asset** instead, then set its download URL in
`model_version.json`:

```json
{ "weights": "best.pt", "url": "https://github.com/<you>/<repo>/releases/download/v2/best.pt", ... }
```

Leave `best.pt` out of the repo; the app downloads it from `url` on first load.

---

## Files

| File | Purpose |
|------|---------|
| `app.py` | The whole app: upload → inference → masks + detection table + download |
| `best.pt` | Trained YOLO26-seg weights (the model) |
| `model_version.json` | Model display name/date + optional remote `url` (the swap knob) |
| `requirements.txt` | Python deps for Streamlit Cloud |
| `sample_images/` | Demo welds so reviewers can one-click run |
