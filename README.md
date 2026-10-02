# Weld Quality and Defect Classifier

Deep-learning image segmentation that classifies weld quality into bad-weld, defect,
and good-weld regions. Upload a weld image, pick a trained model (or compare two side
by side), and the app outlines each region with a confidence score and mask area.

Styling follows `DESIGN.md` (dark editorial theme, teal and orange accents).

> Live app: add your Streamlit Cloud URL here after deploying.

## Models included

Weights are hosted as GitHub Release assets and downloaded by the app on first use.
They are listed in `model_version.json`:

| Model | Arch | Image size | Epochs | Box mAP50 |
|-------|------|-----------|--------|-----------|
| YOLO26s @640 (default) | YOLO26 | 640 | 180 | 0.68 |
| YOLO26m @1024 | YOLO26 | 1024 | 39 | 0.48 |
| YOLOv8m @640 | YOLOv8 | 640 | 300 | 0.74 |

The sidebar "Compare with a second model" option runs two models on the same image so
you can see the difference in detections and overlays.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Self-check that the default model loads (no Streamlit needed):

```bash
python app.py
```

It prints `demo OK: yolo26s_640 {0: 'bad-weld', 1: 'defect', 2: 'good-weld'}`.

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io, sign in with GitHub, choose New app.
3. Pick the repo, branch `main`, main file `app.py`, then Deploy.

Streamlit installs `requirements.txt` and serves `app.py`. Every push to `main`
redeploys automatically. First load of each model downloads its weights once.

## Iterate after the deadline (ship a better model)

Each model is one weights file plus one entry in `model_version.json`. To add or
replace a model when a training run scores better:

1. Strip the new weights to shrink them (run in your training environment):
   ```bash
   python -c "from ultralytics.utils.torch_utils import strip_optimizer; strip_optimizer('best.pt')"
   ```
2. Upload the file as a Release asset, for example a new tag `v2`:
   ```bash
   gh release create v2 best.pt
   ```
   If git or the release upload is blocked on your network, upload the asset through
   the GitHub web UI (Releases, Draft a new release, attach the file) instead.
3. Add or edit an entry in `model_version.json` with the new `url`, `name`, `date`,
   and metrics. Set `"default": true` on the one you want selected first.
4. Commit and push `model_version.json`.

The app keys its model cache on each model id, so the new model appears in the
selector on the next redeploy with no code change.

## Files

| File | Purpose |
|------|---------|
| `app.py` | The app: upload or sample, run or compare models, show masks, table, download |
| `model_version.json` | The list of models (name, metrics, download url). The selector and the swap knob |
| `DESIGN.md` | The visual design spec the theme follows |
| `requirements.txt` | Python dependencies for Streamlit Cloud |
| `sample_images/` | Demo welds so reviewers can run the app in one click |
