"""Streamlit demo: upload a face image, get the ensemble's real/fake prediction and Grad-CAM maps.

    streamlit run client.py

Uses the same models, weights and preprocessing as inference_script.py.
"""
import os

import matplotlib
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from inference_script import DEVICE, get_inference_transforms, load_models

HERE = os.path.dirname(os.path.abspath(__file__))  # relative weight paths are resolved against this folder
DEFAULT_WEIGHTS = "./model_weights.pt" if os.path.exists(os.path.join(HERE, "model_weights.pt")) else "./models"
MODEL_NAMES = {"efficientnet_b4": "EfficientNet-B4", "convnext_tiny": "ConvNeXt-Tiny", "swin_t": "Swin-Tiny"}
MODEL_KIND = {"efficientnet_b4": "CNN", "convnext_tiny": "CNN", "swin_t": "Transformer"}

FAKE_COLOR = "#e5484d"
REAL_COLOR = "#30a46c"

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1180px;}
.hero h1 {font-size: 2.4rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;
          background: linear-gradient(90deg, #6e56cf, #e5484d); -webkit-background-clip: text;
          background-clip: text; color: transparent;}
.hero p {opacity: 0.7; margin: 0.35rem 0 1.4rem 0; font-size: 1.02rem;}
.chip {display: inline-block; padding: 0.18rem 0.65rem; margin: 0 0.35rem 0.35rem 0; border-radius: 999px;
       font-size: 0.78rem; font-weight: 600; border: 1px solid rgba(128,128,128,0.35);}
.card {border: 1px solid rgba(128,128,128,0.25); border-radius: 16px; padding: 1.2rem 1.35rem;
       background: rgba(128,128,128,0.06); margin-bottom: 1rem;}
.verdict {border-radius: 16px; padding: 1.3rem 1.4rem; margin-bottom: 1rem; color: white;}
.verdict .label {font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.12em; opacity: 0.85;}
.verdict .value {font-size: 2.6rem; font-weight: 800; line-height: 1.1;}
.verdict .sub {font-size: 0.95rem; opacity: 0.92; margin-top: 0.25rem;}
.pcard {border: 1px solid rgba(128,128,128,0.25); border-radius: 14px; padding: 0.75rem 1rem 0.6rem;
        background: rgba(128,128,128,0.06); margin-bottom: 1rem;}
.pcard .head {font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.65; margin-bottom: 0.2rem;}
.bar {position: relative; height: 7px; border-radius: 999px; background: rgba(128,128,128,0.22); margin-top: 0.25rem;}
.bar .fill {height: 100%; border-radius: 999px;}
.bar .tick {position: absolute; top: -3px; width: 2px; height: 13px; background: currentColor; opacity: 0.5;}
.row {margin: 0.45rem 0;}
.row .top {display: flex; justify-content: space-between; align-items: baseline; gap: 0.5rem; font-size: 0.86rem;}
.row .name {font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.row .name small {font-weight: 400; opacity: 0.6; font-size: 0.72rem; margin-left: 0.3rem;}
.row .val {font-variant-numeric: tabular-nums; font-weight: 600; white-space: nowrap;}
.row.main .top {font-size: 0.95rem;}
.row.main .bar {height: 9px;}
.section {font-size: 1.15rem; font-weight: 700; margin: 0.4rem 0 0.2rem 0;}
.muted {opacity: 0.65; font-size: 0.88rem;}
.empty {text-align: center; padding: 2.6rem 1rem;}
.empty .big {font-size: 1.25rem; font-weight: 700; margin-bottom: 0.4rem;}
[data-testid="stImageCaption"] {text-align: center;}
</style>
"""


@st.cache_resource(show_spinner="Loading models...")
def get_models(weights_path):
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(HERE, weights_path)
    return load_models(weights_path)


def grad_cam(model, arch, x, sign):
    """Grad-CAM on the last feature stage. sign=+1 explains the 'fake' logit, -1 explains 'real'."""
    store = {}
    layer = model.backbone.features[-1]
    hook = layer.register_forward_hook(lambda module, inputs, output: store.__setitem__("act", output))
    try:
        with torch.enable_grad():
            logit = model(x).float().squeeze()
            grad = torch.autograd.grad(sign * logit, store["act"])[0]
    finally:
        hook.remove()
    act = store["act"]
    if arch == "swin_t":  # Swin stages output [B, H, W, C]
        act, grad = act.permute(0, 3, 1, 2), grad.permute(0, 3, 1, 2)
    weights = grad.float().mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act.float()).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / (cam.max() + 1e-8)).detach().cpu().numpy(), torch.sigmoid(logit).item()


def overlay(image, cam, alpha=0.45):
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return np.clip((1 - alpha) * np.asarray(image, dtype=np.float32) / 255 + alpha * heat, 0, 1)


def bar(p, threshold):
    color = FAKE_COLOR if p >= threshold else REAL_COLOR
    return (f'<div class="bar"><div class="fill" style="width:{p * 100:.1f}%;background:{color};"></div>'
            f'<div class="tick" style="left:calc({threshold * 100:.1f}% - 1px);"></div></div>')


# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Deepfake Detector", page_icon="🛡️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="hero"><h1>Deepfake Detector</h1>'
    "<p>Upload a face image to check whether it is real or manipulated, and see where the models look.</p></div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Settings")
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01,
                          help="Images with ensemble P(fake) at or above this are labelled FAKE.")
    cam_target = st.radio("Grad-CAM explains", ["Predicted class", "Fake", "Real"], horizontal=True)
    cam_alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.45, 0.05)
    with st.expander("Model weights"):
        weights_path = st.text_input("File or folder", DEFAULT_WEIGHTS, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Ensemble")
    st.markdown("".join(f'<span class="chip">{MODEL_NAMES[a]} · {MODEL_KIND[a]}</span>' for a in MODEL_NAMES),
                unsafe_allow_html=True)
    st.markdown('<div class="muted">ImageNet-pretrained backbones, LoRA fine-tuned. '
                f"Validation AUC 0.909. Running on <b>{DEVICE.type.upper()}</b>.</div>", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
if uploaded is None:
    st.markdown(
        '<div class="card empty"><div class="big">Drop a face image above</div>'
        '<div class="muted">JPG or PNG. Best results on a face-centred crop, like the 224×224 training images.</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()

try:
    models = get_models(weights_path)
except Exception as err:
    st.error(f"Could not load models from {weights_path}: {err}")
    st.stop()

image = Image.open(uploaded).convert("RGB")
shown = image.resize((224, 224))
x = get_inference_transforms()(image).unsqueeze(0).to(DEVICE)

with st.spinner("Analysing image..."):
    with torch.no_grad():
        probs = {arch: torch.sigmoid(m(x).float().squeeze()).item() for arch, m in models.items()}
confidence = float(np.mean(list(probs.values())))
is_fake = confidence >= threshold
verdict, color = ("FAKE", FAKE_COLOR) if is_fake else ("REAL", REAL_COLOR)
certainty = confidence if is_fake else 1 - confidence
agree = sum((p >= threshold) == is_fake for p in probs.values())

col_img, col_res = st.columns([1, 1.35], gap="large")
with col_img:
    st.image(image, caption=f"{uploaded.name} · {image.width}×{image.height}", width="stretch")
with col_res:
    st.markdown(
        f'<div class="verdict" style="background:linear-gradient(135deg,{color},{color}cc);">'
        f'<div class="label">Verdict</div><div class="value">{verdict}</div>'
        f'<div class="sub">{certainty:.1%} confident · {agree} of {len(probs)} models agree</div></div>',
        unsafe_allow_html=True,
    )
    def row(name, kind, p, main=False):
        return (f'<div class="row{" main" if main else ""}"><div class="top">'
                f'<span class="name">{name}<small>{kind}</small></span><span class="val">{p:.1%}</span></div>'
                f'{bar(p, threshold)}</div>')

    rows = "".join(row(MODEL_NAMES.get(a, a), MODEL_KIND.get(a, ""), p) for a, p in probs.items())
    st.markdown(
        f'<div class="pcard"><div class="head">P(fake) · threshold {threshold:.2f}</div>'
        f'{row("Ensemble", "mean of 3", confidence, main=True)}'
        f'<hr style="margin:0.5rem 0;opacity:0.2;">{rows}</div>',
        unsafe_allow_html=True,
    )

if cam_target == "Predicted class":
    sign, target = (1, "fake") if is_fake else (-1, "real")
else:
    sign, target = (1, "fake") if cam_target == "Fake" else (-1, "real")

st.markdown(f'<div class="section">Where the models look</div>'
            f'<div class="muted">Grad-CAM: warmer regions push each model towards <b>{target}</b>.</div>',
            unsafe_allow_html=True)
with st.spinner("Computing Grad-CAM..."):
    cams = {arch: grad_cam(m, arch, x, sign)[0] for arch, m in models.items()}
cols = st.columns(len(cams) + 1, gap="small")
for col, (arch, cam) in zip(cols, cams.items()):
    col.image(overlay(shown, cam, cam_alpha), caption=f"{MODEL_NAMES.get(arch, arch)} · {probs[arch]:.0%}", width="stretch")
cols[-1].image(overlay(shown, np.mean(list(cams.values()), axis=0), cam_alpha), caption="Ensemble average", width="stretch")

with st.expander("How to read this"):
    st.markdown(
        "- **P(fake)** is each model's probability that the image is manipulated; the verdict uses the mean of the three.\n"
        "- **Grad-CAM** highlights the image regions that most increased the chosen class's score in each model's last feature layer. "
        "It shows where a model looked, not proof of manipulation.\n"
        "- The models were trained on 224×224 face crops with heavy noise, blur and occlusions; very different images "
        "(full scenes, drawings, screenshots) are outside what they have seen."
    )
