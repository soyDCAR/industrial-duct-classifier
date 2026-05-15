"""
Gradio demo deployed on Hugging Face Spaces.

Model weights are downloaded from GitHub Releases on first run (~25 MB, cached in /tmp/).
Set RELEASE_TAG as a Space secret to pin a specific release; defaults to 'latest'.

Full system (FastAPI + batching + Prometheus):
    https://github.com/soyDCAR/industrial-duct-classifier
"""

import json
import os
import urllib.request
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# model.py is copied into the Space repo root by deploy.sh
from model import MultiEfficientNet, get_transforms

matplotlib.use("Agg")

# ── Model download from GitHub Releases ────────────────────────────────

_REPO = "soyDCAR/industrial-duct-classifier"
_TAG = os.getenv("RELEASE_TAG", "latest")

# GitHub supports /releases/latest/download/<asset> as a permanent redirect
if _TAG == "latest":
    _BASE = f"https://github.com/{_REPO}/releases/latest/download"
else:
    _BASE = f"https://github.com/{_REPO}/releases/download/{_TAG}"

_MODEL_PATH = Path("/tmp/modelo_ductos_multitarea_efnet.pth")
_MAPPING_PATH = Path("/tmp/class_mapping.json")


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    print(f"Downloading {dest.name} …")
    urllib.request.urlretrieve(url, dest)
    print(f"  ✓ {dest.name}  ({dest.stat().st_size / 1e6:.1f} MB)")


_download(f"{_BASE}/modelo_ductos_multitarea_efnet.pth", _MODEL_PATH)
_download(f"{_BASE}/class_mapping.json", _MAPPING_PATH)

# ── Load model (once, at startup) ─────────────────────────────────────

with open(_MAPPING_PATH) as f:
    _mapping = json.load(f)

IDX_TO_CLASS_D = {int(k): v for k, v in _mapping["idx_to_class_d"].items()}
IDX_TO_CLASS_O = {int(k): v for k, v in _mapping["idx_to_class_o"].items()}

DEVICE = torch.device("cpu")  # HF Spaces free tier is CPU-only
MODEL = MultiEfficientNet(len(IDX_TO_CLASS_D), len(IDX_TO_CLASS_O))
MODEL.load_state_dict(torch.load(_MODEL_PATH, map_location=DEVICE, weights_only=True))
MODEL.to(DEVICE).eval()
TRANSFORM = get_transforms(train=False)

MAX_CLASS = 7


# ── Inference ─────────────────────────────────────────────────────────


def _label(val: int, prefix: str) -> str:
    return f"{prefix}{val}" if val < MAX_CLASS else f"{prefix}7+"


def predict(image: Image.Image) -> tuple:
    if image is None:
        return "Upload an image to start.", None

    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out_d, out_o = MODEL(tensor)
        probs_d = torch.softmax(out_d, dim=1).cpu().numpy()[0]
        probs_o = torch.softmax(out_o, dim=1).cpu().numpy()[0]
        pred_d = int(torch.argmax(out_d, dim=1))
        pred_o = int(torch.argmax(out_o, dim=1))

    d_val = IDX_TO_CLASS_D[pred_d]
    o_val = IDX_TO_CLASS_O[pred_o]

    result = (
        f"**Total ducts  (dX):** {_label(d_val, 'd')}\n\n"
        f"**Occupied (oX):** {_label(o_val, 'o')}\n\n"
        f"**Vacant   (vX):** {max(d_val - o_val, 0)}"
    )
    return result, _build_figure(probs_d, probs_o)


def _build_figure(probs_d, probs_o):
    labels_d = [_label(v, "d") for v in IDX_TO_CLASS_D.values()]
    labels_o = [_label(v, "o") for v in IDX_TO_CLASS_O.values()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Probability distribution", fontsize=12, fontweight="bold")
    _bar(ax1, probs_d, labels_d, "Total ducts (dX)", "#4C8BF5")
    _bar(ax2, probs_o, labels_o, "Occupied (oX)", "#34A853")
    fig.tight_layout()
    return fig


def _bar(ax, probs, labels, title, color):
    y = np.arange(len(labels))
    bars = ax.barh(y, probs, color=color, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title(title, fontsize=10)
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    for bar, prob in zip(bars, probs):
        ax.text(
            min(prob + 0.02, 0.95),
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.0%}",
            va="center",
            fontsize=8,
        )


# ── Gradio interface ──────────────────────────────────────────────────

with gr.Blocks(title="Industrial Duct Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏭 Industrial Duct Classifier
        **Model:** EfficientNet-B0 multitask &nbsp;·&nbsp; **Tasks:** total ducts (dX) and occupied (oX)

        Upload an image and the model predicts how many ducts are **total**, **occupied**, and **vacant**.

        > This is an interactive demo. For the production inference service (FastAPI + batching + Prometheus)
        > see the [GitHub repository](https://github.com/soyDCAR/industrial-duct-classifier).
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Input image")
            btn = gr.Button("Classify", variant="primary")
        with gr.Column(scale=1):
            result_out = gr.Markdown(label="Result")
            plot_out = gr.Plot(label="Class probabilities")

    btn.click(fn=predict, inputs=img_input, outputs=[result_out, plot_out])
    img_input.change(fn=predict, inputs=img_input, outputs=[result_out, plot_out])

    gr.Markdown(
        """
        ---
        **Classes:** d0–d6 (total ducts), d7+ (7 or more) &nbsp;·&nbsp; o0–o6 (occupied), o7+
        **Accuracy:** dX 55.4 % · oX 52.4 % — trained on ~840 images, 10 epochs
        **Source:** [soyDCAR/industrial-duct-classifier](https://github.com/soyDCAR/industrial-duct-classifier)
        """
    )

if __name__ == "__main__":
    demo.launch()
