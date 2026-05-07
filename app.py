"""
Demo Gradio del clasificador multitarea de ductos.

Uso local:
    python app.py
    python app.py --model runs/exp1/modelo_ductos_multitarea_efnet.pth \
                  --mapping runs/exp1/class_mapping.json

Hugging Face Spaces:
    Sube este archivo como app.py junto con model.py.
    El .pth y class_mapping.json deben estar en la raíz del Space.
"""

import argparse
import json
import os
import sys

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from model import MultiEfficientNet, get_transforms

matplotlib.use("Agg")  # backend sin pantalla (necesario en servidores/Docker)

# ── Configuración por defecto ────────────────────────────────────────
DEFAULT_MODEL = "modelo_ductos_multitarea_efnet.pth"
DEFAULT_MAPPING = "class_mapping.json"
MAX_CLASS = 7


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo Gradio — Clasificador de Ductos")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--mapping", default=DEFAULT_MAPPING)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Genera link público temporal de Gradio")
    return p.parse_args()


def load_resources(model_path: str, mapping_path: str):
    """Carga modelo y mapeo de clases una sola vez al arrancar."""
    for path in (model_path, mapping_path):
        if not os.path.exists(path):
            sys.exit(
                f"ERROR: No se encontró '{path}'.\n"
                "Asegúrate de haber descargado el modelo desde Releases\n"
                "o haber entrenado con: python train.py --data-dir img/"
            )

    with open(mapping_path) as f:
        data = json.load(f)
    idx_to_class_d = {int(k): v for k, v in data["idx_to_class_d"].items()}
    idx_to_class_o = {int(k): v for k, v in data["idx_to_class_o"].items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiEfficientNet(len(idx_to_class_d), len(idx_to_class_o))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    return model, idx_to_class_d, idx_to_class_o, device


def class_label(val: int, prefix: str) -> str:
    return f"{prefix}{val}" if val < MAX_CLASS else f"{prefix}7+"


def predict(image: Image.Image, model, idx_to_class_d, idx_to_class_o, device, transform) -> tuple:
    """Devuelve texto de resultado + figura de probabilidades."""
    if image is None:
        return "Sube una imagen para comenzar.", None

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out_d, out_o = model(tensor)
        probs_d = torch.softmax(out_d, dim=1).cpu().numpy()[0]
        probs_o = torch.softmax(out_o, dim=1).cpu().numpy()[0]
        pred_d = int(torch.argmax(out_d, dim=1).item())
        pred_o = int(torch.argmax(out_o, dim=1).item())

    d_val = idx_to_class_d[pred_d]
    o_val = idx_to_class_o[pred_o]
    v_val = max(d_val - o_val, 0)

    resultado = (
        f"**Ductos totales  (dX):** {class_label(d_val, 'd')}\n\n"
        f"**Ductos ocupados (oX):** {class_label(o_val, 'o')}\n\n"
        f"**Ductos vacíos   (vX):** {v_val}"
    )

    fig = _build_probs_figure(probs_d, probs_o, idx_to_class_d, idx_to_class_o)
    return resultado, fig


def _build_probs_figure(probs_d, probs_o, idx_to_class_d, idx_to_class_o):
    """Gráfico de barras horizontales con probabilidad por clase (dos paneles)."""
    labels_d = [class_label(v, "d") for v in idx_to_class_d.values()]
    labels_o = [class_label(v, "o") for v in idx_to_class_o.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Distribución de probabilidades", fontsize=12, fontweight="bold")

    _bar_panel(ax1, probs_d, labels_d, "Ductos totales (dX)", "#4C8BF5")
    _bar_panel(ax2, probs_o, labels_o, "Ductos ocupados (oX)", "#34A853")

    fig.tight_layout()
    return fig


def _bar_panel(ax, probs, labels, title, color):
    y = np.arange(len(labels))
    bars = ax.barh(y, probs, color=color, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilidad")
    ax.set_title(title, fontsize=10)
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Etiqueta de porcentaje al final de cada barra
    for bar, prob in zip(bars, probs):
        ax.text(
            min(prob + 0.02, 0.95),
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.0%}",
            va="center",
            fontsize=8,
        )


def build_interface(model, idx_to_class_d, idx_to_class_o, device) -> gr.Blocks:
    transform = get_transforms(train=False)

    def _predict_wrapper(image):
        return predict(image, model, idx_to_class_d, idx_to_class_o, device, transform)

    with gr.Blocks(title="Clasificador de Ductos", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Clasificador de Ductos
            **Modelo:** EfficientNet-B0 multitarea · **Tareas:** ductos totales (dX) y ocupados (oX)

            Sube una imagen de ductos y el modelo predice cuántos hay en total,
            cuántos están ocupados, y calcula los vacíos automáticamente.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(type="pil", label="Imagen de entrada")
                btn = gr.Button("Clasificar", variant="primary")

            with gr.Column(scale=1):
                texto_out = gr.Markdown(label="Resultado")
                plot_out = gr.Plot(label="Probabilidades por clase")

        btn.click(
            fn=_predict_wrapper,
            inputs=img_input,
            outputs=[texto_out, plot_out],
        )
        # También predice al subir la imagen (sin necesidad de hacer clic)
        img_input.change(
            fn=_predict_wrapper,
            inputs=img_input,
            outputs=[texto_out, plot_out],
        )

        # Imagen de ejemplo incluida en el repo
        if os.path.exists("img_predic.jpg"):
            gr.Examples(
                examples=[["img_predic.jpg"]],
                inputs=img_input,
                label="Ejemplo",
            )

        gr.Markdown(
            """
            ---
            **Clases:** d0–d6 (ductos totales), d7+ (7 o más) · o0–o6 (ocupados), o7+

            [Repositorio](https://github.com/TU_USUARIO/industrial-duct-classifier) ·
            [Modelo en Releases](https://github.com/TU_USUARIO/industrial-duct-classifier/releases)
            """
        )

    return demo


def main():
    args = parse_args()
    model, idx_to_class_d, idx_to_class_o, device = load_resources(args.model, args.mapping)
    demo = build_interface(model, idx_to_class_d, idx_to_class_o, device)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
