"""
Predice ductos totales (dX) y ocupados (oX) en una imagen o carpeta.

Uso — imagen individual:
    python predict.py imagen.jpg
    python predict.py imagen.jpg --model runs/modelo_ductos_multitarea_efnet.pth

Uso — carpeta completa:
    python predict.py img/ --batch
    python predict.py img/ --batch --output resultados.csv
"""
import argparse
import csv
import json
import os
import sys

import torch
from PIL import Image

from model import MultiEfficientNet, get_transforms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predicción clasificador de ductos")
    p.add_argument("input",         help="Imagen (.jpg/.png) o carpeta con --batch")
    p.add_argument("--model",       default="modelo_ductos_multitarea_efnet.pth",
                                    help="Ruta al archivo .pth")
    p.add_argument("--mapping",     default="class_mapping.json",
                                    help="Ruta al class_mapping.json generado por train.py")
    p.add_argument("--batch",       action="store_true",
                                    help="Procesar todos los archivos de una carpeta")
    p.add_argument("--output",      default=None,
                                    help="Guardar resultados en CSV (solo con --batch)")
    return p.parse_args()


def load_mapping(path: str) -> tuple[dict, dict]:
    """Carga idx_to_class_d e idx_to_class_o desde class_mapping.json."""
    if not os.path.exists(path):
        sys.exit(
            f"ERROR: No se encontró '{path}'.\n"
            "Ejecuta primero: python train.py --data-dir img/"
        )
    with open(path) as f:
        data = json.load(f)
    # Las claves en JSON son siempre strings; convertimos a int
    idx_to_class_d = {int(k): v for k, v in data["idx_to_class_d"].items()}
    idx_to_class_o = {int(k): v for k, v in data["idx_to_class_o"].items()}
    return idx_to_class_d, idx_to_class_o


def predict_image(image_path: str, model: MultiEfficientNet,
                  idx_to_class_d: dict, idx_to_class_o: dict,
                  transform, device: torch.device) -> dict:
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out_d, out_o = model(tensor)
        pred_d = torch.argmax(out_d, dim=1).item()
        pred_o = torch.argmax(out_o, dim=1).item()

    d_val = idx_to_class_d[pred_d]
    o_val = idx_to_class_o[pred_o]
    v_val = max(d_val - o_val, 0)

    MAX = 7
    return {
        "archivo":  os.path.basename(image_path),
        "d_total":  f"{d_val}" if d_val < MAX else "7+",
        "o_ocup":   f"{o_val}" if o_val < MAX else "7+",
        "v_vacio":  str(v_val),
    }


def print_result(result: dict) -> None:
    print(f"\n📸 {result['archivo']}")
    print(f"   Ductos totales (dX) : {result['d_total']}")
    print(f"   Ductos ocupados (oX): {result['o_ocup']}")
    print(f"   Ductos vacíos   (vX): {result['v_vacio']}")


def main():
    args = parse_args()

    # Verificar que el modelo existe
    if not os.path.exists(args.model):
        sys.exit(
            f"ERROR: No se encontró el modelo '{args.model}'.\n"
            "Descárgalo desde Releases o entrena con: python train.py --data-dir img/"
        )

    idx_to_class_d, idx_to_class_o = load_mapping(args.mapping)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transforms(train=False)

    model = MultiEfficientNet(
        num_classes_d=len(idx_to_class_d),
        num_classes_o=len(idx_to_class_o),
    )
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    # ── Modo batch (carpeta) ────────────────────────────────────────
    if args.batch:
        if not os.path.isdir(args.input):
            sys.exit(f"ERROR: '{args.input}' no es una carpeta. Quita --batch para una imagen.")

        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        files = [f for f in os.listdir(args.input) if f.lower().endswith(extensions)]
        if not files:
            sys.exit(f"No se encontraron imágenes en '{args.input}'.")

        results = []
        for fname in sorted(files):
            path   = os.path.join(args.input, fname)
            result = predict_image(path, model, idx_to_class_d, idx_to_class_o, transform, device)
            print_result(result)
            results.append(result)

        if args.output:
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["archivo", "d_total", "o_ocup", "v_vacio"])
                writer.writeheader()
                writer.writerows(results)
            print(f"\nResultados guardados en: {args.output}")

    # ── Modo imagen individual ──────────────────────────────────────
    else:
        if not os.path.isfile(args.input):
            sys.exit(f"ERROR: No se encontró la imagen '{args.input}'.")
        result = predict_image(args.input, model, idx_to_class_d, idx_to_class_o, transform, device)
        print_result(result)


if __name__ == "__main__":
    main()
