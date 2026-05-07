"""
Evalúa un modelo guardado sobre un conjunto de imágenes.
No requiere reentrenar — útil para comparar checkpoints o auditar el modelo.

Uso:
    python evaluate.py --model modelo_ductos_multitarea_efnet.pth --data-dir img/
    python evaluate.py --model runs/exp1/modelo.pth --data-dir img/ --output-dir runs/exp1
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

from metrics import run_full_evaluation
from model import DuctoDataset, MultiEfficientNet, get_transforms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluación del clasificador de ductos")
    p.add_argument("--model",      required=True,
                                   help="Ruta al archivo .pth")
    p.add_argument("--data-dir",   default="img",
                                   help="Carpeta con las imágenes de evaluación")
    p.add_argument("--mapping",    default=None,
                                   help="Ruta a class_mapping.json (opcional, se infiere del dataset)")
    p.add_argument("--output-dir", default="eval_output",
                                   help="Dónde guardar métricas y matrices de confusión")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"ERROR: No se encontró el modelo '{args.model}'.")
    if not os.path.isdir(args.data_dir):
        sys.exit(f"ERROR: La carpeta '{args.data_dir}' no existe.")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo : {device}")
    print(f"Modelo      : {args.model}")
    print(f"Dataset     : {args.data_dir}")

    # Cargar dataset completo para inferir el mapeo de clases
    dataset = DuctoDataset(args.data_dir, transform=get_transforms(train=False))
    print(f"Imágenes    : {len(dataset)}")
    print(f"Clases dX   : {len(dataset.class_to_idx_d)}  →  {list(dataset.class_to_idx_d.keys())}")
    print(f"Clases oX   : {len(dataset.class_to_idx_o)}  →  {list(dataset.class_to_idx_o.keys())}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Cargar modelo
    model = MultiEfficientNet(
        num_classes_d=len(dataset.class_to_idx_d),
        num_classes_o=len(dataset.class_to_idx_o),
    )
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    # Correr evaluación completa
    results = run_full_evaluation(model, loader, dataset, device, args.output_dir)

    print(f"\n{'═'*55}")
    print("  Resumen final")
    print(f"{'═'*55}")
    print(f"  dX  accuracy={results['dx']['accuracy']:.2%}  "
          f"F1w={results['dx']['f1_weighted']:.4f}  "
          f"F1macro={results['dx']['f1_macro']:.4f}")
    print(f"  oX  accuracy={results['ox']['accuracy']:.2%}  "
          f"F1w={results['ox']['f1_weighted']:.4f}  "
          f"F1macro={results['ox']['f1_macro']:.4f}")
    print(f"\n  Archivos generados en: {args.output_dir}/")
    print("    metrics.json")
    print("    confusion_dx.png")
    print("    confusion_ox.png")


if __name__ == "__main__":
    main()
