"""
Entrena el clasificador multitarea de ductos.

Uso:
    python train.py --data-dir img/
    python train.py --data-dir img/ --epochs 20 --batch-size 16 --lr 0.0005
    python train.py --data-dir img/ --output-dir runs/exp1
"""

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split

from metrics import run_full_evaluation
from model import DuctoDataset, FocalLoss, MultiEfficientNet, get_transforms


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenamiento clasificador de ductos")
    p.add_argument("--data-dir", default="img", help="Carpeta con las imágenes")
    p.add_argument("--output-dir", default="runs", help="Dónde guardar modelo y métricas")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def train_epoch(model, loader, criterion_d, criterion_o, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for x, y_d, y_o in loader:
        x, y_d, y_o = x.to(device), y_d.to(device), y_o.to(device)
        optimizer.zero_grad()
        out_d, out_o = model(x)
        loss = criterion_d(out_d, y_d) + criterion_o(out_o, y_o)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def val_epoch(model, loader, criterion_d, criterion_o, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y_d, y_o in loader:
            x, y_d, y_o = x.to(device), y_d.to(device), y_o.to(device)
            out_d, out_o = model(x)
            total_loss += (criterion_d(out_d, y_d) + criterion_o(out_o, y_o)).item()
    return total_loss / len(loader)


def save_loss_curve(train_losses: list, val_losses: list, path: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="Entrenamiento")
    ax.plot(val_losses, label="Validación")
    ax.set_title("Pérdida por época")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Curva guardada   : {path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo : {device}")

    # Dataset
    dataset = DuctoDataset(args.data_dir)
    print(f"Imágenes    : {len(dataset)}")
    print(f"Clases dX   : {len(dataset.class_to_idx_d)}  →  {list(dataset.class_to_idx_d.keys())}")
    print(f"Clases oX   : {len(dataset.class_to_idx_o)}  →  {list(dataset.class_to_idx_o.keys())}")

    # Guardar class_mapping.json — lo necesitan predict.py y el GUI
    mapping = {
        "idx_to_class_d": {str(k): v for k, v in dataset.idx_to_class_d.items()},
        "idx_to_class_o": {str(k): v for k, v in dataset.idx_to_class_o.items()},
    }
    mapping_path = os.path.join(args.output_dir, "class_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Clase mapping: {mapping_path}")

    # Split train / val con seed fijo para reproducibilidad
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_ds.dataset.transform = get_transforms(train=True)
    val_ds.dataset.transform = get_transforms(train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Modelo + criterios + optimizador
    model = MultiEfficientNet(
        num_classes_d=len(dataset.class_to_idx_d),
        num_classes_o=len(dataset.class_to_idx_o),
    ).to(device)
    criterion_d = FocalLoss()
    criterion_o = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Entrenamiento
    train_losses, val_losses = [], []
    for epoch in range(1, args.epochs + 1):
        t_loss = train_epoch(model, train_loader, criterion_d, criterion_o, optimizer, device)
        v_loss = val_epoch(model, val_loader, criterion_d, criterion_o, device)
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        print(f"Época {epoch:02d}/{args.epochs}  train_loss={t_loss:.4f}  val_loss={v_loss:.4f}")

    # Guardar modelo
    model_path = os.path.join(args.output_dir, "modelo_ductos_multitarea_efnet.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nModelo guardado : {model_path}")

    save_loss_curve(train_losses, val_losses, os.path.join(args.output_dir, "loss_curve.png"))

    # Evaluación completa — importada de metrics.py, sin código duplicado
    print("\nEvaluando en conjunto de validación...")
    run_full_evaluation(model, val_loader, dataset, device, args.output_dir)


if __name__ == "__main__":
    main()
