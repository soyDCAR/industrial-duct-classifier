"""
Funciones de evaluación reutilizables.
Importadas por train.py y evaluate.py para no duplicar código.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def collect_predictions(model, loader, device):
    """Recorre el loader y devuelve listas de verdaderos y predichos."""
    model.eval()
    y_true_d, y_pred_d = [], []
    y_true_o, y_pred_o = [], []

    with torch.no_grad():
        for x, y_d, y_o in loader:
            out_d, out_o = model(x.to(device))
            y_pred_d.extend(torch.argmax(out_d, 1).cpu().tolist())
            y_pred_o.extend(torch.argmax(out_o, 1).cpu().tolist())
            y_true_d.extend(y_d.tolist())
            y_true_o.extend(y_o.tolist())

    return y_true_d, y_pred_d, y_true_o, y_pred_o


def compute_metrics(y_true, y_pred, labels, names, task: str) -> dict:
    """
    Calcula accuracy, F1 macro, F1 weighted y métricas por clase.
    Devuelve un dict estructurado listo para JSON.
    """
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=names,
        output_dict=True, zero_division=0,
    )

    per_class = {
        name: {
            "precision": round(report[name]["precision"], 4),
            "recall":    round(report[name]["recall"],    4),
            "f1":        round(report[name]["f1-score"],  4),
            "support":   int(report[name]["support"]),
        }
        for name in names
    }

    return {
        "task":           task,
        "accuracy":       round(accuracy_score(y_true, y_pred), 4),
        "f1_weighted":    round(f1_score(y_true, y_pred, average="weighted",
                                         labels=labels, zero_division=0), 4),
        "f1_macro":       round(f1_score(y_true, y_pred, average="macro",
                                         labels=labels, zero_division=0), 4),
        "per_class":      per_class,
    }


def print_report(metrics_d: dict, metrics_o: dict) -> None:
    """Imprime un resumen limpio en consola."""
    _print_task_summary(metrics_d)
    _print_task_summary(metrics_o)


def _print_task_summary(m: dict) -> None:
    task = m["task"]
    print(f"\n{'─'*55}")
    print(f"  Tarea: {task}")
    print(f"{'─'*55}")
    print(f"  Accuracy    : {m['accuracy']:.2%}")
    print(f"  F1 weighted : {m['f1_weighted']:.4f}")
    print(f"  F1 macro    : {m['f1_macro']:.4f}")
    print(f"\n  {'Clase':<8} {'Prec':>6} {'Recall':>7} {'F1':>6} {'Soporte':>8}")
    print(f"  {'─'*38}")
    for name, vals in m["per_class"].items():
        print(
            f"  {name:<8} {vals['precision']:>6.2f} "
            f"{vals['recall']:>7.2f} {vals['f1']:>6.2f} "
            f"{vals['support']:>8}"
        )


def save_metrics_json(metrics_d: dict, metrics_o: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"dx": metrics_d, "ox": metrics_o}, f, indent=2, ensure_ascii=False)
    print(f"\n  Métricas guardadas: {path}")


def save_confusion_matrix(
    y_true, y_pred, labels, names,
    title: str, path: str,
    cmap_abs: str = "Blues",
    cmap_norm: str = "Oranges",
) -> None:
    """
    Guarda DOS matrices de confusión lado a lado:
    - Izquierda : conteos absolutos
    - Derecha   : normalizada por fila (recall por clase en %)

    La versión normalizada es crítica con clases desbalanceadas:
    muestra qué porcentaje de cada clase real predijo correctamente el modelo,
    sin que los números grandes de clases frecuentes "tapen" los errores.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Normalización por fila: cada fila suma 1 (divide por total de esa clase real)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(cm.sum(axis=1, keepdims=True) == 0, 0,
                           cm / cm.sum(axis=1, keepdims=True))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    _plot_cm(axes[0], cm,      names, cmap_abs,  "Conteos absolutos")
    _plot_cm(axes[1], cm_norm, names, cmap_norm, "Normalizada por fila (recall %)",
             fmt=".0%")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Matriz guardada  : {path}")


def _plot_cm(ax, cm, names, cmap, subtitle, fmt="d"):
    n = len(names)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap,
                   vmin=0, vmax=(1.0 if fmt == ".0%" else cm.max()))

    ax.set_title(subtitle, fontsize=10)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    tick_marks = range(n)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(names, fontsize=8)

    # Umbral de color para texto legible sobre fondo oscuro/claro
    thresh = cm.max() / 2 if fmt == "d" else 0.5
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            text = format(val, fmt) if fmt != "d" else str(val)
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if val > thresh else "black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def run_full_evaluation(model, loader, dataset, device, output_dir: str) -> dict:
    """
    Pipeline completo: predice → calcula métricas → guarda JSON y PNGs.
    Devuelve el dict con métricas de ambas tareas.
    """
    y_true_d, y_pred_d, y_true_o, y_pred_o = collect_predictions(model, loader, device)

    all_idx_d = sorted(set(y_true_d + y_pred_d))
    all_idx_o = sorted(set(y_true_o + y_pred_o))
    names_d = [dataset.label_name_d(i) for i in all_idx_d]
    names_o = [dataset.label_name_o(i) for i in all_idx_o]

    metrics_d = compute_metrics(y_true_d, y_pred_d, all_idx_d, names_d,
                                task="dX — ductos totales")
    metrics_o = compute_metrics(y_true_o, y_pred_o, all_idx_o, names_o,
                                task="oX — ductos ocupados")

    print_report(metrics_d, metrics_o)
    save_metrics_json(metrics_d, metrics_o, output_dir)

    save_confusion_matrix(
        y_true_d, y_pred_d, all_idx_d, names_d,
        title="Confusión — dX (ductos totales)",
        path=os.path.join(output_dir, "confusion_dx.png"),
    )
    save_confusion_matrix(
        y_true_o, y_pred_o, all_idx_o, names_o,
        title="Confusión — oX (ductos ocupados)",
        path=os.path.join(output_dir, "confusion_ox.png"),
        cmap_abs="Greens", cmap_norm="YlOrRd",
    )

    return {"dx": metrics_d, "ox": metrics_o}
