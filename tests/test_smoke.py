"""
Smoke tests — verifican que el código funciona sin dataset ni modelo guardado.

No testean que el modelo prediga correctamente (eso requiere datos reales).
Testean que la arquitectura, las transformaciones y los CLIs no están rotos.
"""

import subprocess
import sys

import pytest
import torch
from PIL import Image

from model import DuctoDataset, FocalLoss, MultiEfficientNet, get_transforms

# Algunos entornos CI tienen el bridge C torch-numpy roto (numpy instalado pero
# _ARRAY_API no encontrado). Detectamos esto una sola vez y saltamos los tests
# que aplican transforms sobre PIL images en esos entornos.
def _numpy_bridge_ok() -> bool:
    try:
        torch.zeros(1).numpy()
        return True
    except RuntimeError:
        return False

NUMPY_BRIDGE = _numpy_bridge_ok()
requires_numpy = pytest.mark.skipif(
    not NUMPY_BRIDGE,
    reason="torch-numpy C bridge no disponible en este entorno CI",
)

# ── Arquitectura ─────────────────────────────────────────────────────


def test_model_instantiation():
    """MultiEfficientNet se crea con cualquier número de clases."""
    model = MultiEfficientNet(num_classes_d=8, num_classes_o=8)
    assert model is not None


def test_forward_pass_shape():
    """El forward pass devuelve tensores con la forma correcta."""
    model = MultiEfficientNet(num_classes_d=8, num_classes_o=6)
    model.eval()
    batch = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out_d, out_o = model(batch)
    assert out_d.shape == (2, 8), f"Esperado (2,8), obtenido {out_d.shape}"
    assert out_o.shape == (2, 6), f"Esperado (2,6), obtenido {out_o.shape}"


def test_forward_pass_single_image():
    """Funciona con batch de tamaño 1 (caso de inferencia en producción)."""
    model = MultiEfficientNet(num_classes_d=5, num_classes_o=5)
    model.eval()
    img = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out_d, out_o = model(img)
    assert out_d.shape == (1, 5)
    assert out_o.shape == (1, 5)


# ── FocalLoss ────────────────────────────────────────────────────────


def test_focal_loss_positive():
    """FocalLoss devuelve un valor positivo."""
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.randn(4, 8)
    targets = torch.randint(0, 8, (4,))
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0, "FocalLoss debe ser >= 0"


def test_focal_loss_perfect_prediction_near_zero():
    """Con predicciones perfectas el loss tiende a 0."""
    loss_fn = FocalLoss(gamma=2.0)
    # logits muy altos en la clase correcta
    logits = torch.zeros(3, 5)
    targets = torch.tensor([0, 1, 2])
    logits[0, 0] = 100.0
    logits[1, 1] = 100.0
    logits[2, 2] = 100.0
    loss = loss_fn(logits, targets)
    assert loss.item() < 0.01, (
        f"Loss con predicción perfecta debería ser ~0, obtenido {loss.item()}"
    )


# ── Transformaciones ─────────────────────────────────────────────────


@requires_numpy
def test_transforms_return_tensor():
    """Las transformaciones convierten PIL Image en tensor de la forma correcta."""
    img = Image.new("RGB", (300, 200), color=(128, 64, 32))

    for train_mode in (True, False):
        t = get_transforms(train=train_mode)
        tensor = t(img)
        assert isinstance(tensor, torch.Tensor), f"train={train_mode}: esperado Tensor"
        assert tensor.shape == (3, 224, 224), f"train={train_mode}: forma incorrecta {tensor.shape}"


# ── Dataset ──────────────────────────────────────────────────────────


def test_dataset_empty_dir(tmp_path):
    """DuctoDataset sobre carpeta vacía devuelve longitud 0 sin crashear."""
    ds = DuctoDataset(str(tmp_path))
    assert len(ds) == 0


def test_dataset_ignores_unlabeled_files(tmp_path):
    """Archivos sin el patrón _dX_ _oX_ en el nombre son ignorados."""
    (tmp_path / "sin_etiqueta.jpg").touch()
    (tmp_path / "foto.png").touch()
    ds = DuctoDataset(str(tmp_path))
    assert len(ds) == 0


@requires_numpy
def test_dataset_with_synthetic_image(tmp_path):
    """Con una imagen válida el dataset tiene longitud 1 y devuelve un item."""
    img_path = tmp_path / "img001_d3_o1_v2.jpg"
    Image.new("RGB", (224, 224)).save(str(img_path))

    ds = DuctoDataset(str(tmp_path), transform=get_transforms(train=False))
    assert len(ds) == 1

    tensor, label_d, label_o = ds[0]
    assert tensor.shape == (3, 224, 224)
    assert isinstance(label_d, int)
    assert isinstance(label_o, int)


# ── CLIs (argparse) ──────────────────────────────────────────────────


@pytest.mark.parametrize("script", ["train.py", "predict.py", "evaluate.py", "app.py"])
def test_cli_help(script):
    """Todos los scripts responden a --help sin error."""
    result = subprocess.run(
        [sys.executable, script, "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{script} --help retornó código {result.returncode}\nstderr: {result.stderr}"
    )
    assert "usage" in result.stdout.lower(), f"{script} --help no muestra 'usage'"
