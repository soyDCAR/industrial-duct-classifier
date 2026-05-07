import os
import re

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = self.ce(logits, targets)
        p = torch.exp(-logp)
        return (1 - p) ** self.gamma * logp


class DuctoDataset(Dataset):
    """Lee imágenes cuyo nombre contiene _dX_ y _oX_ como etiquetas."""

    MAX_CLASS = 7  # valores > 6 se agrupan en la clase 7 ("7+")

    def __init__(self, img_dir: str, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images: list[str] = []
        self.labels_d: list[int] = []
        self.labels_o: list[int] = []

        for fname in os.listdir(img_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            match_d = re.search(r"_d(\d+)_", fname)
            match_o = re.search(r"_o(\d+)_", fname)
            if match_d and match_o:
                self.images.append(fname)
                self.labels_d.append(min(int(match_d.group(1)), self.MAX_CLASS))
                self.labels_o.append(min(int(match_o.group(1)), self.MAX_CLASS))

        raw_classes_d = sorted(set(self.labels_d))
        raw_classes_o = sorted(set(self.labels_o))
        self.class_to_idx_d = {v: i for i, v in enumerate(raw_classes_d)}
        self.class_to_idx_o = {v: i for i, v in enumerate(raw_classes_o)}
        self.idx_to_class_d = {i: v for v, i in self.class_to_idx_d.items()}
        self.idx_to_class_o = {i: v for v, i in self.class_to_idx_o.items()}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_d = self.class_to_idx_d[self.labels_d[idx]]
        label_o = self.class_to_idx_o[self.labels_o[idx]]
        return image, label_d, label_o

    def label_name_d(self, idx: int) -> str:
        v = self.idx_to_class_d[idx]
        return f"d{v}" if v < self.MAX_CLASS else "d7+"

    def label_name_o(self, idx: int) -> str:
        v = self.idx_to_class_o[idx]
        return f"o{v}" if v < self.MAX_CLASS else "o7+"


class MultiEfficientNet(nn.Module):
    """EfficientNet-B0 con dos cabezas de clasificación: ductos totales y ocupados."""

    def __init__(self, num_classes_d: int, num_classes_o: int):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = base.classifier[1].in_features
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier_d = nn.Linear(in_features, num_classes_d)
        self.classifier_o = nn.Linear(in_features, num_classes_o)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.classifier_d(x), self.classifier_o(x)


def load_model(checkpoint: str, num_classes_d: int, num_classes_o: int,
               device: torch.device) -> MultiEfficientNet:
    model = MultiEfficientNet(num_classes_d, num_classes_o)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()
    return model
