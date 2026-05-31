import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

DEFAULT_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
DEFAULT_IMG_SIZE = 160
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SimpleHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.35):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


@dataclass
class LoadedModelInfo:
    model_path: str
    classes: list[str]
    img_size: int
    architecture: str
    device: str
    metadata: dict[str, Any]


class ModelManager:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: nn.Module | None = None
        self.classes: list[str] = DEFAULT_CLASSES.copy()
        self.img_size: int = DEFAULT_IMG_SIZE
        self.architecture: str = "efficientnet_b3"
        self.model_path: str | None = None
        self.metadata: dict[str, Any] = {}
        self.transform = transforms.Compose(
            [
                transforms.Resize((DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def _build_model(self, num_classes: int) -> nn.Module:
        # Build the exact EfficientNet-B3 backbone and swap in the custom FER head.
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = SimpleHead(in_features=in_features, num_classes=num_classes)
        return model

    def load_model(self, model_path: str) -> LoadedModelInfo:
        # PyTorch 2.6+ defaults to weights_only=True, which can reject older
        # training checkpoints containing richer Python objects. We first try
        # the safe mode, then fallback to full load for trusted local files.
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except Exception:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid checkpoint format: expected a dict")

        classes = checkpoint.get("classes", DEFAULT_CLASSES)
        if not isinstance(classes, list) or len(classes) == 0:
            classes = DEFAULT_CLASSES.copy()

        num_classes = int(checkpoint.get("num_classes", len(classes)))
        if num_classes != len(classes):
            classes = DEFAULT_CLASSES.copy()
            num_classes = len(classes)

        img_size = int(checkpoint.get("img_size", DEFAULT_IMG_SIZE))
        architecture = checkpoint.get("architecture", "efficientnet_b3")

        if architecture != "efficientnet_b3":
            raise ValueError(
                f"Unsupported architecture '{architecture}'. Expected 'efficientnet_b3'."
            )

        model = self._build_model(num_classes=num_classes)

        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            raise ValueError("Checkpoint missing 'model_state_dict'")

        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()

        self.model = model
        self.classes = classes
        self.img_size = img_size
        self.architecture = architecture
        self.model_path = model_path
        self.metadata = {
            "val_accuracy": checkpoint.get("val_accuracy"),
            "raf_accuracy": checkpoint.get("raf_accuracy"),
            "best_epoch": checkpoint.get("best_epoch"),
            "num_classes": num_classes,
        }

        self.transform = transforms.Compose(
            [
                # Inference preprocessing must match training contract:
                # RGB -> resize(img_size) -> tensor -> ImageNet normalization.
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        print("Model loaded successfully")
        print(f"Path: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Architecture: {self.architecture}")
        print(f"Classes: {self.classes}")
        print(f"Image size: {self.img_size}")
        print(f"Metadata: {self.metadata}")

        return LoadedModelInfo(
            model_path=self.model_path,
            classes=self.classes,
            img_size=self.img_size,
            architecture=self.architecture,
            device=self.device,
            metadata=self.metadata,
        )

    def predict_pil(self, image: Image.Image) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        inference_ms = (time.perf_counter() - start) * 1000.0

        percentages = [round(float(p * 100.0), 1) for p in probs]
        top_idx = int(probs.argmax())

        probability_map = {
            class_name: percentages[idx] for idx, class_name in enumerate(self.classes)
        }

        return {
            "top": self.classes[top_idx],
            "confidence": percentages[top_idx],
            "probabilities": probability_map,
            "inference_ms": round(inference_ms, 2),
        }
