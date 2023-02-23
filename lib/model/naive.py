from __future__ import annotations
import json
from pathlib import Path
from random import choices

from .base import Model


class NaiveClassifier(Model):
    def __init__(self, probability: float):
        self.probability = probability

    def fit(self, x_train: list[str], y_train: list[bool]) -> None:
        pass

    def predict(self, x: list[str]) -> list[tuple[bool, float]]:
        scores = self.predict_scores(x)
        classes = [i == 1 for i in scores]
        return [(c, s) for c, s in zip(classes, scores)]

    def predict_classes(self, x: list[str]) -> list[bool]:
        scores = self.predict_scores(x)
        return [i == 1 for i in scores]

    def predict_scores(self, x: list[str]) -> list[float]:
        return choices([0, 1], [1 - self.probability, self.probability], k=len(x))

    def save(self, model_dir: str) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(file=model_dir / "metadata.json", mode="w", encoding="utf-8") as file:
            json.dump({"model_type": "naive"}, file)
        with open(file=model_dir / "probability.txt", mode="w", encoding="utf-8") as file:
            file.write(str(self.probability))

    @classmethod
    def load(cls, model_dir: str) -> NaiveClassifier:
        with open(file=Path(model_dir) / "probability.txt", mode="r", encoding="utf-8") as file:
            probability = float(file.read())
        return cls(probability)
