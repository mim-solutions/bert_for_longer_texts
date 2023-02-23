from __future__ import annotations
from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def fit(self, x_train: list[str], y_train: list[bool]) -> None:
        pass

    @abstractmethod
    def predict(self, x: list[str]) -> list[tuple[bool, float]]:
        pass

    @abstractmethod
    def predict_classes(self, x: list[str]) -> list[bool]:
        pass

    @abstractmethod
    def predict_scores(self, x: list[str]) -> list[float]:
        pass

    @abstractmethod
    def save(self, model_dir: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, model_dir: str) -> Model:
        pass
