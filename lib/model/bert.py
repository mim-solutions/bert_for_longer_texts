from __future__ import annotations
from typing import Any, Optional, Union
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import torch
from torch import Tensor
from torch.nn import BCELoss, DataParallel, Module, Linear, Sigmoid
from torch.optim import AdamW, Optimizer
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, DataLoader
from transformers import AutoModel, AutoTokenizer, BatchEncoding

from lib.entities.learning_curve import LossesSingleEpoch
from lib.model.base import Model

load_dotenv()


class BertClassifier(Model):
    def __init__(
        self,
        params: dict,
        tokenizer: Optional[AutoTokenizer] = None,
        neural_network: Optional[Module] = None,
        device: str = "cuda:0",
        many_gpus: bool = False,
    ):
        if not tokenizer:
            tokenizer = load_tokenizer()
        if not neural_network:
            bert = load_bert()
            neural_network = BertClassifierNN(bert)

        self.params = params
        self.device = device
        self.many_gpus = many_gpus
        self.tokenizer = tokenizer
        self.neural_network = neural_network

        self.neural_network.to(device)
        if device.startswith("cuda") and many_gpus:
            self.neural_network = DataParallel(self.neural_network)

    def fit(self, x_train: list[str], y_train: list[bool], epochs: Optional[int] = None) -> None:
        if not epochs:
            epochs = self.params["epochs"]
        optimizer = AdamW(self.neural_network.parameters(), lr=self.params["learning_rate"])

        tokens = self._tokenize(x_train)
        dataset = TokenizedDataset(tokens, y_train)
        dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=self.params["batch_size"])
        for epoch in range(epochs):
            self._train_single_epoch(dataloader, optimizer)

    def predict(self, x: list[str], batch_size: Optional[int] = None) -> list[tuple[bool, float]]:
        if not batch_size:
            batch_size = self.params["batch_size"]
        scores = self.predict_scores(x, batch_size)
        classes = [i >= 0.5 for i in scores]
        return list(zip(classes, scores))

    def predict_classes(self, x: list[str], batch_size: Optional[int] = None) -> list[bool]:
        if not batch_size:
            batch_size = self.params["batch_size"]
        scores = self.predict_scores(x, batch_size)
        classes = [i >= 0.5 for i in scores]
        return classes

    def predict_scores(self, x: list[str], batch_size: Optional[int] = None) -> list[float]:
        if not batch_size:
            batch_size = self.params["batch_size"]
        tokens = self._tokenize(x)
        dataset = TokenizedDataset(tokens)
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)
        total_predictions = []

        # deactivate dropout layers
        self.neural_network.eval()
        for step, batch in enumerate(dataloader):
            # deactivate autograd
            with torch.no_grad():
                predictions = self._evaluate_single_batch(batch)
                total_predictions.extend(predictions.tolist())
        return total_predictions

    def save(self, model_dir: str) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(file=model_dir / "params.json", mode="w", encoding="utf-8") as file:
            json.dump(self.params, file)
        self.tokenizer.save_pretrained(model_dir)
        if self.many_gpus:
            torch.save(self.neural_network.module, model_dir / "model.bin")
        else:
            torch.save(self.neural_network, model_dir / "model.bin")

    def train_and_evaluate(
        self, x_train: list[str], x_val: list[bool], y_train: list[str], y_val: list[bool], epochs: Optional[int] = None
    ) -> list[LossesSingleEpoch]:
        """Returns history of train and val losses for each epoch"""
        if not epochs:
            epochs = self.params["epochs"]
        optimizer = AdamW(self.neural_network.parameters(), lr=self.params["learning_rate"])

        tokens = self._tokenize(x_train)
        dataset = TokenizedDataset(tokens, y_train)
        dataloader_train = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=self.params["batch_size"])
        tokens = self._tokenize(x_val)
        dataset = TokenizedDataset(tokens, y_val)
        dataloader_val = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=self.params["batch_size"])
        result = []
        for epoch in range(epochs):
            result.append(self._train_and_evaluate_single_epoch(dataloader_train, dataloader_val, optimizer))
        return result

    @classmethod
    def load(cls, model_dir: str, device: str = "cuda:0", many_gpus: bool = False) -> BertClassifier:
        model_dir = Path(model_dir)
        with open(file=model_dir / "params.json", mode="r", encoding="utf-8") as file:
            params = json.load(file)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        neural_network = torch.load(f=model_dir / "model.bin", map_location=device)
        return cls(params, tokenizer, neural_network, device, many_gpus)

    def _tokenize(self, texts: list[str]) -> BatchEncoding:
        """Transforms list of texts to list of tokens (truncated to 512 tokens)."""
        self.tokenizer.pad_token = "<pad>"
        tokens = self.tokenizer.batch_encode_plus(
            texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
        )
        return tokens

    def _train_single_epoch(self, dataloader: DataLoader, optimizer: Optimizer) -> tuple[float, float]:
        self.neural_network.train()
        cross_entropy = BCELoss()

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            labels = batch[-1].float().cpu()
            predictions = self._evaluate_single_batch(batch)

            loss = cross_entropy(predictions, labels)
            loss.backward()
            optimizer.step()

    def _evaluate_single_batch(self, batch: tuple[Tensor]) -> Tensor:
        batch = [t.to(self.device) for t in batch]
        model_input = batch[:2]

        predictions = self.neural_network(*model_input)
        predictions = torch.flatten(predictions).cpu()
        return predictions

    def _train_and_evaluate_single_epoch(
        self, dataloader_train: DataLoader, dataloader_val: DataLoader, optimizer: Optimizer
    ) -> LossesSingleEpoch:
        self.neural_network.train()
        cross_entropy = BCELoss(reduction="sum")

        total_loss = 0
        # activate dropout layers
        self.neural_network.train()
        for step, batch in enumerate(dataloader_train):
            optimizer.zero_grad()
            labels = batch[-1].float().cpu()
            predictions = self._evaluate_single_batch(batch)

            loss = cross_entropy(predictions, labels)
            total_loss += loss.detach().cpu().numpy()
            loss.backward()
            optimizer.step()

        avg_loss_train = total_loss / len(dataloader_train)

        total_loss = 0
        # deactivate dropout layers
        self.neural_network.eval()
        for step, batch in enumerate(dataloader_val):
            # deactivate autograd
            with torch.no_grad():
                labels = batch[-1].float().cpu()
                predictions = self._evaluate_single_batch(batch)

                loss = cross_entropy(predictions, labels)
                total_loss += loss.detach().cpu().numpy()

        avg_loss_val = total_loss / len(dataloader_val)

        return LossesSingleEpoch(loss_train=avg_loss_train, loss_val=avg_loss_val)


class BertClassifierNN(Module):
    def __init__(self, model: AutoModel):
        super().__init__()
        self.model = model

        # classification head
        self.linear = Linear(768, 1)
        self.sigmoid = Sigmoid()

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        x = self.model(input_ids, attention_mask)
        x = x[0][:, 0, :]  # take <s> token (equiv. to [CLS])

        # classification head
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


class TokenizedDataset(Dataset):
    """Dataset for tokens with optional labels."""

    def __init__(self, tokens: BatchEncoding, labels: Optional[list] = None):
        self.input_ids = tokens["input_ids"]
        self.attention_mask = tokens["attention_mask"]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Union[tuple[Tensor, Tensor, Any], tuple[Tensor, Tensor]]:
        if self.labels:
            return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]
        return self.input_ids[idx], self.attention_mask[idx]


def load_tokenizer() -> AutoTokenizer:
    MODEL_LOAD_FROM_FILE = os.environ["MODEL_LOAD_FROM_FILE"] == "True"
    if MODEL_LOAD_FROM_FILE:
        MODEL_PATH = Path(os.environ["MODEL_PATH"])
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer


def load_bert() -> AutoModel:
    MODEL_LOAD_FROM_FILE = os.environ["MODEL_LOAD_FROM_FILE"] == "True"
    if MODEL_LOAD_FROM_FILE:
        MODEL_PATH = Path(os.environ["MODEL_PATH"])
        model = AutoModel.from_pretrained(MODEL_PATH)
    else:
        model = AutoModel.from_pretrained("bert-base-uncased")
    return model
