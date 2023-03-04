from __future__ import annotations
from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch import Tensor
from torch.nn import BCELoss, DataParallel, Module, Linear, Sigmoid
from torch.optim import AdamW, Optimizer
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, DataLoader
from transformers import AutoModel, AutoTokenizer, BatchEncoding, BertModel, PreTrainedTokenizerBase, RobertaModel


class BertClassifier(ABC):
    """
    The "device" parameter can have the following values:
    - "cpu" - the model will be loaded on CPU
    - "cuda" - the model will be loaded on single GPU
    - "cuda:i" - the model will be loaded on the specific single GPU with the index i

    It is also possible to use multiple GPUs. In order to do this:
    - set device to "cuda"
    - set many_gpu flag to True
    - as default it will use all of them.
    To use only selected GPUs - set the environmental variable CUDA_VISIBLE_DEVICES
    """

    @abstractmethod
    def __init__(
        self,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        neural_network: Optional[Module] = None,
        pretrained_model_name_or_path: Optional[str] = "bert-base-uncased",
        device: str = "cuda:0",
        many_gpus: bool = False,
    ):
        if not tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        if not neural_network:
            bert = AutoModel.from_pretrained(pretrained_model_name_or_path)
            neural_network = BertClassifierNN(bert)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        self.many_gpus = many_gpus
        self.tokenizer = tokenizer
        self.neural_network = neural_network
        self.collate_fn = None

        self.neural_network.to(device)
        if device.startswith("cuda") and many_gpus:
            self.neural_network = DataParallel(self.neural_network)

    def fit(self, x_train: list[str], y_train: list[bool], epochs: Optional[int] = None) -> None:
        if not epochs:
            epochs = self.epochs
        optimizer = AdamW(self.neural_network.parameters(), lr=self.learning_rate)

        tokens = self._tokenize(x_train)
        dataset = TokenizedDataset(tokens, y_train)
        dataloader = DataLoader(
            dataset, sampler=RandomSampler(dataset), batch_size=self.batch_size, collate_fn=self.collate_fn
        )
        for epoch in range(epochs):
            self._train_single_epoch(dataloader, optimizer)

    def predict(self, x: list[str], batch_size: Optional[int] = None) -> list[tuple[bool, float]]:
        if not batch_size:
            batch_size = self.batch_size
        scores = self.predict_scores(x, batch_size)
        classes = [i >= 0.5 for i in scores]
        return list(zip(classes, scores))

    def predict_classes(self, x: list[str], batch_size: Optional[int] = None) -> list[bool]:
        if not batch_size:
            batch_size = self.batch_size
        scores = self.predict_scores(x, batch_size)
        classes = [i >= 0.5 for i in scores]
        return classes

    def predict_scores(self, x: list[str], batch_size: Optional[int] = None) -> list[float]:
        if not batch_size:
            batch_size = self.batch_size
        tokens = self._tokenize(x)
        dataset = TokenizedDataset(tokens)
        dataloader = DataLoader(
            dataset, sampler=SequentialSampler(dataset), batch_size=batch_size, collate_fn=self.collate_fn
        )
        total_predictions = []

        # deactivate dropout layers
        self.neural_network.eval()
        for step, batch in enumerate(dataloader):
            # deactivate autograd
            with torch.no_grad():
                predictions = self._evaluate_single_batch(batch)
                total_predictions.extend(predictions.tolist())
        return total_predictions

    @abstractmethod
    def _tokenize(self, texts: list[str]) -> BatchEncoding:
        pass

    def _train_single_epoch(self, dataloader: DataLoader, optimizer: Optimizer) -> None:
        self.neural_network.train()
        cross_entropy = BCELoss()

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            labels = batch[-1].float().cpu()
            predictions = self._evaluate_single_batch(batch)

            loss = cross_entropy(predictions, labels)
            loss.backward()
            optimizer.step()

    @abstractmethod
    def _evaluate_single_batch(self, batch: tuple[Tensor]) -> Tensor:
        pass

    def save(self, model_dir: str) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        params = {"batch_size": self.batch_size, "learning_rate": self.learning_rate, "epochs": self.epochs}
        with open(file=model_dir / "params.json", mode="w", encoding="utf-8") as file:
            json.dump(params, file)
        self.tokenizer.save_pretrained(model_dir)
        if self.many_gpus:
            torch.save(self.neural_network.module, model_dir / "model.bin")
        else:
            torch.save(self.neural_network, model_dir / "model.bin")

    @classmethod
    def load(cls, model_dir: str, device: str = "cuda:0", many_gpus: bool = False) -> BertClassifier:
        model_dir = Path(model_dir)
        with open(file=model_dir / "params.json", mode="r", encoding="utf-8") as file:
            params = json.load(file)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        neural_network = torch.load(f=model_dir / "model.bin", map_location=device)
        return cls(
            **params,
            tokenizer=tokenizer,
            neural_network=neural_network,
            pretrained_model_name_or_path=None,
            device=device,
            many_gpus=many_gpus,
        )


class BertClassifierNN(Module):
    def __init__(self, model: Union[BertModel, RobertaModel]):
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
