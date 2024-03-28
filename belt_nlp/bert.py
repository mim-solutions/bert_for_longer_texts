from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, DataParallel, Linear, Module, MSELoss
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import AutoModel, AutoTokenizer, BatchEncoding, BertModel, PreTrainedTokenizerBase, RobertaModel


class BertClassifier(ABC):
    """
    The "device" parameter can have the following values:
        - "cpu" - The model will be loaded on CPU.
        - "cuda" - The model will be loaded on single GPU.
        - "cuda:i" - The model will be loaded on the specific single GPU with the index i.

    It is also possible to use multiple GPUs. In order to do this:
        - Set device to "cuda".
        - Set many_gpu flag to True.
        - As default it will use all of them.

    To use only selected GPUs - set the environmental variable CUDA_VISIBLE_DEVICES.
    """

    @abstractmethod
    def __init__(
        self,
        num_labels: int,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        accumulation_steps: int = 1,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        neural_network: Optional[Module] = None,
        pretrained_model_name_or_path: Optional[str] = "bert-base-uncased",
        device: str = "cuda:0",
        many_gpus: bool = False,
    ):
        self.num_labels = num_labels

        if not tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        if not neural_network:
            bert = AutoModel.from_pretrained(pretrained_model_name_or_path)
            neural_network = BertClassifierNN(model=bert, num_labels=num_labels)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps
        self._params = {
            "num_labels": self.num_labels,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "accumulation_steps": self.accumulation_steps,
        }
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

    def predict_logits(self, x: list[str], batch_size: Optional[int] = None) -> Tensor:
        """Returns classification (or regression if num_labels==1) scores (before SoftMax)."""
        if not batch_size:
            batch_size = self.batch_size
        tokens = self._tokenize(x)
        dataset = TokenizedDataset(tokens)
        dataloader = DataLoader(
            dataset, sampler=SequentialSampler(dataset), batch_size=batch_size, collate_fn=self.collate_fn
        )
        total_logits = []

        # deactivate dropout layers
        self.neural_network.eval()
        for step, batch in enumerate(dataloader):
            # deactivate autograd
            with torch.no_grad():
                logits = self._evaluate_single_batch(batch)
                total_logits.append(logits)
        return torch.cat(total_logits)

    @abstractmethod
    def _tokenize(self, texts: list[str]) -> BatchEncoding:
        pass

    def _train_single_epoch(self, dataloader: DataLoader, optimizer: Optimizer) -> None:
        self.neural_network.train()

        for step, batch in enumerate(dataloader):
            if self.num_labels > 1:
                labels = batch[-1].long().to(self.device)
                loss_function = CrossEntropyLoss()
                logits = self._evaluate_single_batch(batch)
                loss = loss_function(logits, labels) / self.accumulation_steps
            elif self.num_labels == 1:
                labels = batch[-1].float().to(self.device)
                loss_function = MSELoss()
                scores = torch.flatten(self._evaluate_single_batch(batch))
                loss = loss_function(scores, labels) / self.accumulation_steps
            loss.backward()

            if ((step + 1) % self.accumulation_steps == 0) or (step + 1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()

    @abstractmethod
    def _evaluate_single_batch(self, batch: tuple[Tensor]) -> Tensor:
        pass

    def save(self, model_dir: str) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(file=model_dir / "params.json", mode="w", encoding="utf-8") as file:
            json.dump(self._params, file)
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
    def __init__(self, model: Union[BertModel, RobertaModel], num_labels: int):
        super().__init__()
        self.model = model

        # classification head
        self.linear = Linear(768, num_labels)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        x = self.model(input_ids, attention_mask)
        x = x[0][:, 0, :]  # take <s> token (equiv. to [CLS])

        # classification head
        x = self.linear(x)
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
