from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module
from transformers import AutoTokenizer, BatchEncoding

from lib.model.bert import BertClassifier


class BertClassifierTruncated(BertClassifier):
    def __init__(
        self,
        params: dict,
        tokenizer: Optional[AutoTokenizer] = None,
        neural_network: Optional[Module] = None,
        device: str = "cuda:0",
        many_gpus: bool = False,
    ):
        super().__init__(params, tokenizer, neural_network, device, many_gpus)

    def _tokenize(self, texts: list[str]) -> BatchEncoding:
        """Transforms list of texts to list of tokens (truncated to 512 tokens)."""
        tokens = self.tokenizer.batch_encode_plus(
            texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
        )
        return tokens

    def _evaluate_single_batch(self, batch: tuple[Tensor]) -> Tensor:
        batch = [t.to(self.device) for t in batch]
        model_input = batch[:2]

        predictions = self.neural_network(*model_input)
        predictions = torch.flatten(predictions).cpu()
        return predictions

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

    @classmethod
    def load(cls, model_dir: str, device: str = "cuda:0", many_gpus: bool = False) -> BertClassifierTruncated:
        model_dir = Path(model_dir)
        with open(file=model_dir / "params.json", mode="r", encoding="utf-8") as file:
            params = json.load(file)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        neural_network = torch.load(f=model_dir / "model.bin", map_location=device)
        return cls(params, tokenizer, neural_network, device, many_gpus)
