from __future__ import annotations
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
        pretrained_model_name_or_path: Optional[str] = "bert-base-uncased",
        device: str = "cuda:0",
        many_gpus: bool = False,
    ):
        super().__init__(params, tokenizer, neural_network, pretrained_model_name_or_path, device, many_gpus)

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

    @classmethod
    def load(cls, model_dir: str, device: str = "cuda:0", many_gpus: bool = False) -> BertClassifierTruncated:
        model = super().load(model_dir, device, many_gpus)
        return cls(
            params=model.params,
            tokenizer=model.tokenizer,
            neural_network=model.neural_network,
            pretrained_model_name_or_path=None,
            device=model.device,
            many_gpus=model.many_gpus,
        )
