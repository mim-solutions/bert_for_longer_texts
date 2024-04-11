from __future__ import annotations

from typing import Optional

from torch import argmax
from torch.nn import Module, Softmax
from transformers import PreTrainedTokenizerBase

from belt_nlp.bert_with_pooling import BertBaseWithPooling


class BertClassifierWithPooling(BertBaseWithPooling):
    def __init__(
        self,
        num_labels: int,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        chunk_size: int,
        stride: int,
        minimal_chunk_length: int,
        pooling_strategy: str = "mean",
        accumulation_steps: int = 1,
        maximal_text_length: Optional[int] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        neural_network: Optional[Module] = None,
        pretrained_model_name_or_path: Optional[str] = "bert-base-uncased",
        device: str = "cuda:0",
        many_gpus: bool = False,
    ):
        super().__init__(
            num_labels=num_labels,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            chunk_size=chunk_size,
            stride=stride,
            minimal_chunk_length=minimal_chunk_length,
            pooling_strategy=pooling_strategy,
            accumulation_steps=accumulation_steps,
            maximal_text_length=maximal_text_length,
            tokenizer=tokenizer,
            neural_network=neural_network,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            device=device,
            many_gpus=many_gpus,
        )

    def predict(self, x: list[str], batch_size: Optional[int] = None) -> list[int]:
        """Returns classes."""
        logits = super()._predict_logits(x, batch_size)
        classes = argmax(logits, dim=1).cpu().numpy()
        return classes

    def predict_scores(self, x: list[str], batch_size: Optional[int] = None) -> list[float]:
        """Returns classification probabilities."""
        logits = super()._predict_logits(x, batch_size)
        softmax = Softmax(dim=1)

        probabilities = softmax(logits)
        return probabilities
