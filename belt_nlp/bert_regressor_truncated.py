from __future__ import annotations

from typing import Optional

from torch import Tensor
from torch.nn import Module
from transformers import PreTrainedTokenizerBase

from belt_nlp.bert_truncated import BertBaseTruncated


class BertRegressorTruncated(BertBaseTruncated):
    def __init__(
        self,
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
        super().__init__(
            num_labels=1,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            accumulation_steps=accumulation_steps,
            tokenizer=tokenizer,
            neural_network=neural_network,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            device=device,
            many_gpus=many_gpus,
        )

    def predict(self, x: list[str], batch_size: Optional[int] = None) -> Tensor:
        """Returns regression scores."""
        logits = super()._predict_logits(x, batch_size)
        return logits
