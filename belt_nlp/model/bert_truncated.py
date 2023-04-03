from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor
from torch.nn import Module
from transformers import BatchEncoding, PreTrainedTokenizerBase

from belt_nlp.model.bert import BertClassifier


class BertClassifierTruncated(BertClassifier):
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
        super().__init__(
            batch_size,
            learning_rate,
            epochs,
            tokenizer,
            neural_network,
            pretrained_model_name_or_path,
            device,
            many_gpus,
        )

    def _tokenize(self, texts: list[str]) -> BatchEncoding:
        """
        Transforms list of N texts to the BatchEncoding, that is the dictionary with the following keys:
            - input_ids - Tensor of the size N x 512 of token ids.
                This is stacked Tensor of encodings of each text.
                Values of the tensor are integers.
            - attention_mask - Tensor of the size N x 512 of attention masks.
                This is stacked Tensor of encodings of each text.
                Values of the tensor are booleans.

        If the text is longer than 512 tokens - the rest of it is ignored.
        If the text is shorter than 512 tokens - it is padded to have exactly 512 tokens.
        """
        tokens = self.tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
        return tokens

    def _evaluate_single_batch(self, batch: tuple[Tensor]) -> Tensor:
        batch = [t.to(self.device) for t in batch]
        model_input = batch[:2]

        predictions = self.neural_network(*model_input)
        predictions = torch.flatten(predictions).cpu()
        return predictions
