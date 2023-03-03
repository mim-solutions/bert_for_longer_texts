from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module
from transformers import AutoTokenizer, BatchEncoding

from lib.entities.text_split_params import TextSplitParams
from lib.model.bert import BertClassifier
from lib.model.splitting import transform_list_of_texts


class BertClassifierWithPooling(BertClassifier):
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
        self.text_split_params = TextSplitParams(
            size=params["size"], step=params["step"], minimal_length=params["minimal_length"]
        )
        self.collate_fn = self.collate_fn_pooled_tokens
        self.pooling_strategy = params["pooling_strategy"]

    def _tokenize(self, texts: list[str]) -> BatchEncoding:
        """
        Transforms list of N texts to the BatchEncoding, that is the dictionary with the following keys:
        - input_ids - List of N tensors of the size K(i) x 512 of token ids.
        K(i) is the number of chunks of the text i.
        Each element of the list is stacked Tensor for encoding of each chunk.
        Values of the tensor are integers.
        - attention_mask - List of N tensors of the size K x 512 of token ids.
        K(i) is the number of chunks of the text i.
        Each element of the list is stacked Tensor for encoding of each chunk.
        This is stacked Tensor for encoding of each text.
        Values of the tensor are booleans.
        If the text is longer than 512 tokens - the rest of it is ignored.
        If the text is shorter than 512 tokens - it is padded to have exactly 512 tokens.
        These lists of tensors cannnot be stacked into one tensor,
        because each text can be divided into different number of chunks
        """
        tokens = transform_list_of_texts(texts, self.tokenizer, self.text_split_params)
        return tokens

    def _evaluate_single_batch(self, batch: tuple[Tensor]) -> Tensor:
        input_ids = batch[0]
        attention_mask = batch[1]
        number_of_chunks = [len(x) for x in input_ids]

        # concatenate all input_ids into one batch

        input_ids_combined = []
        for x in input_ids:
            input_ids_combined.extend(x.tolist())

        input_ids_combined_tensors = torch.stack([torch.tensor(x).to(self.device) for x in input_ids_combined])

        # concatenate all attention masks into one batch

        attention_mask_combined = []
        for x in attention_mask:
            attention_mask_combined.extend(x.tolist())

        attention_mask_combined_tensors = torch.stack(
            [torch.tensor(x).to(self.device) for x in attention_mask_combined]
        )

        # get model predictions for the combined batch
        preds = self.neural_network(input_ids_combined_tensors, attention_mask_combined_tensors)

        preds = preds.flatten().cpu()

        # split result preds into chunks

        preds_split = preds.split(number_of_chunks)

        # pooling
        if self.params["pooling_strategy"] == "mean":
            pooled_preds = torch.cat([torch.mean(x).reshape(1) for x in preds_split])
        elif self.params["pooling_strategy"] == "max":
            pooled_preds = torch.cat([torch.max(x).reshape(1) for x in preds_split])

        return pooled_preds

    @classmethod
    def load(cls, model_dir: str, device: str = "cuda:0", many_gpus: bool = False) -> BertClassifierWithPooling:
        model = super().load(model_dir, device, many_gpus)
        return cls(
            params=model.params,
            tokenizer=model.tokenizer,
            neural_network=model.neural_network,
            pretrained_model_name_or_path=None,
            device=model.device,
            many_gpus=model.many_gpus,
        )

    def collate_fn_pooled_tokens(self, data):
        input_ids = [data[i][0] for i in range(len(data))]
        attention_mask = [data[i][1] for i in range(len(data))]
        if len(data[0]) == 2:
            collated = [input_ids, attention_mask]
        else:
            labels = Tensor([data[i][2] for i in range(len(data))])
            collated = [input_ids, attention_mask, labels]
        return collated
