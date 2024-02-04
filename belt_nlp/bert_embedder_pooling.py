import torch

from belt_nlp.bert_with_pooling import BertClassifier
from typing import Optional, List
from torch import Tensor
from torch.nn import Module
from transformers import PreTrainedTokenizerBase, BatchEncoding
from belt_nlp.splitting import transform_list_of_texts


class BertEmbeddingGenerator(BertClassifier):
    def __init__(
        self,
        chunk_size: int,
        stride: int,
        minimal_chunk_length: int,
        pooling_strategy: str = "mean",
        maximal_text_length: Optional[int] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        neural_network: Optional[Module] = None,
        pretrained_model_name_or_path: Optional[str] = "bert-base-uncased",
        trust_remote_code: Optional[bool] = False,
        device: str = "cuda:0",
        many_gpus: bool = False,
    ):

        super().__init__(
            tokenizer=tokenizer,
            neural_network=neural_network,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            device=device,
            many_gpus=many_gpus
        )

        self.chunk_size = chunk_size
        self.stride = stride
        self.minimal_chunk_length = minimal_chunk_length
        self.maximal_text_length = maximal_text_length
        self.pooling_strategy = pooling_strategy
        if pooling_strategy not in ["mean", "max"]:
            raise ValueError("Unknown pooling strategy!")

        self.collate_fn = BertEmbeddingGenerator.collate_fn_pooled_tokens

    def _tokenize(self, texts: list[str]) -> BatchEncoding:
        """
        Transforms list of N texts to the BatchEncoding, that is the dictionary with the following keys:
            - input_ids - List of N tensors of the size K(i) x 512 of token ids.
                K(i) is the number of chunks of the text i.
                Each element of the list is stacked Tensor for encoding of each chunk.
                Values of the tensor are integers.
            - attention_mask - List of N tensors of the size K(i) x 512 of attention masks.
                K(i) is the number of chunks of the text i.
                Each element of the list is stacked Tensor for encoding of each chunk.
                Values of the tensor are booleans.

        These lists of tensors cannot be stacked into one tensor,
        because each text can be divided into different number of chunks.
        """
        tokens = transform_list_of_texts(
            texts, self.tokenizer, self.chunk_size, self.stride, self.minimal_chunk_length, self.maximal_text_length
        )
        return tokens

    def get_embeddings(self, documents: List[str]) -> List[Tensor]:

        all_embeddings = []
        for document in documents:
            tokens = self._tokenize([document])

            input_ids, attention_masks = tokens["input_ids"], tokens["attention_mask"]

            # Process each document's chunks and pool their embeddings
            document_embedding = self.process_and_pool_chunks((input_ids, attention_masks))
            all_embeddings.append(document_embedding)

        return torch.stack(all_embeddings).tolist()

    def process_and_pool_chunks(self, batch: tuple[Tensor]):
        input_ids = batch[0][0].to(self.device)
        attention_mask = batch[1][0].to(self.device)

        model_output = self.neural_network(input_ids, attention_mask=attention_mask, return_embeddings=True)
        sequence_output = model_output[:, 0, :] # Taking CLS token as my pretrained model performs better with it

        if self.pooling_strategy == "mean":
            pooled_output = torch.mean(sequence_output, dim=0).detach().cpu()
        elif self.pooling_strategy == "max":
            pooled_output = torch.max(sequence_output, dim=0).values
        else:
            raise ValueError("Unknown pooling strategy!")

        return pooled_output

    def _evaluate_single_batch(self, batch: tuple[Tensor]) -> Tensor:
        pass

    @staticmethod
    def collate_fn_pooled_tokens(data):
        input_ids = [data[i][0] for i in range(len(data))]
        attention_mask = [data[i][1] for i in range(len(data))]
        if len(data[0]) == 2:
            collated = [input_ids, attention_mask]
        else:
            labels = Tensor([data[i][2] for i in range(len(data))])
            collated = [input_ids, attention_mask, labels]
        return collated




