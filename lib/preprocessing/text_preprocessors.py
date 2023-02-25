from abc import ABC, abstractmethod

from transformers import AutoTokenizer, BatchEncoding

from lib.entities.text_split_params import TextSplitParams
from lib.pooling import transform_text_to_model_input


class Preprocessor(ABC):
    @abstractmethod
    def preprocess(self, array_of_texts: list[str]) -> BatchEncoding:
        pass


class BERTTokenizer(Preprocessor):
    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self.tokenizer = tokenizer

    def preprocess(self, array_of_texts: list[str]) -> BatchEncoding:
        tokens = tokenize_truncated(array_of_texts, self.tokenizer)
        return tokens


class BERTTokenizerPooled(Preprocessor):
    def __init__(self, tokenizer: AutoTokenizer, text_split_params: TextSplitParams) -> None:
        self.tokenizer = tokenizer
        self.text_splits_params = text_split_params

    def preprocess(self, array_of_texts: list[str]) -> BatchEncoding:
        array_of_preprocessed_data = tokenize_pooled(array_of_texts, self.tokenizer, *self.text_splits_params)
        return array_of_preprocessed_data


def tokenize_truncated(texts: list[str], tokenizer: AutoTokenizer) -> BatchEncoding:
    """
    Transforms list of texts to list of tokens (truncated to 512 tokens)
    """
    tokenizer.pad_token = "<pad>"
    tokens = tokenizer.batch_encode_plus(texts, max_length=512, padding=True, truncation=True, return_tensors="pt")

    return tokens


def tokenize_pooled(texts: list[str], tokenizer: AutoTokenizer, text_split_params: TextSplitParams) -> BatchEncoding:
    """
    Tokenizes texts and splits to chunks of 512 tokens
    """
    model_inputs = [transform_text_to_model_input(text, tokenizer, text_split_params) for text in texts]
    input_ids = [model_input[0] for model_input in model_inputs]
    attention_mask = [model_input[1] for model_input in model_inputs]
    tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
    return BatchEncoding(tokens)
