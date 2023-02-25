from transformers import AutoTokenizer, BatchEncoding

from lib.entities.text_split_params import TextSplitParams
from lib.pooling import transform_text_to_model_input


def tokenize_truncated(texts: list[str], tokenizer: AutoTokenizer) -> BatchEncoding:
    """
    Transforms list of texts to list of tokens (truncated to 512 tokens)
    """
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
