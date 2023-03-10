from typing import Optional

import torch
from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizerBase

from lib.entities.exceptions import InconsistentSplittingParamsException


# Functions for preparing input for longer texts - based on
# https://www.kdnuggets.com/2021/04/apply-transformers-any-length-text.html


def transform_list_of_texts(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    stride: int,
    minimal_chunk_length: int,
    maximal_text_length: Optional[int] = None,
) -> BatchEncoding:
    model_inputs = [
        transform_single_text(text, tokenizer, chunk_size, stride, minimal_chunk_length, maximal_text_length)
        for text in texts
    ]
    input_ids = [model_input[0] for model_input in model_inputs]
    attention_mask = [model_input[1] for model_input in model_inputs]
    tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
    return BatchEncoding(tokens)


def transform_single_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    stride: int,
    minimal_chunk_length: int,
    maximal_text_length: Optional[int],
) -> tuple[Tensor, Tensor]:
    """Transforms (the entire) text to model input of BERT model."""
    if maximal_text_length:
        tokens = tokenize_text_with_truncation(text, tokenizer, maximal_text_length)
    else:
        tokens = tokenize_whole_text(text, tokenizer)
    input_id_chunks, mask_chunks = split_tokens_into_smaller_chunks(tokens, chunk_size, stride, minimal_chunk_length)
    add_special_tokens_at_beginning_and_end(input_id_chunks, mask_chunks)
    add_padding_tokens(input_id_chunks, mask_chunks)
    input_ids, attention_mask = stack_tokens_from_all_chunks(input_id_chunks, mask_chunks)
    return input_ids, attention_mask


def tokenize_whole_text(text: str, tokenizer: PreTrainedTokenizerBase) -> BatchEncoding:
    """Tokenizes the entire text without truncation and without special tokens."""
    tokens = tokenizer(text, add_special_tokens=False, truncation=False, return_tensors="pt")
    return tokens


def tokenize_text_with_truncation(
    text: str, tokenizer: PreTrainedTokenizerBase, maximal_text_length: int
) -> BatchEncoding:
    """Tokenizes the text with truncation to maximal_text_length and without special tokens."""
    tokens = tokenizer(
        text, add_special_tokens=False, max_length=maximal_text_length, truncation=True, return_tensors="pt"
    )
    return tokens


def split_tokens_into_smaller_chunks(
    tokens: BatchEncoding,
    chunk_size: int,
    stride: int,
    minimal_chunk_length: int,
) -> tuple[list[Tensor], list[Tensor]]:
    """Splits tokens into overlapping chunks with given size and step."""
    input_id_chunks = split_overlapping(tokens["input_ids"][0], chunk_size, stride, minimal_chunk_length)
    mask_chunks = split_overlapping(tokens["attention_mask"][0], chunk_size, stride, minimal_chunk_length)
    return input_id_chunks, mask_chunks


def add_special_tokens_at_beginning_and_end(input_id_chunks: list[Tensor], mask_chunks: list[Tensor]) -> None:
    """
    Adds special CLS token (token id = 101) at the beginning.
    Adds SEP token (token id = 102) at the end of each chunk.
    Adds corresponding attention masks equal to 1 (attention mask is boolean).
    """
    for i in range(len(input_id_chunks)):
        # adding CLS (token id 101) and SEP (token id 102) tokens
        input_id_chunks[i] = torch.cat([Tensor([101]), input_id_chunks[i], Tensor([102])])
        # adding attention masks  corresponding to special tokens
        mask_chunks[i] = torch.cat([Tensor([1]), mask_chunks[i], Tensor([1])])


def add_padding_tokens(input_id_chunks: list[Tensor], mask_chunks: list[Tensor]) -> None:
    """Adds padding tokens (token id = 0) at the end to make sure that all chunks have exactly 512 tokens."""
    for i in range(len(input_id_chunks)):
        # get required padding length
        pad_len = 512 - input_id_chunks[i].shape[0]
        # check if tensor length satisfies required chunk size
        if pad_len > 0:
            # if padding length is more than 0, we must add padding
            input_id_chunks[i] = torch.cat([input_id_chunks[i], Tensor([0] * pad_len)])
            mask_chunks[i] = torch.cat([mask_chunks[i], Tensor([0] * pad_len)])


def stack_tokens_from_all_chunks(input_id_chunks: list[Tensor], mask_chunks: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Reshapes data to a form compatible with BERT model input."""
    input_ids = torch.stack(input_id_chunks)
    attention_mask = torch.stack(mask_chunks)

    return input_ids.long(), attention_mask.int()


def split_overlapping(tensor: Tensor, chunk_size: int, stride: int, minimal_chunk_length: int) -> list[Tensor]:
    """Helper function for dividing 1-dimensional tensors into overlapping chunks."""
    check_split_parameters_consistency(chunk_size, stride, minimal_chunk_length)
    result = [tensor[i : i + chunk_size] for i in range(0, len(tensor), stride)]
    if len(result) > 1:
        # ignore chunks with less than minimal_length number of tokens
        result = [x for x in result if len(x) >= minimal_chunk_length]
    return result


def check_split_parameters_consistency(chunk_size: int, stride: int, minimal_chunk_length: int) -> None:
    if chunk_size > 510:
        raise InconsistentSplittingParamsException("Size of each chunk cannot be bigger than 510!")
    if minimal_chunk_length > chunk_size:
        raise InconsistentSplittingParamsException("Minimal length cannot be bigger than size!")
    if stride > chunk_size:
        raise InconsistentSplittingParamsException(
            "Step cannot be bigger than size! Chunks must overlap or be near each other!"
        )
