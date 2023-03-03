from typing import TypeVar

import torch
from torch import Tensor
from transformers import AutoTokenizer, BatchEncoding

from lib.entities.exceptions import InconsinstentSplitingParamsException
from lib.entities.text_split_params import TextSplitParams

T = TypeVar("T")

# Functions for preparing input for longer texts - based on
# https://www.kdnuggets.com/2021/04/apply-transformers-any-length-text.html


def transform_list_of_texts(
    texts: list[str], tokenizer: AutoTokenizer, text_split_params: TextSplitParams
) -> BatchEncoding:
    model_inputs = [transform_single_text(text, tokenizer, text_split_params) for text in texts]
    input_ids = [model_input[0] for model_input in model_inputs]
    attention_mask = [model_input[1] for model_input in model_inputs]
    tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
    return BatchEncoding(tokens)


def transform_single_text(
    text: str, tokenizer: AutoTokenizer, text_split_params: TextSplitParams
) -> tuple[Tensor, Tensor]:
    """Transforms the entire text to model input of BERT model"""
    tokens = tokenize_all_text(text, tokenizer)
    input_id_chunks, mask_chunks = split_tokens_into_smaller_chunks(tokens, text_split_params)
    add_special_tokens_at_beginning_and_end(input_id_chunks, mask_chunks)
    add_padding_tokens(input_id_chunks, mask_chunks)
    input_ids, attention_mask = stack_tokens_from_all_chunks(input_id_chunks, mask_chunks)
    return input_ids, attention_mask


def tokenize_all_text(text: str, tokenizer: AutoTokenizer) -> BatchEncoding:
    """
    Tokenizes the entire text without truncation and without special tokens
    """
    tokens = tokenizer.encode_plus(text, add_special_tokens=False, return_tensors="pt")
    return tokens


def split_tokens_into_smaller_chunks(
    tokens: BatchEncoding, text_split_params: TextSplitParams
) -> tuple[list[Tensor], list[Tensor]]:
    """Splits tokens into overlapping chunks with given size and step"""
    assert text_split_params.size <= 510
    input_id_chunks = split_overlapping(tokens["input_ids"][0], text_split_params)
    mask_chunks = split_overlapping(tokens["attention_mask"][0], text_split_params)
    return input_id_chunks, mask_chunks


def add_special_tokens_at_beginning_and_end(input_id_chunks: list[Tensor], mask_chunks: list[Tensor]) -> None:
    """
    Adds special CLS token (token id = 101) at the beginning
    Adds SEP token (token id = 102) at the end of each chunk
    """
    for i in range(len(input_id_chunks)):
        # adding CLS (token id 101) and SEP (token id 102) tokens
        input_id_chunks[i] = torch.cat([Tensor([101]), input_id_chunks[i], Tensor([102])])
        mask_chunks[i] = torch.cat([Tensor([1]), mask_chunks[i], Tensor([1])])


def add_padding_tokens(input_id_chunks: list[Tensor], mask_chunks: list[Tensor]) -> None:
    """Adds padding tokens (token id = 0) at the end to make sure that all chunks have exactly 512 tokens"""
    for i in range(len(input_id_chunks)):
        # get required padding length
        pad_len = 512 - input_id_chunks[i].shape[0]
        # check if tensor length satisfies required chunk size
        if pad_len > 0:
            # if padding length is more than 0, we must add padding
            input_id_chunks[i] = torch.cat([input_id_chunks[i], Tensor([0] * pad_len)])
            mask_chunks[i] = torch.cat([mask_chunks[i], Tensor([0] * pad_len)])


def stack_tokens_from_all_chunks(input_id_chunks: list[Tensor], mask_chunks: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Reshapes data to a form compatible with BERT model input"""
    input_ids = torch.stack(input_id_chunks)
    attention_mask = torch.stack(mask_chunks)

    return input_ids.long(), attention_mask.int()


def split_overlapping(array: list[T], text_split_params) -> list[list[T]]:
    """Helper function for dividing arrays into overlapping chunks"""
    result = [array[i : i + text_split_params.size] for i in range(0, len(array), text_split_params.step)]
    if len(result) > 1:
        # ignore chunks with less then minimal_length number of tokens
        result = [x for x in result if len(x) >= text_split_params.minimal_length]
    return result


def check_split_parameters_consistency(text_split_params: TextSplitParams) -> None:
    if text_split_params.size > 510:
        raise InconsinstentSplitingParamsException("Size of each chunk cannot be bigger than 510!")
    if text_split_params.minimal_length > text_split_params.size:
        raise InconsinstentSplitingParamsException("Minimal length cannot be bigger than size!")
    if text_split_params.step > text_split_params.size:
        raise InconsinstentSplitingParamsException(
            "Step cannot be bigger than size! Chunks must overlap or be near each other!"
        )
