from lib.entities.exceptions import InconsinstentSplitingParamsException

import pytest
from torch import equal, Tensor

from lib.entities.text_split_params import TextSplitParams
from lib.model.splitting import split_overlapping

EXAMPLE_TENSOR = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def test_split_overlapping():
    spliting_params = TextSplitParams(chunk_size=5, stride=5, minimal_chunk_length=5)
    expected_result = [Tensor([1, 2, 3, 4, 5]), Tensor([6, 7, 8, 9, 10])]
    splitted = split_overlapping(EXAMPLE_TENSOR, spliting_params)
    assert list_of_tensors_deep_equal(splitted, expected_result)

    spliting_params = TextSplitParams(chunk_size=5, stride=1, minimal_chunk_length=5)
    expected_result = [
        Tensor([1, 2, 3, 4, 5]),
        Tensor([2, 3, 4, 5, 6]),
        Tensor([3, 4, 5, 6, 7]),
        Tensor([4, 5, 6, 7, 8]),
        Tensor([5, 6, 7, 8, 9]),
        Tensor([6, 7, 8, 9, 10]),
    ]
    splitted = split_overlapping(EXAMPLE_TENSOR, spliting_params)
    assert list_of_tensors_deep_equal(splitted, expected_result)

    spliting_params = TextSplitParams(chunk_size=5, stride=1, minimal_chunk_length=3)
    expected_result = [
        Tensor([1, 2, 3, 4, 5]),
        Tensor([2, 3, 4, 5, 6]),
        Tensor([3, 4, 5, 6, 7]),
        Tensor([4, 5, 6, 7, 8]),
        Tensor([5, 6, 7, 8, 9]),
        Tensor([6, 7, 8, 9, 10]),
        Tensor([7, 8, 9, 10]),
        Tensor([8, 9, 10]),
    ]
    splitted = split_overlapping(EXAMPLE_TENSOR, spliting_params)
    assert list_of_tensors_deep_equal(splitted, expected_result)

    spliting_params = TextSplitParams(chunk_size=9, stride=1, minimal_chunk_length=3)
    expected_result = [
        Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        Tensor([2, 3, 4, 5, 6, 7, 8, 9, 10]),
        Tensor([3, 4, 5, 6, 7, 8, 9, 10]),
        Tensor([4, 5, 6, 7, 8, 9, 10]),
        Tensor([5, 6, 7, 8, 9, 10]),
        Tensor([6, 7, 8, 9, 10]),
        Tensor([7, 8, 9, 10]),
        Tensor([8, 9, 10]),
    ]
    splitted = split_overlapping(EXAMPLE_TENSOR, spliting_params)
    assert list_of_tensors_deep_equal(splitted, expected_result)


def test_too_large_chunk_exception():
    spliting_params = TextSplitParams(chunk_size=511, stride=2, minimal_chunk_length=1)

    with pytest.raises(InconsinstentSplitingParamsException):
        split_overlapping(EXAMPLE_TENSOR, spliting_params)


def test_stride_larger_than_chunk_size_exception():
    spliting_params = TextSplitParams(chunk_size=3, stride=4, minimal_chunk_length=1)

    with pytest.raises(InconsinstentSplitingParamsException):
        split_overlapping(EXAMPLE_TENSOR, spliting_params)


def test_minimal_length_larger_than_chunk_size_exception():
    spliting_params = TextSplitParams(chunk_size=3, stride=2, minimal_chunk_length=4)

    with pytest.raises(InconsinstentSplitingParamsException):
        split_overlapping(EXAMPLE_TENSOR, spliting_params)


def list_of_tensors_deep_equal(list_1: list[Tensor], list_2: list[Tensor]) -> bool:
    if len(list_1) != len(list_2):
        return False
    for x, y in zip(list_1, list_2):
        if not equal(x, y):
            return False
    return True
