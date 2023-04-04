import pytest
from torch import Tensor

from belt_nlp.exceptions import InconsistentSplittingParamsException
from belt_nlp.splitting import split_overlapping
from belt_nlp.tensor_utils import list_of_tensors_deep_equal

EXAMPLE_TENSOR = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def test_split_overlapping():
    expected_result = [Tensor([1, 2, 3, 4, 5]), Tensor([6, 7, 8, 9, 10])]
    splitted = split_overlapping(EXAMPLE_TENSOR, chunk_size=5, stride=5, minimal_chunk_length=5)
    assert list_of_tensors_deep_equal(splitted, expected_result)

    expected_result = [
        Tensor([1, 2, 3, 4, 5]),
        Tensor([2, 3, 4, 5, 6]),
        Tensor([3, 4, 5, 6, 7]),
        Tensor([4, 5, 6, 7, 8]),
        Tensor([5, 6, 7, 8, 9]),
        Tensor([6, 7, 8, 9, 10]),
    ]
    splitted = split_overlapping(EXAMPLE_TENSOR, chunk_size=5, stride=1, minimal_chunk_length=5)
    assert list_of_tensors_deep_equal(splitted, expected_result)

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
    splitted = split_overlapping(EXAMPLE_TENSOR, chunk_size=5, stride=1, minimal_chunk_length=3)
    assert list_of_tensors_deep_equal(splitted, expected_result)

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
    splitted = split_overlapping(EXAMPLE_TENSOR, chunk_size=9, stride=1, minimal_chunk_length=3)
    assert list_of_tensors_deep_equal(splitted, expected_result)


def test_too_large_chunk_exception():
    with pytest.raises(InconsistentSplittingParamsException):
        split_overlapping(EXAMPLE_TENSOR, chunk_size=511, stride=2, minimal_chunk_length=1)


def test_stride_larger_than_chunk_size_exception():
    with pytest.raises(InconsistentSplittingParamsException):
        split_overlapping(EXAMPLE_TENSOR, chunk_size=3, stride=4, minimal_chunk_length=1)


def test_minimal_length_larger_than_chunk_size_exception():
    with pytest.raises(InconsistentSplittingParamsException):
        split_overlapping(EXAMPLE_TENSOR, chunk_size=3, stride=2, minimal_chunk_length=4)
