from torch import equal, Tensor

from lib.entities.text_split_params import TextSplitParams
from lib.model.splitting import split_overlapping


def test_split_overlapping():
    example_list = Tensor([1, 2, 3, 4, 5])
    spliting_params = TextSplitParams(chunk_size=3, stride=2, minimal_chunk_length=1)
    expected_result = [Tensor([1, 2, 3]), Tensor([3, 4, 5]), Tensor([5])]

    splitted = split_overlapping(example_list, spliting_params)
    assert list_of_tensors_deep_equal(splitted, expected_result)


def list_of_tensors_deep_equal(list_1: list[Tensor], list_2: list[Tensor]) -> bool:
    if len(list_1) != len(list_2):
        return False
    for x, y in zip(list_1, list_2):
        if not equal(x, y):
            return False
    return True
