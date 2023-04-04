from torch import equal, Tensor


def list_of_tensors_deep_equal(list_1: list[Tensor], list_2: list[Tensor]) -> bool:
    if len(list_1) != len(list_2):
        return False
    for x, y in zip(list_1, list_2):
        if not equal(x, y):
            return False
    return True
