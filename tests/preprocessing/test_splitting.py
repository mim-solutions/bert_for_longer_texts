from lib.entities.text_split_params import TextSplitParams
from lib.preprocessing.splitting import split_overlapping


def test_split_overlapping():
    example_list = [1, 2, 3, 4, 5]
    spliting_params = TextSplitParams(size=3, step=2, minimal_length=1)
    expected_result = [[1, 2, 3], [3, 4, 5], [5]]

    splitted = split_overlapping(example_list, spliting_params)
    assert splitted == expected_result
