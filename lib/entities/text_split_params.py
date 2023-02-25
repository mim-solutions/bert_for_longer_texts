from typing import NamedTuple


class TextSplitParams(NamedTuple):
    size: int
    step: int
    minimal_length: int
