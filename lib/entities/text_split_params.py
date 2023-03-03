from typing import NamedTuple


class TextSplitParams(NamedTuple):
    chunk_size: int
    stride: int
    minimal_chunk_length: int
