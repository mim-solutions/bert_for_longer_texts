from typing import NamedTuple


class LossesSingleEpoch(NamedTuple):
    loss_train: float
    loss_val: float
