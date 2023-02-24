from typing import NamedTuple


class TrainingInfoForSingleEpoch(NamedTuple):
    loss_train: float
    predictions_val: list[float]
