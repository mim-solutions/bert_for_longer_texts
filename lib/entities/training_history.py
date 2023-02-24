from typing import List, NamedTuple


class TrainingInfoForSingleEpoch(NamedTuple):
    loss_train: float
    predictions_val: List[float]
