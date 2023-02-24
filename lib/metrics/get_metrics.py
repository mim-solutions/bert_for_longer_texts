import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss

from lib.entities.training_history import TrainingInfoForSingleEpoch


def plot_loss_curves(train_history: list[TrainingInfoForSingleEpoch], y_val: list[bool]):
    metrics = get_metrics(train_history, y_val)
    plot_metrics(metrics)


def get_metrics(train_history: list[TrainingInfoForSingleEpoch], y_val: list[bool]) -> dict[str, list[float]]:
    result = {}
    result["train_loss"] = [training_info.loss_train for training_info in train_history]
    result["val_loss"] = [log_loss(y_val, training_info.predictions_val) for training_info in train_history]
    result["val_accuracy"] = [accuracy_score(y_val, training_info.predictions_val) for training_info in train_history]
    return result


def plot_metrics(metrics: dict[str, list[float]]):
    cmap = plt.get_cmap("tab10")
    _, ax = plt.subplots(figsize=(10, 10))

    for i, (key, value) in enumerate(metrics.items()):
        ax.plot(value, "-", label=key, color=cmap(i))
        ax.legend()
