import matplotlib.pyplot as plt

from lib.entities.learning_curve import LossesSingleEpoch


def plot_learning_curve(train_history: list[LossesSingleEpoch]):
    metrics = get_learning_curve(train_history)
    plot_metrics(metrics)


def get_learning_curve(train_history: list[LossesSingleEpoch]) -> dict[str, list[float]]:
    result = {}
    result["train_loss"] = [training_info.loss_train for training_info in train_history]
    result["val_loss"] = [training_info.loss_val for training_info in train_history]
    return result


def plot_metrics(metrics: dict[str, list[float]]):
    cmap = plt.get_cmap("tab10")
    _, ax = plt.subplots(figsize=(10, 10))

    for i, (key, value) in enumerate(metrics.items()):
        ax.plot(value, "-", label=key, color=cmap(i))
        ax.legend()
