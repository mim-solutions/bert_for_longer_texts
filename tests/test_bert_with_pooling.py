from pathlib import Path
from shutil import rmtree

import numpy as np
import torch

from belt_nlp.bert_classifier_with_pooling import BertClassifierWithPooling

MODEL_PARAMS = {
    "num_labels": 2,
    "batch_size": 1,
    "learning_rate": 5e-5,
    "epochs": 1,
    "chunk_size": 510,
    "stride": 256,
    "minimal_chunk_length": 1,
    "pooling_strategy": "mean",
    "device": "cpu",
}


def test_fit_and_predict():
    """The test is quite naive, but it goes through all the methods."""
    params = MODEL_PARAMS
    model = BertClassifierWithPooling(**params)
    x_train = ["carrot", "cucumber", "tomato", "potato"]
    y_train = [1] * len(x_train)

    x_test = ["pepper", "eggplant"]
    expected_classes = [1, 1]

    model.fit(x_train, y_train)
    classes = model.predict(x_test)
    scores = model.predict_scores(x_test)

    assert scores.shape == torch.Size([2, 2])

    assert np.array_equal(classes, expected_classes)


def test_prediction_order():
    """Check if the order of predictions is preserved."""
    params = MODEL_PARAMS
    model = BertClassifierWithPooling(**params)
    x_train = ["carrot", "cucumber", "tomato", "potato"]
    y_train = [1] * len(x_train)

    x_test = ["pepper"] * 99 + ["chair"] * 1

    model.fit(x_train, y_train)
    predicted_scores = model.predict_scores(x_test)

    expected_score_pepper = model.predict_scores(["pepper"])
    expected_score_chair = model.predict_scores(["chair"])

    expected_result_tensor = torch.cat([expected_score_pepper] * 99 + [expected_score_chair])

    assert torch.equal(predicted_scores, expected_result_tensor)


def test_save_and_load():
    params = MODEL_PARAMS
    model = BertClassifierWithPooling(**params)
    path = Path("tmp_bert_model_test_dir")

    model.save(str(path))

    try:
        model_loaded = BertClassifierWithPooling.load(str(path))
        # assert types to be more specific than 'isinstance()'
        assert type(model_loaded.tokenizer) == type(model.tokenizer)  # noqa: E721
        assert type(model_loaded.neural_network) == type(model.neural_network)  # noqa: E721
    finally:
        rmtree(path)
