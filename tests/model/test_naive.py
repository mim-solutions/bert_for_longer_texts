from pathlib import Path
from shutil import rmtree

from lib.model.naive import NaiveClassifier


def test_predict():
    model = NaiveClassifier(1)
    predictions = model.predict(["a", "b", "c"])

    assert len(predictions) == 3
    for cls, score in predictions:
        assert cls
        assert score == 1.0


def test_save_and_load():
    model = NaiveClassifier(0.6)
    path = Path("tmp_naive_model_test_dir")

    model.save(str(path))

    try:
        model_loaded = NaiveClassifier.load(str(path))
        assert model_loaded.probability == model.probability
    finally:
        rmtree(path)
