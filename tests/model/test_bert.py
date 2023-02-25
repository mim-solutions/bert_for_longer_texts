from pathlib import Path
from shutil import rmtree

from lib.model.bert_truncated import BertClassifierTruncated


def test_fit_and_predict():
    """The test is quite naive, but it goes through all the methods."""
    params = {"batch_size": 1, "learning_rate": 5e-5, "epochs": 1}
    model = BertClassifierTruncated(params, device='cpu')
    x_train = ['carrot', 'cucumber', 'tomato', 'potato']
    y_train = [True] * len(x_train)

    x_test = ['pepper', 'eggplant']
    expected_classes = [True, True]

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    assert len(predictions) == 2
    assert [x[0] for x in predictions] == expected_classes
    for cls, score in predictions:
        assert isinstance(cls, bool)
        assert isinstance(score, float)


def test_prediction_order():
    """Check if the order of predictions is preserved."""
    params = {"batch_size": 1, "learning_rate": 5e-5, "epochs": 1}
    model = BertClassifierTruncated(params, device='cpu')
    x_train = ['carrot', 'cucumber', 'tomato', 'potato']
    y_train = [True] * len(x_train)

    x_test = ['pepper'] * 99 + ['chair'] * 1

    model.fit(x_train, y_train)
    predicted_scores = model.predict_scores(x_test)

    expected_score_pepper = model.predict_scores(['pepper'])[0]
    expected_score_chair = model.predict_scores(['chair'])[0]

    # Test if only last prediction is different
    # This will fail if the predictions are shuffled
    for i in range(len(x_test) - 1):
        assert predicted_scores[i] == expected_score_pepper
    assert predicted_scores[-1] == expected_score_chair


def test_save_and_load():
    params = {"batch_size": 1, "learning_rate": 5e-5, "epochs": 1}
    model = BertClassifierTruncated(params, device='cpu')
    path = Path('tmp_roberta_model_test_dir')

    model.save(str(path))

    try:
        model_loaded = BertClassifierTruncated.load(str(path), device='cpu')
        assert model_loaded.params == params
        # assert types to be more specific than 'isinstance()'
        assert type(model_loaded.tokenizer) == type(model.tokenizer)  # noqa: E721
        assert type(model_loaded.neural_network) == type(model.neural_network)  # noqa: E721
    finally:
        rmtree(path)
