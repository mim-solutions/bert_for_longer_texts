from lib.model.bert import load_tokenizer
from lib.preprocessing.tokenize import tokenize_truncated


def test_tokenize_truncated():
    texts = ["Hello world"]
    tokenizer = load_tokenizer()
    expected_input_ids = [101, 7592, 2088, 102]

    tokens = tokenize_truncated(texts, tokenizer)
    input_ids = tokens['input_ids'][0].numpy().tolist()

    assert input_ids == expected_input_ids
