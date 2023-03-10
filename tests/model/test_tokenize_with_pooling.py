import pandas as pd

from lib.model.bert_with_pooling import BertClassifierWithPooling
from lib.model.tensor_utils import list_of_tensors_deep_equal

MODEL_PARAMS = {
    "batch_size": 1,
    "learning_rate": 5e-5,
    "epochs": 1,
    "chunk_size": 510,
    "stride": 510,
    "minimal_chunk_length": 510,
    "pooling_strategy": "mean",
}
MODEL = BertClassifierWithPooling(**MODEL_PARAMS, device="cpu")

SAMPLE_DATA_PATH = "sample_data/sample_data_eng.csv"
SHORT_REVIEW = pd.read_csv(SAMPLE_DATA_PATH).loc[0, "sentence"]
LONG_REVIEW = pd.read_csv(SAMPLE_DATA_PATH).loc[961, "sentence"]


def test_tokenize():
    tokens_short = MODEL._tokenize([SHORT_REVIEW])
    assert list(tokens_short["input_ids"][0].shape) == [1, 512]
    assert list(tokens_short["attention_mask"][0].shape) == [1, 512]

    tokens_long = MODEL._tokenize([LONG_REVIEW])
    expected_number_of_chunks = 3
    assert list(tokens_long["input_ids"][0].shape) == [expected_number_of_chunks, 512]
    assert list(tokens_long["attention_mask"][0].shape) == [expected_number_of_chunks, 512]

    tokens_two_texts_in_batch = MODEL._tokenize([SHORT_REVIEW, LONG_REVIEW])
    expected_input_ids = tokens_short["input_ids"] + tokens_long["input_ids"]
    expected_attention_mask = tokens_short["attention_mask"] + tokens_long["attention_mask"]
    assert list_of_tensors_deep_equal(tokens_two_texts_in_batch["input_ids"], expected_input_ids)
    assert list_of_tensors_deep_equal(tokens_two_texts_in_batch["attention_mask"], expected_attention_mask)
