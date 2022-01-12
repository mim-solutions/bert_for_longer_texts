"""
Unit test - this was done for polish roberta model, as default these tests are skipped
"""

import unittest

import torch
import pandas as pd
import numpy as np

from lib.base_model import create_test_dataloader
from lib.main import load_tokenizer
from lib.text_preprocessors import tokenize, tokenize_pooled, BERTTokenizer
from lib.custom_datasets import TokenizedDataset

from lib.pooling import (tokenize_all_text, split_overlapping, split_tokens_into_smaller_chunks,
 add_special_tokens_at_beginning_and_end, add_padding_tokens, stack_tokens_from_all_chunks,
 transform_text_to_model_input)

SAMPLE_LONGER_TEXT_PATH = 'test/sample_data/sample.txt'

RUN_UNIT_TESTS = False

@unittest.skipIf(RUN_UNIT_TESTS is False, "skip test for polish language")
class TestBaseModelUnits(unittest.TestCase):
    """
    Tests for single functions and objects
    """
    
    def test_tokenize_function(self):
        tokenizer = load_tokenizer()
        # Test tokenizing
        text = 'Ala ma kota'
        expected_result = [0, 14065, 55, 22098, 2]

        texts = [text]

        tokenized = tokenize(texts,tokenizer)
        result_list = tokenized['input_ids'][0].numpy().tolist()

        self.assertEqual(result_list,expected_result)

    def test_preprocessor(self):
        tokenizer = load_tokenizer()
        preprocessor = BERTTokenizer(tokenizer)

        text = 'Ala ma kota'
        texts = [text]

        preprocessed = preprocessor.preprocess(texts)
        input_ids = preprocessed['input_ids'][0].numpy().tolist()
        expected_result = [0, 14065, 55, 22098, 2]
        self.assertEqual(input_ids,expected_result)
        attention_mask = preprocessed['attention_mask'][0].numpy().tolist()
        expected_result = [1, 1, 1, 1, 1]
        self.assertEqual(attention_mask,expected_result)

    def test_create_dataset(self):
        tokenizer = load_tokenizer()
        preprocessor = BERTTokenizer(tokenizer)

        texts = ['Ala ma kota.','Chrząszcz brzmi w trzcinie.']
        labels = [0,1]

        X_preprocessed = preprocessor.preprocess(texts)
        dataset = TokenizedDataset(X_preprocessed,labels)
        # Test dataset creating
        self.assertEqual(len(dataset),2)

        # Test dataloaders
        test_dataloader = create_test_dataloader(dataset,2)
        loaded_sample = next(iter(test_dataloader))
        input_ids = loaded_sample[0].numpy().tolist()
        expected_result = [[0, 14065, 55, 22098, 5, 2, 1, 1], [0, 40946, 6265, 6, 25090, 68, 5, 2]]
        self.assertEqual(input_ids,expected_result)
        attention_mask = loaded_sample[1].numpy().tolist()
        expected_result = [[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]]
        self.assertEqual(attention_mask,expected_result)
        labels_obtained = loaded_sample[2].numpy().tolist()
        expected_result =[0,1]
        self.assertEqual(labels_obtained,expected_result)


@unittest.skipIf(RUN_UNIT_TESTS is False, "skip test for polish language")
class TestPoolingUnits(unittest.TestCase):
    """
    Tests for single functions and objects
    """

    def test_pooling_functions(self):
        tokenizer = load_tokenizer()
        # Load example text
        with open(SAMPLE_LONGER_TEXT_PATH, 'r') as file:
            longer_text = file.read()
        # Test tokenizing the entire texts
        tokens = tokenize_all_text(longer_text,tokenizer)
        number_of_tokens =  len(tokens['input_ids'][0].numpy())
        expected_result = 3194

        self.assertEqual(number_of_tokens,expected_result)
        # Test splitting tokens into overlapping chunks
        size = 510
        step = 256
        minimal_length = 256

        input_id_chunks, mask_chunks = split_tokens_into_smaller_chunks(tokens,size,step,minimal_length)
        number_of_chunks = len(input_id_chunks)
        expected_result = 12

        self.assertEqual(number_of_chunks,expected_result)
        # Test adding special tokens at the beginning and end
        add_special_tokens_at_beginning_and_end(input_id_chunks, mask_chunks)
        first_token_id = input_id_chunks[0][0].item()
        expected_result = 101

        self.assertEqual(first_token_id,expected_result)

        last_token_id = input_id_chunks[0][-1].item()
        expected_result = 102

        self.assertEqual(last_token_id,expected_result)
        # Test adding padding tokens to make sure all chunks have exactly 512 tokens
        add_padding_tokens(input_id_chunks, mask_chunks)
        last_chunk = input_id_chunks[-1]
        result = len(last_chunk)
        expected_result = 512

        self.assertEqual(result,expected_result)

        last_token_id = last_chunk[-1]
        expected_result = 0

        self.assertEqual(last_token_id,expected_result)

        # Test reshaping tokens for model input
        input_ids, attention_mask = stack_tokens_from_all_chunks(input_id_chunks, mask_chunks)
        result = list(input_ids.shape)
        expected_result = [12,512]

        self.assertEqual(result,expected_result)

        # Integrated test for combined method transform_text_to_model_input
        input_ids, attention_mask = transform_text_to_model_input(longer_text,tokenizer,size,step,minimal_length)

        result = list(input_ids.shape)
        expected_result = [12,512]

        self.assertEqual(result,expected_result)

    def test_tokenize_pooled(self):
        tokenizer = load_tokenizer()
        # Load example text
        with open(SAMPLE_LONGER_TEXT_PATH, 'r') as file:
            longer_text = file.read()

        size = 510
        step = 256
        minimal_length = 256

        texts = [longer_text, longer_text]

        tokens = tokenize_pooled(texts, tokenizer, size, step, minimal_length)
        # Test if the result has an expected shape
        self.assertEqual(len(tokens['input_ids']),len(texts))
        result = list(tokens['input_ids'][0].shape)
        expected_result = [12,512]

        self.assertEqual(result, expected_result)

    def test_split_overlapping(self):
        example_list = [1,2,3,4,5]
        splitted = split_overlapping(example_list,3,2,1)

        expected_result = [[1,2,3],[3,4,5],[5]]
        
        self.assertEqual(splitted,expected_result)

if __name__ == '__main__':
    unittest.main()