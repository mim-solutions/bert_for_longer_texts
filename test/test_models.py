"""
Test for models
"""

import unittest

import pandas as pd
import numpy as np

from lib.config import VISIBLE_GPUS


import os
os.environ["CUDA_VISIBLE_DEVICES"]= VISIBLE_GPUS
import torch


from lib.base_model import create_test_dataloader
from lib.roberta_main import load_tokenizer
from lib.input_transforms import tokenize
from lib.text_preprocessors import RobertaTokenizer
from lib.custom_datasets import TokenizedDataset

class TestModel(unittest.TestCase):
    """
    Basic test case for model
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
        preprocessor = RobertaTokenizer(tokenizer)

        text = 'Ala ma kota'
        texts = [text]
        expected_result = [[[0, 14065, 55, 22098, 2], [1, 1, 1, 1, 1]]]

        preprocessed = preprocessor.preprocess(texts)

        result_list = preprocessed.tolist()

        self.assertEqual(result_list,expected_result)

    def test_create_dataset(self):
        tokenizer = load_tokenizer()
        preprocessor = RobertaTokenizer(tokenizer)

        texts = ['Ala ma kota.','Chrząszcz brzmi w trzcinie.']
        labels = [0,1]

        X_preprocessed = preprocessor.preprocess(texts)
        dataset = TokenizedDataset(X_preprocessed,labels)
        # Test dataset creating
        self.assertEqual(len(dataset),2)

        # Test dataloaders
        test_dataloader = create_test_dataloader(dataset,2)
        loaded_sample = next(iter(test_dataloader))

        expected_result = [[0, 14065, 55, 22098, 5, 2, 1, 1], [0, 40946, 6265, 6, 25090, 68, 5, 2]]
        result_list = loaded_sample[0].numpy().tolist()

        self.assertEqual(result_list,expected_result)








if __name__ == '__main__':
    unittest.main()
        