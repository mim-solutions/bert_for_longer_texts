"""
Test for models
"""

import unittest

import pandas as pd
import numpy as np

from config import VISIBLE_GPUS


import os
os.environ["CUDA_VISIBLE_DEVICES"]= VISIBLE_GPUS
import torch

from sklearn.model_selection import train_test_split
from lib.roberta_main import RobertaClassificationModelWithPooling

SAMPLE_DATA_PATH = 'test/sample_data/sample_data.csv'

class TestModel(unittest.TestCase):
    """
    Tests the model on simple data
    """
    
    def test_model(self):
        df = pd.read_csv(SAMPLE_DATA_PATH)

        texts = df['sentence'].tolist()
        labels = df['target'].tolist()

        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

        model = RobertaClassificationModelWithPooling()

        model.fit(X_train,y_train,epochs = 5)

        preds = model.predict(X_test)

        predicted_classes = (np.array(preds) >= 0.5)
        accurate = sum(predicted_classes == np.array(y_test).astype(bool))
        accuracy = accurate/len(y_test)

        print(f'Test accuracy: {accuracy}')

if __name__ == '__main__':
    unittest.main()
        