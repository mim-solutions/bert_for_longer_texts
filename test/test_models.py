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
from lib.main import BERTClassificationModel, BERTClassificationModelWithPooling

SAMPLE_DATA_PATH = 'test/sample_data/sample_data_eng.csv'
MODEL_CLASSES_TO_TEST = [BERTClassificationModel, BERTClassificationModelWithPooling]
EPOCHS = 2

class TestModels(unittest.TestCase):
    """
    Tests for model methods
    """
    

    def test_model_methods(self):
        df = pd.read_csv(SAMPLE_DATA_PATH)

        texts = df['sentence'].tolist()
        labels = df['target'].tolist()

        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

        model_classes = MODEL_CLASSES_TO_TEST

        for model_class in model_classes:
            print(f'Testing model class {model_class}')
            # Test fit and predict methods
            model = model_class()

            model.fit(X_train,y_train,epochs = EPOCHS)

            preds = model.predict(X_test)

            predicted_classes = (np.array(preds) >= 0.5)
            accurate = sum(predicted_classes == np.array(y_test).astype(bool))
            accuracy = accurate/len(y_test)

            print(f'Test accuracy: {accuracy}')
            # Test train and evaluate method
            _ = model.train_and_evaluate(X_train, X_test, y_train, y_test,epochs = EPOCHS)

if __name__ == '__main__':
    unittest.main()
        