import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass

from transformers import RobertaModel, PreTrainedTokenizerFast, AutoModel, AdamW
from lib.roberta_architecture import RobertaSequenceClassificationArch
from lib.base_model import Model

from lib.custom_datasets import TokenizedDataset

from lib.text_preprocessors import RobertaTokenizer
from lib.config import ROBERTA_PATH

## Model default params

DefaultParamsRoberta = {
    'device' : 'cuda:0',
    'batch_size' : 6,
    'learning_rate' : 5e-6
}

## Main class

class RobertaClassificationModel(Model):
    def __init__(self, params = DefaultParamsRoberta):
        super().__init__()
        self.params = params
        tokenizer, roberta = load_pretrained_model()
        self.preprocessor = RobertaTokenizer(tokenizer)
        self.dataset_class = TokenizedDataset
        self.nn = initialize_model(roberta,self.params['device'])
        self.optimizer = AdamW(self.nn.parameters(),
                  lr = self.params['learning_rate'])          # learning rate
        
## Helper functions

def load_pretrained_model():
    tokenizer = load_tokenizer()
    model = load_roberta()

    return tokenizer, model

def load_tokenizer():
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(ROBERTA_PATH, "tokenizer.json"))
    return tokenizer

def load_roberta():
    model: RobertaModel = AutoModel.from_pretrained(ROBERTA_PATH)
    return model


def initialize_model(roberta,device):
    # pass the pre-trained roberta model to our defined architecture
    model = RobertaSequenceClassificationArch(roberta)
    # push the model to GPU/CPU
    model = model.to(device)
    # run on multiple GPU's
    model = nn.DataParallel(model)
    return model