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

from lib.custom_datasets import TokenizedDataset, collate_fn_pooled_tokens

from lib.text_preprocessors import RobertaTokenizer, RobertaTokenizerPooled
from config import ROBERTA_PATH

## Model default params

DefaultParamsRoberta = {
    'device' : 'cuda:0',
    'batch_size' : 6,
    'learning_rate' : 5e-6
}

DefaultParamsRobertaWithPooling = {
    'device' : 'cuda:0',
    'batch_size' : 6,
    'learning_rate' : 5e-6,
    'pooling_strategy': 'max',
    'size': 510,
    'step': 256,
    'minimal_length': 1
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

class RobertaClassificationModelWithPooling(Model):
    def __init__(self, params = DefaultParamsRobertaWithPooling):
        super().__init__()
        self.params = params
        tokenizer, roberta = load_pretrained_model()
        self.preprocessor = RobertaTokenizerPooled(tokenizer,params['size'],params['step'],params['minimal_length'])
        self.dataset_class = TokenizedDataset
        self.collate_fn = collate_fn_pooled_tokens
        self.nn = initialize_model(roberta,self.params['device'])
        self.optimizer = AdamW(self.nn.parameters(),
                  lr = self.params['learning_rate'])          # learning rate
    def evaluate_single_batch(self,batch,model,device):
        input_ids = batch[0]
        attention_mask = batch[1]
        number_of_chunks = [len(x) for x in input_ids]
        labels = batch[2]

        # concatenate all input_ids into one batch

        input_ids_combined = []
        for x in input_ids:
            input_ids_combined.extend(x.tolist())

        input_ids_combined_tensors = torch.stack([torch.tensor(x).to(device) for x in input_ids_combined])

        # concatenate all attention maska into one batch

        attention_mask_combined = []
        for x in attention_mask:
            attention_mask_combined.extend(x.tolist())

        attention_mask_combined_tensors = torch.stack([torch.tensor(x).to(device) for x in attention_mask_combined])

        # get model predictions for the combined batch
        preds = model(input_ids_combined_tensors,attention_mask_combined_tensors)

        preds = preds.flatten().cpu()

        # split result preds into chunks

        preds_split = preds.split(number_of_chunks)

        # pooling
        if self.params['pooling_strategy'] == 'mean':
            pooled_preds = torch.cat([torch.mean(x).reshape(1) for x in preds_split])
        elif self.params['pooling_strategy'] == 'max':
            pooled_preds = torch.cat([torch.max(x).reshape(1) for x in preds_split])

        labels_detached = torch.tensor(labels).float()

        return pooled_preds, labels_detached
        
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