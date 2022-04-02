import os
import torch
import torch.nn as nn

from transformers import PreTrainedTokenizerFast, AutoModel, AdamW
from transformers import BertTokenizer, BertModel

from lib.architecture import BERTSequenceClassificationArch
from lib.base_model import Model

from lib.custom_datasets import TokenizedDataset, collate_fn_pooled_tokens

from lib.text_preprocessors import BERTTokenizer, BERTTokenizerPooled
from config import MODEL_LOAD_FROM_FILE, MODEL_PATH, DEFAULT_PARAMS_BERT, DEFAULT_PARAMS_BERT_WITH_POOLING


class BERTClassificationModel(Model):
    def __init__(self, params=DEFAULT_PARAMS_BERT):
        super().__init__()
        self.params = params
        tokenizer, bert = load_pretrained_model()
        self.preprocessor = BERTTokenizer(tokenizer)
        self.dataset_class = TokenizedDataset
        self.nn = initialize_model(bert, self.params['device'])
        self.optimizer = AdamW(self.nn.parameters(),
                               lr=self.params['learning_rate'])

    def evaluate_single_batch(self, batch, model, device):
        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        model_input = batch[:-1]

        labels = batch[-1]

        # model predictions
        preds = model(*model_input)
        preds = torch.flatten(preds).cpu()
        labels = labels.float().cpu()
        return preds, labels


class BERTClassificationModelWithPooling(Model):
    def __init__(self, params=DEFAULT_PARAMS_BERT_WITH_POOLING):
        super().__init__()
        self.params = params
        tokenizer, bert = load_pretrained_model()
        self.preprocessor = BERTTokenizerPooled(
            tokenizer, params['size'], params['step'], params['minimal_length'])
        self.dataset_class = TokenizedDataset
        self.collate_fn = collate_fn_pooled_tokens
        self.nn = initialize_model(bert, self.params['device'])
        self.optimizer = AdamW(self.nn.parameters(),
                               lr=self.params['learning_rate'])

    def evaluate_single_batch(self, batch, model, device):
        input_ids = batch[0]
        attention_mask = batch[1]
        number_of_chunks = [len(x) for x in input_ids]
        labels = batch[2]

        # concatenate all input_ids into one batch

        input_ids_combined = []
        for x in input_ids:
            input_ids_combined.extend(x.tolist())

        input_ids_combined_tensors = torch.stack(
            [torch.tensor(x).to(device) for x in input_ids_combined])

        # concatenate all attention maska into one batch

        attention_mask_combined = []
        for x in attention_mask:
            attention_mask_combined.extend(x.tolist())

        attention_mask_combined_tensors = torch.stack(
            [torch.tensor(x).to(device) for x in attention_mask_combined])

        # get model predictions for the combined batch
        preds = model(
            input_ids_combined_tensors,
            attention_mask_combined_tensors)

        preds = preds.flatten().cpu()

        # split result preds into chunks

        preds_split = preds.split(number_of_chunks)

        # pooling
        if self.params['pooling_strategy'] == 'mean':
            pooled_preds = torch.cat(
                [torch.mean(x).reshape(1) for x in preds_split])
        elif self.params['pooling_strategy'] == 'max':
            pooled_preds = torch.cat([torch.max(x).reshape(1)
                                     for x in preds_split])

        labels_detached = torch.tensor(labels).float()

        return pooled_preds, labels_detached

# Helper functions


def load_pretrained_model():
    tokenizer = load_tokenizer()
    model = load_bert()

    return tokenizer, model


def load_tokenizer():
    if MODEL_LOAD_FROM_FILE:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(
                MODEL_PATH, "tokenizer.json"))
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer


def load_bert():
    if MODEL_LOAD_FROM_FILE:
        model = AutoModel.from_pretrained(MODEL_PATH)
    else:
        model = BertModel.from_pretrained("bert-base-uncased")
    return model


def initialize_model(bert, device):
    # pass the pre-trained BERT model to our defined architecture
    model = BERTSequenceClassificationArch(bert)
    # push the model to GPU/CPU
    model = model.to(device)
    # run on multiple GPU's
    model = nn.DataParallel(model)
    return model
