import torch
import numpy as np

#from joblib import Memory

#memory = Memory('_cache/text_preprocessed_cache')

#@memory.cache(ignore = ['tokenizer'])
def tokenize(texts, tokenizer):
    ''' Transforms list of texts to list of tokens (truncated to 512 tokens) '''
    texts = list(texts)
    tokenizer.pad_token = "<pad>"
    tokenized = tokenizer.batch_encode_plus(
        texts,
        max_length = 512,
        padding =True,
        truncation=True,
        return_tensors = 'pt')
    return tokenized