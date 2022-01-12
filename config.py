MODEL_LOAD_FROM_FILE = False
MODEL_PATH = "../resources/roberta"
VISIBLE_GPUS = "6"

DEFAULT_PARAMS_BERT = {
    'device' : 'cuda',
    'batch_size' : 6,
    'learning_rate' : 5e-6
}

DEFAULT_PARAMS_BERT_WITH_POOLING = {
    'device' : 'cuda',
    'batch_size' : 6,
    'learning_rate' : 5e-6,
    'pooling_strategy': 'mean', # options: ['mean','max']
    'size': 510,
    'step': 256,
    'minimal_length': 1
}