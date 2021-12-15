import numpy as np

from lib.input_transforms import tokenize


class Preprocessor():
    def __init__(self):
        pass
    def preprocess(self,array_of_texts):
        raise NotImplementedError("Preprocessing is implemented for subclasses only")

class RobertaTokenizer(Preprocessor):
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
    def preprocess(self,array_of_texts):
        tokenized = tokenize(array_of_texts,self.tokenizer)
        input_ids = tokenized['input_ids'].numpy()
        attention_mask = tokenized['attention_mask'].numpy()
        array_of_preprocessed_data = np.array(list(zip(input_ids,attention_mask)))
        return array_of_preprocessed_data