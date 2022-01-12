# BERT For Longer Texts

## Project description and motivation

Goals:

1. Wrap the BERT model for binary text classification in a simple, convenient interface for immediate use
2. Implement support for longer text sequences - BERT automatically truncates every text to 512 tokens. The truncation leads to loss of information and makes BERT model not suitable for longer texts. We use pooling method as described in the [comment](https://github.com/google-research/bert/issues/27#issuecomment-435265194) to solve this problem

### 1. BERT with a simple, minimal interface

The main procedure of using the BERT model is to apply the method called transfer learning, that is:
- download the model with pretrained weights for a specified language
- train this model using the labelled train set (fine-tuning)

There are several resources describing the procedure in more details about text preprocessing, tokenization, et cetera. However it is often useful to have a model as a blackbox ready to use on raw texts. Here we wrap the BERT model into a simple class with basic methods `fit` and `predict` which are used on the raw text without any pre-existing knowledge of BERT model theory and implementation.

Let us briefly describe existing tutorials to illustrate the need for a simple, ready to use solution:
- [Transfer Learning for NLP: Fine-Tuning BERT for Text Classification](https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/) - long, detailed tutorial, recommended to understand what is under the hood of the BERT transfer learning, however unnecessary if we are only interested in building a fast prototype
- [Fine-tuning a pretrained model](https://huggingface.co/docs/transformers/training) - relatively simple solution based on the `Trainer` object implemented in the `transformers` library. However this still needs some explicit steps to preprocess the texts and the steps of fitting and prediction are not separated which may be unconvenient.

The advantages of our wrapper:
- the input is just a list of raw texts and its labels. All preprocessing, tokenization, dataset preparation is done under the hood
- simple interface - model class with high level methods `fit` and `predict`

### 2. BERT for longer texts

The BERT model can process texts of maximal length of 512 tokens (roughly speaking tokens are equivalent to words). It is a consequence of the model architecture and cannot be directly adjusted. More details are described in a [document](docs/roberta_for_longer_texts.md).

## Installation and dependencies

Because of the size of BERT model, it is recommended to train it on GPU. Hence it is necessary to install `torch` version compatible with the machine. Other libraries are installed by the bash file `bash env_setup.sh`. This project is built on top of the `transformers` library which provides BERT and BERT-like model implementations compatible with PyTorch framework. More detailed instruction can be found in [Environment setup](docs/setup_env.md).

## Loading the pretrained BERT model

In the first step we need to download the BERT model pretrained on corpora of texts in a given language.

For polish it is the file ```roberta_base_transformers.zip``` from  [here](https://github.com/sdadas/polish-roberta/releases). After download, unzip files and copy the path to the config file ```config.py``` eg. ```ROBERTA_PATH = "../resources/roberta"```.

## Configuration

 In the file ```config.py``` set the path to the pretrained model described above. Also here we specify the available GPU we want to run the model on. It is allowed to use multiple GPUs, eg. ```VISIBLE_GPUS = "0,2,3"```.

## Tests
To make sure that everything works properly, run the command ```pytest test```.

## Model classes
Two main classes are implemented:
- `RobertaClassificationModel` - base binary classification model, longer texts are truncated to 512 tokens
- `RobertaClassificationModelWithPooling` - extended model for longer texts ([more details here](docs/roberta_for_longer_texts.md))

## Interfaced
The main methods are:
- `fit` - fine-tune the model to the training set, uses list of raw texts and labels
- `predict` - calculate the list of probabilities for given list of raw texts. Model must fine-tuned before that.
- `train_and_evaluate` - train and evaluate model on given train and test sets. Useful for example for obtaining the learning curve

## Przykład użycia - metody fit i predict

```
import pandas as pd
import numpy as np

from config import VISIBLE_GPUS

import os
os.environ["CUDA_VISIBLE_DEVICES"]= VISIBLE_GPUS
import torch

from sklearn.model_selection import train_test_split
from lib.roberta_main import RobertaClassificationModel

SAMPLE_DATA_PATH = 'test/sample_data/sample_data.csv'

# Loading data for tests
df = pd.read_csv(SAMPLE_DATA_PATH)

texts = df['sentence'].tolist() # list of texts
labels = df['target'].tolist() # list of 0/1 labels

# Train test split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Loading model
model = RobertaClassificationModel()
# Fitting a model to training data for 5 epochs
model.fit(X_train,y_train,epochs = 5)
# Predicted probability for test set
preds = model.predict(X_test)

predicted_classes = (np.array(preds) >= 0.5)
accurate = sum(predicted_classes == np.array(y_test).astype(bool))
accuracy = accurate/len(y_test)

print(f'Test accuracy: {accuracy}')
```

Wynik powyższego kodu (przykładowy, może się różnić z uwagi na losowość):
 ```
Epoch: 0, Train accuracy: 0.590625, Train loss: 0.6770698979496956
Epoch: 1, Train accuracy: 0.721875, Train loss: 0.58055414929986
Epoch: 2, Train accuracy: 0.89375, Train loss: 0.3515076955780387
Epoch: 3, Train accuracy: 0.9125, Train loss: 0.2523562053218484
Epoch: 4, Train accuracy: 0.940625, Train loss: 0.164382476080209
Test accuracy: 0.925
 ```

## Przykład użycia - metoda train_and_evaluate

```
import pandas as pd
import numpy as np

from config import VISIBLE_GPUS

import os
os.environ["CUDA_VISIBLE_DEVICES"]= VISIBLE_GPUS
import torch

from sklearn.model_selection import train_test_split
from lib.roberta_main import RobertaClassificationModel

SAMPLE_DATA_PATH = 'test/sample_data/sample_data.csv'

# Loading data for tests
df = pd.read_csv(SAMPLE_DATA_PATH)

texts = df['sentence'].tolist() # list of texts
labels = df['target'].tolist() # list of 0/1 labels

# Train test split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Loading model
model = RobertaClassificationModel()
# Fitting a model to training data for 5 epochs
result = model.train_and_evaluate(X_train, X_test, y_train, y_test,epochs = 5)
```

 Wynik powyższego kodu (przykładowy, może się różnić z uwagi na losowość):

```
Epoch: 0, Train accuracy: 0.5875, Train loss: 0.6660354651510716
Epoch: 0, Test accuracy: 0.65, Test loss: 0.5904278859496117
Epoch: 1, Train accuracy: 0.78125, Train loss: 0.5293301593512296
Epoch: 1, Test accuracy: 0.925, Test loss: 0.35317784547805786
Epoch: 2, Train accuracy: 0.88125, Train loss: 0.34443826507776976
Epoch: 2, Test accuracy: 0.95, Test loss: 0.1830226019024849
Epoch: 3, Train accuracy: 0.9375, Train loss: 0.20902621131390334
Epoch: 3, Test accuracy: 0.9375, Test loss: 0.17358638979494573
Epoch: 4, Train accuracy: 0.96875, Train loss: 0.12159209074452519
Epoch: 4, Test accuracy: 0.95, Test loss: 0.12857716977596284
```

Dodatkowo metryki w kolejnych epokach są zapisane w zmiennej `result`.