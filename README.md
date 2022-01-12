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

## Loading the pretrained model

As a default the standard english `bert-base-uncased` model is used as a pretrained model.

It is possible to use custom models for other languages. Below we describe how to do this for polish roberta model:

- In the first step we need to download the BERT model pretrained on corpora of texts in a given language.
- For polish it is the file ```roberta_base_transformers.zip``` from  [here](https://github.com/sdadas/polish-roberta/releases). After download, unzip files and copy the path to the config file ```config.py``` eg. ```ROBERTA_PATH = "../resources/roberta"``` and set `MODEL_LOAD_FROM_FILE = True`.

## Configuration

 In the file ```config.py``` we specify the available GPU we want to run the model on. It is allowed to use multiple GPUs, eg. ```VISIBLE_GPUS = "0,2,3"```.

## Tests
To make sure that everything works properly, run the command ```pytest test```.

## Model classes
Two main classes are implemented:
- `RobertaClassificationModel` - base binary classification model, longer texts are truncated to 512 tokens
- `RobertaClassificationModelWithPooling` - extended model for longer texts ([more details here](docs/roberta_for_longer_texts.md))

## Interface
The main methods are:
- `fit` - fine-tune the model to the training set, uses list of raw texts and labels
- `predict` - calculate the list of probabilities for given list of raw texts. Model must fine-tuned before that.
- `train_and_evaluate` - train and evaluate model on given train and test sets. Useful for example for obtaining the learning curve

## Examples
- [fit and predict method for base model](ipython/example_base_model_fit_predict.ipynb)
- [train and evaluate method for base model](ipython/example_base_model_train_and_evaluate.ipynb)
- [fit and predict method for model with pooling](ipython/example_model_with_pooling_fit_predict.ipynb)
- [train and evaluate method for model with pooling](ipython/example_model_with_pooling_train_and_evaluate.ipynb)