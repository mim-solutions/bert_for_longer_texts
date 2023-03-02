# BERT For Longer Texts

## Project description and motivation

The BERT model can process texts of maximal length of 512 tokens (roughly speaking tokens are equivalent to words). It is a consequence of the model architecture and cannot be directly adjusted. Discussion of this issue can be found [here](https://github.com/google-research/bert/issues/27). Method to overcome this issue was proposed by Devlin (one of the authors of BERT) in the aforementioned discussion: [comment](https://github.com/google-research/bert/issues/27#issuecomment-435265194). The main goal of our project is to implement this method and allow the BERT model to process longer texts during both prediction and fine-tuning.

More technical details are described in a [document](docs/bert_for_longer_texts.md).

## Installation and dependencies

The project requires Python 3.9+ to run. Because of the size of BERT model, it is recommended to train it on GPU. Hence it is necessary to install `torch` version compatible with the machine. Other libraries are installed from the `requirements.txt` file. More detailed instruction can be found in [Environment setup](docs/setup_env.md).

## Loading the pretrained model

As a default the standard english `bert-base-uncased` model is used as a pretrained model.

It is possible to use custom models for other languages. Below we describe how to do this for polish roberta model:

- In the first step we need to download the BERT model pretrained on corpora of texts in a given language.
- For polish it is the file ```roberta_base_transformers.zip``` from  [here](https://github.com/sdadas/polish-roberta/releases). After download, unzip files and copy the path to the config file ```.env``` eg. ```MODEL_PATH = "../resources/roberta"``` and set `MODEL_LOAD_FROM_FILE = "True"`.

## Tests
To make sure that everything works properly, run the command ```pytest tests -rA```. As a default during tests models are trained on small samples on CPU.

## Model classes
Two main classes are implemented:
- `BertClassifierTruncated` - base binary classification model, longer texts are truncated to 512 tokens
- `BertClassifierWithPooling` - extended model for longer texts ([more details here](docs/bert_for_longer_texts.md))

## Interface
The main methods are:
- `fit` - fine-tune the model to the training set, uses list of raw texts and labels
- `predict_classes` - calculate the list of classification for given list of raw texts. Model must be fine-tuned before that.
- `predict_scores` - calculate the list of probabilities for given list of raw texts. Model must be fine-tuned before that.

## Examples
- [fit and predict method for base model](notebooks/example_base_model_fit_predict.ipynb)
- [fit and predict method for model with pooling](notebooks/example_model_with_pooling_fit_predict.ipynb)