# BERT For Longer Texts

## Project description and motivation

The BERT model can process texts of the maximal length of 512 tokens (roughly speaking tokens are equivalent to words). It is a consequence of the model architecture and cannot be directly adjusted. Discussion of this issue can be found [here](https://github.com/google-research/bert/issues/27). Method to overcome this issue was proposed by Devlin (one of the authors of BERT) in the previously mentioned discussion: [comment](https://github.com/google-research/bert/issues/27#issuecomment-435265194). The main goal of our project is to implement this method and allow the BERT model to process longer texts during prediction and fine-tuning.

More technical details are described in a [document](docs/bert_for_longer_texts.md).

## Installation and dependencies

The project requires Python 3.9+ to run. We recommend training the models on the GPU. Hence, it is necessary to install `torch` version compatible with the machine. Other libraries are installed from the `requirements.txt` file. More detailed instruction is in [Environment setup](docs/setup_env.md).

## Loading the pre-trained model
 
As a default, the standard English `bert-base-uncased` model is used as a pre-trained model. However, it is possible to use any Bert or Roberta model. To do this, use the parameter `pretrained_model_name_or_path`.
It can be either:
- a string with the name of a pre-trained model configuration to download from huggingface library, e.g.: `roberta-base`.
- a path to a directory with the downloaded model, e.g.: `./my_model_directory/`.

## Tests
To make sure everything works properly, run the command ```pytest tests -rA```. As a default, during tests, models are trained on small samples on the CPU.

## Model classes
Two main classes are implemented:
- `BertClassifierTruncated` - base binary classification model, longer texts are truncated to 512 tokens
- `BertClassifierWithPooling` - extended model for longer texts ([more details here](docs/bert_for_longer_texts.md))

## Interface
The main methods are:
- `fit` - fine-tune the model to the training set, use the list of raw texts and labels
- `predict_classes` - calculate the list of classifications for the given list of raw texts. The model must be fine-tuned before that.
- `predict_scores` - calculate the list of probabilities for the given list of raw texts. The model must be fine-tuned before that.

## Examples
- [fit and predict method for base model](notebooks/example_base_model_fit_predict.ipynb)
- [fit and predict method for model with pooling](notebooks/example_model_with_pooling_fit_predict.ipynb)

## Contributors
The project was created at [MIM AI](https://www.mim.ai/) by:
- [Michał Brzozowski](https://github.com/MichalBrzozowski91) 
- [Marek Wachnicki](https://github.com/mwachnicki)

All contributions and ideas are welcome. Feel free to report any [issue](https://github.com/mim-solutions/bert_for_longer_texts/issues) or suggest a [pull request](https://github.com/mim-solutions/bert_for_longer_texts/pulls).