# **BELT** (**BE**RT For **L**onger **T**exts)

## Project description and motivation

### The BELT approach

The BERT model can process texts of the maximal length of 512 tokens (roughly speaking tokens are equivalent to words). It is a consequence of the model architecture and cannot be directly adjusted. Discussion of this issue can be found [here](https://github.com/google-research/bert/issues/27). Method to overcome this issue was proposed by Devlin (one of the authors of BERT) in the previously mentioned discussion: [comment](https://github.com/google-research/bert/issues/27#issuecomment-435265194). The main goal of our project is to implement this method and allow the BERT model to process longer texts during prediction and fine-tuning. We dub this approach BELT (**BE**RT For **L**onger **T**exts).

More technical details are described in a [document](https://github.com/mim-solutions/bert_for_longer_texts/blob/main/docs/bert_for_longer_texts.md). We also prepared the comprehensive blog post: [part 1](https://www.mim.ai/fine-tuning-bert-model-for-arbitrarily-long-texts-part-1/), [part 2](https://www.mim.ai/fine-tuning-bert-model-for-arbitrarily-long-texts-part-2/).

### Attention is all you need, but 512 words is all you have

The limitations of the BERT model to the 512 tokens come from the very beginning of the transformers models. Indeed, the attention mechanism, invented in the groundbreaking 2017 paper [Attention is all you need](https://arxiv.org/abs/1706.03762), scales quadratically with the sequence length. Unlike RNN or CNN models, which can process sequences of arbitrary length, transformers with the full attention (like BERT) are infeasible (or very expensive) to process long sequences.
To overcome the issue, alternative approaches with sparse attention mechanisms were proposed in 2020: [BigBird](https://arxiv.org/abs/2007.14062) and [Longformer](https://arxiv.org/abs/2004.05150).

### BELT vs. BigBird vs. LongFormer

Let us now clarify the key differences between the BELT approach to fine-tuning and the sparse attention models BigBird and Longformer:
- The main difference is that BigBird and Longformers are not modified BERTs. They are models with different architectures. Hence, they need to be pre-trained from scratch or downloaded.
- This leads to the main advantage of the BELT approach - it uses any pre-trained BERT or RoBERTa models. A quick look at the HuggingFace Hub confirms that there are about 100 times more resources for [BERT](https://huggingface.co/models?other=bert) than for [Longformer](https://huggingface.co/models?other=longformer). It might be easier to find the one appropriate for the specific task or language.
- On the other hand, we have not done any benchmark tests yet. We believe that the comparison of the BELT approach with the models with sparse attention might be very instructive. Some work in this direction was done in the 2022 paper [Extend and Explain: Interpreting Very Long Language Models](https://proceedings.mlr.press/v193/stremmel22a/stremmel22a.pdf). The authors cited our implementation under the former name `roberta_for_longer_texts`. We encourage more research in this direction.

## Installation and dependencies

The project requires Python 3.9+ to run. We recommend training the models on the GPU. Hence, it is necessary to install `torch` version compatible with the machine. The version of the driver depends on the machine - first, check the version of GPU drivers by the command `nvidia-smi` and choose the newest version compatible with these drivers according to [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) (e.g.: 11.1). Then we install `torch` to get the compatible build. [Here](https://pytorch.org/get-started/previous-versions/), we find which torch version is compatible with the CUDA version on our machine.

Another option is to use the CPU-only version of torch:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
Next, we recommend installing via pip:
```
pip3 install belt-nlp
```

If you want to clone the repo in order to run tests or notebooks, you can use the `requirements.txt` file.

## Model classes

Two main classes are implemented:
- `BertClassifierTruncated` - base binary classification model, longer texts are truncated to 512 tokens
- `BertClassifierWithPooling` - extended model for longer texts ([more details here](https://github.com/mim-solutions/bert_for_longer_texts/blob/main/docs/bert_for_longer_texts.md))

## Interface

The main methods are:
- `fit` - fine-tune the model to the training set, use the list of raw texts and labels
- `predict_classes` - calculate the list of classifications for the given list of raw texts. The model must be fine-tuned before that.
- `predict_scores` - calculate the list of probabilities for the given list of raw texts. The model must be fine-tuned before that.

## Loading the pre-trained model
 
As a default, the standard English `bert-base-uncased` model is used as a pre-trained model. However, it is possible to use any Bert or Roberta model. To do this, use the parameter `pretrained_model_name_or_path`.
It can be either:
- a string with the name of a pre-trained model configuration to download from huggingface library, e.g.: `roberta-base`.
- a path to a directory with the downloaded model, e.g.: `./my_model_directory/`.

## Tests
To make sure everything works properly, run the command ```pytest tests -rA```. As a default, during tests, models are trained on small samples on the CPU.

## Examples
- [fit and predict method for base model](https://github.com/mim-solutions/bert_for_longer_texts/blob/main/notebooks/example_base_model_fit_predict.ipynb)
- [fit and predict method for model with pooling](https://github.com/mim-solutions/bert_for_longer_texts/blob/main/notebooks/example_model_with_pooling_fit_predict.ipynb)

## Contributors
The project was created at [MIM AI](https://www.mim.ai/) by:
- [Michał Brzozowski](https://github.com/MichalBrzozowski91) 
- [Marek Wachnicki](https://github.com/mwachnicki)

If you want to contribute to the library, see the [contributing info](https://github.com/mim-solutions/bert_for_longer_texts/blob/main/CONTRIBUTING.md).

## License
See the [LICENSE](https://github.com/mim-solutions/bert_for_longer_texts/blob/main/LICENSE.txt) file for license rights and limitations (MIT).

## For Maintainers

File `requirements.txt` can be updated using the command:
```
bash pip-freeze-without-torch.sh > requirements.txt
```
This script saves all dependencies of the current active environment except `torch`.

In order to add the next version of the package to pypi, do the following steps:
- First, increment the package version in `pyproject.toml`.
- Then build the new version: run `python3.9 -m build` from the main folder.
- Finally, upload to pypi: `twine upload dist/*` (two newly created files).
