# BERT modification for longer texts

## Motivation
The BERT model can only use the text of the maximal length of 512 tokens (roughly speaking: token = word). It is built in the model architecture and cannot be directly changed. Discussion of this issue can be found [here](https://github.com/google-research/bert/issues/27).

## Method
Method to overcome this issue was proposed by Devlin (one of the authors of BERT) in the previously mentioned discussion: [comment](https://github.com/google-research/bert/issues/27#issuecomment-435265194).

The procedure of splitting and pooling is determined by the hyperparameters of the class `BertClassifierWithPooling`. These are `maximal_text_length`, `chunk_size`, `stride`, `minimal_chunk_length`,  and `pooling_strategy`.
They are used in the following way:
- The parameter `maximal_text_length` is used to truncate the tokens. It can be either `None`, which means no truncation, or an integer, determining the number of tokens to consider. Standard BERT truncates to 510 tokens because it needs 2 additional tokens at the beginning and the end.
- The integer parameter `chunk_size` determines the size (in number of tokens) of each chunk. This parameter cannot be larger than 510. Otherwise, we will not be able to fit the chunk into the input of BERT.
- Tokens may overlap depending on the parameter `stride`.
- In other words: we get chunks by moving the window of the size `chunk_size` by the length equal to `stride`. Stride cannot be bigger than chunk size. Chunks must overlap or be near each other.
- Stride has the analogous meaning here to that in [convolutional neural networks](https://deepai.org/machine-learning-glossary-and-terms/stride).
- The `chunk_size` is analogous to [kernel_size](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) in 1D CNNs.
- We ignore chunks which are too small - smaller than `minimal_chunk_length`. This parameter cannot be set larger than `chunk_size`.
- See the example in [the aforementioned comment](https://github.com/google-research/bert/issues/27#issuecomment-435265194).
- More examples of splitting with different sets of parameters are in [test_splitting](https://github.com/mim-solutions/bert_for_longer_texts/blob/main/tests/test_splitting.py).
- The string parameter `pooling_strategy` is used at the end to aggregate the model results. It can be either `mean` or `max`.

### 1. Preparing a single text
We follow [this instruction](https://www.kdnuggets.com/2021/04/apply-transformers-any-length-text.html). The main difference is that we allow the text chunks to overlap.
- Tokenize the whole text (if `maximal_text_length=None`) or truncate to the size `maximal_text_length`.
- Split the tokens into chunks based on the model hyperparameters `chunk_size`, `stride`, and `minimal_chunk_length`.
- For each chunk add special tokens at the beginning and the end.
- Add padding tokens to make all tokenized sequences the same length.
- Stack the tensor chunks into one via `torch.stack`.

### 2. Model evaluation
- The stacked tensor is then fed into the model as a mini-batch.
- We get $N$ probabilities, one for each text chunk.
- We obtain the final probability by using the aggregation function on these probabilities (this function is mean or maximum - it depends on the hyperparameter `pooling_strategy`).

### 3. Fine-tuning the classifier
- During training, we do the same steps as above. The crucial part is that all the operations of the type `cat/stack/split/mean/max` must be done on tensors with the attached gradient. That is, we use built-in torch tensor transformations. Any intermediate conversions to lists or arrays are not allowed. Otherwise, the crucial backpropagation command `loss.backward()` won't work. More precisely, we override the standard `torch` training loop in the method `_evaluate_single_batch` in the [bert_with_pooling.py](https://github.com/mim-solutions/bert_for_longer_texts/blob/main/belt_nlp/bert_with_pooling.py).
- Because the number of chunks for the given input text is variable, texts after tokenization are tensors with variable length. The default torch class `Dataloader` cannot allow this (because it automatically wants to stack the tensors). That is why we create custom dataloaders with overwritten method `collate_fn` - more details can be found [here](https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418).

## Remarks
- Because we fed all the text chunks as a mini-batch, the procedure may use a lot of GPU memory to fit all the gradients during fine-tuning even with `batch_size=1`. In this case, we recommend setting the parameter `maximal_text_length` to truncate longer texts. Naturally, this is the trade-off between the context we want the model to look at and the available resources. Setting `maximal_text_length=510` is equivalent to using the standard BERT model with truncation.