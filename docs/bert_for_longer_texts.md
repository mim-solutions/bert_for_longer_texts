# BERT modification for longer texts

## Motivation
BERT model can only use the text of the maximal length of $512$ tokens (roughly speaking: token = word). It is built in the model architecture and cannot be directly changed. Discussion of this issue can be found [here](https://github.com/google-research/bert/issues/27).

## Method
Method to overcome this issue was proposed by Devlin (one of the authors of BERT) in the aforementioned discussion: [comment](https://github.com/google-research/bert/issues/27#issuecomment-435265194)
According to this, we do the following steps
1. Prepare the single text
- We follow [this instruction](https://www.kdnuggets.com/2021/04/apply-transformers-any-length-text.html). The key difference is that we allow the text chunks to overlap.
The splitting procedure is the following:
- The procedure of splitting and pooling is determined by the hyperparameters of the class `BertClassifierWithPooling`. These are `chunk_size`, `stride`, `minimal_chunk_length`, `maximal_text_length` and `pooling_strategy`.
- Tokenize the whole text (if `maximal_text_length=None`) or truncate to the size `maximal_text_length`.
- Split the tokens to chunks of the size `chunk_size`.
- Tokens may overlap dependent on the parameter `stride`.
- In other words: we get chunks by moving the window of the size `chunk_size` by the length equal to `stride`.
- See the example in [the comment](https://github.com/google-research/bert/issues/27#issuecomment-435265194).
- More examples of splitting with differents sets of parameters are in [test_splitting](../tests/model/test_splitting.py).
- Stride has the analogous meaning here that in convolutional neural networks.
- The `chunk_size` is analogous to kernel_size in CNNs.
- We ignore chunks which are too small - smaller than `minimal_chunk_length`.
- For each chunk add special tokens at the beginning and the end.
- Add padding tokens to make all tokenized sequences the same length.
- Stack the tensor chunks into one via `torch.stack`.
2. Model evaluation
- The stacked tensor is then fed into the model as a mini-batch.
- As a result we get $N$ probabilities, one for each text chunk.
- Final probability is obtained by using the aggregation function on these probabilities (this function is mean or maximum - depends on the hyperparameter `pooling_strategy`).
3. Fine-tuning the classifier
- During training we do the same steps as above, the crucial part is that all the operations of the type `cat/stack/split/mean/max` are done on tensors with attached gradient, that is we use built-in torch tensor transformations. Any intermediate conversions to lists or array are not allowed. Otherwise the key backpropagation command `loss.backward()` won't work. More precisely, we override the standard `torch` training loop in the method `_evaluate_single_batch` in the [bert_with_pooling.py](../lib/model/bert_with_pooling.py).
- Because number of chunks for the given input text is variable, texts after tokenization are tensors with variable length. Default torch class `Dataloader` cannot allow this (because it automatically wants to stack the tensors). That is why we create custom dataloaders with overwritten method `collate_fn` - more details about this can be found [here](https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418).

## Remarks
- Because we fed all the text chunks as a mini-batch, the procedure may use a lot of GPU memory to fit all the gradients during fine-tuning even with `batch_size=1`. In this case we recommend setting the parameter `maximal_text_length` to truncate longer texts. Naturally, this is the trade-off between the context we want the model to look at and the available resources. Setting `maximal_text_length=510` is equivalent to using the standard BERT model with truncation.