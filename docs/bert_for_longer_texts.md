# BERT modification for longer texts

## Motivation
BERT model can only use the text of the maximal length of 512 tokens (roughly speaking: token = word). It is built in the model architecture and cannot be directly changed. Discussion of this issue can be found [here](https://github.com/google-research/bert/issues/27).

## Method
Method to overcome this issue was proposed by Devlin (one of the authors of BERT) in the aforementioned discussion: [comment](https://github.com/google-research/bert/issues/27#issuecomment-435265194)
According to this, we do the following steps
1. Prepare the single text - we follow [this instruction](https://www.kdnuggets.com/2021/04/apply-transformers-any-length-text.html)
- tokenize all text
- divide all tokens into chunks of the size `size`, overlapping with step `step`, with minimal chunk length `minimal_length`
- for each chunk we add special tokens at the beginning and the end
- add padding tokens to make all tokenized sequences the same length
- stack the tensor chunks into one via `torch.stack`
2. Model evaluation
- the stacked tensor is then fed into the model as a mini-batch
- as a result we get $N$ probabilities, one for each text chunk
- final probability is obtained by using the aggregation function on these probabilities (this function is mean or maximum - depends on the hyperparameter `pooling_strategy`)
3. Transfer learning
- during training we do the same steps as above, the crucial part is that all the operations of the type `cat/stack/split/mean/max` are done on tensors with attached gradient, that is we use built-in torch tensor transformations. Any intermediate conversions to lists or array are not allowed. Otherwise the key backpropagation command `loss.backward()` won't work.
- because number of chunks for the given input text is variable, texts after tokenization are tensors with variable length. Default torch class `Dataloader` cannot allow this (because it automatically wants to stack the tensors). That is why we create custom dataloaders with overwritten method `collate_fn` - more details about this can be found [here](https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418)

## Remarks
- mentioned pooling hyperparameters `size`, `step`, `minimal_length`, `pooling_strategy` are parameters of the class `BertClassifierWithPooling`.