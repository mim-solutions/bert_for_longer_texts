from torch import nn


class BERTSequenceClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_proj = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, cls_token_hidden_state):
        x = cls_token_hidden_state
        x = self.out_proj(x)
        x = self.sigmoid(x)
        return x


class BERTSequenceClassificationArch(nn.Module):

    def __init__(self, bert):

        super().__init__()

        self.bert = bert
        self.classification_head = BERTSequenceClassificationHead()

    def forward(self, input_ids, attention_mask):

        x = bert_vectorize(self.bert, input_ids, attention_mask)
        x = self.classification_head(x)
        return x


def bert_vectorize(bert, input_ids, attention_mask):
    outputs = bert(input_ids, attention_mask)
    sequence_output = outputs[0]

    vectorized = sequence_output[:, 0, :]  # take <s> token (equiv. to [CLS])
    return vectorized
