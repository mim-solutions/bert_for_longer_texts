import torch
import torch.nn as nn

class RobertaSequenceClassificationHead(nn.Module):
    def __init__(self):

        super().__init__()

        # dropout layer
        #self.dropout = nn.Dropout(0.1)

        # dense layer 1
        #self.dense = nn.Linear(768,768)
        
        # dense layer 2 (Output layer)
        self.out_proj = nn.Linear(768,1)

        #sigmoid activation function
        self.sigmoid = nn.Sigmoid()


    def forward(self, cls_token_hidden_state):
        x = cls_token_hidden_state
        #x = self.dropout(x)
        #x = self.dense(x)
        #x = torch.tanh(x)
        #x = self.dropout(x)
        x = self.out_proj(x)
        
        # apply softmax activation
        x = self.sigmoid(x)

        return x


class RobertaSequenceClassificationArch(nn.Module):

    def __init__(self, roberta):

        super().__init__()

        self.roberta = roberta
        self.classification_head = RobertaSequenceClassificationHead()
        
    #define the forward pass
    def forward(self, input_ids, attention_mask):

        #pass the inputs to the model

        x = roberta_vectorize(self.roberta,input_ids,attention_mask)

        # pass vectorized output to classification head

        x = self.classification_head(x)
        return x

def roberta_vectorize(roberta, input_ids, attention_mask):
    outputs = roberta(input_ids,attention_mask)
    sequence_output = outputs[0]

    vectorized = sequence_output[:, 0, :]  # take <s> token (equiv. to [CLS])
    return vectorized