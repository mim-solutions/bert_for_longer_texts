from torch.utils.data import Dataset

class TextDataset(Dataset):
    ''' Dataset for raw texts with labels'''
    def __init__(self,texts,labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx],self.labels[idx]

class TokenizedDataset(Dataset):
    ''' Dataset for tokens with labels'''
    def __init__(self,tokens,labels):
        self.input_ids = tokens[:,0]
        self.attention_mask = tokens[:,1]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx],self.attention_mask[idx],self.labels[idx]

def collate_fn_pooled_tokens(data):
    input_ids = [data[i][0] for i in range(len(data))]
    attention_mask = [data[i][1] for i in range(len(data))]
    labels = [data[i][2] for i in range(len(data))]
    collated = [input_ids, attention_mask, labels]
    return collated