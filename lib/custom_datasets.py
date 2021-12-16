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