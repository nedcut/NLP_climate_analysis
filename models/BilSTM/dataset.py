import torch
from torch.utils.data import Dataset

class TweetDataset(Dataset):
    """
    Custom PyTorch dataset for sentiment classification.
    """
    def __init__(self, inputs, labels):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]
