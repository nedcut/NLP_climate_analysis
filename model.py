import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class Climodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.model.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        logits = self.classifier(cls)
        return logits

    def tokenize(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", model_max_length=512)