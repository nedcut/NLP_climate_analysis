import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torch.nn import CrossEntropyLoss

class ClimateModel(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained("distilbert-base-uncased", num_labels=3)
        self.model = AutoModel.from_pretrained("distilbert-base-uncased", config=config)
        self.classifier = nn.Linear(self.model.config.hidden_size, 3) # Output size is 3 for anti, neutral, pro
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Get the hidden state of the [CLS] token (first token)
        pooled_output = outputs.last_hidden_state[:, 0]
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        # Apply classification layer
        logits = self.classifier(pooled_output)

        loss = None
        # Calculate loss if labels are provided
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Ensure logits and labels shapes are compatible for CrossEntropyLoss
            # Logits: [batch_size, num_labels], Labels: [batch_size]
            loss = loss_fct(logits.view(-1, 3), labels.view(-1)) # Use 3 for num_labels

        # Return loss and logits in a tuple if loss was calculated
        # The Trainer expects this format when labels are provided
        if loss is not None:
            return (loss, logits)
        else:
            # During evaluation or prediction without labels, just return logits
            return logits # Or return SequenceClassifierOutput(logits=logits)

# Dataset class for the climate sentiment data
class ClimateDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)





