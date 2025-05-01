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

# Function to compute metrics for evaluation
def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Training arguments for the Trainer
def get_training_args(output_dir='./results'):
    from transformers import TrainingArguments
    
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        logging_dir='./logs',
        logging_steps=10
    )

# Prediction function to use after training
def predict_sentiment(model, tokenizer, texts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    encodings = {key: val.to(device) for key, val in encodings.items()}

    with torch.no_grad():
        # Pass only input_ids and attention_mask for prediction
        outputs = model(input_ids=encodings['input_ids'], attention_mask=encodings['attention_mask'])
        # The model should return only logits when labels are not provided
        logits = outputs

    predictions = torch.argmax(logits, dim=1).cpu().numpy()
    # Ensure the sentiment map covers all possible prediction indices (0, 1, 2)
    sentiment_map = {0: 'anti', 1: 'neutral', 2: 'pro'}
    return [sentiment_map[pred] for pred in predictions]

