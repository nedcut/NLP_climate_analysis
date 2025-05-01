import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TrainingArguments

# Function to compute metrics for evaluation
def compute_metrics(eval_pred):
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
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy='epoch',
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