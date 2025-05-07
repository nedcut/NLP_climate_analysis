from transformers import TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

def get_training_args(output_dir='./results/BERT'):
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=42
    )

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def predict_sentiment(model, tokenizer, texts):
    model.eval()
    inputs = tokenizer(
        texts,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        predictions = torch.argmax(logits, dim=-1).tolist()

    label_map = {0: 'anti', 1: 'neutral', 2: 'pro'}
    return [label_map[p] for p in predictions]
