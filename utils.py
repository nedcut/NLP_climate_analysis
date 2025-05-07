from transformers import TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import numpy as np

def get_training_args(output_dir='./results'):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def predict_sentiment(model, tokenizer, texts, device='cpu', batch_size=32):
    model.eval()
    model.to(device)
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        if not batch_texts: # Should not happen with correct loop logic but good for safety
            continue
            
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        with torch.no_grad():
            outputs = model(**encodings)
            if isinstance(outputs, tuple):
                logits = outputs[1] if len(outputs) > 1 and outputs[1] is not None else outputs[0]
            else:
                logits = outputs
            
            if hasattr(logits, 'logits'): # Handle cases where logits are an attribute of the output object
                logits = logits.logits

            preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        label_map = {0: 'anti', 1: 'neutral', 2: 'pro'}
        all_preds.extend([label_map[p] for p in preds])
            
    return all_preds