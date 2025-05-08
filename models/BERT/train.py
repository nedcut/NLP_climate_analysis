import pandas as pd
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, Trainer, EarlyStoppingCallback
from sklearn.metrics import classification_report
from models.BERT.model import ClimateModel, ClimateDataset
from models.BERT.utils import get_training_args, compute_metrics, predict_sentiment
import argparse

def main():
    parser = argparse.ArgumentParser(description='Fine-tune BERT climate sentiment model')
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--patience', type=int, default=2, help='Early stopping patience in eval epochs')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_df = pd.read_csv('data/train_data.csv')
    dev_df = pd.read_csv('data/dev_data.csv')
    test_df = pd.read_csv('data/test_data.csv')

    # Label remapping: -1 → 0, 0 → 1, 1 → 2
    label_map = {-1: 0, 0: 1, 1: 2}
    train_df['sentiment'] = train_df['sentiment'].dropna().map(label_map)
    dev_df['sentiment'] = dev_df['sentiment'].dropna().map(label_map)
    test_df['sentiment'] = test_df['sentiment'].dropna().map(label_map)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize_texts(df):
        df['message'] = df['message'].fillna('').astype(str)
        encodings = tokenizer(
            df['message'].tolist(),
            truncation=True,
            padding=True,
            max_length=args.max_length,
            return_tensors='pt'
        )
        return ClimateDataset(
            {k: v.squeeze() for k, v in encodings.items()},
            torch.tensor(df['sentiment'].tolist())
        )

    # Create datasets
    train_dataset = tokenize_texts(train_df)
    dev_dataset = tokenize_texts(dev_df)
    test_dataset = tokenize_texts(test_df)

    # Model
    model = ClimateModel()

    # Training args
    training_args = get_training_args(
        output_dir='./results/BERT',
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    # Train
    trainer.train()

    # Save best model
    trainer.save_model('./saved_models/BERT')

    # Evaluate on test set
    test_outputs = trainer.predict(test_dataset)
    predicted_labels = test_outputs.predictions.argmax(-1)
    true_labels = test_df['sentiment'].values

    target_names = ['anti', 'neutral', 'pro']
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=target_names))

    # Save results
    bert_results = pd.DataFrame({
        'message': test_df['message'],
        'true_sentiment': test_df['sentiment'].map({0: 'anti', 1: 'neutral', 2: 'pro'}),
        'bert_sentiment': pd.Series(predicted_labels).map({0: 'anti', 1: 'neutral', 2: 'pro'})
    })
    bert_results.to_csv('results/BERT/bert_results.csv', index=False)

    # Example predictions
    print("\nExample predictions:")
    example_texts = [
        "Climate change is a real threat that requires immediate action.", #Pro
        "I'm not sure if humans are causing climate change or if it's natural.", #Neutral
        "Global warming is a hoax created by scientists for grant money." #anti
    ]
    predictions = predict_sentiment(model, tokenizer, example_texts, batch_size=args.batch_size)
    for text, pred in zip(example_texts, predictions):
        print(f"Text: {text}")
        print(f"Prediction: {pred}\n")

if __name__ == "__main__":
    main()
