import pandas as pd
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, Trainer
from sklearn.metrics import classification_report
from model import ClimateModel, ClimateDataset
from utils import get_training_args, compute_metrics, predict_sentiment

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_df = pd.read_csv('data/train_data.csv')
    dev_df = pd.read_csv('data/dev_data.csv')
    test_df = pd.read_csv('data/test_data.csv')

    # Label remapping: -1 → 0, 0 → 1, 1 → 2
    label_map = {-1: 0, 0: 1, 1: 2}
    train_df['sentiment'] = train_df['sentiment'].map(label_map)
    dev_df['sentiment'] = dev_df['sentiment'].map(label_map)
    test_df['sentiment'] = test_df['sentiment'].map(label_map)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize_texts(df):
        df['message'] = df['message'].fillna('').astype(str)
        encodings = tokenizer(
            df['message'].tolist(),
            truncation=True,
            padding=True,
            max_length=128,
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
    training_args = get_training_args(output_dir='./results/BERT')

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
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
    predictions = predict_sentiment(model, tokenizer, example_texts)
    for text, pred in zip(example_texts, predictions):
        print(f"Text: {text}")
        print(f"Prediction: {pred}\n")

if __name__ == "__main__":
    main()
