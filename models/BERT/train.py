import pandas as pd
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, Trainer
from sklearn.metrics import classification_report
from models.BERT.model import ClimateModel, ClimateDataset
from utils import get_training_args, compute_metrics, predict_sentiment

def main():
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('data/train_data.csv')
    test_df = pd.read_csv('data/test_data.csv')

    # Remap sentiment labels: -1 -> 0 (anti), 0 -> 1 (neutral), 1 -> 2 (pro)
    train_df['sentiment'] = train_df['sentiment'].map({-1: 0, 0: 1, 1: 2})
    test_df['sentiment'] = test_df['sentiment'].map({-1: 0, 0: 1, 1: 2})
    print("Train sentiment unique values (remapped):", train_df['sentiment'].unique())
    print("Test sentiment unique values (remapped):", test_df['sentiment'].unique())
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Ensure all messages are strings and handle potential NaN values
    train_df['message'] = train_df['message'].fillna('').astype(str)
    test_df['message'] = test_df['message'].fillna('').astype(str)
    
    # Check for class imbalance and compute class weights
    class_labels = np.unique(train_df['sentiment'])
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=class_labels,
        y=train_df['sentiment']
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print("Class weights:", class_weights)
    
    # Tokenize texts
    print("Tokenizing texts...")
    train_encodings = tokenizer(
        train_df['message'].tolist(), 
        truncation=True, 
        padding=True, 
        max_length=128,
        return_tensors='pt'
    )
    
    test_encodings = tokenizer(
        test_df['message'].tolist(), 
        truncation=True, 
        padding=True, 
        max_length=128,
        return_tensors='pt'
    )
    
    # Convert encodings to dataset format
    train_dataset = ClimateDataset(
        {k: v.squeeze() for k, v in train_encodings.items()}, 
        torch.tensor(train_df['sentiment'].tolist())
    )
    
    test_dataset = ClimateDataset(
        {k: v.squeeze() for k, v in test_encodings.items()}, 
        torch.tensor(test_df['sentiment'].tolist())
    )
    
    # Initialize model
    model = ClimateModel(class_weights=class_weights)
    
    # Get training arguments
    training_args = get_training_args(output_dir='./results/BERT')
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train model
    print("Training model...")
    trainer.train()
    
    # Evaluate model
    print("Evaluating model...")
    evaluation = trainer.evaluate()
    print(f"Evaluation results: {evaluation}")
    
    # Generate predictions for test data
    print("Generating predictions...")
    test_outputs = trainer.predict(test_dataset)
    predicted_labels = test_outputs.predictions.argmax(-1)
    true_labels = test_df['sentiment'].values
    
    # Generate classification report
    target_names = ['anti', 'neutral', 'pro']
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=target_names))
    
    # Save BERT results for comparison
    bert_results = pd.DataFrame({
        'message': test_df['message'],
        'true_sentiment': test_df['sentiment'].map({0: 'anti', 1: 'neutral', 2: 'pro'}),
        'bert_sentiment': pd.Series(predicted_labels).map({0: 'anti', 1: 'neutral', 2: 'pro'})
    })
    bert_results.to_csv('results/BERT/bert_results.csv', index=False)
    print("BERT results saved to results/BERT/bert_results.csv")
    
    # Save model
    print("Saving model...")
    trainer.save_model('./saved_models/BERT')
    tokenizer.save_pretrained('./saved_models/BERT')
    
    print("Model training and evaluation complete!")
    
    # Example prediction
    example_texts = [
        "Climate change is a real threat that requires immediate action.",          # pro
        "I'm not sure if humans are causing climate change or if it's natural.",    # neutral
        "Global warming is a hoax created by scientists for grant money."           # anti
    ]
    
    predictions = predict_sentiment(model, tokenizer, example_texts)
    
    print("\nExample predictions:")
    for text, pred in zip(example_texts, predictions):
        print(f"Text: {text}")
        print(f"Prediction: {pred}\n")

if __name__ == "__main__":
    main()