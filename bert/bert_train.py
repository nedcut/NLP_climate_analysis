import pandas as pd
import torch
from transformers import AutoTokenizer, Trainer
from sklearn.metrics import classification_report
from NLP_climate_analysis.models.BERT.model import ClimateModel, ClimateDataset, compute_metrics, get_training_args, predict_sentiment

def main():
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data.csv')
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Ensure all messages are strings and handle potential NaN values
    train_df['message'] = train_df['message'].fillna('').astype(str)
    test_df['message'] = test_df['message'].fillna('').astype(str)
    
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
    model = ClimateModel()
    
    # Get training arguments
    training_args = get_training_args(output_dir='./results')
    
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
    
    # Save model
    print("Saving model...")
    trainer.save_model('./saved_model')
    tokenizer.save_pretrained('./saved_model')
    
    print("Model training and evaluation complete!")
    
    # Example prediction
    example_texts = [
        "Climate change is a real threat that requires immediate action.",
        "I'm not sure if humans are causing climate change or if it's natural.",
        "Global warming is a hoax created by scientists for grant money."
    ]
    
    predictions = predict_sentiment(model, tokenizer, example_texts)
    
    print("\nExample predictions:")
    for text, pred in zip(example_texts, predictions):
        print(f"Text: {text}")
        print(f"Prediction: {pred}\n")

if __name__ == "__main__":
    main()