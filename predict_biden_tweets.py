import pandas as pd
import torch
from transformers import AutoTokenizer
from utils import predict_sentiment
import matplotlib.pyplot as plt
import os

def main():
    # Define paths
    model_path = './saved_models/BERT'
    input_csv_path = 'data/biden_cleaned_tweets.csv'
    output_csv_path = 'results/biden_tweets_predictions.csv'
    output_chart_path = 'results/biden_vs_training_sentiment_comparison.png'
    training_data_path = 'data/train_data.csv'

    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load ClimateModel
    print(f"Loading custom ClimateModel from {model_path}...")
    from models.BERT.model import ClimateModel # Path to ClimateModel definition
    
    # Instantiate model
    model = ClimateModel() 
    
    # Determine the path to the model weights file
    state_dict_path = os.path.join(model_path, 'model.safetensors')
    if not os.path.exists(state_dict_path):
        state_dict_path = os.path.join(model_path, 'pytorch_model.bin') # Fallback

    if os.path.exists(state_dict_path):
        print(f"Loading model weights from {state_dict_path}...")
        if state_dict_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(state_dict_path)
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))
        print("Custom ClimateModel weights loaded successfully.")
    else:
        print(f"Error: Could not find model weights file (model.safetensors or pytorch_model.bin) in {model_path}.")
        print("Please ensure the model was saved correctly and the path is accurate.")
        return # Exit if weights are not found

    model.to(device)
    model.eval()

    # Load data
    print(f"Loading data from {input_csv_path}...")
    try:
        biden_df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_path}")
        return
    
    text_column = 'Text'
    print(f"Using column '{text_column}' for tweet messages.")
    tweets = biden_df[text_column].fillna('').astype(str).tolist()

    if not tweets:
        print("No tweets found in the input file.")
        return

    # Predict sentiment
    print("Predicting sentiments...")
    predictions = predict_sentiment(model, tokenizer, tweets, device=device)
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'message': tweets,
        'sentiment': predictions
    })

    # Save predictions
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

    # Visualize sentiment distribution
    print("Visualizing sentiment distribution...")
    
    # Load training data for comparison
    train_df = pd.read_csv(training_data_path)
    sentiment_map_train = {-1: 'anti', 0: 'neutral', 1: 'pro'}
    train_df['sentiment_label'] = train_df['sentiment'].map(sentiment_map_train)
    train_sentiment_counts = train_df['sentiment_label'].value_counts(normalize=True)

    biden_sentiment_counts = output_df['sentiment'].value_counts(normalize=True)
    
    labels = biden_sentiment_counts.index.tolist()
    biden_sizes = [biden_sentiment_counts.get(label, 0) for label in labels]

    fig, axs = plt.subplots(1, 2 if train_sentiment_counts is not None else 1, figsize=(12 if train_sentiment_counts is not None else 6, 6))
    
    if train_sentiment_counts is not None:
        train_labels = train_sentiment_counts.index.tolist()
        # Ensure consistent ordering and colors for labels
        all_labels = sorted(list(set(labels + train_labels)))
        
        biden_sizes_ordered = [biden_sentiment_counts.get(label, 0) for label in all_labels]
        train_sizes_ordered = [train_sentiment_counts.get(label, 0) for label in all_labels]

        axs[0].pie(train_sizes_ordered, labels=all_labels, autopct='%1.1f%%', startangle=90)
        axs[0].set_title('Training Data Sentiment (2015-2018)')
        axs[0].axis('equal')

        axs[1].pie(biden_sizes_ordered, labels=all_labels, autopct='%1.1f%%', startangle=90)
        axs[1].set_title('Biden Tweets Sentiment (2023)')
        axs[1].axis('equal')
    else:
        ax_single = axs # if only one subplot, axs is not an array
        ax_single.pie(biden_sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax_single.set_title('Biden Tweets Sentiment (2023)')
        ax_single.axis('equal')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_chart_path), exist_ok=True)
    plt.savefig(output_chart_path)
    print(f"Sentiment comparison chart saved to {output_chart_path}")
    plt.show()

if __name__ == "__main__":
    main()