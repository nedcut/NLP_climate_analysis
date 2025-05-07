import pandas as pd
import torch
from transformers import AutoTokenizer
from utils import predict_sentiment
import matplotlib.pyplot as plt
import os

def main():
    # Define paths
    model_path = './saved_models/BERT'
    input_path = 'data/cleaned_news_tweets.csv'
    output_csv_path = 'results/news_tweets_predictions.csv'
    output_chart_path = 'results/news_sentiment.png'

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
    
    state_dict_path = os.path.join(model_path, 'model.safetensors')
    
    print(f"Loading model weights from {state_dict_path}...")
    from safetensors.torch import load_file
    state_dict = load_file(state_dict_path)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    df = df['message'].astype(str).tolist()
    
    # predict sentiment
    print("Predicting sentiment...")
    predictions = predict_sentiment(model, tokenizer, df, device=device)
    
    # output dataframe
    output_df = pd.DataFrame({
        'message': df,
        'predicted_sentiment': predictions
    })
    
    # save predictions to CSV
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")
    
    # visualize results
    print("Visualizing results...")
    
    sentiment_counts = output_df['predicted_sentiment'].value_counts(normalize=True)
    labels = sentiment_counts.index.tolist()
    sizes = sentiment_counts.values.tolist()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title('News Tweets Sentiment Distribution')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Define the output path for the chart, ensuring the directory exists
    os.makedirs(os.path.dirname(output_chart_path), exist_ok=True)
    plt.savefig(output_chart_path)
    print(f"Sentiment distribution chart saved to {output_chart_path}")
    plt.show()

if __name__ == "__main__":
    main()

