import os
import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report

# Setup
nltk.download('vader_lexicon')
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, '../../data/test_data.csv')
df = pd.read_csv(file_path)

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Label mapping
label_map = {-1: 'anti', 0: 'neutral', 1: 'pro'}
df['true_sentiment'] = df['sentiment'].map(label_map)

def get_vader_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score > 0:
        return 'pro'
    elif score < 0:
        return 'anti'
    else:
        return 'neutral'

# Apply sentiment classification
df['vader_sentiment'] = df['message'].astype(str).apply(get_vader_sentiment)

# Ensure clean string types
df['true_sentiment'] = df['true_sentiment'].astype(str)
df['vader_sentiment'] = df['vader_sentiment'].astype(str)

# Save VADER results
vader_results_path = os.path.join(script_dir, '../../results/VADER/vader_results.csv')
df[['message', 'true_sentiment', 'vader_sentiment']].to_csv(vader_results_path, index=False)
print(f"VADER results saved to {vader_results_path}")

# Evaluation
print("Classification Report")
print(classification_report(
    df['true_sentiment'],
    df['vader_sentiment'],
    labels=['anti', 'neutral', 'pro'],
    target_names=['anti', 'neutral', 'pro']
))

print(df['true_sentiment'].value_counts())

# Print common misclassifications
print("\nTop Misclassified Examples:")
misclassified = df[df['true_sentiment'] != df['vader_sentiment']]
print(misclassified[['message', 'true_sentiment', 'vader_sentiment']].sample(10, random_state=42))
