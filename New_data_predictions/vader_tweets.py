import os
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.VADER.vader import get_compound_score, get_vader_sentiment

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '../../data/biden_cleaned_tweets.csv')
output_dir = os.path.join(script_dir, '../../results/VADER')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'biden_vader_predictions.csv')

# Load data
df = pd.read_csv(data_path)
df['message'] = df['message'].astype(str)


# Apply predictions
df['compound'] = df['message'].apply(get_compound_score)
df['vader_sentiment'] = df['compound'].apply(get_vader_sentiment)

# Save predictions
df[['message', 'compound', 'vader_sentiment']].to_csv(output_path, index=False)

# Preview
print("\nSample predictions:")
print(df[['message', 'vader_sentiment']].sample(5, random_state=42))

# Plot sentiment distribution
sentiment_counts = df['vader_sentiment'].value_counts().reindex(['anti', 'neutral', 'pro'], fill_value=0)
plt.figure(figsize=(8, 5))
sentiment_counts.plot(kind='bar')
plt.title("VADER Sentiment Distribution (Biden Tweets)")
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.grid(axis='y')
plt_path = os.path.join(output_dir, 'biden_vader_sentiment_distribution.png')
plt.savefig(plt_path)
plt.show()

print(f"\nSentiment distribution plot saved to {plt_path}")
