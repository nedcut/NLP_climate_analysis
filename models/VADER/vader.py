import os
import nltk
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, f1_score

# Setup
nltk.download('vader_lexicon')

# Initialize VADER
sia = SentimentIntensityAnalyzer()

def get_compound_score(text):
    return sia.polarity_scores(text)['compound']

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../../data')
results_dir = os.path.join(script_dir, '../../results/VADER')
os.makedirs(results_dir, exist_ok=True)

# Load data
train_df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
dev_df = pd.read_csv(os.path.join(data_dir, 'dev_data.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
biden_df = pd.read_csv(os.path.join(data_dir, 'biden_cleaned_tweets.csv'))

# Label mapping
label_map = {-1: 'anti', 0: 'neutral', 1: 'pro'}
for df in [train_df, dev_df, test_df]:
    df['true_sentiment'] = df['sentiment'].map(label_map).astype(str)
    df['message'] = df['message'].astype(str)
    df['compound'] = df['message'].apply(lambda x: sia.polarity_scores(x)['compound'])


def get_vader_sentiment(score, pos_threshold, neg_threshold):
    if score >= pos_threshold:
        return 'pro'
    elif score <= neg_threshold:
        return 'anti'
    else:
        return 'neutral'


best_f1 = 0
best_thresh = (None, None)
true_dev_labels = dev_df['true_sentiment']

#Tuning thresholds on dev set
for pos in np.arange(0.01, 0.5, 0.05):
    for neg in np.arange(-0.01, -0.5, -0.05):
        preds = dev_df['compound'].apply(lambda s: get_vader_sentiment(s, pos, neg))
        f1 = f1_score(true_dev_labels, preds, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = (neg, pos)

neg_thresh, pos_thresh = best_thresh
print(f"Best thresholds found: neg={neg_thresh}, pos={pos_thresh}, F1={best_f1:.4f}")

# Apply best thresholds to all sets
for name, df in zip(['Train', 'Dev', 'Test'], [train_df, dev_df, test_df]):
    df['vader_sentiment'] = df['compound'].apply(lambda s: get_vader_sentiment(s, pos_thresh, neg_thresh))
    print(f"\n{name} Set Classification Report:")
    print(classification_report(
        df['true_sentiment'],
        df['vader_sentiment'],
        labels=['anti', 'neutral', 'pro'],
        target_names=['anti', 'neutral', 'pro']
    ))

# Test on unlabeled biden data, giving each tweet a sentiment label
biden_df['compound'] = biden_df['Text'].apply(lambda x: sia.polarity_scores(x)['compound'])
biden_df['vader_sentiment'] = biden_df['compound'].apply(lambda s: get_vader_sentiment(s, pos_thresh, neg_thresh))
biden_results_path = os.path.join(results_dir, 'biden_results.csv')
biden_df[['Text', 'vader_sentiment']].to_csv(biden_results_path, index=False)

# Save test results
test_results_path = os.path.join(results_dir, 'vader_results.csv')
test_df[['message', 'true_sentiment', 'vader_sentiment']].to_csv(test_results_path, index=False)

# Show common misclassifications
print("\n🔎 Top Misclassified Examples (Test Set):")
misclassified = test_df[test_df['true_sentiment'] != test_df['vader_sentiment']]
print(misclassified[['message', 'true_sentiment', 'vader_sentiment']].sample(10, random_state=42))
