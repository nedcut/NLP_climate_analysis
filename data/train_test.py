import pandas as pd
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def clean_message(text):
    text = re.sub(r'^RT\s+@\w+:\s*', '', text)  # remove retweet prefix
    text = re.sub(r'@\w+', '', text)            # remove usernames
    text = text.lower().strip()                 # convert to lowercase and strip whitespace
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # remove URLs
    return text

# Load and clean data
df = pd.read_csv('data/twitter_sentiment_data.csv')
df['message'] = df['message'].astype(str).apply(clean_message)
df = df[df['sentiment'] != 2]  # Remove 'news' category

# First split: Train+Dev and Test
train_val_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['sentiment']
)

# Second split: Train and Dev (80% train, 20% dev of the 80%)
train_df, dev_df = train_test_split(
    train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['sentiment']
)

# Save to CSV
train_df.to_csv('data/train_data.csv', index=False)
dev_df.to_csv('data/dev_data.csv', index=False)
test_df.to_csv('data/test_data.csv', index=False)

# Optional: Visualize sentiment distributions
for name, split in [('Training', train_df), ('Development', dev_df), ('Test', test_df)]:
    sentiment_counts = split['sentiment'].value_counts()
    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title(f'Sentiment Distribution in {name} Set')
    plt.ylabel('')
    plt.show()
