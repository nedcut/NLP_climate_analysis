import re
import pandas as pd

def clean_message(text):
    text = re.sub(r'^RT\s+@\w+:\s*', '', text)  # remove retweet prefix
    text = re.sub(r'@\w+', '', text)            # remove usernames
    text = text.lower().strip()                 # convert to lowercase and strip whitespace
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # remove URLs
    return text

# paths
path_in = 'data/Biden_Climate Change tweets_sentiments.csv'
path_out = 'data/biden_cleaned_tweets.csv'

# load data
df = pd.read_csv(path_in)

# clean data
df = df["Text"].astype(str).apply(clean_message)

# remove duplicates
df = df.drop_duplicates()

df.to_csv(path_out, index=False)