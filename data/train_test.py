import pandas as pd
import re
from sklearn.model_selection import train_test_split

def clean_message(text):
    text = re.sub(r'^RT\s+@\w+:\s*', '', text)  # remove retweet prefix
    text = re.sub(r'@\w+', '', text)            # remove usernames
    text = text.lower().strip()                 # convert to lowercase and strip whitespace
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # remove URLs
    return text

# load data
df = pd.read_csv('twitter_sentiment_data.csv')

# clean data
df['message'] = df['message'].astype(str).apply(clean_message)

# get rid of rows with sentiment == 2 (news)
df = df[df['sentiment'] != 2]

# print top 10 rows (for testing)
# print(df.head(10))

# split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentiment'])

# save train and test sets to csv files
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

# show a pie chart of the sentiment distribution in the training set
# kinda just did this for practice with matplotlib
import matplotlib.pyplot as plt
train_sentiment_counts = train_df['sentiment'].value_counts()
train_sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Sentiment Distribution in Training Set')
plt.ylabel('')
plt.show()
# show a pie chart of the sentiment distribution in the test set
test_sentiment_counts = test_df['sentiment'].value_counts()     