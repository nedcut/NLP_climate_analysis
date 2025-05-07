import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Load the results
df = pd.read_csv(r'results\BERT\bert_results.csv')

# Drop rows with missing values (if any)
df = df.dropna(subset=['true_sentiment', 'bert_sentiment'])

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(df['true_sentiment'], df['bert_sentiment'], labels=['pro', 'neutral', 'anti']))

# Print classification report
report = classification_report(
    df['true_sentiment'],
    df['bert_sentiment'],
    labels=['pro', 'neutral', 'anti'],
    output_dict=True,
    zero_division=0
)
report_df = pd.DataFrame(report).transpose()
print("\nClassification Report:")
print(report_df.round(3))