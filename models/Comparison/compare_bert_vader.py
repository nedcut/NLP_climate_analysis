import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Load results
vader = pd.read_csv('results/VADER/vader_results.csv')
bert = pd.read_csv('results/BERT/bert_results.csv')

# Merge on message and true_sentiment
merged = pd.merge(vader, bert, on=['message', 'true_sentiment'], suffixes=('_vader', '_bert'))

# Agreement statistics
agreement = (merged['vader_sentiment'] == merged['bert_sentiment']).mean()
print(f"Agreement between VADER and BERT: {agreement:.2%}")

# Confusion matrix: VADER vs BERT
cm = confusion_matrix(merged['vader_sentiment'], merged['bert_sentiment'], labels=['anti', 'neutral', 'pro'])
print("\nVADER vs BERT Confusion Matrix:")
print(pd.DataFrame(cm, index=['VADER_anti', 'VADER_neutral', 'VADER_pro'], columns=['BERT_anti', 'BERT_neutral', 'BERT_pro']))

# Classification reports
print("\nClassification Report (BERT vs True):")
print(classification_report(merged['true_sentiment'], merged['bert_sentiment'], labels=['anti', 'neutral', 'pro']))

print("\nClassification Report (VADER vs True):")
print(classification_report(merged['true_sentiment'], merged['vader_sentiment'], labels=['anti', 'neutral', 'pro']))

# Visualization: Heatmap of confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['BERT_anti', 'BERT_neutral', 'BERT_pro'],
            yticklabels=['VADER_anti', 'VADER_neutral', 'VADER_pro'])
plt.title('VADER vs BERT Sentiment Confusion Matrix')
plt.xlabel('BERT Prediction')
plt.ylabel('VADER Prediction')
plt.tight_layout()
plt.show()