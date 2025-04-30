import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from model import ClimateModel

def load_model(model_path):
    """Load the saved model"""
    model = ClimateModel()
    model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location=torch.device('cpu')))
    return model

def predict(model, tokenizer, texts, device='cpu'):
    """Run prediction on a list of texts"""
    model.to(device)
    model.eval()
    
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
    encodings = {k: v.to(device) for k, v in encodings.items()}
    
    with torch.no_grad():
        outputs = model(**encodings)
    
    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    return predictions

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    model_path = './saved_model'
    test_data_path = 'test_data.csv'
    class_names = ['anti', 'neutral', 'pro']
    
    # Load the model and tokenizer
    print("Loading model and tokenizer...")
    model = load_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(test_data_path)
    
    # Make predictions
    print("Making predictions...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions = predict(model, tokenizer, test_df['message'].tolist(), device)
    
    # Evaluate predictions
    print("Evaluating predictions...")
    true_labels = test_df['sentiment'].values
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    report = classification_report(true_labels, predictions, target_names=class_names)
    print(report)
    
    # Create and save confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plot_confusion_matrix(cm, class_names)
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    # Interactive prediction mode
    print("\nEnter text to classify (or type 'exit' to quit):")
    while True:
        text = input("> ")
        if text.lower() == 'exit':
            break
        
        pred = predict(model, tokenizer, [text], device)[0]
        sentiment = class_names[pred]
        print(f"Predicted sentiment: {sentiment}")

if __name__ == "__main__":
    main()