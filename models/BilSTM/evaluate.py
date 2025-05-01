import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def predict(model, tokenizer, text_list, word_index, embedding_dim=100, max_len=100, device='cpu'):
    """
    Predict sentiment for a list of input texts using the trained BiLSTM model.
    Returns the predicted sentiment labels.
    """
    from keras.preprocessing.sequence import pad_sequences # type: ignore
    from keras.preprocessing.text import Tokenizer # type: ignore

    model.eval()
    model.to(device)

    # Use the same tokenizer that was fit on training data
    sequences = tokenizer.texts_to_sequences(text_list)
    padded = pad_sequences(sequences, maxlen=max_len)
    inputs = torch.tensor(padded, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    label_map = {0: 'anti', 1: 'neutral', 2: 'pro'}
    return [label_map[p] for p in preds]

def evaluate_model(model, test_loader, device, label_names):
    """
    Evaluate the model on the test set and print metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_names))
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, label_names)


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("bilstm_confusion_matrix.png")
    plt.close()
