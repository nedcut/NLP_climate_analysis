import pandas as pd
import torch
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from BilSTM.dataset import TweetDataset
from BilSTM.model import BiLSTMModel
from BilSTM.train import train_model
from BilSTM.evaluate import evaluate_model
from BilSTM.utils import load_glove_embeddings

def main():
    MAX_LEN = 100
    BATCH_SIZE = 64
    EMBEDDING_DIM = 100
    NUM_EPOCHS = 5
    GLOVE_PATH = "embeddings/glove.6B.100d.txt"
    CLASS_NAMES = ['anti', 'neutral', 'pro']

    # Load data
    train_df = pd.read_csv("data/train_data.csv")
    test_df = pd.read_csv("data/test_data.csv")

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_df['message'])
    word_index = tokenizer.word_index

    X_train = pad_sequences(tokenizer.texts_to_sequences(train_df['message']), maxlen=MAX_LEN)
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_df['message']), maxlen=MAX_LEN)
    y_train = train_df['sentiment'].values
    y_test = test_df['sentiment'].values

    # Load embeddings
    embedding_matrix = load_glove_embeddings(GLOVE_PATH, word_index, EMBEDDING_DIM)

    # Model and training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMModel(embedding_matrix).to(device)

    train_loader = DataLoader(TweetDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TweetDataset(X_test, y_test), batch_size=BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train loop
    for epoch in range(NUM_EPOCHS):
        loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss:.4f}")

    # Evaluate
    evaluate_model(model, test_loader, device, CLASS_NAMES)

if __name__ == "__main__":
    main()
