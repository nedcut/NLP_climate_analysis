import torch
import torch.nn as nn
import torch.nn.functional as F

def train_model(model, train_loader, optimizer, device):
    """
    Train the BiLSTM model on the training set using CrossEntropyLoss.
    """
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # Raw logits
        loss = criterion(outputs, labels)  
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)
