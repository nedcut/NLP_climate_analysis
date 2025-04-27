import os
import pandas as pd
import torch
import kagglehub
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model import Climodel
from torch import nn, optim
from tqdm import tqdm

# use gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# custom dataset
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

# load data
data_path = kagglehub.dataset_download("edqian/twitter-climate-change-sentiment-dataset")
df = pd.read_csv(data_path)

# filter to only positive, negative, neutral
# sentiment: 1=pro, 0=neutral, -1=anti, 2=news (drop)
df = df[df['sentiment'].isin([1, 0, -1])].copy()

# use 'message' as text and 'sentiment' as label
texts = df['message'].tolist()
labels = df['sentiment'].tolist()

# encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# train/dev split
X_train, X_dev, y_train, y_dev = train_test_split(texts, labels, test_size=0.1, random_state=42)

# initialize model
model = Climodel().to(device)

# datasets and loaders
train_dataset = TweetDataset(X_train, y_train, model.tokenizer)
dev_dataset = TweetDataset(X_dev, y_dev, model.tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total

# training loop
EPOCHS = 3
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = eval_epoch(model, dev_loader, criterion)
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# save model
torch.save(model.state_dict(), 'climodel_sentiment.pt')
print("Training complete. Model saved as climodel_sentiment.pt")
