import pandas as pd
import json
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader

# Custom dataset for training BERT
class BERTDataset(torch.utils.data.Dataset):
    """
      Dataset for BBC News data.

      Args:
          - encodings (dict): A dictionary containing the encoded inputs by Tokenizer.
          - labels (list or array): A list or array containing the labels corresponding to the data.

    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class BERTReg(nn.Module):
	def __init__(self, model):
		super(BERTReg, self).__init__()
		self.bert = model
		for param in self.bert.parameters():
			param.requires_grad = False
		self.regressor = nn.Linear(768, 1)

	def forward(self, input_ids):
		outputs = self.bert(**input_ids)
		pooled_output = outputs.pooler_output
		rating = self.regressor(pooled_output)
		return rating


if __name__ == "__main__":
    
    df = pd.read_csv('ratebeer_subset.csv')
    texts = df['review/text'].values
    ratings = df['review/overall'].values

    X_train, X_test, y_train, y_test = train_test_split(texts, ratings, test_size=0.2, random_state=42)
    
    # Example usage
    torch.cuda.empty_cache()
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    model_name = 'bert-base-uncased'
    bert_model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Define the model
    model = BERTReg(bert_model)
    model.to(device)

    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss(reduction='sum')
    batch_size = 256

    # Training loop
    epochs = 10
    
    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True)
    
    train_dataset = BERTDataset(train_encodings, y_train)
    val_dataset = BERTDataset(val_encodings, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].float().to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(X_train)}")
        train_loss.append(total_loss/len(X_train))

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation"):
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].float().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                total_loss += loss.item()
            print(f"Validation Loss: {total_loss/len(X_test)}")
            val_loss.append(total_loss/len(X_test))

    figure = plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Curve')
    plt.savefig('loss_curve.png')