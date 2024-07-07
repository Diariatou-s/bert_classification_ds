import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel
import torch
import pandas as pd
from torch.optim import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class NetflixDataset(Dataset):
    def __init__(self, df, model_name_or_path="bert-base-uncased", max_len=250):
        self.df = df.dropna(subset=['Summary', 'Series or Movie'])
        self.labels = sorted(self.df["Series or Movie"].unique())
        self.labels_dict = {label: i for i, label in enumerate(self.labels)}
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        summary_text = self.df["Summary"].iloc[index]
        labels = self.labels_dict[self.df["Series or Movie"].iloc[index]]
        inputs = self.tokenizer(summary_text, padding="max_length", max_length=self.max_len, truncation=True, return_tensors="pt")
        
        return {
            "inputs": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels
        }
        
class CustomBert(nn.Module):
    def __init__(self, name_or_model_path="bert-base-uncased", num_classes=2):
        super(CustomBert, self).__init__()
        self.bert_pretrained = BertModel.from_pretrained(name_or_model_path)
        self.classifier = nn.Linear(self.bert_pretrained.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        x = self.bert_pretrained(input_ids=input_ids, attention_mask=attention_mask)
        x = self.classifier(x.pooler_output)
        return x
        
def training_step(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for data in tqdm(dataloader, total=len(dataloader)):
        print(data)
        input_ids = data["input_ids"].squeeze(0)
        attention_mask = data["attention_mask"].squeeze(0)
        labels = data["labels"]
        
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        
        loss = loss_fn(output, labels)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader.dataset)

def evaluation(model, test_dataloader, loss_fn):
    model.eval()
    correct_predictions = 0
    losses = []
    for data in tqdm(test_dataloader, total=len(test_dataloader)):
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        labels = data["labels"]
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        _, pred = output.max(1)
        correct_predictions += torch(pred==labels)
        loss = loss_fn(output, labels)
        losses.append(loss.item())
        
    return np.mean(losses), correct_predictions / len(test_dataloader.dataset)

def main():

    full_df = pd.read_csv("data/netflix-rotten-tomatoes-metacritic-imdb.csv")
    train_df, test_df = train_test_split(full_df, test_size=0.3, random_state=42)
    train_dataset = NetflixDataset(train_df, max_len=100)
    test_dataset = NetflixDataset(test_df, max_len=100)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=2)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=2)
    
    n_epoch = 8
    
    model = CustomBert()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.005)
    for _ in range(n_epoch):
        loss_train = training_step(model, train_dataloader, loss_fn, optimizer)
        loss_eval, accuracy = evaluation(model, test_dataloader, loss_fn)
        
    print(
        f"Train Loss : {loss_train} | Eval Loss : {loss_eval} | Accuracy : {accuracy}"
    )

    torch.save(model.state_dict(), 'models/model_ds_netflix.pth')

if __name__ == "__main__":
    main()