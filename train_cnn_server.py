import json
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from cnn_builder import render_stroke_entry
from cnn_model import get_cnn_model

class StrokeDataset(Dataset):
    def __init__(self, data_path, size=64):
        self.entries = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    entry = json.loads(line)
                    if 'label' in entry and 'strokes' in entry:
                        self.entries.append(entry)
                except: pass
        self.size = size

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img = render_stroke_entry(entry, size=self.size)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        label = ord(entry['label'].upper()) - 65
        return img_tensor, label

def train_cnn():
    dataset = StrokeDataset("dataset.jsonl")
    if len(dataset) < 30:
        print("Not enough data for CNN training")
        return

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = get_cnn_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for imgs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

    torch.save(model.state_dict(), "cnn_model.pth")
    print("CNN model saved to cnn_model.pth")

if __name__ == "__main__":
    train_cnn()
