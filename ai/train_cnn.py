import json
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
from cnn_builder import render_stroke_entry
from cnn_model import get_cnn_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset.jsonl"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


def iter_dataset_files():
    dataset_path = Path(os.getenv("DATA_PATH", str(DEFAULT_DATASET_PATH))).expanduser()
    dataset_dir = Path(os.getenv("DATA_DIR", str(DEFAULT_DATASET_DIR))).expanduser()
    files = []

    if dataset_path.exists() and dataset_path.is_file():
        files.append(dataset_path)

    if dataset_dir.exists():
        files.extend(sorted(
            path for path in dataset_dir.glob("*.jsonl")
            if path.is_file()
        ))

    return files

class StrokeDataset(Dataset):
    def __init__(self, data_paths, size=64):
        self.entries = []
        for data_path in data_paths:
            if os.path.exists(data_path):
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            entry = json.loads(line)
                            if 'label' in entry and 'strokes' in entry:
                                self.entries.append(entry)
                        except Exception:
                            pass
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
    models_dir = os.getenv("MODELS_DIR", str(DEFAULT_MODELS_DIR))
    os.makedirs(models_dir, exist_ok=True)

    dataset = StrokeDataset(iter_dataset_files())
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

    # Save PTH
    pth_path = os.path.join(models_dir, "cnn_model.pth")
    torch.save(model.state_dict(), pth_path)
    os.chmod(pth_path, 0o644)

    # Save ONNX
    model.eval()
    dummy_input = torch.randn(1, 1, 64, 64)
    onnx_path = os.path.join(models_dir, "cnn.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=17,
        dynamo=False,
    )
    os.chmod(onnx_path, 0o644)
    print(f"CNN model saved to ONNX at {onnx_path}")

if __name__ == "__main__":
    train_cnn()
