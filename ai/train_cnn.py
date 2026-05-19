import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from cnn_builder import render_stroke_entry
from cnn_model import get_cnn_model
from training_data import load_training_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


class StrokeDataset(Dataset):
    def __init__(self, entries, size: int = 64):
        self.entries = list(entries)
        self.size = size

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img = render_stroke_entry(entry, size=self.size)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        label = ord(entry["label"].upper()) - 65
        return img_tensor, label


def train_cnn() -> None:
    models_dir = Path(os.getenv("MODELS_DIR", str(DEFAULT_MODELS_DIR))).expanduser()
    models_dir.mkdir(parents=True, exist_ok=True)

    split = load_training_split()
    dataset = StrokeDataset(split["train_samples"])
    if len(dataset) < 30:
        print("Not enough training data for CNN training.")
        return

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = get_cnn_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(10):
        total_loss = 0.0
        for imgs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

    pth_path = models_dir / "cnn_model.pth"
    torch.save(model.state_dict(), pth_path)
    os.chmod(pth_path, 0o644)

    model.eval()
    dummy_input = torch.randn(1, 1, 64, 64)
    onnx_path = models_dir / "cnn.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )
    os.chmod(onnx_path, 0o644)
    print(
        "CNN retrained and exported to ONNX "
        f"with {len(split['train_samples'])} train samples and {len(split['high_quality_eval'])} HQ-eval samples."
    )


if __name__ == "__main__":
    train_cnn()
