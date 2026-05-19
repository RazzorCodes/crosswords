import json
import pickle
import os
import sys
from trainer_1 import train

def load_dataset(path):
    data = []
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data.append(json.loads(line))
            except: pass
    return data

if __name__ == "__main__":
    dataset_path = "dataset.jsonl"
    data = load_dataset(dataset_path)

    if len(data) < 10:
        sys.exit(0)

    clf = train(data)

    # Save to a temporary file first then rename to avoid corruption
    with open("letter_clf.pkl.tmp", "wb") as f:
        pickle.dump(clf, f)
    os.rename("letter_clf.pkl.tmp", "letter_clf.pkl")
    print(f"SVM retrained and saved with {len(data)} samples.")
