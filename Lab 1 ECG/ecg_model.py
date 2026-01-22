import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


LABEL_MAP = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}


class ECGCSVDataset(Dataset):
    def __init__(self, csv_path: str):
        arr = pd.read_csv(csv_path, header=None).values.astype(np.float32)
        self.X = arr[:, :-1]
        self.y = arr[:, -1].astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).unsqueeze(0)  # (1, L)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


class SimpleECGCNN(nn.Module):
    def __init__(self, input_len: int, n_classes: int,
                 c1: int = 32, c2: int = 64, kernel: int = 5,
                 dropout: float = 0.2, dense: int = 64):
        super().__init__()
        pad = kernel // 2
        self.features = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=kernel, padding=pad),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(c1, c2, kernel_size=kernel, padding=pad),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            out = self.features(dummy)
            flat = out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flat, dense),
            nn.ReLU(),
            nn.Linear(dense, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def make_class_weights_from_csv(train_csv: str, n_classes: int):
    y = pd.read_csv(train_csv, header=None).iloc[:, -1].astype(int).values
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    total = counts.sum()
    weights = total / (n_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * yb.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_y = []
    all_p = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += loss.item() * yb.size(0)
        pred = torch.argmax(logits, dim=1)

        correct += (pred == yb).sum().item()
        total += yb.size(0)

        all_y.append(yb.cpu().numpy())
        all_p.append(pred.cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)
    return total_loss / total, correct / total, y_true, y_pred


def plot_history(hist):
    epochs = np.arange(1, len(hist["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, hist["train_loss"], label="train_loss")
    plt.plot(epochs, hist["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, hist["train_acc"], label="train_acc")
    plt.plot(epochs, hist["val_acc"], label="val_acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def plot_confusion(cm, labels):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.show()


def run_experiment(
    train_csv="mitbih_train.csv",
    test_csv="mitbih_test.csv",
    epochs=15,
    batch_size=256,
    lr=1e-3,
    c1=32,
    c2=64,
    kernel=5,
    dropout=0.2,
    dense=64,
    seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ECGCSVDataset(train_csv)
    test_ds = ECGCSVDataset(test_csv)

    input_len = train_ds.X.shape[1]
    n_classes = int(np.max(train_ds.y)) + 1

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    class_weights = make_class_weights_from_csv(train_csv, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = SimpleECGCNN(
        input_len=input_len,
        n_classes=n_classes,
        c1=c1, c2=c2, kernel=kernel,
        dropout=dropout, dense=dense
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, _, _ = eval_model(model, test_loader, criterion, device)

        hist["train_loss"].append(tr_loss)
        hist["train_acc"].append(tr_acc)
        hist["val_loss"].append(va_loss)
        hist["val_acc"].append(va_acc)

        print(f"Epoch {ep:02d}/{epochs} | train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | test_loss={va_loss:.4f} test_acc={va_acc:.4f}")

    test_loss, test_acc, y_true, y_pred = eval_model(model, test_loader, criterion, device)

    labels = [LABEL_MAP[i] for i in range(n_classes)]
    cm = confusion_matrix(y_true, y_pred)

    print("\nFinal Test Accuracy:", float(test_acc))
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))

    plot_history(hist)
    plot_confusion(cm, labels)

    return {
        "test_acc": float(test_acc),
        "params": {"epochs": epochs, "batch_size": batch_size, "lr": lr, "c1": c1, "c2": c2, "kernel": kernel, "dropout": dropout, "dense": dense},
        "cm": cm
    }


if __name__ == "__main__":
    run_experiment()
