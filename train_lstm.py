from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

project_dir = Path(__file__).resolve().parent
raw_dir = project_dir / "data" / "raw"
models_dir = project_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

cols = [
    "engine_id", "cycle",
    "setting_1", "setting_2", "setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21"
]

df = pd.read_csv(raw_dir / "train_FD001.txt", sep=r"\s+", header=None)
df.columns = cols
df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

max_cycle = df.groupby("engine_id")["cycle"].transform("max")
df["RUL"] = max_cycle - df["cycle"]
df["RUL"] = df["RUL"].clip(upper=125)

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
window_size = 30

X_seq, y_seq = [], []

for engine_id, group in df.groupby("engine_id"):
    group = group.sort_values("cycle").reset_index(drop=True)
    values = group[sensor_cols].values
    rul_values = group["RUL"].values

    for i in range(len(values) - window_size + 1):
        X_seq.append(values[i:i+window_size])
        y_seq.append(rul_values[i+window_size-1])

X_seq = np.asarray(X_seq, dtype=np.float32)
y_seq = np.asarray(y_seq, dtype=np.float32)

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=64, num_layers=2, dropout=0.2, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out.squeeze(-1)

idx = np.arange(len(X_seq))
np.random.seed(42)
np.random.shuffle(idx)

split = int(0.8 * len(X_seq))
train_idx = idx[:split]
val_idx = idx[split:]

X_train = torch.tensor(X_seq[train_idx], dtype=torch.float32)
y_train = torch.tensor(y_seq[train_idx], dtype=torch.float32)
X_val = torch.tensor(X_seq[val_idx], dtype=torch.float32)
y_val = torch.tensor(y_seq[val_idx], dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)

model = LSTMRegressor(input_dim=X_seq.shape[2])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

best_val_loss = float("inf")
patience = 5
counter = 0

for epoch in range(15):
    model.train()
    train_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            val_loss += loss.item() * xb.size(0)

    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}/15 | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping.")
        break

torch.save(model.state_dict(), models_dir / "lstm_model.pt")
print("LSTM model saved.")