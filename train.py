import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# =========================
# 参数
# =========================
SEQ_LEN = 24
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model/transformer_model.pth"

# =========================
# 创建目录
# =========================
os.makedirs("model", exist_ok=True)

# =========================
# 数据读取
# =========================
df = pd.read_csv("data/load_data.csv")

values = df['load'].values.reshape(-1,1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

# =========================
# 构造时间序列
# =========================
def create_sequences(data, seq_len):
    X = []
    y = []

    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])

    return np.array(X), np.array(y)

X, y = create_sequences(scaled, SEQ_LEN)

split = int(len(X)*0.8)

X_train = X[:split]
y_train = y[:split]

class LoadDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(
    LoadDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =========================
# Positional Encoding
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# =========================
# Transformer
# =========================
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(1, 64)

        self.pos_encoder = PositionalEncoding(64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        x = self.embedding(x)

        x = self.pos_encoder(x)

        x = self.transformer(x)

        x = x[:, -1, :]

        return self.fc(x)

# =========================
# 初始化
# =========================
model = TransformerModel().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================
# 训练
# =========================
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss/len(train_loader):.6f}")

# =========================
# 保存模型
# =========================
torch.save(model.state_dict(), MODEL_PATH)

print("模型训练完成")
print(f"模型保存至: {MODEL_PATH}")