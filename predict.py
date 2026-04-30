import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

SEQ_LEN = 24
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 读取数据
# =========================
df = pd.read_csv("data/load_data.csv")

values = df['load'].values.reshape(-1,1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

# =========================
# PositionalEncoding
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
# 模型
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
# 加载模型
# =========================
model = TransformerModel().to(DEVICE)

model.load_state_dict(
    torch.load("model/transformer_model.pth", weights_only=False)
)

model.eval()

# =========================
# 最近24小时预测下一小时
# =========================
latest_seq = scaled[-SEQ_LEN:]

input_data = torch.FloatTensor(latest_seq).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred = model(input_data)

pred_value = scaler.inverse_transform(pred.cpu().numpy())

print("下一小时预测负荷:")
print(pred_value[0][0])

# =========================
# 可视化
# =========================
plt.figure(figsize=(10,5))

plt.plot(values[-100:], label="Historical Load")
plt.scatter(len(values), pred_value[0][0], s=100, label="Forecast")

plt.legend()
plt.title("Transformer Load Forecast")
plt.show()