"""
数据处理工具
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch


def load_data(data_path):
    """加载数据"""
    df = pd.read_csv(data_path)
    values = df['load'].values.reshape(-1, 1)
    return values


def preprocess_data(values):
    """数据预处理"""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    return scaled, scaler


def create_sequences(data, seq_len):
    """创建时间序列"""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


class LoadDataset(Dataset):
    """负荷数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_data_loaders(X, y, batch_size, train_ratio=0.8):
    """创建数据加载器"""
    split = int(len(X) * train_ratio)
    
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    
    train_dataset = LoadDataset(X_train, y_train)
    val_dataset = LoadDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader