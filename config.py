"""
配置文件
"""
import torch

# 数据参数
SEQ_LEN = 24
DATA_PATH = "data/load_data.csv"

# 训练参数
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model/transformer_model.pth"

# 模型参数
D_MODEL = 64
N_HEAD = 8
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
NUM_LAYERS = 3

# 可视化参数
VISUALIZATION_DIR = "visualizations"
SAVE_VISUALIZATION = True
SHOW_VISUALIZATION = False