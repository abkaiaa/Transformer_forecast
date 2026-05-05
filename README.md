# 电力负荷预测MVP
基于Transformer的短期电力负荷预测模型，使用历史负荷数据实现逐时预测。

## 技术栈
- Python 3.12
- PyTorch 2.11
- Pandas / NumPy / Scikit-learn
- Matplotlib

## 项目结构
```
load_forecast_mvp/
├── config.py            # 配置文件（参数集中管理）
├── models.py            # 模型定义（Transformer + 位置编码）
├── data_utils.py        # 数据处理工具（加载、预处理、序列化）
├── visualization.py     # 可视化工具（训练曲线、预测对比、流程图）
├── train.py             # 训练脚本
├── predict.py           # 预测脚本
├── data/
│   └── load_data.csv    # 历史负荷数据
├── model/
│   └── transformer_model.pth  # 训练好的模型权重
└── visualizations/      # 自动生成的可视化图片（每次迭代）
```

## 项目功能
- **模块化架构**：配置、模型、数据、可视化分离，易于维护和扩展
- **数据预处理**：归一化、时间序列构造、训练/验证集划分
- **模型训练**：Transformer编码器+全连接层，支持梯度裁剪、学习率调度、验证集评估
- **模型预测**：单步预测（下一小时）+ 多步预测（未来24小时）
- **迭代可视化**：每5个epoch自动生成训练Loss曲线、预测对比图、训练摘要图
- **预测可视化**：历史数据+预测值、多步预测曲线、置信区间、预测分布

## 运行方式
1. 安装依赖：`pip install -r requirements.txt`
2. 运行训练：`python train.py`
3. 运行预测：`python predict.py`

训练过程中，每5个epoch会在 `visualizations/` 目录下自动生成可视化图片，包括：
- `training_loss_epoch_X.png` — 训练/验证损失曲线
- `prediction_epoch_X.png` — 验证集预测对比散点图
- `training_summary_epoch_X.png` — 训练摘要（损失曲线+损失分布+趋势）
- `data_distribution.png` — 数据分布图

## 项目进度
- ✅ 基础模型训练完成
- ✅ 数据预处理模块
- ✅ 模块化重构
- ✅ 验证集评估 + 学习率调度
- ✅ 每次迭代流程图输出
- ✅ 多步预测（24小时）
- ⬜ 多模型对比（LSTM/GRU/Transformer）
- ⬜ 网页端可视化界面
