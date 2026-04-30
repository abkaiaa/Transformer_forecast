# 电力负荷预测MVP
基于Transformer的短期电力负荷预测模型，使用历史负荷数据+气象数据实现72小时逐时预测。

## 技术栈
- Python 3.12
- PyTorch 2.11
- Pandas / NumPy / Scikit-learn
- Matplotlib

## 项目功能
- 数据预处理：缺失值填充、归一化、时序特征工程
- 模型训练：Transformer编码器+全连接层预测
- 模型评估：MAE、RMSE、MAPE指标计算
- 结果可视化：训练Loss曲线、预测值与真实值对比

## 运行方式
1.  安装依赖：`pip install -r requirements.txt`
2.  运行训练：`python train.py`
3.  运行预测：`python predict.py`

## 项目进度
- ✅ 基础模型训练完成
- ✅ 数据预处理模块
- ⬜ 多模型对比（LSTM/GRU/Transformer）
- ⬜ 网页端可视化界面
