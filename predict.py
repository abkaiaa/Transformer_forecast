"""
预测脚本 - 优化版
支持：模块化结构、多步预测、可视化输出
"""
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from config import *
from models import TransformerModel
from data_utils import load_data, preprocess_data
from visualization import Visualizer


def load_model(model_path):
    """加载模型"""
    model = TransformerModel(
        d_model=D_MODEL,
        nhead=N_HEAD,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(model_path, weights_only=False, map_location=DEVICE)
    )
    model.eval()
    return model


def predict_next_hour(model, scaler, values, seq_len=SEQ_LEN):
    """预测下一小时负荷"""
    # 取最近 seq_len 小时的数据
    latest_seq = values[-seq_len:]
    scaled_seq = scaler.transform(latest_seq.reshape(-1, 1))

    input_data = torch.FloatTensor(scaled_seq).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(input_data)

    pred_value = scaler.inverse_transform(pred.cpu().numpy())
    return pred_value[0][0]


def predict_multi_step(model, scaler, values, steps=24, seq_len=SEQ_LEN):
    """多步预测"""
    predictions = []
    current_seq = values[-seq_len:].copy()

    for _ in range(steps):
        scaled_seq = scaler.transform(current_seq.reshape(-1, 1))
        input_data = torch.FloatTensor(scaled_seq).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(input_data)

        pred_value = scaler.inverse_transform(pred.cpu().numpy())[0][0]
        predictions.append(pred_value)

        # 滑动窗口更新
        current_seq = np.append(current_seq[1:], pred_value)

    return np.array(predictions)


def main():
    """主函数"""
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    # 初始化可视化器
    visualizer = Visualizer(save_dir=VISUALIZATION_DIR, show=SHOW_VISUALIZATION)

    # 加载数据
    print("=" * 50)
    print("📥 加载数据...")
    values = load_data(DATA_PATH)
    scaled, scaler = preprocess_data(values)
    print(f"数据总量: {len(values)} 条")

    # 加载模型
    print("\n🔧 加载模型...")
    model = load_model(MODEL_PATH)
    print(f"模型加载完成: {MODEL_PATH}")

    # 单步预测：下一小时
    print("\n" + "=" * 50)
    print("🔮 预测下一小时负荷...")
    next_hour_pred = predict_next_hour(model, scaler, values)
    print(f"下一小时预测负荷: {next_hour_pred:.2f}")
    print(f"最近一小时实际负荷: {values[-1][0]:.2f}")

    # 多步预测：未来24小时
    print("\n" + "=" * 50)
    print("🔮 预测未来24小时负荷...")
    multi_step_preds = predict_multi_step(model, scaler, values, steps=24)
    for i, pred in enumerate(multi_step_preds, 1):
        print(f"  +{i:2d}小时: {pred:.2f}")

    # =========================
    # 可视化
    # =========================
    print("\n" + "=" * 50)
    print("🖼️ 生成可视化...")

    # 图1: 历史数据 + 单步预测
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 子图1: 最近100小时历史 + 下一小时预测
    ax1 = axes[0, 0]
    recent = values[-100:]
    ax1.plot(range(len(recent)), recent, 'b-', label="历史负荷", linewidth=1.5)
    ax1.scatter(len(recent), next_hour_pred, s=150, c='red', zorder=5,
               label=f"预测值: {next_hour_pred:.2f}", edgecolors='black')
    ax1.set_xlabel("时间步 (小时)")
    ax1.set_ylabel("负荷")
    ax1.set_title("下一小时负荷预测")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 子图2: 未来24小时预测
    ax2 = axes[0, 1]
    hours = np.arange(1, 25)
    ax2.plot(hours, multi_step_preds, 'r-o', linewidth=2, markersize=5,
            label="预测负荷")
    ax2.fill_between(hours,
                     multi_step_preds * 0.95,
                     multi_step_preds * 1.05,
                     alpha=0.2, color='red', label="±5% 置信区间")
    ax2.set_xlabel("未来时间 (小时)")
    ax2.set_ylabel("负荷")
    ax2.set_title("未来24小时负荷预测")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 子图3: 最近200小时趋势 + 24小时预测衔接
    ax3 = axes[1, 0]
    historical = values[-200:]
    all_values = np.concatenate([historical.flatten(), multi_step_preds])
    ax3.plot(range(len(historical)), historical, 'b-', label="历史负荷", linewidth=1.5)
    ax3.plot(range(len(historical) - 1, len(historical) + 24),
             np.concatenate([[historical[-1][0]], multi_step_preds]),
             'r--o', linewidth=2, markersize=4, label="预测负荷")
    ax3.axvline(x=len(historical) - 1, color='gray', linestyle=':', alpha=0.7)
    ax3.set_xlabel("时间步 (小时)")
    ax3.set_ylabel("负荷")
    ax3.set_title("历史负荷与预测衔接")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 子图4: 预测值分布
    ax4 = axes[1, 1]
    ax4.hist(multi_step_preds, bins=12, color='coral', edgecolor='black', alpha=0.7)
    ax4.axvline(x=next_hour_pred, color='red', linestyle='--', linewidth=2,
               label=f"下一小时: {next_hour_pred:.2f}")
    ax4.set_xlabel("负荷值")
    ax4.set_ylabel("频数")
    ax4.set_title("未来24小时预测分布")
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(VISUALIZATION_DIR, f"prediction_{timestamp}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 预测可视化已保存: {save_path}")

    plt.show()

    print("\n✅ 预测完成！")


if __name__ == "__main__":
    main()