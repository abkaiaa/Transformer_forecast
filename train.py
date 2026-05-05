"""
训练脚本 - 优化版
支持：模块化结构、验证集、学习率调度、梯度裁剪、迭代可视化
"""
import os
import torch
import torch.nn as nn
from config import *
from models import TransformerModel
from data_utils import load_data, preprocess_data, create_sequences, create_data_loaders
from visualization import Visualizer


def train_model():
    """训练模型"""
    # 创建目录
    os.makedirs("model", exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    # 初始化可视化器
    visualizer = Visualizer(save_dir=VISUALIZATION_DIR, show=SHOW_VISUALIZATION)

    # 加载数据
    print("=" * 50)
    print("📥 加载数据...")
    print("=" * 50)
    values = load_data(DATA_PATH)
    scaled, scaler = preprocess_data(values)

    # 绘制数据分布
    visualizer.plot_data_distribution(values, "原始负荷数据")
    print(f"数据总量: {len(values)} 条")
    print(f"数据范围: [{values.min():.2f}, {values.max():.2f}]")

    # 创建序列
    print("\n📊 创建时间序列...")
    X, y = create_sequences(scaled, SEQ_LEN)
    print(f"序列数量: {len(X)} 条 (序列长度: {SEQ_LEN})")

    # 创建数据加载器（含验证集）
    train_loader, val_loader = create_data_loaders(X, y, BATCH_SIZE)
    print(f"训练集: {len(train_loader.dataset)} 条")
    print(f"验证集: {len(val_loader.dataset)} 条")

    # 初始化模型
    print("\n🏗️ 初始化模型...")
    model = TransformerModel(
        d_model=D_MODEL,
        nhead=N_HEAD,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 绘制模型架构
    visualizer.plot_model_architecture(model, (SEQ_LEN, 1))

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 训练循环
    print("\n🚀 开始训练...")
    print("=" * 50)
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        avg_val_loss = None
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.to(DEVICE)
                    y_val = y_val.to(DEVICE)
                    outputs = model(X_val)
                    loss = criterion(outputs, y_val)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            # 学习率调度
            scheduler.step(avg_val_loss)

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), MODEL_PATH)

        # 更新可视化历史
        visualizer.update_training_history(epoch, avg_train_loss, avg_val_loss)

        # 定期生成可视化
        if epoch % 5 == 0 or epoch == EPOCHS:
            # 绘制训练损失曲线
            visualizer.plot_training_loss(epoch)

            # 创建训练摘要图
            current_lr = optimizer.param_groups[0]['lr']
            visualizer.create_training_summary(
                epoch, avg_train_loss, avg_val_loss,
                learning_rate=current_lr
            )

            # 验证集预测对比
            if val_loader:
                model.eval()
                with torch.no_grad():
                    sample_X, sample_y = next(iter(val_loader))
                    sample_X = sample_X.to(DEVICE)
                    predictions = model(sample_X).cpu().numpy()
                    true_values = sample_y.numpy()

                visualizer.plot_prediction_comparison(
                    true_values, predictions, epoch,
                    title="验证集预测对比"
                )

        # 打印训练信息
        if avg_val_loss is not None:
            status = "⭐ 最佳" if avg_val_loss <= best_val_loss else ""
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"训练损失: {avg_train_loss:.6f} | "
                  f"验证损失: {avg_val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f} {status}")
        else:
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"训练损失: {avg_train_loss:.6f}")

    # 保存最终模型
    if not val_loader:
        torch.save(model.state_dict(), MODEL_PATH)

    # 生成最终可视化报告
    print("\n" + "=" * 50)
    print("📈 生成最终可视化报告...")
    visualizer.create_training_summary(
        EPOCHS, avg_train_loss,
        avg_val_loss if val_loader else None,
        additional_info=f"最佳验证损失: {best_val_loss:.6f}"
    )

    print("\n✅ 训练完成！")
    print(f"📊 最佳验证损失: {best_val_loss:.6f}")
    print(f"💾 模型保存至: {MODEL_PATH}")
    print(f"🖼️ 可视化保存至: {VISUALIZATION_DIR}/")

    return model, scaler


if __name__ == "__main__":
    train_model()