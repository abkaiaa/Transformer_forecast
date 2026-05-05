"""
可视化工具
"""
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch
from datetime import datetime

# 尝试设置中文字体
def _setup_chinese_font():
    """设置中文字体"""
    candidates = [
        'Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi',
        'FangSong', 'STHeiti', 'WenQuanYi Micro Hei',
        'Noto Sans CJK SC', 'Source Han Sans CN'
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available:
            plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return font_name
    # 如果没找到中文字体，使用英文标签
    return None

_CN_FONT = _setup_chinese_font()


class Visualizer:
    """可视化类"""
    def __init__(self, save_dir="visualizations", show=False):
        self.save_dir = save_dir
        self.show = show
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练历史记录
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
    def update_training_history(self, epoch, train_loss, val_loss=None):
        """更新训练历史"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
    
    def plot_training_loss(self, epoch):
        """绘制训练损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, 'b-', label='训练损失')
        if self.val_losses:
            plt.plot(self.epochs, self.val_losses, 'r-', label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.title(f'训练损失曲线 (Epoch {epoch})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if self.save_dir:
            filepath = os.path.join(self.save_dir, f'training_loss_epoch_{epoch}.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
        
        if self.show:
            plt.show()
        else:
            plt.close()
    
    def plot_prediction_comparison(self, y_true, y_pred, epoch, title="预测对比"):
        """绘制预测对比图"""
        plt.figure(figsize=(12, 6))
        
        # 绘制真实值
        plt.subplot(1, 2, 1)
        plt.plot(y_true, 'b-', label='真实值', alpha=0.7)
        plt.plot(y_pred, 'r--', label='预测值', alpha=0.7)
        plt.xlabel('时间步')
        plt.ylabel('负荷')
        plt.title(f'{title} (Epoch {epoch})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 绘制散点图
        plt.subplot(1, 2, 2)
        plt.scatter(y_true, y_pred, alpha=0.5, s=20)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('预测 vs 真实')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_dir:
            filepath = os.path.join(self.save_dir, f'prediction_epoch_{epoch}.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
        
        if self.show:
            plt.show()
        else:
            plt.close()
    
    def plot_data_distribution(self, data, title="数据分布"):
        """绘制数据分布图"""
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(data, bins=50, alpha=0.7, color='blue')
        plt.xlabel('负荷值')
        plt.ylabel('频数')
        plt.title(f'{title} - 直方图')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(data, 'b-', alpha=0.7)
        plt.xlabel('时间')
        plt.ylabel('负荷值')
        plt.title(f'{title} - 时序图')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_dir:
            filepath = os.path.join(self.save_dir, 'data_distribution.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
        
        if self.show:
            plt.show()
        else:
            plt.close()
    
    def plot_model_architecture(self, model, input_shape):
        """绘制模型架构图（简化版）"""
        plt.figure(figsize=(12, 8))
        
        # 获取模型信息
        layers = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                layers.append((name, module))
        
        # 绘制模型架构
        y_positions = np.linspace(0.1, 0.9, len(layers))
        
        for i, (name, layer) in enumerate(layers):
            plt.text(0.5, y_positions[i], f'{name}: {layer.__class__.__name__}', 
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.title('模型架构')
        plt.axis('off')
        
        if self.save_dir:
            filepath = os.path.join(self.save_dir, 'model_architecture.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
        
        if self.show:
            plt.show()
        else:
            plt.close()
    
    def create_training_summary(self, epoch, train_loss, val_loss=None, 
                              learning_rate=None, additional_info=None):
        """创建训练摘要图"""
        plt.figure(figsize=(12, 8))
        
        # 训练信息
        info_text = f"Epoch: {epoch}\n"
        info_text += f"训练损失: {train_loss:.6f}\n"
        if val_loss is not None:
            info_text += f"验证损失: {val_loss:.6f}\n"
        if learning_rate is not None:
            info_text += f"学习率: {learning_rate:.6f}\n"
        if additional_info:
            info_text += additional_info
        
        plt.subplot(2, 2, 1)
        plt.text(0.1, 0.5, info_text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.axis('off')
        plt.title('训练信息')
        
        # 损失曲线
        plt.subplot(2, 2, 2)
        if self.train_losses:
            plt.plot(self.epochs, self.train_losses, 'b-', label='训练损失')
            if self.val_losses:
                plt.plot(self.epochs, self.val_losses, 'r-', label='验证损失')
            plt.xlabel('Epoch')
            plt.ylabel('损失')
            plt.title('损失曲线')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 损失分布
        plt.subplot(2, 2, 3)
        if len(self.train_losses) > 1:
            plt.hist(self.train_losses, bins=20, alpha=0.7, color='blue')
            plt.xlabel('损失值')
            plt.ylabel('频数')
            plt.title('损失分布')
            plt.grid(True, alpha=0.3)
        
        # 最近损失趋势
        plt.subplot(2, 2, 4)
        if len(self.train_losses) > 5:
            recent_losses = self.train_losses[-10:]
            plt.plot(recent_losses, 'b-o', label='最近损失')
            plt.xlabel('最近Epoch')
            plt.ylabel('损失')
            plt.title('最近损失趋势')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.save_dir, f'training_summary_epoch_{epoch}_{timestamp}.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
        
        if self.show:
            plt.show()
        else:
            plt.close()