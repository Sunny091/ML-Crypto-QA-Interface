"""
Transformer 模型訓練腳本

功能：
1. 訓練 Transformer 模型
2. 記錄訓練過程 (loss, accuracy, metrics)
3. 輸出訓練報告到檔案
4. 儲存最佳模型

輸出檔案：
- models/artifacts/transformer.pt (模型權重)
- evaluation/results/training_log.csv (訓練日誌)
- evaluation/results/training_report.txt (訓練報告)
- evaluation/results/training_curves.png (訓練曲線圖)

Usage:
    python -m models.transformer.train
"""

import sys
import json
import time
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer.model import create_model
from models.transformer.data_processor import DataProcessor


class TrainingLogger:
    """訓練日誌記錄器"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
            "timestamp": []
        }
        self.start_time = None
        self.config = {}

    def set_config(self, config: dict):
        """設定訓練配置"""
        self.config = config

    def start(self):
        """開始訓練"""
        self.start_time = time.time()
        print("=" * 60)
        print("Transformer 模型訓練開始")
        print(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float
    ):
        """記錄 epoch 結果"""
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        self.history["learning_rate"].append(lr)
        self.history["timestamp"].append(datetime.now().isoformat())

        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {lr:.6f}")

    def save_csv(self):
        """儲存訓練日誌為 CSV"""
        import pandas as pd
        df = pd.DataFrame(self.history)
        csv_path = self.output_dir / "training_log.csv"
        df.to_csv(csv_path, index=False)
        print(f"訓練日誌已儲存: {csv_path}")

    def save_report(self, test_metrics: dict):
        """儲存訓練報告"""
        elapsed = time.time() - self.start_time

        report = f"""
================================================================================
                    Transformer 模型訓練報告
================================================================================

訓練時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
總耗時: {elapsed:.2f} 秒 ({elapsed/60:.2f} 分鐘)

--------------------------------------------------------------------------------
                              模型配置
--------------------------------------------------------------------------------
{json.dumps(self.config.get('model', {}), indent=2, ensure_ascii=False)}

--------------------------------------------------------------------------------
                              訓練配置
--------------------------------------------------------------------------------
- Epochs: {self.config.get('epochs', 'N/A')}
- Batch Size: {self.config.get('batch_size', 'N/A')}
- Learning Rate: {self.config.get('learning_rate', 'N/A')}
- Sequence Length: {self.config.get('seq_len', 'N/A')}
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau

--------------------------------------------------------------------------------
                              資料集資訊
--------------------------------------------------------------------------------
{json.dumps(self.config.get('data_info', {}), indent=2, ensure_ascii=False)}

--------------------------------------------------------------------------------
                              訓練結果
--------------------------------------------------------------------------------
最佳驗證準確率: {max(self.history['val_acc']):.4f} (Epoch {self.history['val_acc'].index(max(self.history['val_acc'])) + 1})
最低驗證損失: {min(self.history['val_loss']):.4f}
最終訓練準確率: {self.history['train_acc'][-1]:.4f}
最終驗證準確率: {self.history['val_acc'][-1]:.4f}

--------------------------------------------------------------------------------
                              測試集評估
--------------------------------------------------------------------------------
- Test Accuracy: {test_metrics.get('accuracy', 0):.4f}
- Test Precision: {test_metrics.get('precision', 0):.4f}
- Test Recall: {test_metrics.get('recall', 0):.4f}
- Test F1 Score: {test_metrics.get('f1', 0):.4f}

混淆矩陣:
{test_metrics.get('confusion_matrix', 'N/A')}

--------------------------------------------------------------------------------
                              學習心得
--------------------------------------------------------------------------------
1. 模型使用 Transformer 架構，能夠捕捉時間序列中的長期依賴關係
2. 使用多種技術指標作為特徵 (RSI, 波動率, MA ratio 等)
3. 採用 Early Stopping 避免過擬合
4. 使用 Learning Rate Scheduler 動態調整學習率

================================================================================
"""
        report_path = self.output_dir / "training_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"訓練報告已儲存: {report_path}")

    def plot_curves(self):
        """繪製訓練曲線"""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Loss 曲線
            axes[0].plot(self.history['epoch'], self.history['train_loss'], label='Train Loss')
            axes[0].plot(self.history['epoch'], self.history['val_loss'], label='Val Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training & Validation Loss')
            axes[0].legend()
            axes[0].grid(True)

            # Accuracy 曲線
            axes[1].plot(self.history['epoch'], self.history['train_acc'], label='Train Acc')
            axes[1].plot(self.history['epoch'], self.history['val_acc'], label='Val Acc')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Training & Validation Accuracy')
            axes[1].legend()
            axes[1].grid(True)

            plt.tight_layout()
            plot_path = self.output_dir / "training_curves.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"訓練曲線已儲存: {plot_path}")

        except ImportError:
            print("matplotlib 未安裝，跳過繪圖")


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None, grad_clip=1.0):
    """訓練一個 epoch（帶梯度裁剪）"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()

        # 梯度裁剪
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # 更新學習率（每個 step）
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * len(y)
        _, predicted = outputs.max(1)
        correct += predicted.eq(y).sum().item()
        total += len(y)

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    """評估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item() * len(y)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += len(y)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def compute_metrics(preds: List, labels: List) -> Dict:
    """計算評估指標"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    preds = np.array(preds)
    labels = np.array(labels)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "confusion_matrix": str(confusion_matrix(labels, preds))
    }


def main():
    """主訓練流程（優化版）"""
    # 設定裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # 優化配置（v3 - 最終版）
    config = {
        "model": {
            "input_dim": 20,        # 特徵數
            "d_model": 64,          # 模型容量
            "nhead": 4,             # 注意力頭
            "num_layers": 2,        # 淺層（避免過擬合）
            "dim_feedforward": 128, # FFN 維度
            "dropout": 0.4,         # 高正則化
            "seq_len": 30,          # 序列長度
            "num_classes": 2
        },
        "epochs": 200,              # 更多 epochs
        "batch_size": 32,           # batch size
        "learning_rate": 0.0002,    # 低學習率
        "weight_decay": 0.1,        # 強 L2 正則化
        "seq_len": 30,
        "patience": 30,             # 更多耐心
        "warmup_epochs": 15,        # 長預熱
        "label_smoothing": 0.1      # 標籤平滑
    }

    # 準備資料
    processor = DataProcessor(seq_len=config["seq_len"])
    train_loader, val_loader, test_loader, data_info = processor.prepare_data(
        "BTC",
        batch_size=config["batch_size"]
    )
    config["data_info"] = data_info

    # 建立模型
    model = create_model(config["model"]).to(device)
    print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")

    # 計算類別權重（處理不平衡）
    train_labels = []
    for _, y in train_loader:
        train_labels.extend(y.numpy())
    train_labels = np.array(train_labels)
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * 2  # 歸一化
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"類別權重: DOWN={class_weights[0]:.4f}, UP={class_weights[1]:.4f}")

    # 損失函數（帶標籤平滑和類別權重）
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config["label_smoothing"]
    )

    # 優化器（AdamW 帶權重衰減）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    # 學習率排程器（帶預熱的餘弦退火）
    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = len(train_loader) * config["warmup_epochs"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 日誌記錄器
    output_dir = PROJECT_ROOT / "evaluation" / "results"
    logger = TrainingLogger(output_dir)
    logger.set_config(config)
    logger.start()

    # 訓練
    best_val_acc = 0
    patience_counter = 0
    model_save_path = PROJECT_ROOT / "models" / "artifacts" / "transformer.pt"

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scheduler=scheduler, grad_clip=1.0
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)

        # 儲存最佳模型（根據驗證準確率）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config
            }, model_save_path)
            patience_counter = 0
            print(f"  ✓ 新最佳模型 (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config["patience"]:
            print(f"\nEarly stopping at epoch {epoch} (patience={config['patience']})")
            break

    print("\n" + "=" * 60)
    print("訓練完成！")
    print("=" * 60)

    # 載入最佳模型進行測試
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 測試集評估
    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_preds, test_labels)
    print(f"\n測試集準確率: {test_metrics['accuracy']:.4f}")
    print(f"測試集 F1 Score: {test_metrics['f1']:.4f}")

    # 儲存結果
    logger.save_csv()
    logger.save_report(test_metrics)
    logger.plot_curves()

    # 儲存 scaler 參數（用於預測時的正規化）
    scaler_path = PROJECT_ROOT / "models" / "artifacts" / "transformer_scaler.json"
    with open(scaler_path, 'w') as f:
        json.dump({
            'mean': processor.scaler_params['mean'].tolist(),
            'std': processor.scaler_params['std'].tolist(),
            'features': processor.FEATURES
        }, f)
    print(f"Scaler 參數已儲存: {scaler_path}")

    print(f"\n模型已儲存: {model_save_path}")
    print("訓練完成！")


if __name__ == "__main__":
    main()
