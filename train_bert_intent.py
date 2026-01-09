#!/usr/bin/env python3
"""
BERT 意圖分類器訓練腳本

支援環境：
- M1/M2 Mac (MPS 加速)
- CUDA GPU
- CPU (較慢)

模型：hfl/rbt3 (38M 參數，輕量級中文 BERT)
"""

import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import argparse

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# ==============================================================================
# 配置
# ==============================================================================

@dataclass
class Config:
    model_name: str = "hfl/rbt3"  # 輕量級中文 BERT
    max_length: int = 64  # 智慧家居指令通常很短
    num_labels: int = 5  # turn_on, turn_off, climate_set_mode, get_state, chat
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # 路徑
    train_file: str = "bert_training_data/intent_train.jsonl"
    test_file: str = "bert_training_data/intent_test.jsonl"
    output_dir: str = "bert_intent_model"

# 標籤名稱
LABEL_NAMES = ["turn_on", "turn_off", "climate_set_mode", "get_state", "chat"]

# ==============================================================================
# 資料集
# ==============================================================================

class IntentDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self.samples.append({
                        "text": data["text"],
                        "label": data["label"]
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        encoding = self.tokenizer(
            sample["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(sample["label"], dtype=torch.long)
        }

# ==============================================================================
# 訓練函數
# ==============================================================================

def compute_metrics(eval_pred):
    """計算評估指標"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def get_device():
    """自動檢測最佳設備"""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🚀 使用 CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("🍎 使用 Apple MPS")
    else:
        device = "cpu"
        print("💻 使用 CPU")
    
    return device

def main():
    parser = argparse.ArgumentParser(description="訓練 BERT 意圖分類器")
    parser.add_argument("--epochs", type=int, default=5, help="訓練輪數")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=2e-5, help="學習率")
    parser.add_argument("--model", type=str, default="hfl/rbt3", help="預訓練模型")
    args = parser.parse_args()
    
    config = Config(
        model_name=args.model,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
    )
    
    print("=" * 60)
    print("BERT 意圖分類器訓練")
    print("=" * 60)
    print(f"模型: {config.model_name}")
    print(f"批次大小: {config.batch_size}")
    print(f"學習率: {config.learning_rate}")
    print(f"訓練輪數: {config.num_epochs}")
    print("=" * 60)
    
    # 檢測設備
    device = get_device()
    
    # 載入 tokenizer 和模型
    print("\n📥 載入模型和 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        id2label={i: name for i, name in enumerate(LABEL_NAMES)},
        label2id={name: i for i, name in enumerate(LABEL_NAMES)},
    )
    
    # 載入資料集
    print("\n📂 載入資料集...")
    train_dataset = IntentDataset(config.train_file, tokenizer, config.max_length)
    test_dataset = IntentDataset(config.test_file, tokenizer, config.max_length)
    
    print(f"   訓練集: {len(train_dataset)} 筆")
    print(f"   測試集: {len(test_dataset)} 筆")
    
    # 統計標籤分布
    train_labels = [s["label"] for s in train_dataset.samples]
    print("\n📊 訓練集標籤分布:")
    for label_id, label_name in enumerate(LABEL_NAMES):
        count = train_labels.count(label_id)
        print(f"   {label_name}: {count}")
    
    # 訓練參數
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",  # 不使用 wandb 等
        # MPS 特定設定
        use_mps_device=(device == "mps"),
        dataloader_num_workers=0 if device == "mps" else 4,
    )
    
    # 建立 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    # 開始訓練
    print("\n🏋️ 開始訓練...")
    trainer.train()
    
    # 評估
    print("\n📈 評估結果:")
    results = trainer.evaluate()
    for key, value in results.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
    
    # 儲存模型
    print(f"\n💾 儲存模型到: {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # 顯示混淆矩陣
    print("\n🔢 混淆矩陣:")
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids
    
    cm = confusion_matrix(true_labels, pred_labels)
    print("       " + "  ".join([f"{name[:6]:>6}" for name in LABEL_NAMES]))
    for i, row in enumerate(cm):
        print(f"{LABEL_NAMES[i][:6]:>6} " + "  ".join([f"{val:>6}" for val in row]))
    
    print("\n✅ 訓練完成！")
    print(f"   模型位置: {config.output_dir}")
    print(f"   標籤數量: {config.num_labels}")
    print(f"   標籤列表: {LABEL_NAMES}")

if __name__ == "__main__":
    main()
