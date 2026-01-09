#!/usr/bin/env python3
"""
BERT 意圖分類 + 填槽 聯合訓練腳本

架構：
- 意圖分類：CLS token → Linear → 5 類
- 填槽：Span Extraction（類似 QA 模型，預測 slot 在文本中的位置）

支援環境：M1/M2 Mac (MPS) 或 CUDA GPU
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import argparse

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ==============================================================================
# 配置
# ==============================================================================

@dataclass
class Config:
    model_name: str = "bert-base-chinese"  # 改用 bert-base-chinese（更穩定）
    max_length: int = 64
    num_intents: int = 5
    slot_types: List[str] = None  # 動態設定
    batch_size: int = 32
    learning_rate: float = 3e-5
    num_epochs: int = 8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    train_file: str = "bert_training_data/joint_train.jsonl"
    test_file: str = "bert_training_data/joint_test.jsonl"
    output_dir: str = "bert_joint_model"
    
    def __post_init__(self):
        if self.slot_types is None:
            self.slot_types = ["name", "area", "mode"]

INTENT_NAMES = ["turn_on", "turn_off", "climate_set_mode", "get_state", "chat"]

# ==============================================================================
# 模型定義
# ==============================================================================

class JointIntentSlotModel(nn.Module):
    """
    聯合意圖分類 + 填槽模型
    
    使用 Span Extraction 方法：
    - 對每個 slot type，預測其在文本中的 start/end 位置
    - 如果該 slot 不存在，start 和 end 都指向 [CLS] (位置 0)
    """
    
    def __init__(self, model_name: str, num_intents: int, slot_types: List[str]):
        super().__init__()
        
        self.slot_types = slot_types
        self.num_slots = len(slot_types)
        
        # 載入預訓練模型（跳過 safetensors 轉換檢查）
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, use_safetensors=False)
        hidden_size = self.config.hidden_size
        
        # 意圖分類頭
        self.intent_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_intents)
        )
        
        # 填槽頭：每個 slot type 有自己的 start/end 預測器
        self.slot_start = nn.ModuleDict({
            slot: nn.Linear(hidden_size, 1) for slot in slot_types
        })
        self.slot_end = nn.ModuleDict({
            slot: nn.Linear(hidden_size, 1) for slot in slot_types
        })
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        intent_labels: torch.Tensor = None,
        slot_start_labels: Dict[str, torch.Tensor] = None,
        slot_end_labels: Dict[str, torch.Tensor] = None,
    ):
        # BERT 編碼
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden)
        pooled_output = sequence_output[:, 0, :]  # CLS token
        
        # 意圖分類
        intent_logits = self.intent_classifier(pooled_output)
        
        # 填槽預測
        slot_start_logits = {}
        slot_end_logits = {}
        
        for slot in self.slot_types:
            start_logits = self.slot_start[slot](sequence_output).squeeze(-1)  # (batch, seq_len)
            end_logits = self.slot_end[slot](sequence_output).squeeze(-1)
            
            # 應用 attention mask（忽略 padding）
            start_logits = start_logits + (1 - attention_mask.float()) * -10000.0
            end_logits = end_logits + (1 - attention_mask.float()) * -10000.0
            
            slot_start_logits[slot] = start_logits
            slot_end_logits[slot] = end_logits
        
        # 計算損失
        loss = None
        if intent_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            intent_loss = loss_fct(intent_logits, intent_labels)
            loss = intent_loss
            
            if slot_start_labels is not None and slot_end_labels is not None:
                for slot in self.slot_types:
                    if slot in slot_start_labels:
                        start_loss = loss_fct(slot_start_logits[slot], slot_start_labels[slot])
                        end_loss = loss_fct(slot_end_logits[slot], slot_end_labels[slot])
                        loss = loss + (start_loss + end_loss) * 0.5
        
        return {
            "loss": loss,
            "intent_logits": intent_logits,
            "slot_start_logits": slot_start_logits,
            "slot_end_logits": slot_end_logits,
        }

# ==============================================================================
# 資料集
# ==============================================================================

class JointDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, config: Config):
        self.samples = []
        self.tokenizer = tokenizer
        self.config = config
        self.slot_types = config.slot_types
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self.samples.append({
                        "text": data["text"],
                        "intent_label": data["intent_label"],
                        "slots": data.get("slots", {})
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offset_mapping = encoding["offset_mapping"].squeeze(0)
        
        # 找到每個 slot 在 token 序列中的位置
        slot_start_labels = {}
        slot_end_labels = {}
        
        for slot in self.slot_types:
            if slot in sample["slots"] and sample["slots"][slot]:
                slot_value = sample["slots"][slot]
                # 找到 slot_value 在原文中的位置
                char_start = text.find(slot_value)
                
                if char_start != -1:
                    char_end = char_start + len(slot_value) - 1
                    
                    # 將字元位置轉換為 token 位置
                    token_start, token_end = self._find_token_span(
                        offset_mapping, char_start, char_end
                    )
                    
                    slot_start_labels[slot] = token_start
                    slot_end_labels[slot] = token_end
                else:
                    # 找不到，指向 CLS
                    slot_start_labels[slot] = 0
                    slot_end_labels[slot] = 0
            else:
                # 該 slot 不存在，指向 CLS
                slot_start_labels[slot] = 0
                slot_end_labels[slot] = 0
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "intent_label": torch.tensor(sample["intent_label"], dtype=torch.long),
            "slot_start_labels": {k: torch.tensor(v, dtype=torch.long) for k, v in slot_start_labels.items()},
            "slot_end_labels": {k: torch.tensor(v, dtype=torch.long) for k, v in slot_end_labels.items()},
        }
    
    def _find_token_span(self, offset_mapping, char_start, char_end):
        """將字元位置轉換為 token 位置"""
        token_start = 0
        token_end = 0
        
        for i, (start, end) in enumerate(offset_mapping.tolist()):
            if start == 0 and end == 0:  # 特殊 token
                continue
            if start <= char_start < end:
                token_start = i
            if start <= char_end < end:
                token_end = i
                break
        
        return token_start, token_end

def collate_fn(batch):
    """自訂 collate 函數處理 slot labels"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    intent_labels = torch.stack([item["intent_label"] for item in batch])
    
    slot_start_labels = {}
    slot_end_labels = {}
    
    slot_types = list(batch[0]["slot_start_labels"].keys())
    for slot in slot_types:
        slot_start_labels[slot] = torch.stack([item["slot_start_labels"][slot] for item in batch])
        slot_end_labels[slot] = torch.stack([item["slot_end_labels"][slot] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "intent_labels": intent_labels,
        "slot_start_labels": slot_start_labels,
        "slot_end_labels": slot_end_labels,
    }

# ==============================================================================
# 訓練與評估
# ==============================================================================

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else 
                 {kk: vv.to(device) for kk, vv in v.items()} 
                 for k, v in batch.items()}
        
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            intent_labels=batch["intent_labels"],
            slot_start_labels=batch["slot_start_labels"],
            slot_end_labels=batch["slot_end_labels"],
        )
        
        loss = outputs["loss"]
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, tokenizer, device, config):
    model.eval()
    
    all_intent_preds = []
    all_intent_labels = []
    slot_correct = {slot: 0 for slot in config.slot_types}
    slot_total = {slot: 0 for slot in config.slot_types}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else 
                     {kk: vv.to(device) for kk, vv in v.items()} 
                     for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
            )
            
            # Intent
            intent_preds = outputs["intent_logits"].argmax(dim=-1).cpu().numpy()
            intent_labels = batch["intent_labels"].numpy()
            all_intent_preds.extend(intent_preds)
            all_intent_labels.extend(intent_labels)
            
            # Slots
            for slot in config.slot_types:
                start_preds = outputs["slot_start_logits"][slot].argmax(dim=-1).cpu()
                end_preds = outputs["slot_end_logits"][slot].argmax(dim=-1).cpu()
                start_labels = batch["slot_start_labels"][slot]
                end_labels = batch["slot_end_labels"][slot]
                
                for i in range(len(start_preds)):
                    slot_total[slot] += 1
                    if start_preds[i] == start_labels[i] and end_preds[i] == end_labels[i]:
                        slot_correct[slot] += 1
    
    intent_acc = accuracy_score(all_intent_labels, all_intent_preds)
    intent_f1 = f1_score(all_intent_labels, all_intent_preds, average='weighted')
    
    slot_accs = {slot: slot_correct[slot] / max(slot_total[slot], 1) 
                 for slot in config.slot_types}
    
    return {
        "intent_accuracy": intent_acc,
        "intent_f1": intent_f1,
        "slot_accuracies": slot_accs,
        "avg_slot_accuracy": np.mean(list(slot_accs.values())),
    }

def main():
    parser = argparse.ArgumentParser(description="訓練 BERT 意圖分類 + 填槽模型")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--model", type=str, default="bert-base-chinese", help="預訓練模型")
    args = parser.parse_args()
    
    config = Config(
        model_name=args.model,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
    )
    
    print("=" * 60)
    print("BERT 意圖分類 + 填槽 聯合訓練")
    print("=" * 60)
    print(f"模型: {config.model_name}")
    print(f"Slots: {config.slot_types}")
    print(f"批次大小: {config.batch_size}")
    print(f"學習率: {config.learning_rate}")
    print(f"訓練輪數: {config.num_epochs}")
    print("=" * 60)
    
    device = get_device()
    print(f"設備: {device}")
    
    # 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # 載入資料集
    print("\n📂 載入資料集...")
    train_dataset = JointDataset(config.train_file, tokenizer, config)
    test_dataset = JointDataset(config.test_file, tokenizer, config)
    
    print(f"   訓練集: {len(train_dataset)} 筆")
    print(f"   測試集: {len(test_dataset)} 筆")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 建立模型
    print("\n🔧 建立模型...")
    model = JointIntentSlotModel(
        model_name=config.model_name,
        num_intents=config.num_intents,
        slot_types=config.slot_types,
    )
    model.to(device)
    
    # 優化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 訓練
    print("\n🏋️ 開始訓練...")
    best_f1 = 0
    
    for epoch in range(config.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{config.num_epochs} ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        metrics = evaluate(model, test_loader, tokenizer, device, config)
        print(f"Intent Accuracy: {metrics['intent_accuracy']:.4f}")
        print(f"Intent F1: {metrics['intent_f1']:.4f}")
        print(f"Slot Accuracies: {metrics['slot_accuracies']}")
        print(f"Avg Slot Accuracy: {metrics['avg_slot_accuracy']:.4f}")
        
        if metrics['intent_f1'] > best_f1:
            best_f1 = metrics['intent_f1']
            # 儲存最佳模型
            Path(config.output_dir).mkdir(exist_ok=True)
            torch.save(model.state_dict(), f"{config.output_dir}/model.pt")
            tokenizer.save_pretrained(config.output_dir)
            
            # 儲存配置
            with open(f"{config.output_dir}/config.json", 'w') as f:
                json.dump({
                    "model_name": config.model_name,
                    "num_intents": config.num_intents,
                    "slot_types": config.slot_types,
                    "intent_names": INTENT_NAMES,
                }, f, indent=2)
            
            print(f"💾 儲存最佳模型 (F1: {best_f1:.4f})")
    
    print(f"\n✅ 訓練完成！最佳 F1: {best_f1:.4f}")
    print(f"   模型位置: {config.output_dir}")

if __name__ == "__main__":
    main()
