#!/usr/bin/env python3
"""
從 Qwen 訓練資料轉換成 BERT 意圖分類 + 填槽格式

輸入：training_data_v7_get_state.jsonl
輸出：
  - bert_training_data/joint_train.jsonl
  - bert_training_data/joint_test.jsonl

格式範例：
{
    "text": "打開書房大燈",
    "intent": "turn_on",
    "slots": {"name": "大燈", "area": "書房"}
}
"""

import json
import re
import random
from pathlib import Path
from collections import Counter

# 意圖標籤對應
INTENT_LABELS = {
    "turn_on": 0,
    "turn_off": 1,
    "climate_set_mode": 2,
    "get_state": 3,
    "chat": 4,
}

# Slot 類型
SLOT_TYPES = ["name", "area", "mode", "temperature", "brightness"]

# 用於統計
stats = Counter()
slot_stats = Counter()

def extract_action_and_params(assistant_response: str) -> tuple:
    """從助理回應中提取 ACTION 名稱和參數
    
    Returns:
        (action_name, params_dict)
    """
    lines = assistant_response.strip().split('\n')
    action = None
    params = {}
    
    for line in lines:
        line = line.strip()
        
        # 提取 ACTION
        if line.startswith('ACTION '):
            action = line[7:].strip()
            continue
        
        # 提取參數
        for slot_type in SLOT_TYPES:
            if line.startswith(f'{slot_type} '):
                value = line[len(slot_type)+1:].strip()
                params[slot_type] = value
                break
    
    return action, params

def extract_user_text(user_content: str) -> str:
    """提取純用戶輸入"""
    # 跳過二次對話的系統訊息
    if user_content.startswith("State result:"):
        return None
    if user_content.startswith("Search result:"):
        return None
    
    if user_content.startswith("User request:\n"):
        return user_content[len("User request:\n"):].strip()
    return user_content.strip()

def process_jsonl(input_path: str, output_train: str, output_test: str, test_ratio: float = 0.1):
    """處理 JSONL 並分割成訓練/測試集"""
    
    samples = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            messages = data.get('messages', [])
            
            # 找到 user 和 assistant 訊息
            user_msg = None
            assistant_msg = None
            
            for msg in messages:
                if msg['role'] == 'user':
                    user_msg = msg['content']
                elif msg['role'] == 'assistant':
                    assistant_msg = msg['content']
            
            if not user_msg or not assistant_msg:
                continue
            
            # 提取用戶輸入（過濾系統訊息）
            text = extract_user_text(user_msg)
            if text is None:
                stats["skipped_system_msg"] += 1
                continue
            
            # 提取 ACTION 和參數
            action, params = extract_action_and_params(assistant_msg)
            
            # 只保留我們關心的 4 種 ACTION + chat
            if action is None:
                intent_label = INTENT_LABELS["chat"]
                intent_name = "chat"
                stats["chat"] += 1
            elif action in INTENT_LABELS:
                intent_label = INTENT_LABELS[action]
                intent_name = action
                stats[action] += 1
            else:
                # 跳過其他 ACTION
                stats[f"skipped_{action}"] += 1
                continue
            
            # 記錄 slot 統計
            for slot_type in params:
                slot_stats[slot_type] += 1
            
            # ⭐ 過濾「未分類」區域
            if params.get("area") == "未分類" or "未分類" in text:
                stats["skipped_uncategorized"] += 1
                continue
            
            samples.append({
                "text": text,
                "intent": intent_name,
                "intent_label": intent_label,
                "slots": params,
            })
    
    # 打亂順序
    random.shuffle(samples)
    
    # 分割訓練/測試
    split_idx = int(len(samples) * (1 - test_ratio))
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]
    
    # 寫入檔案
    with open(output_train, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    with open(output_test, 'w', encoding='utf-8') as f:
        for sample in test_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    return len(train_samples), len(test_samples)

def add_chat_samples(output_train: str, output_test: str, num_samples: int = 400):
    """添加純聊天樣本"""
    
    chat_examples = [
        "你好", "嗨", "早安", "午安", "晚安", "謝謝", "感謝",
        "你叫什麼名字", "你是誰", "你好嗎", "今天過得好嗎",
        "吃飽了嗎", "天氣好熱", "好冷喔", "累死了", "無聊",
        "陪我聊天", "講個笑話", "說故事給我聽", "你喜歡什麼",
        "你會做什麼", "你厲害嗎", "晚餐吃什麼好", "推薦電影",
        "有什麼好玩的", "明天要幹嘛", "我好累", "肚子餓",
        "想睡覺", "睡不著", "心情不好", "開心", "難過",
        "你真可愛", "喵喵喵", "哈哈哈", "好笑", "無聊死了",
    ]
    
    variations = ["{text}", "{text}喔", "{text}啦", "{text}呢", "欸{text}"]
    
    chat_samples = []
    for text in chat_examples:
        for var in variations:
            chat_samples.append({
                "text": var.format(text=text),
                "intent": "chat",
                "intent_label": INTENT_LABELS["chat"],
                "slots": {},  # 純聊天沒有 slots
            })
    
    random.shuffle(chat_samples)
    chat_samples = chat_samples[:num_samples]
    
    # 分割並追加到檔案
    split_idx = int(len(chat_samples) * 0.9)
    train_chats = chat_samples[:split_idx]
    test_chats = chat_samples[split_idx:]
    
    with open(output_train, 'a', encoding='utf-8') as f:
        for sample in train_chats:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    with open(output_test, 'a', encoding='utf-8') as f:
        for sample in test_chats:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    return len(train_chats), len(test_chats)

def main():
    input_file = "generate_dataset/training_data_v7_get_state.jsonl"
    output_dir = Path("bert_training_data")
    output_dir.mkdir(exist_ok=True)
    
    output_train = output_dir / "joint_train.jsonl"
    output_test = output_dir / "joint_test.jsonl"
    
    print("=" * 60)
    print("BERT 意圖分類 + 填槽 訓練資料生成器")
    print("=" * 60)
    
    # 轉換主要資料
    print(f"\n📂 讀取: {input_file}")
    train_count, test_count = process_jsonl(
        input_file,
        str(output_train),
        str(output_test)
    )
    
    print(f"\n📊 Intent 統計:")
    for action, count in sorted(stats.items()):
        if action.startswith("skipped_"):
            print(f"   ⏭️  {action}: {count} (跳過)")
        else:
            print(f"   ✅ {action}: {count}")
    
    print(f"\n🏷️  Slot 統計:")
    for slot_type, count in sorted(slot_stats.items()):
        print(f"   📌 {slot_type}: {count}")
    
    # 添加聊天樣本
    print(f"\n💬 添加聊天樣本...")
    chat_train, chat_test = add_chat_samples(str(output_train), str(output_test), 400)
    
    train_count += chat_train
    test_count += chat_test
    
    print(f"\n📁 輸出:")
    print(f"   訓練集: {output_train} ({train_count} 筆)")
    print(f"   測試集: {output_test} ({test_count} 筆)")
    
    # 顯示範例
    print(f"\n📝 資料範例:")
    with open(output_train, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            sample = json.loads(line)
            print(f"   {sample['text']}")
            print(f"      → intent: {sample['intent']}, slots: {sample['slots']}")
    
    print("\n✅ 完成！")

if __name__ == "__main__":
    main()
