#!/usr/bin/env python3
"""
BERT 意圖分類 + 填槽 推理模組

可獨立使用或整合到 qwen_model_server.py
"""

import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Dict, List, Tuple, Optional
import time

class JointIntentSlotModel(nn.Module):
    """聯合意圖分類 + 填槽模型（與訓練腳本相同結構）"""
    
    def __init__(self, model_name: str, num_intents: int, slot_types: List[str]):
        super().__init__()
        
        self.slot_types = slot_types
        self.num_slots = len(slot_types)
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.config.hidden_size
        
        self.intent_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_intents)
        )
        
        self.slot_start = nn.ModuleDict({
            slot: nn.Linear(hidden_size, 1) for slot in slot_types
        })
        self.slot_end = nn.ModuleDict({
            slot: nn.Linear(hidden_size, 1) for slot in slot_types
        })
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]
        
        intent_logits = self.intent_classifier(pooled_output)
        
        slot_start_logits = {}
        slot_end_logits = {}
        
        for slot in self.slot_types:
            start_logits = self.slot_start[slot](sequence_output).squeeze(-1)
            end_logits = self.slot_end[slot](sequence_output).squeeze(-1)
            
            start_logits = start_logits + (1 - attention_mask.float()) * -10000.0
            end_logits = end_logits + (1 - attention_mask.float()) * -10000.0
            
            slot_start_logits[slot] = start_logits
            slot_end_logits[slot] = end_logits
        
        return {
            "intent_logits": intent_logits,
            "slot_start_logits": slot_start_logits,
            "slot_end_logits": slot_end_logits,
        }


class BertJointClassifier:
    """BERT 意圖分類 + 填槽 推理器"""
    
    def __init__(
        self, 
        model_path: str = "bert_joint_model",
        device: Optional[str] = None,
        confidence_threshold: float = 0.85
    ):
        self.confidence_threshold = confidence_threshold
        
        # 自動檢測設備
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"🔧 BERT 聯合分類器初始化")
        print(f"   模型路徑: {model_path}")
        print(f"   運行設備: {self.device}")
        
        # 載入配置
        with open(f"{model_path}/config.json", 'r') as f:
            config = json.load(f)
        
        self.intent_names = config["intent_names"]
        self.slot_types = config["slot_types"]
        self.num_intents = config["num_intents"]
        base_model = config["model_name"]
        
        print(f"   意圖類別: {self.intent_names}")
        print(f"   Slot 類型: {self.slot_types}")
        
        # 載入 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 載入模型
        self.model = JointIntentSlotModel(
            model_name=base_model,
            num_intents=self.num_intents,
            slot_types=self.slot_types,
        )
        self.model.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"   載入完成 ✅")
    
    def predict(self, text: str) -> Dict:
        """
        預測意圖和填槽
        
        Args:
            text: 用戶輸入文字
            
        Returns:
            {
                "intent": "turn_on",
                "intent_confidence": 0.95,
                "slots": {"name": "大燈", "area": "書房"},
                "raw_text": "打開書房大燈"
            }
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"][0].tolist()
        
        # 推理
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            
            # Intent
            intent_probs = torch.softmax(outputs["intent_logits"], dim=-1)
            intent_confidence, intent_id = torch.max(intent_probs, dim=-1)
            intent = self.intent_names[intent_id.item()]
            
            # Slots
            slots = {}
            for slot in self.slot_types:
                start_logits = outputs["slot_start_logits"][slot][0]
                end_logits = outputs["slot_end_logits"][slot][0]
                
                start_idx = torch.argmax(start_logits).item()
                end_idx = torch.argmax(end_logits).item()
                
                # 如果 start 和 end 都是 0（CLS），表示該 slot 不存在
                if start_idx == 0 and end_idx == 0:
                    continue
                
                # 確保 end >= start
                if end_idx < start_idx:
                    end_idx = start_idx
                
                # 從 offset_mapping 提取原文
                if start_idx < len(offset_mapping) and end_idx < len(offset_mapping):
                    char_start = offset_mapping[start_idx][0]
                    char_end = offset_mapping[end_idx][1]
                    
                    if char_start < char_end:
                        slot_value = text[char_start:char_end]
                        slots[slot] = slot_value
        
        return {
            "intent": intent,
            "intent_confidence": intent_confidence.item(),
            "slots": slots,
            "raw_text": text,
        }
    
    def build_action(self, result: Dict) -> Optional[Dict]:
        """
        根據預測結果建構 ACTION
        
        Returns:
            None 如果是 chat 或信心不足
            否則返回 ACTION 結構
        """
        if result["intent"] == "chat":
            return None
        
        if result["intent_confidence"] < self.confidence_threshold:
            return None
        
        action = {
            "action": result["intent"],
            "params": result["slots"],
            "confidence": result["intent_confidence"],
        }
        
        return action
    
    def should_use_llm(self, text: str) -> Tuple[bool, Dict]:
        """
        判斷是否需要使用 LLM
        
        Returns:
            (should_use_llm, prediction_result)
        """
        result = self.predict(text)
        action = self.build_action(result)
        
        if action is None:
            return True, result
        
        # 檢查必要參數
        intent = result["intent"]
        slots = result["slots"]
        
        # 根據意圖檢查必要的 slots
        required_slots = {
            "turn_on": ["name"],
            "turn_off": ["name"],
            "get_state": ["name"],
            "climate_set_mode": ["mode"],
        }
        
        if intent in required_slots:
            for required_slot in required_slots[intent]:
                if required_slot not in slots or not slots[required_slot]:
                    # 缺少必要參數，交給 LLM
                    return True, result
        
        return False, result
    
    def benchmark(self, texts: list, num_runs: int = 100):
        """效能測試"""
        print(f"\n⏱️  效能測試 ({num_runs} 次)")
        
        # 預熱
        for _ in range(10):
            self.predict(texts[0])
        
        start = time.time()
        for _ in range(num_runs):
            for text in texts:
                self.predict(text)
        
        elapsed = time.time() - start
        total_predictions = num_runs * len(texts)
        avg_ms = (elapsed / total_predictions) * 1000
        
        print(f"   總預測數: {total_predictions}")
        print(f"   總耗時: {elapsed:.2f}s")
        print(f"   平均延遲: {avg_ms:.2f}ms")
        
        return avg_ms


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="BERT 聯合分類器測試")
    parser.add_argument("--model", type=str, default="bert_joint_model")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("-i", "--interactive", action="store_true", help="互動模式")
    args = parser.parse_args()
    
    classifier = BertJointClassifier(model_path=args.model)
    
    # 互動模式
    if args.interactive:
        print("\n🎤 互動模式（輸入 'q' 或 'exit' 退出）")
        print("-" * 50)
        
        while True:
            try:
                text = input("\n請輸入: ").strip()
                if not text:
                    continue
                if text.lower() in ['q', 'exit', 'quit']:
                    print("👋 再見！")
                    break
                
                result = classifier.predict(text)
                use_llm, _ = classifier.should_use_llm(text)
                
                llm_tag = "→ LLM" if use_llm else "✓ 直接處理"
                slots_str = json.dumps(result["slots"], ensure_ascii=False)
                
                print(f"   意圖: {result['intent']} ({result['intent_confidence']:.2%})")
                print(f"   Slots: {slots_str}")
                print(f"   {llm_tag}")
                
            except KeyboardInterrupt:
                print("\n👋 再見！")
                break
            except Exception as e:
                print(f"❌ 錯誤: {e}")
        
        return
    
    # 預設測試模式
    test_texts = [
        "打開書房大燈",
        "關掉客廳燈",
        "冷氣開著嗎",
        "把冷氣設定成冷氣模式",
        "你好",
        "開風扇",
        "關臥室冷氣",
        "燈亮著嗎",
        "設定暖氣模式",
        "客廳燈關一下",
    ]
    
    print("\n📋 測試預測:")
    print("-" * 70)
    
    for text in test_texts:
        result = classifier.predict(text)
        use_llm, _ = classifier.should_use_llm(text)
        
        llm_tag = "→ LLM" if use_llm else "✓ 直接處理"
        slots_str = json.dumps(result["slots"], ensure_ascii=False)
        
        print(f"輸入: {text}")
        print(f"   意圖: {result['intent']} ({result['intent_confidence']:.2%})")
        print(f"   Slots: {slots_str}")
        print(f"   {llm_tag}")
        print()
    
    if args.benchmark:
        classifier.benchmark(test_texts)
    
    print("✅ 測試完成！")


if __name__ == "__main__":
    main()
