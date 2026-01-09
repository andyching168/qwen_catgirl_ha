#!/usr/bin/env python3
"""
BERT 意圖分類器推理模組

可獨立使用或整合到 qwen_model_server.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Tuple, Optional
import time

# 標籤定義
LABEL_NAMES = ["turn_on", "turn_off", "climate_set_mode", "get_state", "chat"]

class BertIntentClassifier:
    """BERT 意圖分類器"""
    
    def __init__(
        self, 
        model_path: str = "bert_intent_model",
        device: Optional[str] = None,
        confidence_threshold: float = 0.85
    ):
        """
        初始化分類器
        
        Args:
            model_path: 訓練好的模型路徑
            device: 運行設備 (cuda/mps/cpu)，None 為自動檢測
            confidence_threshold: 高確定性閾值
        """
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
        
        print(f"🔧 BERT 意圖分類器初始化")
        print(f"   模型路徑: {model_path}")
        print(f"   運行設備: {self.device}")
        
        # 載入模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"   載入完成 ✅")
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        預測意圖
        
        Args:
            text: 用戶輸入文字
            
        Returns:
            (intent, confidence): 意圖名稱和信心分數
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            confidence, predicted_id = torch.max(probs, dim=-1)
            intent = LABEL_NAMES[predicted_id.item()]
            
        return intent, confidence.item()
    
    def should_use_llm(self, text: str) -> Tuple[bool, str, float]:
        """
        判斷是否需要使用 LLM
        
        Args:
            text: 用戶輸入文字
            
        Returns:
            (should_use_llm, intent, confidence)
        """
        intent, confidence = self.predict(text)
        
        # 高確定性且非聊天 → 不需要 LLM
        if confidence >= self.confidence_threshold and intent != "chat":
            return False, intent, confidence
        
        # 需要 LLM 處理
        return True, intent, confidence
    
    def benchmark(self, texts: list, num_runs: int = 100):
        """效能測試"""
        print(f"\n⏱️  效能測試 ({num_runs} 次)")
        
        # 預熱
        for _ in range(10):
            self.predict(texts[0])
        
        # 計時
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
    """測試推理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="BERT 意圖分類器測試")
    parser.add_argument("--model", type=str, default="bert_intent_model", help="模型路徑")
    parser.add_argument("--benchmark", action="store_true", help="執行效能測試")
    args = parser.parse_args()
    
    # 初始化分類器
    classifier = BertIntentClassifier(model_path=args.model)
    
    # 測試樣本
    test_texts = [
        "打開書房燈",
        "關掉客廳燈",
        "冷氣開著嗎",
        "把冷氣設定成冷氣模式",
        "你好",
        "今天天氣如何",
        "開風扇",
        "關冷氣",
        "燈亮著嗎",
        "設定暖氣模式",
    ]
    
    print("\n📋 測試預測:")
    for text in test_texts:
        intent, confidence = classifier.predict(text)
        use_llm, _, _ = classifier.should_use_llm(text)
        llm_tag = "→ LLM" if use_llm else "→ 直接處理"
        print(f"   [{intent:>16}] ({confidence:.2%}) {text} {llm_tag}")
    
    if args.benchmark:
        classifier.benchmark(test_texts)
    
    print("\n✅ 測試完成！")


if __name__ == "__main__":
    main()
