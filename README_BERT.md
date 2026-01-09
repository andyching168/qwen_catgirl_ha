# BERT 意圖分類器

在 LLM 前方的輕量級意圖分類器，用於快速路由請求。

## 支援的意圖

| ID | 意圖 | 說明 |
|----|------|------|
| 0 | `turn_on` | 開啟設備 |
| 1 | `turn_off` | 關閉設備 |
| 2 | `climate_set_mode` | 設定冷氣模式 |
| 3 | `get_state` | 查詢設備狀態 |
| 4 | `chat` | 純聊天 |

## 快速開始

### 1. 安裝依賴

```bash
# M1 Mac
pip install -r requirements_bert.txt

# Ubuntu + CUDA
pip install -r requirements_bert.txt
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 2. 生成訓練資料

```bash
python generate_bert_data.py
```

輸出：
- `bert_training_data/intent_train.jsonl`
- `bert_training_data/intent_test.jsonl`

### 3. 訓練模型

```bash
# M1 Mac (自動使用 MPS)
python train_bert_intent.py --epochs 5 --batch_size 32

# 自訂參數
python train_bert_intent.py --epochs 10 --batch_size 16 --lr 3e-5
```

訓練完成後，模型儲存在 `bert_intent_model/`

### 4. 測試推理

```bash
# 基本測試
python bert_intent_classifier.py --model bert_intent_model

# 效能測試
python bert_intent_classifier.py --model bert_intent_model --benchmark
```

## 整合到模型伺服器

在 `qwen_model_server.py` 中：

```python
from bert_intent_classifier import BertIntentClassifier

# 初始化（在載入 Qwen 之前或之後）
bert_classifier = BertIntentClassifier(
    model_path="bert_intent_model",
    confidence_threshold=0.85
)

# 在 /process endpoint 中使用
@app.post("/process")
async def process(request: InferenceRequest):
    # 1. BERT 快速分類
    use_llm, intent, confidence = bert_classifier.should_use_llm(request.text)
    
    if not use_llm:
        # 高確定性：可直接組裝 ACTION（需要額外的 NER 或規則）
        logger.info(f"⚡ BERT 快速路由: {intent} ({confidence:.2%})")
        # ... 組裝 ACTION 回應
    
    # 低確定性或聊天：交給 LLM
    response = run_inference(request.text, ...)
    return response
```

## 效能預期

| 指標 | 預期值 |
|------|--------|
| 準確率 | >95% |
| 單次推理延遲 | <10ms (GPU) |
| VRAM 使用 | ~150MB |

## 檔案結構

```
qwen貓娘new/
├── generate_bert_data.py      # 資料轉換
├── train_bert_intent.py       # 訓練腳本
├── bert_intent_classifier.py  # 推理模組
├── requirements_bert.txt      # 依賴
├── bert_training_data/        # 訓練資料
│   ├── intent_train.jsonl
│   └── intent_test.jsonl
└── bert_intent_model/         # 訓練好的模型
    ├── config.json
    ├── model.safetensors
    └── tokenizer.json
```
