# BERT + Qwen 整合伺服器部署指南

## 架構概覽

```
用戶請求
    │
    ▼
┌─────────────────┐
│ BERT Pre-Router │  ← 快速分類 (~10ms)
│  意圖 + 填槽    │
└─────────────────┘
    │
    ├─────────────────────────────────────┐
    │ 高確定性 (>90%)                     │ 低確定性 / 聊天
    │ + 必要 slots 齊全                   │
    ▼                                     ▼
┌─────────────────┐            ┌─────────────────┐
│ 直接返回 ACTION │            │  Qwen LLM 處理  │
│ + 喵化回應      │            │  完整對話能力   │
└─────────────────┘            └─────────────────┘
```

## 檔案結構

```
qwen貓娘new/
├── qwen_bert_model_server.py   # 整合伺服器（新）
├── bert_joint_model/            # BERT 訓練好的模型
│   ├── config.json
│   ├── model.pt
│   ├── tokenizer.json
│   └── ...
└── [your_qwen_model]/           # Qwen fine-tuned 模型
```

---

## 部署步驟

### 1. 複製模型到伺服器

```bash
# 在 M1 Mac 上
cd /Users/ac/Downloads/qwen貓娘new

# 壓縮 BERT 模型
tar -czvf bert_joint_model.tar.gz bert_joint_model/

# 複製到 Ubuntu 伺服器
scp bert_joint_model.tar.gz user@your-server:/path/to/models/
scp qwen_bert_model_server.py user@your-server:/path/to/project/
```

### 2. 在 Ubuntu 伺服器上解壓

```bash
# 在伺服器上
cd /path/to/models
tar -xzvf bert_joint_model.tar.gz
```

### 3. 安裝依賴

```bash
# 建立虛擬環境（建議）
python -m venv venv
source venv/bin/activate

# 安裝依賴
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers fastapi uvicorn pydantic opencc-python-reimplemented
pip install unsloth  # 如果使用 unsloth 載入 Qwen
```

### 4. 修改配置

編輯 `qwen_bert_model_server.py`：

```python
# 修改這些路徑
BERT_MODEL_PATH = "/path/to/bert_joint_model"
QWEN_MODEL_PATH = "/path/to/your/qwen/model"

# 調整信心閾值（可選）
BERT_CONFIDENCE_THRESHOLD = 0.90
```

### 5. 啟動伺服器

```bash
python qwen_bert_model_server.py
```

預期輸出：

```
🔧 載入 BERT 聯合分類器...
   意圖類別: ['turn_on', 'turn_off', 'climate_set_mode', 'get_state', 'chat']
   Slot 類型: ['name', 'area', 'mode']
   設備: cuda
✅ BERT 模型載入完成
🚀 載入 Qwen 模型...
✅ Qwen 模型載入完成
🚀 啟動 Qwen + BERT Catgirl Home Assistant API Server
📡 監聽: 0.0.0.0:8124
```

---

## API 端點

| 端點 | 方法 | 說明 |
|------|------|------|
| `/` | GET | 伺服器資訊 |
| `/health` | GET | 健康檢查 |
| `/process` | POST | 處理用戶請求（主要端點） |
| `/process_with_state` | POST | get_state 二次對話 |
| `/search_result` | POST | 搜尋結果喵化 |
| `/fallback_assistant` | POST | Fallback 回應喵化 |

### `/process` 請求範例

```json
{
  "text": "打開書房大燈",
  "history": []
}
```

### 回應範例（BERT 直接處理）

```json
{
  "action": "turn_on",
  "params": {"name": "大燈", "area": "書房"},
  "response_text": "好的喵！正在開啟書房的大燈～",
  "has_action": true,
  "raw_response": "好的喵！正在開啟書房的大燈～\nACTION turn_on\nname 大燈\narea 書房",
  "processed_by": "bert"
}
```

### 回應範例（Qwen LLM 處理）

```json
{
  "action": null,
  "params": {},
  "response_text": "你好呀主人！今天過得好嗎喵？",
  "has_action": false,
  "raw_response": "你好呀主人！今天過得好嗎喵？",
  "processed_by": "qwen"
}
```

---

## Home Assistant 整合

### 修改 conversation_agent.py

在 `conversation_agent.py` 中，回應現在包含 `processed_by` 欄位，可以用來追蹤：

```python
# 在 _call_model_api 返回後
if action_result.get("processed_by") == "bert":
    _LOGGER.info("⚡ BERT 快速處理")
else:
    _LOGGER.info("🧠 Qwen LLM 處理")
```

---

## VRAM 使用預估

| 組件 | VRAM |
|------|------|
| BERT (bert-base-chinese) | ~400MB |
| Qwen 1.5B (4bit) | ~3GB |
| AI TTS | ~7GB |
| **總計** | ~10.4GB ✅ |

> RTX 3060 12GB 可以順利運行！

---

## 效能預期

| 情境 | 處理者 | 延遲 |
|------|--------|------|
| 簡單開關指令 | BERT | ~10-20ms |
| 複雜指令/聊天 | Qwen | ~200-500ms |

BERT 可以處理約 **60-70%** 的日常智慧家居指令，大幅減少 LLM 調用。

---

## 監控日誌

伺服器會輸出詳細日誌：

```
📥 收到請求: 打開書房大燈
🔍 BERT: intent=turn_on, conf=99.87%, slots={'name': '大燈', 'area': '書房'}
⚡ BERT 直接處理（高確定性）
📤 BERT 回應: 好的喵！正在開啟書房的大燈～
```

```
📥 收到請求: 你好啊
🔍 BERT: intent=chat, conf=99.82%, slots={}
🧠 交給 Qwen LLM 處理...
📤 Qwen 回應: 你好呀主人！有什麼需要幫忙的嗎喵？
```

---

## Systemd 服務（可選）

建立 `/etc/systemd/system/qwen-catgirl.service`：

```ini
[Unit]
Description=Qwen + BERT Catgirl Model Server
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/project
ExecStart=/path/to/venv/bin/python qwen_bert_model_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable qwen-catgirl
sudo systemctl start qwen-catgirl
sudo systemctl status qwen-catgirl
```

---

## 故障排除

### 1. BERT 模型載入失敗

確認路徑正確：
```bash
ls -la bert_joint_model/
# 應該有 config.json, model.pt, tokenizer.json 等
```

### 2. Qwen 模型載入失敗

確認 unsloth 安裝正確：
```bash
pip install unsloth
```

### 3. CUDA 記憶體不足

降低 Qwen batch size 或使用更激進的量化。

### 4. BERT 判斷錯誤太多

調低信心閾值：
```python
BERT_CONFIDENCE_THRESHOLD = 0.85  # 從 0.90 降到 0.85
```
