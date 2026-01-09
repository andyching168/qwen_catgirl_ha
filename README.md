# Qwen Catgirl Home Assistant Project

貓娘語氣的家居助理（Qwen finetune）與 Home Assistant 自訂整合，包含資料集生成腳本與推論 API 服務。

> [!IMPORTANT]
> **BERT-first Architecture (2026-01-10)**
> 自本版本起，系統全面採用 BERT-first 架構作為未來開發方向。引入 BERT 作為 Pre-Router 處理高頻、高確定性的簡單指令（需 10ms），僅在需要複雜語意理解或閒聊時 Fallback 至 Qwen LLM。
> 舊版純 Qwen 架構已封存於 `legacy_qwen_only` 分支。

## 特色
- **Hybrid 架構**：BERT 極速路由 + Qwen 深度理解 + Gemini/HA 終極備援。
- **模板回覆**：時間、日期、天氣查詢 <1ms 瞬間回應。
- Qwen 模型微調：以 name+area 格式控制設備，避免 entity_id 幻覺。
- Home Assistant 整合：自訂 conversation agent，支援 fallback（HA/Gemini）與多輪對話記錄。
- 資料生成工具：可依動作分佈自動產生訓練資料，含虛擬設備與隨機化狀態。
- 隱私防護：設備清單檔案不應上傳，提供清理指引。

## 目錄概覽
- `qwen_model_server.py`：FastAPI 推論服務，載入微調後模型並提供 /process 等端點。
- `training.py`：最簡化的微調腳本（Unsloth + TRL）。
- `generate_dataset/`：資料生成工具與說明（見 `generate_dataset/CONFIG_README.md`）。
- `home assistant custom compoment/qwen_catgirl_conversation/`：HA 自訂組件，對話代理與設定流程。

## 部署指引
### 1) 準備模型
- 依 `training.py` 微調並輸出至 `./qwen-catgirl-ha-switch-v2`（或自定路徑）。
- 在 `qwen_model_server.py` 調整 `MODEL_PATH`、`HOST`、`PORT`（預設 8124）。
- 以 GPU/支援環境執行：
  ```bash
  python qwen_model_server.py
  ```

### 2) Home Assistant 自訂整合
- 將資料夾 `home assistant custom compoment/qwen_catgirl_conversation/` 複製到 HA 的 `config/custom_components/`。
- 在 HA 介面新增整合，`model_url` 指向上面的推論服務（例：`http://<server-ip>:8124`）。
- 可於選項中調整 fallback / removable keywords。

### 3) 必要的 Home Assistant Scripts
此整合會呼叫以下 scripts，需在 HA 中自行建立：

| Script 名稱 | 用途 | 必要參數 |
|------------|------|---------|
| `script.assist_ha_fallback` | Fallback 第一層：使用 HA 內建 Assist 處理請求 | `query`（用戶輸入文字） |
| `script.assist_gemini_fallback` | Fallback 第二層：使用 Gemini AI 處理請求 | `query`（用戶輸入文字） |
| `script.assist_search_google` | 執行 Google 搜尋並回傳結果 | `query`（搜尋關鍵字） |
| `script.set_climate_mode` | 設定空調模式 | `area`（區域名稱）、`mode`（模式如 cool/heat/auto） |

#### 範例：assist_ha_fallback
```yaml
alias: Assist HA Fallback
mode: single
fields:
  query:
    description: 用戶輸入
    example: "開燈"
sequence:
  - service: conversation.process
    data:
      agent_id: conversation.home_assistant  # 內建 Assist
      text: "{{ query }}"
    response_variable: result
  - stop: ""
    response_variable: response
    variables:
      response: "{{ result.response.speech.plain.speech }}"
```

#### 範例：assist_gemini_fallback
```yaml
alias: Assist Gemini Fallback
mode: single
fields:
  query:
    description: 用戶輸入
    example: "今天天氣如何"
sequence:
  - service: conversation.process
    data:
      agent_id: conversation.gemini_conversation_agent  # 你的 Gemini Agent ID
      text: "{{ query }}"
    response_variable: result
  - stop: ""
    response_variable: response
    variables:
      response: "{{ result.response.speech.plain.speech }}"
```

#### 範例：assist_search_google
```yaml
alias: Assist Search Google
mode: single
fields:
  query:
    description: 搜尋關鍵字
    example: "明天台北天氣"
sequence:
  - service: rest_command.google_search  # 需自行設定 REST command
    data:
      query: "{{ query }}"
    response_variable: search_result
  - stop: ""
    response_variable: response
    variables:
      response: "{{ search_result.content }}"
```

#### 範例：set_climate_mode
```yaml
alias: Set Climate Mode
mode: single
fields:
  area:
    description: 區域名稱
    example: "客廳"
  mode:
    description: 空調模式
    example: "cool"
sequence:
  - service: climate.set_hvac_mode
    target:
      area_id: "{{ area }}"
    data:
      hvac_mode: "{{ mode }}"
```

> **注意**：以上範例僅供參考，請依據你的實際環境（Agent ID、REST API 設定等）調整。

### 4) 產生訓練資料（選擇性）
- 參考 `generate_dataset/CONFIG_README.md`，設定生成參數並執行 `generate_training_data.py`。
- 若需使用真實設備清單，先用 `fetch_ha_exposed_devices.py` 產生 `ha_exposed_devices.json` / `ha_device_list.txt`，**用後請刪除或忽略版本控制**。
- 上傳前僅保留生成的 jsonl 訓練集（確認不含敏感資訊）。

### 5) 隱私與版控
- **不要提交** 含設備名稱/區域/狀態的檔案：`generate_dataset/tools/ha_exposed_devices.json`、`generate_dataset/tools/ha_device_list.txt` 等。
- 模型與服務 URL 建議改為環境變數或占位值，再行發布。

## 用途範例
- 在 HA 中：語音/文字對話「關掉書房大燈」→ agent 經模型解析為 ACTION，轉 Intent 執行。
- 聊天模式：無控制關鍵字時以貓娘語氣回應，不輸出 ACTION。
- 搜尋模式：模型輸出 ACTION search → HA 腳本搜尋 → `/search_result` 生成最終回應。

## 疑難排解
- 模型未載入：確認 `MODEL_PATH`、GPU 記憶體、環境套件（unsloth、fastapi、uvicorn）。
- HA 未連線模型：檢查 `model_url`、防火牆與 CORS；可在瀏覽器/`curl` 測試 `/health`。
- 回應錯誤或找不到設備：確保 HA 標記了 assist label 的實體，或調整 fallback / removable keywords。
