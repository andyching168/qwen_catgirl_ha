# 訓練資料生成配置說明

## 配置參數位置

所有生成條目的數量變數都集中在 `generate_training_data.py` 檔案的最上方（第 17-33 行）。

## 可配置參數

### 基礎控制資料
```python
TURN_ON_COUNT = 800      # turn_on 開啟控制資料數量
TURN_OFF_COUNT = 800     # turn_off 關閉控制資料數量
```

這兩個參數控制開關設備的訓練資料數量，兩者數量建議保持相等以避免模型偏向某一種操作。

### 其他功能資料
```python
CHAT_COUNT = 400         # 純聊天資料數量
SEARCH_COUNT = 300       # ACTION search 搜尋資料數量
CLIMATE_MODE_COUNT = 100 # climate_set_mode 冷氣模式資料數量
```

- **CHAT_COUNT**: 純聊天對話（打招呼、感謝、閒聊等），不涉及設備控制
- **SEARCH_COUNT**: 需要搜尋外部資訊的對話（天氣、新聞、股價等）
- **CLIMATE_MODE_COUNT**: 冷氣模式切換（冷氣、暖氣、送風、除濕等）

### 虛擬設備配置
```python
VIRTUAL_DEVICES_COUNT = 150  # 生成的虛擬設備數量
```

虛擬設備用於補充真實設備，增加訓練資料的多樣性。

### 設備採樣配置
```python
MIN_DEVICES_PER_SAMPLE = 25  # 最少設備數
MAX_DEVICES_PER_SAMPLE = 31  # 最多設備數
```

每條訓練資料會隨機採樣這個範圍內的設備數量，避免每次都使用全部設備，減少 token 使用並增加多樣性。

## 目前配置的資料分布

以預設配置為例：

| 類型 | 數量 | 比例 |
|------|------|------|
| 控制資料 (turn_on/off) | 1600 | 66% |
| 純聊天 | 400 | 16% |
| ACTION search | 300 | 12% |
| climate_set_mode | 100 | 4% |
| **總計** | **2400** | **100%** |

## 調整建議

### 場景 1：增加控制功能訓練
如果想讓模型更擅長控制設備：
```python
TURN_ON_COUNT = 1000
TURN_OFF_COUNT = 1000
CHAT_COUNT = 300
SEARCH_COUNT = 200
CLIMATE_MODE_COUNT = 100
```

### 場景 2：平衡各類型功能
如果希望各功能更平衡：
```python
TURN_ON_COUNT = 600
TURN_OFF_COUNT = 600
CHAT_COUNT = 500
SEARCH_COUNT = 500
CLIMATE_MODE_COUNT = 200
```

### 場景 3：著重聊天互動
如果想讓模型更有個性、更會聊天：
```python
TURN_ON_COUNT = 500
TURN_OFF_COUNT = 500
CHAT_COUNT = 800
SEARCH_COUNT = 400
CLIMATE_MODE_COUNT = 200
```

## 注意事項

1. **turn_on 和 turn_off 建議保持 1:1 比例**，避免模型偏向只開不關或只關不開
2. **總數量建議控制在 2000-3000 條**之間，太少可能訓練效果不佳，太多可能訓練時間過長
3. **調整後記得檢查輸出的比例資訊**，確保符合預期
4. **設備採樣範圍建議維持在 25-31**，太少可能上下文資訊不足，太多會浪費 token

## 使用方式

1. 編輯 `generate_training_data.py` 檔案頂部的配置參數
2. 執行腳本：
   ```bash
   python generate_training_data.py
   ```
3. 檢查輸出的統計資訊，確認各類型資料數量符合預期
4. 查看生成的 `training_data_v6_domain_grouped.jsonl` 檔案

## 準備設備資料（避免洩漏）

1) 從 Home Assistant 匯出設備清單：
- 在 `generate_dataset/tools/` 下，將從 Web UI 複製的清單貼到 `exposed_entities_list.txt`（可自建）。
- 設定環境變數：
   ```bash
   export HA_URL="http://homeassistant.local:8123"  # 你的 HA 位址
   export HA_TOKEN="<long-lived access token>"
   ```
- 執行 `fetch_ha_exposed_devices.py` 產生 `ha_exposed_devices.json` 與 `ha_device_list.txt`（含實際設備名稱/區域/狀態）。

2) 生成前請先清理：
- 這兩個輸出檔含個人裝置資訊，**不要上傳或提交到 Git**。用完可移除，或改用匿名示例檔。
- 若要公開專案，請確保 `ha_exposed_devices.json`、`ha_device_list.txt` 已刪除並確認版本控制忽略它們。

3) 生成訓練資料：
- `generate_training_data.py` 會讀取 `ACTIONS.yaml` 與 `ha_exposed_devices.json`，並寫出 `training_data_v6_domain_grouped.jsonl`。
- 上傳前僅保留合成出的 jsonl（若其中未含私人資訊），其餘裝置原始檔請刪除。

## 部署與測試（概要）

- **模型服務**：在專案根目錄執行 `python qwen_model_server.py`，依需要調整 `MODEL_PATH`、`HOST`、`PORT`（預設 8124）。確保 GPU/環境可載入模型。
- **Home Assistant 外掛**：將 `home assistant custom compoment/qwen_catgirl_conversation/` 放入 HA `custom_components/`，在整合中新增並填入 `model_url` 指向上方服務，例如 `http://<server-ip>:8124`。
- **隱私檢查**：部署前再次確認未將內網位址、存取 Token 或裝置清單檔案提交到版本庫。
