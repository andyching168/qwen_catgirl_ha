#!/usr/bin/env python3
"""
qwen_bert_model_server.py
整合 BERT Pre-Router + Qwen LLM 的模型伺服器

架構：
  用戶請求 → BERT 意圖分類 + 填槽 → 高確定性? → 直接返回 ACTION
                                   ↓ 否
                                 Qwen LLM 處理
                                   ↓
                                喵化回應
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
import uvicorn
import re
import logging
from typing import Optional, List, Dict
from datetime import datetime
from opencc import OpenCC

# ==============================================================================
# 配置
# ==============================================================================
HOST = "0.0.0.0"
PORT = 8124
MAX_SEQ_LENGTH = 512

# BERT 配置
BERT_MODEL_PATH = "./bert_joint_model"
BERT_CONFIDENCE_THRESHOLD = 0.90  # 高於此信心才直接處理

# Qwen 配置 (保持與原版相同)
QWEN_MODEL_PATH = "./qwen-catgirl-ha-switch-v2"

# ==============================================================================
# Logging
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 全局變數
# ==============================================================================
bert_model = None
bert_tokenizer = None
bert_config = None
qwen_model = None
qwen_tokenizer = None
s2t_converter = OpenCC('s2t')

app = FastAPI(title="Qwen + BERT Catgirl Home Assistant API")

# ==============================================================================
# BERT 模型定義
# ==============================================================================

class JointIntentSlotModel(nn.Module):
    """聯合意圖分類 + 填槽模型"""
    
    def __init__(self, model_name: str, num_intents: int, slot_types: List[str]):
        super().__init__()
        
        self.slot_types = slot_types
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, use_safetensors=False)
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

# ==============================================================================
# Request/Response Models
# ==============================================================================

class Device(BaseModel):
    entityId: str
    friendlyName: str
    domain: str
    state: str
    area: Optional[str] = None
    brightnessPct: Optional[int] = None

class InferenceRequest(BaseModel):
    """處理用戶請求"""
    text: str
    devices: Optional[List[Device]] = None
    language: Optional[str] = "zh"
    history: Optional[List[dict]] = None

class StateResultRequest(BaseModel):
    """v7: 處理 get_state 二次對話請求"""
    original_question: str
    state_result: str
    device_name: str
    area: str

class SearchResultRequest(BaseModel):
    """搜尋結果回應請求"""
    user_question: str
    search_result: str

class FallbackAssistantRequest(BaseModel):
    """Fallback 助理回應喵化請求"""
    user_question: str
    assistant_response: str

class ActionResult(BaseModel):
    action: Optional[str] = None
    params: Dict[str, str] = {}
    response_text: str
    has_action: bool
    raw_response: Optional[str] = None
    processed_by: str = "unknown"  # "bert" 或 "qwen"

class SearchResultResponse(BaseModel):
    response_text: str

# ==============================================================================
# 載入模型
# ==============================================================================

def load_bert_model():
    """載入 BERT 意圖分類 + 填槽模型"""
    global bert_model, bert_tokenizer, bert_config
    
    import json
    
    logger.info("🔧 載入 BERT 聯合分類器...")
    
    # 載入配置
    with open(f"{BERT_MODEL_PATH}/config.json", 'r') as f:
        bert_config = json.load(f)
    
    logger.info(f"   意圖類別: {bert_config['intent_names']}")
    logger.info(f"   Slot 類型: {bert_config['slot_types']}")
    
    # 載入 tokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    
    # 載入模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_model = JointIntentSlotModel(
        model_name=bert_config["model_name"],
        num_intents=bert_config["num_intents"],
        slot_types=bert_config["slot_types"],
    )
    bert_model.load_state_dict(torch.load(f"{BERT_MODEL_PATH}/model.pt", map_location=device))
    bert_model.to(device)
    bert_model.eval()
    
    logger.info(f"   設備: {device}")
    logger.info("✅ BERT 模型載入完成")
    
    return device

def load_qwen_model():
    """載入 Qwen LLM 模型"""
    global qwen_model, qwen_tokenizer
    
    from unsloth import FastLanguageModel
    
    logger.info("🚀 載入 Qwen 模型...")
    
    qwen_model, qwen_tokenizer = FastLanguageModel.from_pretrained(
        model_name=QWEN_MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(qwen_model)
    
    logger.info("✅ Qwen 模型載入完成")

# ==============================================================================
# BERT 推理
# ==============================================================================

def bert_predict(text: str) -> Dict:
    """使用 BERT 進行意圖分類 + 填槽"""
    device = next(bert_model.parameters()).device
    
    # Tokenize
    encoding = bert_tokenizer(
        text,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offset_mapping = encoding["offset_mapping"][0].tolist()
    
    # 推理
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask)
        
        # Intent
        intent_probs = torch.softmax(outputs["intent_logits"], dim=-1)
        intent_confidence, intent_id = torch.max(intent_probs, dim=-1)
        intent = bert_config["intent_names"][intent_id.item()]
        
        # Slots
        slots = {}
        for slot in bert_config["slot_types"]:
            start_logits = outputs["slot_start_logits"][slot][0]
            end_logits = outputs["slot_end_logits"][slot][0]
            
            start_idx = torch.argmax(start_logits).item()
            end_idx = torch.argmax(end_logits).item()
            
            if start_idx == 0 and end_idx == 0:
                continue
            
            if end_idx < start_idx:
                end_idx = start_idx
            
            if start_idx < len(offset_mapping) and end_idx < len(offset_mapping):
                char_start = offset_mapping[start_idx][0]
                char_end = offset_mapping[end_idx][1]
                
                if char_start < char_end:
                    slot_value = text[char_start:char_end]
                    slots[slot] = slot_value
    
    return {
        "intent": intent,
        "confidence": intent_confidence.item(),
        "slots": slots,
    }

# 應該直接交給 LLM 的關鍵字（不適合 BERT 處理）
LLM_KEYWORDS = [
    "天氣", "下雨", "氣溫", "溫度多少",  # 天氣查詢
    "幾點", "幾號", "星期幾", "今天日期",  # 時間查詢
    "搜尋", "查一下", "找一下", "幫我查",  # 搜尋請求
    "怎麼", "為什麼", "什麼是", "如何",    # 知識問答
    "新聞", "股票", "匯率",                # 資訊查詢
]

def should_use_bert(result: Dict, original_text: str = "") -> bool:
    """判斷是否可以直接使用 BERT 結果"""
    intent = result["intent"]
    confidence = result["confidence"]
    slots = result["slots"]
    
    # 關鍵字預過濾：包含特定關鍵字 → 交給 LLM
    for keyword in LLM_KEYWORDS:
        if keyword in original_text:
            return False
    
    # 低信心度 → 交給 LLM
    if confidence < BERT_CONFIDENCE_THRESHOLD:
        return False
    
    # 純聊天 → 交給 LLM
    if intent == "chat":
        return False
    
    # 檢查必要 slots
    required_slots = {
        "turn_on": ["name"],
        "turn_off": ["name"],
        "get_state": ["name"],
        "climate_set_mode": ["mode"],
    }
    
    if intent in required_slots:
        for required_slot in required_slots[intent]:
            if required_slot not in slots or not slots[required_slot]:
                return False
    
    return True

def generate_catgirl_response(intent: str, slots: Dict) -> str:
    """為 BERT 結果生成喵化回應
    
    注意：回應不再包含區域名稱，因為：
    1. BERT 可能錯誤拆分（如「床頭燈」→ area=床, name=床頭燈）
    2. HA 組件會再次修正區域
    3. 使用者通常不需要在回應中看到區域確認
    """
    name = slots.get("name", "設備")
    mode = slots.get("mode", "")
    
    # 只使用 name，不加 area（避免重複和錯誤）
    templates = {
        "turn_on": [
            f"好的喵！正在開啟{name}～",
            f"收到喵！{name}開起來囉～",
            f"沒問題喵！幫你開啟{name}了～",
        ],
        "turn_off": [
            f"好的喵！正在關閉{name}～",
            f"收到喵！{name}關掉囉～",
            f"沒問題喵！幫你關閉{name}了～",
        ],
        "get_state": [
            f"讓我看看{name}的狀態喵～",
            f"我來查一下{name}喵！",
        ],
        "climate_set_mode": [
            f"好的喵！正在設定{mode}模式～",
            f"收到喵！切換到{mode}模式囉～",
        ],
    }
    
    import random
    responses = templates.get(intent, [f"好的喵！"])
    return random.choice(responses)

def build_action_string(intent: str, slots: Dict, response_text: str) -> str:
    """建構 ACTION 字串（用於歷史記錄）"""
    lines = [response_text, f"ACTION {intent}"]
    
    for key, value in slots.items():
        lines.append(f"{key} {value}")
    
    return "\n".join(lines)

# ==============================================================================
# Qwen 推理（保持與原版相同的邏輯）
# ==============================================================================

SYSTEM_PROMPT = """你是日和喵，可愛的貓娘智慧家居助理喵！

輸出格式：
<回應文字（加入「喵」增加萌感）>
ACTION <action_name>
name <設備中文名稱>
area <區域中文名稱>
[其他參數...]

可用 ACTION：
- turn_on/off: 開關設備 (name, area)
- get_state: 查詢設備狀態 (name, area) - 當用戶詢問設備狀態時必須使用
- search: 搜尋資訊 (query) - 當需要搜尋天氣、新聞、網路資訊時使用
- climate_set_mode: 設定空調模式 (area, mode)

重要原則：
1. 你不知道任何設備的當前狀態，必須使用 get_state 來查詢
2. 用戶詢問「...開著嗎」「...是什麼狀態」時，輸出 ACTION get_state
3. 用戶詢問天氣、新聞、需要上網查詢的資訊時，輸出 ACTION search
4. name 是設備名稱（例如：大燈、風扇、冷氣）
5. area 是區域名稱（例如：書房、客廳、臥室），預設用書房
6. 純聊天時不輸出 ACTION
7. 回應要親切可愛，適當加入「喵」

範例：
用戶：關掉書房大燈
助理：好的，正在關閉書房大燈喵
ACTION turn_off
name 大燈
area 書房

用戶：明天會下雨嗎
助理：讓我查一下天氣預報喵～
ACTION search
query 明天天氣預報

用戶：中和區今天天氣如何
助理：我來查一下中和區的天氣喵！
ACTION search
query 中和區今天天氣"""

def qwen_inference(user_input: str, history: list = None) -> str:
    """使用 Qwen 進行推理"""
    current_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"現在時間: {current_dt}"},
    ]
    
    if history:
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": f"User request:\n{user_input}"})
    
    prompt = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = qwen_tokenizer(
        prompt, 
        return_tensors="pt",
        max_length=MAX_SEQ_LENGTH,
        truncation=True
    ).to("cuda")
    
    outputs = qwen_model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.3,
        top_p=0.95,
        do_sample=True,
        pad_token_id=qwen_tokenizer.eos_token_id,
    )
    
    # 不跳過特殊 token，這樣才能正確分割
    response = qwen_tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取最後一個助理回應
    # Qwen 格式: <|im_start|>assistant\n{回應}<|im_end|>
    if "<|im_start|>assistant" in response:
        # 取得最後一個 assistant 區塊
        parts = response.split("<|im_start|>assistant")
        last_response = parts[-1]
        
        # 移除結尾標記
        if "<|im_end|>" in last_response:
            last_response = last_response.split("<|im_end|>")[0]
        
        # 移除開頭的換行
        response = last_response.strip()
    else:
        # 備用：嘗試其他格式
        if "assistant\n" in response:
            parts = response.split("assistant\n")
            response = parts[-1].strip()
    
    # 簡轉繁
    response = s2t_converter.convert(response)
    
    return response

def parse_action_from_response(response: str) -> Optional[Dict]:
    """從 Qwen 回應解析 ACTION"""
    lines = response.strip().split('\n')
    action = None
    params = {}
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('ACTION '):
            action = line[7:].strip()
            continue
        
        for param in ['name', 'area', 'mode', 'temperature', 'brightness', 'query']:
            if line.startswith(f'{param} '):
                params[param] = line[len(param)+1:].strip()
                break
    
    if action:
        return {"action": action, "params": params}
    return None

# ==============================================================================
# API Endpoints
# ==============================================================================

@app.on_event("startup")
async def startup():
    """啟動時載入模型"""
    load_bert_model()
    load_qwen_model()

@app.get("/")
def root():
    return {
        "service": "Qwen + BERT Catgirl Home Assistant API",
        "bert_model": BERT_MODEL_PATH,
        "bert_confidence_threshold": BERT_CONFIDENCE_THRESHOLD,
    }

@app.get("/health")
def health():
    return {"status": "healthy", "bert_loaded": bert_model is not None, "qwen_loaded": qwen_model is not None}

@app.post("/process", response_model=ActionResult)
async def process_request(request: InferenceRequest):
    """
    處理用戶請求
    
    流程：
    1. BERT 意圖分類 + 填槽
    2. 高確定性 → 直接返回 ACTION
    3. 低確定性/聊天 → Qwen LLM 處理
    """
    logger.info("=" * 60)
    logger.info(f"📥 收到請求: {request.text}")
    
    try:
        # ===== Step 0: 簡單模板回覆（最快，不需要任何模型）=====
        
        # 時間詢問
        TIME_PATTERN = re.compile(r'(幾點幾分|現在幾點|現在是幾點|現在幾點幾分|幾點了)')
        if TIME_PATTERN.search(request.text):
            now = datetime.now()
            hour = now.hour
            minute = now.minute
            period = '上午' if hour < 12 else '下午'
            hour12 = hour if 1 <= hour <= 12 else (hour - 12 if hour > 12 else 12)
            response_text = f"現在是{period}{hour12}點{minute:02d}分喵～"
            logger.info(f"⏰ 時間詢問，模板回覆: {response_text}")
            
            return ActionResult(
                action=None,
                params={},
                response_text=response_text,
                has_action=False,
                raw_response=response_text,
                processed_by="template",
            )
        
        # 日期詢問
        DATE_PATTERN = re.compile(r'(幾月幾號|今天幾號|現在是幾號|今天是幾月幾號|今天日期|今天星期幾)')
        if DATE_PATTERN.search(request.text):
            now = datetime.now()
            weekdays = ['一', '二', '三', '四', '五', '六', '日']
            weekday = weekdays[now.weekday()]
            response_text = f"今天是{now.year}年{now.month}月{now.day}日，星期{weekday}喵～"
            logger.info(f"📅 日期詢問，模板回覆: {response_text}")
            
            return ActionResult(
                action=None,
                params={},
                response_text=response_text,
                has_action=False,
                raw_response=response_text,
                processed_by="template",
            )
        
        # 天氣查詢 → 直接構造 search ACTION
        WEATHER_PATTERN = re.compile(r'(天氣|下雨|氣溫|會不會下雨|降雨)')
        if WEATHER_PATTERN.search(request.text):
            # 提取查詢內容（直接用原始輸入作為 query）
            query = request.text
            response_text = f"讓我查一下天氣資訊喵～"
            logger.info(f"🌤️ 天氣查詢，直接構造 search ACTION")
            
            return ActionResult(
                action="search",
                params={"query": query},
                response_text=response_text,
                has_action=True,
                raw_response=f"{response_text}\nACTION search\nquery {query}",
                processed_by="template",
            )
        
        # ===== Step 1: BERT 預處理 =====
        bert_result = bert_predict(request.text)
        logger.info(f"🔍 BERT: intent={bert_result['intent']}, conf={bert_result['confidence']:.2%}, slots={bert_result['slots']}")
        
        # Step 2: 判斷是否可以直接使用 BERT 結果
        if should_use_bert(bert_result, request.text):
            logger.info("⚡ BERT 直接處理（高確定性）")
            
            intent = bert_result["intent"]
            slots = bert_result["slots"]
            
            # 生成喵化回應
            response_text = generate_catgirl_response(intent, slots)
            raw_response = build_action_string(intent, slots, response_text)
            
            logger.info(f"📤 BERT 回應: {response_text}")
            
            return ActionResult(
                action=intent,
                params=slots,
                response_text=response_text,
                has_action=True,
                raw_response=raw_response,
                processed_by="bert",
            )
        
        # Step 3: 交給 Qwen LLM 處理
        logger.info("🧠 交給 Qwen LLM 處理...")
        
        response = qwen_inference(request.text, request.history)
        action_data = parse_action_from_response(response)
        
        # 提取純文字回應
        response_lines = response.split('\n')
        response_text_lines = []
        for line in response_lines:
            if not line.strip().startswith(('ACTION', 'name', 'area', 'mode', 'temperature', 'brightness', 'query')):
                response_text_lines.append(line)
        response_text = '\n'.join(response_text_lines).strip()
        
        logger.info(f"📤 Qwen 回應: {response_text}")
        
        return ActionResult(
            action=action_data['action'] if action_data else None,
            params=action_data['params'] if action_data else {},
            response_text=response_text,
            has_action=action_data is not None,
            raw_response=response,
            processed_by="qwen",
        )
        
    except Exception as e:
        import traceback
        logger.error(f"❌ 處理請求時發生錯誤: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_with_state", response_model=SearchResultResponse)
async def process_with_state(request: StateResultRequest):
    """處理 get_state 二次對話（使用 LLM 生成回應）"""
    logger.info(f"🔍 get_state 二次對話: {request.device_name}@{request.area}")
    
    full_name = f"{request.area}{request.device_name}" if request.area else request.device_name
    
    # 構建多輪對話 prompt
    current_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"現在時間: {current_dt}"},
        {"role": "user", "content": f"User request:\n{request.original_question}"},
        {"role": "assistant", "content": f"我來幫你看一下喵～\nACTION get_state\nname {request.device_name}\narea {request.area}"},
        {"role": "user", "content": f"State result:\n{full_name}: {request.state_result}"},
    ]
    
    prompt = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = qwen_tokenizer(
        prompt, 
        return_tensors="pt",
        max_length=MAX_SEQ_LENGTH,
        truncation=True
    ).to("cuda")
    
    outputs = qwen_model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.5,  # 稍高的溫度增加多樣性
        top_p=0.9,
        do_sample=True,
        pad_token_id=qwen_tokenizer.eos_token_id
    )
    
    response = qwen_tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取最後一個 assistant 回應
    if "<|im_start|>assistant" in response:
        parts = response.split("<|im_start|>assistant")
        last_part = parts[-1].strip()
        if last_part.startswith("\n"):
            last_part = last_part[1:]
        if "<|im_end|>" in last_part:
            last_part = last_part.split("<|im_end|>")[0]
        response = last_part.strip()
    
    # 清理
    response = response.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    response = s2t_converter.convert(response)
    
    logger.info(f"📤 二次對話回應: {response}")
    
    return SearchResultResponse(response_text=response)

@app.post("/search_result", response_model=SearchResultResponse)
async def process_search_result(request: SearchResultRequest):
    """處理搜尋結果回應"""
    # 使用 Qwen 喵化搜尋結果
    # 這裡可以直接用模板或調用 Qwen
    response_text = f"{request.search_result}喵～"
    return SearchResultResponse(response_text=response_text)

@app.post("/fallback_assistant", response_model=SearchResultResponse)
async def process_fallback_assistant(request: FallbackAssistantRequest):
    """處理 Fallback 助理回應喵化"""
    # 簡單喵化
    response_text = request.assistant_response
    if "喵" not in response_text:
        response_text = f"{response_text}喵～"
    return SearchResultResponse(response_text=response_text)

# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == "__main__":
    logger.info("🚀 啟動 Qwen + BERT Catgirl Home Assistant API Server")
    logger.info(f"📡 監聽: {HOST}:{PORT}")
    logger.info(f"🔧 BERT 信心閾值: {BERT_CONFIDENCE_THRESHOLD}")
    uvicorn.run(app, host=HOST, port=PORT)
