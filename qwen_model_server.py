#!/usr/bin/env python3
"""
qwen_model_server.py
提供 Qwen 模型的 REST API 服務 (v6 - name + area 格式)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from unsloth import FastLanguageModel
import uvicorn
from typing import Optional, List, Dict
import re
from collections import defaultdict
import logging
from datetime import datetime
from opencc import OpenCC




# ==============================================================================
# 配置
# 可以選擇"./qwen-catgirl-ha-switch-v2"
# 或"./qwen2.5-1.5b-home-assistant"
# ==============================================================================
MODEL_PATH = "./qwen-catgirl-ha-switch-v2"
MAX_SEQ_LENGTH = 2048
HOST = "0.0.0.0"
PORT = 8124  # 改成 8124 避免與 HA 衝突

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True  # 強制覆蓋現有的日誌配置
)
logger = logging.getLogger(__name__)

# 同時設定 uvicorn 的 logger
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_access_logger = logging.getLogger("uvicorn.access")

app = FastAPI(title="Qwen Catgirl Home Assistant API v6")

# ==============================================================================
# 全局變數
# ==============================================================================
model = None
tokenizer = None
s2t_converter = OpenCC('s2t')  # 簡體轉繁體

# 聊天回應詞彙修正字典（修正高隨機性導致的不當用詞）
CHAT_WORD_FIXES = {
    r'夢魘': '夢境',
    r'噩夢': '好夢',
}

# ==============================================================================
# System Prompt (v7 版本 - 不預先給設備列表)
# ==============================================================================
current_dt = datetime.now().strftime("%Y-%m-%d %H:%M")
current_time = datetime.now().strftime("%H:%M")
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
- search: 搜尋資訊 (query) 
- climate_set_mode: 設定空調模式 (area, mode)

重要原則：
1. 你不知道任何設備的當前狀態，必須使用 get_state 來查詢
2. 用戶詢問「...開著嗎」「...是什麼狀態」「現在幾度」等問題時，必須輸出 ACTION get_state
3. 只輸出中文 name 和 area，不要輸出 entity_id
4. name 是設備名稱（例如：大燈、風扇、冷氣）
5. area 是區域名稱（例如：書房、客廳、臥室），如果使用者沒有說區域的話，預設用書房
6. 純聊天時不輸出 ACTION
7. 回應要親切可愛，適當加入「喵」

範例：
用戶：客廳燈開著嗎
助理：我來幫你看一下喵～
ACTION get_state
name 燈
area 客廳

用戶：關掉書房大燈
助理：好的，正在關閉書房大燈喵
ACTION turn_off
name 大燈
area 書房

用戶：設定客廳冷氣為冷氣模式  
助理：好的喵！正在設定客廳冷氣為冷氣模式喵
ACTION climate_set_mode
area 客廳
mode cool

用戶：你好
助理：你好呀主人！有什麼需要幫忙的嗎喵？"""
# ==============================================================================
# 搜尋結果回答之簡易提示詞
# ==============================================================================
SEARCH_RESULT_PROMPT = """你是日和喵，可愛的貓娘智慧家居助理喵！

你的任務是根據搜尋結果，用貓娘風格回答使用者的問題。

【重要原則】
1. **絕對不能改變任何數字、時間、日期、溫度、百分比等具體資訊**
2. 保留搜尋結果中的所有重要資訊（降雨機率、溫度、濕度等）
3. 用親切可愛的語氣表達，加入「喵」增加萌感（1-2個喵即可）
4. **直接轉述資訊，不要額外解析或總結**
5. 保持簡潔，不要過度解釋
6. 不要添加搜尋結果中沒有的資訊（如「適宜外出」等建議）
7. 避免使用項目符號或複雜格式，用自然語句表達

【範例】
問題：明天會下雨嗎
搜尋結果：明天有60%的降雨機率，溫度25度
回答：明天有60%的降雨機率，溫度25度喔喵～記得帶傘喵！

問題：台北現在幾度
搜尋結果：台北現在氣溫28度，濕度75%
回答：台北現在氣溫28度，濕度75%喔喵～

問題：中和區今天的天氣
搜尋結果：根據AccuWeather的資料，今晚中和區晚上6點到11點的溫度大約在72-75°F (約22-24°C) 之間。中央氣象署的資料顯示，新北市今晚降雨機率為10%
回答：根據AccuWeather，今晚中和區晚上6點到11點的溫度大約在72-75°F (約22-24°C) 之間喵～中央氣象署說新北市今晚降雨機率為10%喔！"""

# ==============================================================================
# Fallback 助理回應喵化提示詞
# ==============================================================================
FALLBACK_ASSISTANT_PROMPT = """你是日和喵，可愛的貓娘智慧家居助理喵！

你的任務是將其他助理（如 Gemini、Home Assistant Assist）的回應，用你獨特的貓娘風格重新表達。

【重要原則】
1. **絕對不能改變任何數字、時間、日期、溫度等具體資訊**
2. 保留原始回應的完整內容和意思，一字不漏
3. 只調整語氣，加入「喵」增加萌感（但不要過度使用，1-2個喵即可）
4. 使用親切可愛的語氣
5. 保持簡短自然
6. 不要添加原本沒有的資訊
7. 不要省略任何重要資訊

【範例】
原始：溫度已設定27
喵化：好的喵！溫度已經設定為27度了喵～

原始：提醒已設定在明天下午3點
喵化：好的喵！已經幫你設定明天下午3點的提醒了～

原始：已開啟客廳的燈
喵化：好的喵！已經幫你開啟客廳的燈了～"""

# ==============================================================================
# Request/Response Models
# ==============================================================================
class Device(BaseModel):
    entityId: str
    friendlyName: str
    domain: str
    state: str
    area: Optional[str] = None  # 新增 area 欄位
    brightnessPct: Optional[int] = None
    color: Optional[str] = None
    currentTemp: Optional[float] = None
    targetTemp: Optional[float] = None
    position: Optional[int] = None
    percentage: Optional[int] = None

class InferenceRequest(BaseModel):
    """處理用戶請求（v7: devices 變成可選）"""
    text: str
    devices: Optional[List[Device]] = None  # v7: 變成可選，不再強制要求
    language: Optional[str] = "zh"
    history: Optional[List[dict]] = None  # 對話歷史：[{"role": "user", "content": "..."}, ...]

class StateResultRequest(BaseModel):
    """v7: 處理 get_state 二次對話請求"""
    original_question: str  # 用戶的原始問題
    state_result: str  # get_state 返回的狀態結果
    device_name: str  # 設備名稱
    area: str  # 區域名稱

class SearchResultRequest(BaseModel):
    """搜尋結果回應請求（不需要裝置列表）"""
    user_question: str  # 使用者的原始問題
    search_result: str  # 搜尋工具返回的結果

class FallbackAssistantRequest(BaseModel):
    """Fallback 助理回應喵化請求"""
    user_question: str  # 使用者的原始問題
    assistant_response: str  # 其他助理（Google Assistant、Alexa等）的回應

class ActionResult(BaseModel):
    action: Optional[str] = None
    params: Dict[str, str] = {}
    response_text: str
    has_action: bool
    raw_response: Optional[str] = None  # 保存包含 ACTION 的原始回應，用於對話歷史

class SearchResultResponse(BaseModel):
    """搜尋結果回應（只包含文字回應）"""
    response_text: str

# ==============================================================================
# 載入模型
# ==============================================================================
@app.on_event("startup")
async def load_model():
    global model, tokenizer
    logger.info("=" * 80)
    logger.info("🚀 正在載入 Qwen v6 模型 (name + area 格式)...")
    logger.info(f"📂 模型路徑: {MODEL_PATH}")
    logger.info("=" * 80)
    
    # 根據 MODEL_PATH 自動為 1.5B 模型選擇全精度推理
    if "1.5b" in MODEL_PATH.lower():
        logger.info("🔧 偵測到 1.5B 模型，啟用全精度 (float32) 推理，關閉 4-bit 加速")
        dtype = torch.float32
        load_in_4bit = False
    else:
        logger.info("🔧 未偵測到 1.5B，使用預設量化/省記憶體設定（4-bit）")
        dtype = None
        load_in_4bit = True
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    FastLanguageModel.for_inference(model)
    
    logger.info("✅ 模型載入完成！")
    logger.info(f"📊 max_seq_length: {MAX_SEQ_LENGTH}")
    logger.info(f"📡 監聽端口: {PORT}")
    logger.info("=" * 80)

# ==============================================================================
# Device List 格式化（Domain 分組 + area）
# ==============================================================================
def devices_to_domain_grouped_format(devices: List[Device]) -> str:
    """將設備列表轉換為 Domain 分組格式（v6 - 包含 area）"""
    grouped = defaultdict(list)
    
    for device in devices:
        grouped[device.domain].append(device)
    
    lines = []
    for domain in sorted(grouped.keys()):
        lines.append(f"{domain}:")
        for device in grouped[domain]:
            short_id = device.entityId.split('.', 1)[1] if '.' in device.entityId else device.entityId
            name = device.friendlyName
            state = device.state
            
            # v6: 加入 area 資訊
            area = f"[{device.area}]" if device.area else ""
            
            line = f"  {short_id} '{name}' {area} {state}"
            
            # 添加額外屬性
            if device.domain == 'light' and device.brightnessPct:
                line += f" {device.brightnessPct}%"
                if device.color:
                    line += f" {device.color}"
            elif device.domain == 'climate':
                if device.currentTemp:
                    line += f" curr={device.currentTemp}"
                if device.targetTemp:
                    line += f" target={device.targetTemp}"
            elif device.domain == 'cover' and device.position is not None:
                line += f" pos={device.position}%"
            elif device.domain == 'fan' and device.percentage:
                line += f" {device.percentage}%"
            
            lines.append(line)
    
    return '\n'.join(lines)

# ==============================================================================
# ACTION 解析 (v6 格式：支援 name + area)
# ==============================================================================
def parse_action_from_response(response: str) -> Optional[Dict[str, any]]:
    """從模型輸出解析 ACTION (v6 格式)"""
    if "ACTION" not in response:
        return None
    
    lines = response.split('\n')
    action = None
    params = {}
    
    for line in lines:
        line = line.strip()
        
        # 提取 ACTION
        if line.startswith("ACTION "):
            action = line.replace("ACTION ", "").strip()
        
        # v6: 提取 name（不帶引號）
        elif line.startswith("name "):
            params['name'] = line.replace("name ", "").strip()
        
        # v6: 提取 area（不帶引號）
        elif line.startswith("area "):
            params['area'] = line.replace("area ", "").strip()
        
        # 提取其他參數
        elif line.startswith("brightness "):
            params['brightness'] = int(line.replace("brightness ", "").strip())
        
        elif line.startswith("color "):
            params['color'] = line.replace("color ", "").strip()
        
        elif line.startswith("temperature "):
            params['temperature'] = float(line.replace("temperature ", "").strip())
        
        elif line.startswith("command "):
            params['command'] = line.replace("command ", "").strip()
        
        elif line.startswith("position "):
            params['position'] = int(line.replace("position ", "").strip())
        
        elif line.startswith("volume "):
            params['volume'] = int(line.replace("volume ", "").strip())
        elif line.startswith("query "):
            params['query'] = line.replace("query ", "").strip()
        elif line.startswith('mode '):  # ⭐ 加入這行！
            params['mode'] = line.replace('mode ', '').strip()
    if not action:
        return None
    
    return {
        "action": action,
        "params": params
    }

# ==============================================================================
# 推理
# ==============================================================================
# 控制關鍵字正則表達式
CONTROL_KEYWORDS_PATTERN = re.compile(
    r'(溫度|濕度|開|關|把|設定|調整|打開|關閉|切換|調成|變成|設為|啟動|停止|改成)'
)

# 時間/日期詢問偵測
TIME_QUESTION_PATTERN = re.compile(r'(幾點幾分|現在幾點|現在是幾點|現在幾點幾分)')
DATE_QUESTION_PATTERN = re.compile(r'(幾月幾號|今天幾號|現在是幾號|今天是幾月幾號|今天日期)')


def run_inference(user_input: str, device_list: str = None, history: list = None) -> str:
    """執行推理（支援純聊天時用高 temp 重新生成）
    
    v7 更新：device_list 變成可選參數
    - 如果提供 device_list，會包含在 prompt 中（向後兼容）
    - 如果不提供，使用純粹的 user request（新架構）
    
    Args:
        user_input: 用戶輸入
        device_list: 設備列表（可選）
        history: 對話歷史 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    print(f"🎯 開始推理 - 使用者輸入: {user_input}")
    logger.info(f"🎯 開始推理 - 使用者輸入: {user_input}")
    
    if history:
        logger.info(f"📝 收到 {len(history)} 條歷史訊息")
    
    # 檢查是否包含控制關鍵字或狀態詢問關鍵字
    has_control_keyword = bool(CONTROL_KEYWORDS_PATTERN.search(user_input))
    
    # v7: 新增狀態詢問關鍵字
    STATE_QUERY_PATTERN = re.compile(r'(開著嗎|關著嗎|亮著嗎|是什麼狀態|現在幾度|溫度多少|有開嗎|有鎖嗎|鎖好了嗎)')
    has_state_query = bool(STATE_QUERY_PATTERN.search(user_input))
    
    if has_control_keyword or has_state_query:
        logger.info("🔧 檢測到控制/狀態查詢關鍵字，使用低 temperature (0.3)")
        print("🔧 檢測到控制/狀態查詢關鍵字，使用低 temperature")
        initial_temp = 0.3
    else:
        logger.info("💬 未檢測到控制關鍵字，先用中低 temperature (0.3) 嘗試")
        print("💬 未檢測到控制關鍵字，先用中低 temperature 嘗試")
        initial_temp = 0.3
    
    # v7: 根據是否有 device_list 決定 prompt 格式
    if device_list:
        user_message = f"Available devices:\n{device_list}\n\nUser request: {user_input}"
        logger.debug(f"📋 設備列表:\n{device_list}")
    else:
        user_message = f"User request:\n{user_input}"
        logger.info("📋 v7 模式：不使用設備列表")
    
    current_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 構建 messages（包含歷史對話）
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"現在時間: {current_dt}"},
    ]
    
    # ⭐ 添加對話歷史
    if history:
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        logger.info(f"📝 已添加 {len(history)} 條歷史到 prompt")
    
    # 添加當前用戶訊息
    messages.append({"role": "user", "content": user_message})
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    logger.debug(f"📝 Prompt 長度: {len(prompt)} 字元")
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        max_length=MAX_SEQ_LENGTH,
        truncation=True
    ).to("cuda")
    
    print(f"🔄 模型生成中 (temperature={initial_temp})...")
    logger.info(f"🔄 模型生成中 (temperature={initial_temp})...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=initial_temp,
        top_p=0.85,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取 assistant 的回應
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
    elif "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    print(f"✅ 初步推理完成 - 回應: {response[:100]}{'...' if len(response) > 100 else ''}")
    logger.info(f"✅ 初步推理完成 - 回應: {response[:100]}{'...' if len(response) > 100 else ''}")
    
    # ⭐ 檢測回顧性/上下文相關問題（這些問題的第一次回答通常是正確的，不應重新生成）
    CONTEXT_QUESTION_PATTERN = re.compile(r'(剛才|剛剛|之前|上一個|上次|剛說|剛做|說了什麼|做了什麼|什麼指令|什麼命令)')
    is_context_question = bool(CONTEXT_QUESTION_PATTERN.search(user_input))
    
    # 如果沒有控制/狀態查詢關鍵字 且 沒有 ACTION 且 不是回顧性問題，用高 temp 重新生成
    if not has_control_keyword and not has_state_query and "ACTION" not in response and not is_context_question:
        logger.info("🔄 檢測到純聊天（無 ACTION），使用高隨機性參數重新生成")
        print("🔄 檢測到純聊天，使用高隨機性參數重新生成以增加變化性")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.85,  # 提高 temperature 增加隨機性
            top_p=0.9,  # 提高 top_p 考慮更多選項
            top_k=50,  # 加入 top_k 限制在前 50 個 token 中選擇
            do_sample=True,
            repetition_penalty=1.15,  # 提高 repetition_penalty 避免重複
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 再次提取 assistant 的回應
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        elif "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        # 如果高隨機性導致出現 ACTION（任何格式），只保留第一行文字
        action_patterns = ["ACTION", "Action", "action:", "Action:", "Action to be taken"]
        has_unwanted_action = any(pattern in response for pattern in action_patterns)
        
        if has_unwanted_action:
            logger.warning("⚠️  高隨機性生成出現意外 ACTION，僅保留第一行文字")
            print("⚠️  高隨機性生成出現意外 ACTION，僅保留第一行文字")
            # 只保留第一行（純聊天回應）
            response = response.split('\n')[0].strip()
        
        # 簡轉繁（高隨機性可能導致簡體輸出）
        response = s2t_converter.convert(response)
        
        # 修正不當用詞（高隨機性可能導致奇怪的詞彙）
        for pattern, replacement in CHAT_WORD_FIXES.items():
            if re.search(pattern, response):
                logger.info(f"🔧 修正用詞: '{pattern}' → '{replacement}'")
                print(f"🔧 修正用詞: '{pattern}' → '{replacement}'")
                response = re.sub(pattern, replacement, response)
        
        print(f"✨ 高 temp 重新生成完成 - 回應: {response[:100]}{'...' if len(response) > 100 else ''}")
        logger.info(f"✨ 高 temp 重新生成完成 - 回應: {response[:100]}{'...' if len(response) > 100 else ''}")
    
    return response


def run_state_result_inference(original_question: str, state_result: str, device_name: str, area: str) -> str:
    """
    v7: 執行 get_state 二次對話推理
    
    當模型輸出 ACTION get_state 後，系統查詢設備狀態並返回結果。
    此函數負責根據狀態結果生成回答。
    
    Args:
        original_question: 用戶的原始問題
        state_result: 設備狀態結果
        device_name: 設備名稱
        area: 區域名稱
    """
    print(f"🔍 二次對話推理 - 問題: {original_question}")
    print(f"📊 狀態結果: {device_name}@{area} = {state_result}")
    logger.info(f"🔍 二次對話推理 - 問題: {original_question}")
    logger.info(f"📊 狀態結果: {device_name}@{area} = {state_result}")
    
    full_name = f"{area}{device_name}"
    
    # 構建多輪對話 prompt
    current_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"現在時間: {current_dt}"},
        {"role": "user", "content": f"User request:\n{original_question}"},
        {"role": "assistant", "content": f"我來幫你看一下喵～\nACTION get_state\nname {device_name}\narea {area}"},
        {"role": "user", "content": f"State result:\n{full_name}: {state_result}"},
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        max_length=MAX_SEQ_LENGTH,
        truncation=True
    ).to("cuda")
    
    print("🔄 模型生成中（二次對話模式）...")
    logger.info("🔄 模型生成中（二次對話模式）...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.3,  # 低溫度確保準確性
        top_p=0.85,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取 assistant 的最後一個回應
    # 格式應該是: ...<|im_start|>assistant\n{回應}<|im_end|>
    if "<|im_start|>assistant" in response:
        # 取最後一個 assistant 回應
        parts = response.split("<|im_start|>assistant")
        last_part = parts[-1].strip()
        
        # 移除開頭的換行
        if last_part.startswith("\n"):
            last_part = last_part[1:]
        
        # 移除 <|im_end|> 和之後的內容
        if "<|im_end|>" in last_part:
            last_part = last_part.split("<|im_end|>")[0]
        
        response = last_part.strip()
    else:
        # 備用方案：嘗試用 "assistant" 分割
        if "assistant\n" in response:
            parts = response.split("assistant\n")
            response = parts[-1].strip()
    
    # 移除可能殘留的特殊符號
    response = response.replace("<|im_end|>", "").strip()
    response = response.replace("<|im_start|>", "").strip()
    
    print(f"✅ 二次對話推理完成 - 回應: {response}")
    logger.info(f"✅ 二次對話推理完成 - 回應: {response}")
    
    return response


def run_search_result_inference(user_question: str, search_result: str) -> str:
    """
    執行搜尋結果回應推理（專用函數）
    
    特點：
    - 使用簡短的 SEARCH_RESULT_PROMPT
    - 不需要裝置列表
    - 只需要：原始問題 + 搜尋結果
    - 生成簡短可愛的回應
    """
    print(f"🔍 搜尋結果推理 - 問題: {user_question}")
    print(f"📊 搜尋結果: {search_result[:100]}{'...' if len(search_result) > 100 else ''}")
    logger.info(f"🔍 搜尋結果推理 - 問題: {user_question}")
    logger.info(f"📊 搜尋結果: {search_result[:100]}{'...' if len(search_result) > 100 else ''}")
    
    # 清理輸入（去除多餘空格和換行）
    user_question_clean = ' '.join(user_question.strip().split())
    search_result_clean = ' '.join(search_result.strip().split())
    
    # 構建 user prompt - 強調保留所有數據並保持簡潔
    user_message = f'使用者問："{user_question_clean}"，搜尋結果："{search_result_clean}"。請用貓娘風格簡潔地轉述這些資訊，【絕對不能改變任何數字】，【不要額外解析或總結】，直接用可愛的語氣說出來就好喵！'
    
    current_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    messages = [
        {"role": "system", "content": SEARCH_RESULT_PROMPT},
        {"role": "system", "content": f"現在時間: {current_dt}"},
        {"role": "user", "content": user_message}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        max_length=MAX_SEQ_LENGTH,
        truncation=True
    ).to("cuda")
    
    print("🔄 模型生成中（搜尋結果模式）...")
    logger.info("🔄 模型生成中（搜尋結果模式）...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,  # 適中長度，避免過度解釋
        temperature=0.3,  # 降低溫度以保持資訊準確性
        top_p=0.85,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取 assistant 的回應
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
    elif "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    # 移除尾部特殊符號
    response = response.rstrip()
    if response.endswith("<|im_end|>"):
        response = response[:-10].rstrip()
    
    # 完整輸出回應（不截斷）
    print(f"✅ 搜尋結果推理完成 - 回應:")
    print(response)
    logger.info(f"✅ 搜尋結果推理完成 - 回應:")
    logger.info(response)
    
    return response


def run_fallback_assistant_inference(user_question: str, assistant_response: str) -> str:
    """
    執行 Fallback 助理回應喵化推理
    
    使用場景：
    - 當指令 fallback 到其他助理（Google Assistant、Alexa 等）
    - 收到其他助理的回應後
    - 將回應喵化後再送出給使用者
    
    特點：
    - 使用專用的 FALLBACK_ASSISTANT_PROMPT
    - 保留原始回應的完整資訊
    - 加入貓娘風格和「喵」
    """
    print(f"🎭 Fallback 助理喵化 - 問題: {user_question}")
    print(f"💬 助理回應: {assistant_response}")
    logger.info(f"🎭 Fallback 助理喵化 - 問題: {user_question}")
    logger.info(f"💬 助理回應: {assistant_response}")
    
    # 清理輸入
    user_question_clean = ' '.join(user_question.strip().split())
    assistant_response_clean = ' '.join(assistant_response.strip().split())
    
    # 構建 user prompt - 強調不能改變數字
    user_message = f'使用者問："{user_question_clean}"，其他助理回答："{assistant_response_clean}"。請用貓娘風格重新表達，但【絕對不能改變任何數字、溫度、時間等資訊】，只改變語氣加入「喵」！'
    
    messages = [
        {"role": "system", "content": FALLBACK_ASSISTANT_PROMPT},
        {"role": "user", "content": user_message}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        max_length=MAX_SEQ_LENGTH,
        truncation=True
    ).to("cuda")
    
    print("🔄 模型生成中（Fallback 助理喵化模式）...")
    logger.info("🔄 模型生成中（Fallback 助理喵化模式）...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,  # 稍長一些以保留完整資訊
        temperature=0.3,  # 降低溫度以保持資訊準確性（從 0.6 改為 0.3）
        top_p=0.85,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取 assistant 的回應
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
    elif "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    # 移除尾部特殊符號
    response = response.rstrip()
    if response.endswith("<|im_end|>"):
        response = response[:-10].rstrip()
    
    # 完整輸出回應（不截斷）
    print(f"✅ Fallback 助理喵化完成 - 回應:")
    print(response)
    logger.info(f"✅ Fallback 助理喵化完成 - 回應:")
    logger.info(response)
    
    return response

# ==============================================================================
# API Endpoints
# ==============================================================================
@app.get("/")
async def root():
    return {
        "status": "ok", 
        "message": "Qwen Catgirl Home Assistant API v7",
        "version": "v7 (get_state architecture)",
        "model": MODEL_PATH
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/process", response_model=ActionResult)
async def process_request(request: InferenceRequest):
    """
    處理用戶請求 (v7 版本 - 不強制要求 devices)
    
    v7 變更：
    - devices 變成可選參數
    - 如果不提供 devices，模型將不知道設備狀態，會主動調用 get_state 查詢
    """
    print("=" * 60)
    print(f"📨 收到 /process 請求")
    print(f"💬 使用者輸入: {request.text}")
    
    # v7: devices 是可選的
    device_count = len(request.devices) if request.devices else 0
    print(f"🔧 設備數量: {device_count} {'(v7: 不使用設備列表)' if device_count == 0 else ''}")
    
    logger.info("=" * 60)
    logger.info(f"📨 收到 /process 請求")
    logger.info(f"💬 使用者輸入: {request.text}")
    logger.info(f"🔧 設備數量: {device_count}")
    
    if model is None:
        logger.error("❌ 模型未載入")
        print("❌ 模型未載入")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # v7: 格式化設備列表（如果有提供）
        device_list = None
        if request.devices:
            device_list = devices_to_domain_grouped_format(request.devices)
        
        # 檢查是否為時間或日期詢問，若是直接用模板回覆再交由 fallback 助理進行喵化處理
        if TIME_QUESTION_PATTERN.search(request.text):
            now = datetime.now()
            hour = now.hour
            minute = now.minute
            period = '上午' if hour < 12 else '下午'
            hour12 = hour if 1 <= hour <= 12 else (hour - 12 if hour > 12 else 12)
            assistant_response = f"現在是{period}{hour12}:{minute:02d}"
            logger.info(f"⏰ 偵測到時間詢問，回傳模板: {assistant_response}")
            print(f"⏰ 偵測到時間詢問，回傳模板: {assistant_response}")

            # 使用 fallback 助理進行喵化處理
            response = run_fallback_assistant_inference(request.text, assistant_response)
            action_data = None
        elif DATE_QUESTION_PATTERN.search(request.text):
            today = datetime.now()
            # 直接使用喵化模板回覆，不經由 LLM 處理
            assistant_response = f"今天是{today.year}年{today.month}月{today.day}日喵～"
            logger.info(f"📅 偵測到日期詢問，回傳模板（已喵化）: {assistant_response}")
            print(f"📅 偵測到日期詢問，回傳模板（已喵化）: {assistant_response}")

            # 不呼叫 LLM，直接回傳已喵化的文字
            response = assistant_response
            action_data = None
        else:
            # 執行推理（一般流程，包含對話歷史）
            response = run_inference(request.text, device_list, request.history)
            
            # 解析 ACTION (v6 格式)
            action_data = parse_action_from_response(response)
        
        if action_data:
            logger.info(f"⚡ 解析到 ACTION: {action_data['action']}")
            logger.info(f"📋 參數: {action_data['params']}")
            print(f"⚡ 解析到 ACTION: {action_data['action']}")
            print(f"📋 參數: {action_data['params']}")
        else:
            logger.info("💭 無 ACTION（純聊天）")
            print("💭 無 ACTION（純聊天）")
        
        # 提取純文字回應（去除 ACTION 部分）
        response_lines = response.split('\n')
        response_text = []
        for line in response_lines:
            if not line.strip().startswith(('ACTION', 'name', 'area', 'brightness', 'color', 'temperature', 'command', 'position', 'volume', 'mode')):
                response_text.append(line)
        response_text = '\n'.join(response_text).strip()
        
        logger.info(f"📤 返回回應: {response_text}")
        logger.info("=" * 60)
        print(f"📤 返回回應: {response_text}")
        print("=" * 60)
        
        return ActionResult(
            action=action_data['action'] if action_data else None,
            params=action_data['params'] if action_data else {},
            response_text=response_text,
            has_action=action_data is not None,
            raw_response=response,  # 保存完整回應（包含 ACTION）
        )
        
    except Exception as e:
        import traceback
        logger.error("❌ 處理請求時發生錯誤:")
        logger.error(traceback.format_exc())
        print("❌ 處理請求時發生錯誤:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_with_state", response_model=SearchResultResponse)
async def process_with_state(request: StateResultRequest):
    """
    v7: 處理 get_state 二次對話
    
    使用場景：
    1. 用戶詢問設備狀態（例如：「客廳燈開著嗎」）
    2. 模型輸出 ACTION get_state
    3. Home Assistant 查詢設備狀態
    4. 呼叫此 API，傳入原始問題和狀態結果
    5. 模型根據狀態結果生成回答
    
    參數：
    - original_question: 用戶的原始問題
    - state_result: 設備狀態（例如：「on」「off」「cool, 26°C」）
    - device_name: 設備名稱
    - area: 區域名稱
    """
    print("=" * 60)
    print("📨 收到 /process_with_state 請求")
    print(f"❓ 原始問題: {request.original_question}")
    print(f"📊 狀態結果: {request.device_name}@{request.area} = {request.state_result}")
    
    logger.info("=" * 60)
    logger.info("📨 收到 /process_with_state 請求")
    logger.info(f"❓ 原始問題: {request.original_question}")
    logger.info(f"📊 狀態結果: {request.device_name}@{request.area} = {request.state_result}")
    
    if model is None:
        logger.error("❌ 模型未載入")
        print("❌ 模型未載入")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 執行二次對話推理
        response_text = run_state_result_inference(
            original_question=request.original_question,
            state_result=request.state_result,
            device_name=request.device_name,
            area=request.area
        )
        
        print("=" * 60)
        logger.info("=" * 60)
        
        return SearchResultResponse(
            response_text=response_text
        )
        
    except Exception as e:
        import traceback
        logger.error("❌ 處理 get_state 二次對話時發生錯誤:")
        logger.error(traceback.format_exc())
        print("❌ 處理 get_state 二次對話時發生錯誤:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search_result", response_model=SearchResultResponse)
async def process_search_result(request: SearchResultRequest):
    """
    處理搜尋結果回應（不需要裝置列表）
    
    使用場景：
    1. 使用者提出需要搜尋的問題（例如：「週五中和區會下雨嗎」）
    2. 系統使用搜尋工具獲得結果
    3. 呼叫此 API，傳入原始問題和搜尋結果
    4. 模型生成簡短、可愛的回應
    
    特點：
    - 使用簡短的 SEARCH_RESULT_PROMPT（而非完整的 SYSTEM_PROMPT）
    - 不需要提供裝置列表
    - 自動清理換行和多餘空格
    - 只返回文字回應（不包含 ACTION）
    """
    print("=" * 60)
    print("📨 收到 /search_result 請求")
    logger.info("=" * 60)
    logger.info("📨 收到 /search_result 請求")
    
    if model is None:
        logger.error("❌ 模型未載入")
        print("❌ 模型未載入")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 執行搜尋結果推理
        response_text = run_search_result_inference(
            user_question=request.user_question,
            search_result=request.search_result
        )
        
        print("=" * 60)
        logger.info("=" * 60)
        
        return SearchResultResponse(
            response_text=response_text
        )
        
    except Exception as e:
        import traceback
        logger.error("❌ 處理搜尋結果時發生錯誤:")
        logger.error(traceback.format_exc())
        print("❌ 處理搜尋結果時發生錯誤:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fallback_assistant", response_model=SearchResultResponse)
async def process_fallback_assistant(request: FallbackAssistantRequest):
    """
    處理 Fallback 助理回應喵化
    
    使用場景：
    1. 使用者的指令因各種因素 fallback 到其他助理（Google Assistant、Alexa 等）
    2. 其他助理執行完畢並返回回應
    3. 呼叫此 API，傳入原始問題和其他助理的回應
    4. 模型將回應喵化後返回
    
    範例：
    - 使用者：「提醒我明天下午3點開會」
    - Google Assistant：「好的，我已經為你設定明天下午3點的提醒」
    - 本 API 喵化後：「好的喵！已經幫你設定明天下午3點的提醒了喵～」
    
    特點：
    - 使用專用的 FALLBACK_ASSISTANT_PROMPT
    - 保留原始回應的完整資訊和意思
    - 加入貓娘風格和「喵」
    - 只返回文字回應（不包含 ACTION）
    """
    print("=" * 60)
    print("📨 收到 /fallback_assistant 請求")
    logger.info("=" * 60)
    logger.info("📨 收到 /fallback_assistant 請求")
    
    if model is None:
        logger.error("❌ 模型未載入")
        print("❌ 模型未載入")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 執行 Fallback 助理回應喵化推理
        response_text = run_fallback_assistant_inference(
            user_question=request.user_question,
            assistant_response=request.assistant_response
        )
        
        print("=" * 60)
        logger.info("=" * 60)
        
        return SearchResultResponse(
            response_text=response_text
        )
        
    except Exception as e:
        import traceback
        logger.error("❌ 處理 Fallback 助理喵化時發生錯誤:")
        logger.error(traceback.format_exc())
        print("❌ 處理 Fallback 助理喵化時發生錯誤:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == "__main__":
    logger.info("🚀 啟動 Qwen Catgirl Home Assistant API Server v6")
    logger.info(f"📡 監聽: {HOST}:{PORT}")
    logger.info("💡 v6 特性: name + area 格式，零 entity_id 幻覺")
    uvicorn.run(app, host=HOST, port=PORT)

