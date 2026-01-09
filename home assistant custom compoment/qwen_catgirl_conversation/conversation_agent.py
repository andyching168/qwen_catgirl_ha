"""Qwen Catgirl conversation agent implementation (v7 - with trace support)."""
from __future__ import annotations

import asyncio
import logging
from typing import Literal
from difflib import SequenceMatcher

import aiohttp
import async_timeout

from homeassistant.components import conversation
from homeassistant.components.conversation import trace
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import label_registry as lr
from homeassistant.helpers import device_registry as dr
from homeassistant.util import ulid

from .const import (
    DOMAIN,
    CONF_MODEL_URL,
    CONF_FALLBACK_KEYWORDS,
    CONF_REMOVABLE_KEYWORDS,
    DEFAULT_MODEL_URL,
    DEFAULT_FALLBACK_KEYWORDS,
    DEFAULT_REMOVABLE_KEYWORDS,
)

_LOGGER = logging.getLogger(__name__)

# ⭐ 對話歷史快取（記憶體存儲，按 conversation_id 分組）
# 格式: {conversation_id: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
_conversation_history = {}

# 每個對話最多保留 3 輪（避免記憶體爆炸）
MAX_HISTORY_TURNS = 3


class QwenCatgirlConversationAgent(conversation.AbstractConversationAgent):
    """Qwen Catgirl conversation agent (v7 - with trace support)."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self.model_url = entry.data.get(CONF_MODEL_URL, DEFAULT_MODEL_URL)
        
        # ⭐ 獲取 fallback 關鍵字（優先使用 options，其次 data，最後預設值）
        self.fallback_keywords = (
            entry.options.get(CONF_FALLBACK_KEYWORDS)
            or entry.data.get(CONF_FALLBACK_KEYWORDS)
            or DEFAULT_FALLBACK_KEYWORDS
        )
        
        # ⭐ 獲取可移除關鍵字
        self.removable_keywords = (
            entry.options.get(CONF_REMOVABLE_KEYWORDS)
            or entry.data.get(CONF_REMOVABLE_KEYWORDS)
            or DEFAULT_REMOVABLE_KEYWORDS
        )
        
        # 確保是列表
        if isinstance(self.fallback_keywords, str):
            self.fallback_keywords = [
                k.strip() for k in self.fallback_keywords.split(",") if k.strip()
            ]
        
        if isinstance(self.removable_keywords, str):
            self.removable_keywords = [
                k.strip() for k in self.removable_keywords.split(",") if k.strip()
            ]
        
        _LOGGER.warning("=" * 60)
        _LOGGER.warning("🚀 Qwen Catgirl Agent Initialized")
        _LOGGER.warning("📍 Model URL: %s", self.model_url)
        _LOGGER.warning("🔑 Fallback keywords loaded: %s", self.fallback_keywords)
        _LOGGER.warning("✂️  Removable keywords loaded: %s", self.removable_keywords)
        _LOGGER.warning("=" * 60)

    @property
    def attribution(self) -> dict:
        """Return the attribution."""
        return {
            "name": "Qwen Catgirl v6",
            "url": "https://github.com/yourusername/qwen-catgirl",
        }

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return ["zh", "en"]

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        language = user_input.language or "zh"
        
        # 初始化調試資訊字典 (會添加到 Intent Response 中,在 Voice Assistant Debug 可見)
        self._debug_info = {
            "agent": "Qwen Catgirl v7",
            "model_url": self.model_url,
            "processing_steps": [],
        }
        
        _LOGGER.info("=== Qwen v7 Processing (with trace + multi-turn) ===")
        _LOGGER.info("User input: %s", user_input.text)
        
        # ⭐ 獲取或創建 conversation_id
        conversation_id = user_input.conversation_id or ulid.ulid()
        _LOGGER.info("Conversation ID: %s", conversation_id)
        
        # ⭐ 獲取對話歷史
        history = _conversation_history.get(conversation_id, [])
        if history:
            _LOGGER.info("📝 Found %d previous messages in this conversation", len(history))
            self._debug_info["processing_steps"].append({
                "step": "history_loaded",
                "message_count": len(history),
            })
        
        # ✅ Trace: 開始處理
        trace.async_conversation_trace_append(
            trace.ConversationTraceEventType.AGENT_DETAIL,
            {
                "agent": "Qwen Catgirl v7",
                "model_url": self.model_url,
                "step": "start_processing",
            }
        )
        
        # 🎵 檢測 fallback 關鍵字，直接交給 Gemini 處理（避免誤判）
        _LOGGER.info("🔍 Checking fallback keywords...")
        _LOGGER.info("Current fallback keywords: %s", self.fallback_keywords)
        _LOGGER.info("Current removable keywords: %s", self.removable_keywords)
        _LOGGER.info("User input: %s", user_input.text)
        
        import re
        
        # 檢測 fallback 關鍵字（完全匹配，不區分大小寫）
        detected_keywords = []
        for keyword in self.fallback_keywords:
            # 使用 \b 確保完整詞匹配（對中文則使用完整字串匹配）
            # 對於中文或混合文字，直接檢查是否包含（但要完整包含）
            if keyword.lower() in user_input.text.lower():
                detected_keywords.append(keyword)
        
        # 檢測並移除可移除關鍵字（完全匹配，不區分大小寫）
        detected_removable = []
        modified_text = user_input.text
        
        for keyword in self.removable_keywords:
            if keyword.lower() in user_input.text.lower():
                detected_removable.append(keyword)
        
        if detected_removable:
            _LOGGER.info("✂️  Detected removable keywords: %s", detected_removable)
            # 移除所有檢測到的可移除關鍵字（不區分大小寫）
            for keyword in detected_removable:
                # 使用正則表達式移除關鍵字（不區分大小寫）
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                modified_text = pattern.sub('', modified_text)
            
            # 清理多餘的空格
            modified_text = ' '.join(modified_text.split())
            _LOGGER.info("✂️  Text after removal: %s", modified_text)
        
        _LOGGER.info("Detected keywords: %s", detected_keywords if detected_keywords else "None")
        
        if detected_keywords:
            _LOGGER.warning("⚠️⚠️⚠️ FALLBACK TRIGGERED! Bypassing Qwen to Gemini ⚠️⚠️⚠️")
            _LOGGER.warning("Keywords found: %s in: %s", detected_keywords, user_input.text)
            
            # 記錄到調試資訊
            self._debug_info["processing_steps"].append({
                "step": "fallback_keyword_bypass",
                "detected_keywords": detected_keywords,
                "detected_removable": detected_removable,
                "original_text": user_input.text,
                "modified_text": modified_text if detected_removable else user_input.text,
                "action": "fallback_to_gemini",
            })
            
            # ✅ Trace: Fallback 關鍵字檢測
            trace.async_conversation_trace_append(
                trace.ConversationTraceEventType.AGENT_DETAIL,
                {
                    "step": "fallback_keyword_bypass",
                    "detected_keywords": detected_keywords,
                    "detected_removable": detected_removable,
                    "original_text": user_input.text,
                    "modified_text": modified_text if detected_removable else user_input.text,
                    "action": "bypass_to_gemini_fallback",
                    "reason": "User configured fallback keywords detected",
                }
            )
            
            # 如果有移除關鍵字，使用修改後的文字
            if detected_removable:
                # 創建一個新的 ConversationInput 物件，使用修改後的文字
                modified_input = conversation.ConversationInput(
                    text=modified_text,
                    conversation_id=user_input.conversation_id,
                    device_id=user_input.device_id,
                    language=user_input.language,
                    agent_id=user_input.agent_id,
                    context=user_input.context,
                    satellite_id=user_input.satellite_id,
                )
                return await self._fallback_to_builtin_assist(modified_input, language)
            
            return await self._fallback_to_builtin_assist(user_input, language)
        
        # ✅ v7: 不再預先收集所有設備狀態
        # 模型會在需要時調用 ACTION get_state 來查詢特定設備
        _LOGGER.info("v7 mode: Not collecting device states upfront")
        
        # 記錄到調試資訊
        self._debug_info["processing_steps"].append({
            "step": "v7_no_devices_mode",
            "reason": "Model will use get_state when needed",
        })
        
        # ✅ Trace: v7 模式
        trace.async_conversation_trace_append(
            trace.ConversationTraceEventType.AGENT_DETAIL,
            {
                "step": "v7_mode_enabled",
                "description": "Not sending device list, model will use get_state",
            }
        )
        
        # 呼叫模型 API（v7: 不傳設備列表）
        try:
            # ⭐ v7: 不傳入 devices，只傳 history
            action_result = await self._call_model_api(user_input.text, [], history)
            _LOGGER.info("Model response: action=%s, params=%s", 
                        action_result.get('action'), action_result.get('params'))
            
            # 記錄到調試資訊
            self._debug_info["processing_steps"].append({
                "step": "model_response",
                "action": action_result.get('action'),
                "params": action_result.get('params'),
                "has_action": action_result.get('has_action'),
            })
            
            # ✅ Trace: 模型回應
            trace.async_conversation_trace_append(
                trace.ConversationTraceEventType.AGENT_DETAIL,
                {
                    "step": "qwen_model_response",
                    "has_action": action_result.get('has_action'),
                    "action": action_result.get('action'),
                    "params": action_result.get('params'),
                    "response_text": action_result.get('response_text', '')[:100],  # 限制長度
                }
            )
        except Exception as err:
            _LOGGER.error("Error calling model API: %s", err)
            # 模型 API 失敗，使用內建 assist 作為備援
            _LOGGER.info("Falling back to built-in assist parser")
            return await self._fallback_to_builtin_assist(user_input, language)
        
        # ⭐ v7: 如果有 ACTION，處理對應動作
        if action_result.get('has_action'):
            action = action_result.get('action')
            
            # v7: 處理 get_state 動作（二次對話流程）
            if action == 'get_state':
                try:
                    params = action_result.get('params', {})
                    name = params.get('name', '')
                    area = params.get('area', '書房')  # 預設書房
                    
                    _LOGGER.info("🔍 v7: Executing get_state for %s@%s", name, area)
                    
                    # ✅ Trace: get_state 開始
                    trace.async_conversation_trace_append(
                        trace.ConversationTraceEventType.AGENT_DETAIL,
                        {
                            "step": "get_state_start",
                            "device_name": name,
                            "area": area,
                        }
                    )
                    
                    # 查詢單一設備狀態（返回 tuple: state, actual_area）
                    state_result, actual_area = await self._get_single_device_state(name, area)
                    _LOGGER.info("🔍 State result: %s, Actual area: %s", state_result, actual_area)
                    
                    # 調用 /process_with_state 進行二次對話（使用實際區域）
                    response_text = await self._call_model_with_state(
                        original_question=user_input.text,
                        state_result=state_result,
                        device_name=name,
                        area=actual_area  # 使用實際匹配到的區域
                    )
                    
                    intent_response = intent.IntentResponse(language=language)
                    intent_response.async_set_speech(response_text)
                    
                    # ✅ Trace: get_state 完成
                    trace.async_conversation_trace_append(
                        trace.ConversationTraceEventType.AGENT_DETAIL,
                        {
                            "step": "get_state_complete",
                            "state_result": state_result,
                            "response": response_text[:100],
                        }
                    )
                    
                except Exception as err:
                    _LOGGER.error("Error executing get_state: %s", err, exc_info=True)
                    # get_state 失敗，返回友善錯誤訊息
                    intent_response = intent.IntentResponse(language=language)
                    intent_response.async_set_speech("抱歉喵，我找不到這個設備的狀態喵...")
                    
            # 處理搜尋動作
            elif action == 'search':
                try:
                    intent_response = await self._execute_search(
                        action_result,
                        language,
                        user_input
                    )
                except Exception as err:
                    _LOGGER.error("Error executing search: %s", err, exc_info=True)
                    # 搜尋失敗，使用內建 assist 作為備援
                    _LOGGER.warning("=" * 60)
                    _LOGGER.warning("Search failed, falling back to built-in assist")
                    _LOGGER.warning("=" * 60)
                    try:
                        return await self._fallback_to_builtin_assist(user_input, language)
                    except Exception as fallback_err:
                        _LOGGER.error("Fallback also failed: %s", fallback_err, exc_info=True)
                        # 如果 fallback 也失敗，返回友善錯誤訊息
                        intent_response = intent.IntentResponse(language=language)
                        intent_response.async_set_speech("抱歉，我現在無法處理這個請求喵...")
                        return conversation.ConversationResult(
                            response=intent_response,
                            conversation_id=user_input.conversation_id or ulid.ulid(),
                        )
            else:
                # 處理其他設備控制動作
                try:
                    # v7: 由於沒有 devices 列表，跳過名稱修正
                    # 直接使用模型輸出的參數
                    _LOGGER.info("v7: Using model output params directly (no device list for correction)")
                    
                    intent_response = await self._execute_via_intent(
                        action_result, 
                        language, 
                        user_input
                    )
                    
                    # ✅ Trace: Intent 執行成功
                    intent_name = f"Hass{action.replace('_', ' ').title().replace(' ', '')}"
                    trace.async_conversation_trace_append(
                        trace.ConversationTraceEventType.TOOL_CALL,
                        {
                            "intent_name": intent_name,
                            "slots": action_result.get('params', {}),  # v7: 從 action_result 獲取 params
                        }
                    )
                except Exception as err:
                    _LOGGER.error("Error executing intent: %s", err, exc_info=True)
                    
                    # ✅ Trace: Intent 失敗，觸發 fallback
                    trace.async_conversation_trace_append(
                        trace.ConversationTraceEventType.AGENT_DETAIL,
                        {
                            "step": "intent_execution_failed",
                            "error": str(err),
                            "error_type": type(err).__name__,
                            "fallback_triggered": True,
                        }
                    )
                    
                    # 執行失敗，使用內建 assist 作為備援
                    _LOGGER.warning("=" * 60)
                    _LOGGER.warning("Intent execution failed, falling back to built-in assist")
                    _LOGGER.warning("=" * 60)
                    try:
                        return await self._fallback_to_builtin_assist(user_input, language)
                    except Exception as fallback_err:
                        _LOGGER.error("Fallback also failed: %s", fallback_err, exc_info=True)
                        # 如果 fallback 也失敗，返回友善錯誤訊息
                        intent_response = intent.IntentResponse(language=language)
                        intent_response.async_set_speech("抱歉，我現在無法處理這個請求喵...")
                        return conversation.ConversationResult(
                            response=intent_response,
                            conversation_id=user_input.conversation_id or ulid.ulid(),
                        )
        else:
            # 純聊天
            response_text = action_result.get('response_text', '收到')
            intent_response = intent.IntentResponse(language=language)
            intent_response.async_set_speech(response_text)
        
        # ⭐ 保存對話歷史（使用原始回應，包含 ACTION）
        history_response = action_result.get('raw_response') or intent_response.speech.get("plain", {}).get("speech", "")
        self._save_to_history(conversation_id, user_input.text, history_response)
        
        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=conversation_id,
        )

    async def _get_all_devices_with_area(self) -> list[dict]:
        """獲取有 'assist' 標籤的設備狀態（包含 area）"""
        from homeassistant.helpers import label_registry as lr
        from homeassistant.helpers import entity_registry as er
        from homeassistant.helpers import area_registry as ar
        
        devices = []
        
        try:
            # 獲取 registries
            label_reg = lr.async_get(self.hass)
            entity_registry = er.async_get(self.hass)
            area_reg = ar.async_get(self.hass)
            
            # 找到 assist 標籤
            assist_label_id = None
            for label_id, label_entry in label_reg.labels.items():
                if label_entry.name.lower() == "assist":
                    assist_label_id = label_id
                    break
            
            if assist_label_id is None:
                _LOGGER.warning("Label 'assist' not found")
                return []
            
            _LOGGER.info("Found 'assist' label: %s", assist_label_id)
            
            # 收集有標籤的 entity
            labeled_entities = []
            for entity_id, entity_entry in entity_registry.entities.items():
                if assist_label_id in entity_entry.labels:
                    labeled_entities.append((entity_id, entity_entry))
            
            _LOGGER.info("Found %d entities with 'assist' label", len(labeled_entities))
            
            # 收集設備狀態
            controllable_domains = {"light", "switch", "fan", "cover", "climate", "media_player", "vacuum", "lock"}
            
            for entity_id, entity_entry in labeled_entities:
                state = self.hass.states.get(entity_id)
                if state is None:
                    continue
                
                if state.domain not in controllable_domains:
                    continue
                
                # 獲取 area 名稱
                area_name = None
                if entity_entry.area_id:
                    area_entry = area_reg.async_get_area(entity_entry.area_id)
                    if area_entry:
                        area_name = area_entry.name
                
                device = {
                    "entityId": state.entity_id,
                    "friendlyName": state.attributes.get("friendly_name", state.entity_id),
                    "domain": state.domain,
                    "state": state.state,
                    "area": area_name,  # v6: 加入 area
                }
                
                # 安全地添加屬性
                try:
                    if state.domain == "light":
                        brightness = state.attributes.get("brightness")
                        if brightness is not None:
                            device["brightnessPct"] = int(brightness / 255 * 100)
                    
                    elif state.domain == "climate":
                        if "current_temperature" in state.attributes:
                            device["currentTemp"] = float(state.attributes["current_temperature"])
                        if "temperature" in state.attributes:
                            device["targetTemp"] = float(state.attributes["temperature"])
                    
                    elif state.domain == "cover":
                        if "current_position" in state.attributes:
                            device["position"] = int(state.attributes["current_position"])
                    
                    elif state.domain == "fan":
                        if "percentage" in state.attributes:
                            device["percentage"] = int(state.attributes["percentage"])
                
                except Exception as err:
                    _LOGGER.debug("Error processing %s: %s", entity_id, err)
                
                devices.append(device)
            
            _LOGGER.info("Collected %d controllable devices", len(devices))
            return devices
            
        except Exception as err:
            _LOGGER.error("Fatal error in _get_all_devices_with_area: %s", err, exc_info=True)
            return []

    async def _correct_area_in_response(self, response_text: str, matched_states: list | None) -> str:
        """修正回應文字中的區域名稱
        
        檢測回應中的區域名稱，若與實際設備區域不同則替換。
        
        Args:
            response_text: 原始回應文字
            matched_states: Intent 匹配到的設備狀態
            
        Returns:
            修正後的回應文字
        """
        _LOGGER.debug("🔍 _correct_area_in_response called, matched_states=%s", 
                     len(matched_states) if matched_states else 0)
        
        if not matched_states:
            return response_text
        
        try:
            from homeassistant.helpers import entity_registry as er
            from homeassistant.helpers import area_registry as ar
            from homeassistant.helpers import device_registry as dr
            
            entity_registry = er.async_get(self.hass)
            area_reg = ar.async_get(self.hass)
            device_reg = dr.async_get(self.hass)
            
            # 獲取第一個匹配設備的區域
            matched_entity_id = matched_states[0].entity_id
            entity_entry = entity_registry.async_get(matched_entity_id)
            
            actual_area_name = None
            if entity_entry:
                # 優先使用 entity 的區域
                area_id = entity_entry.area_id
                # 如果 entity 沒有區域，嘗試從 device 獲取
                if not area_id and entity_entry.device_id:
                    device_entry = device_reg.async_get(entity_entry.device_id)
                    if device_entry:
                        area_id = device_entry.area_id
                
                if area_id:
                    area_entry = area_reg.async_get_area(area_id)
                    if area_entry:
                        actual_area_name = area_entry.name
            
            if not actual_area_name:
                return response_text
            
            # 收集所有 HA 區域名稱（用於檢測回應中的錯誤區域）
            all_areas = area_reg.async_list_areas()
            all_area_names = [area.name for area in all_areas]
            
            # 加入常見的不友善區域名稱
            all_area_names.extend(["未分類", "未分类", "Unassigned", "unassigned"])
            
            _LOGGER.info("🏠 Device actual area: '%s'", actual_area_name)
            _LOGGER.info("📝 Original response: '%s'", response_text)
            _LOGGER.debug("🗂️ All areas to check: %s", all_area_names)
            
            # 檢查回應中是否包含任何區域名稱（不同於實際區域）
            corrected = response_text
            
            # 獲取設備名稱，用於避免替換設備名內的文字
            device_name = matched_states[0].name if matched_states[0].name else ""
            
            # 方法 1：替換已知的 HA 區域名稱（但不替換設備名稱內的部分）
            for area_name in all_area_names:
                if area_name in corrected and area_name != actual_area_name:
                    # 檢查是否在設備名稱內（如「床頭燈」裡的「床」）
                    if device_name and area_name in device_name:
                        _LOGGER.debug("📍 Skipping area '%s' inside device name '%s'", area_name, device_name)
                        continue
                    corrected = corrected.replace(area_name, actual_area_name)
                    _LOGGER.info("📍 Corrected known area: '%s' → '%s'", area_name, actual_area_name)
            
            # 方法 2：使用正則檢測「任何中文區域名 + 設備名」模式
            # 例如：「更衣室螢幕燈」「走廊大燈」等，即使區域不在 HA 列表中
            import re
            # 提取設備名稱（從 matched_states）
            device_name_pattern = matched_states[0].name if matched_states[0].name else ""
            if device_name_pattern and actual_area_name:
                # 常見的動詞前綴（不應該被當作區域）
                action_verbs = ["正在開啟", "正在關閉", "已開啟", "已關閉", "開啟", "關閉", 
                               "幫你開", "幫你關", "幫你開啟", "幫你關閉", "打開", "關掉"]
                
                # 匹配「X + 設備名」模式，其中 X 是 2-4 個中文字
                pattern = re.compile(rf'([\u4e00-\u9fff]{{2,4}}){re.escape(device_name_pattern)}')
                match = pattern.search(corrected)
                if match:
                    found_area = match.group(1)
                    # 檢查是否為動詞（不是區域）
                    is_action_verb = any(found_area.endswith(verb) or verb.endswith(found_area) 
                                        for verb in action_verbs)
                    if is_action_verb:
                        _LOGGER.debug("📍 Skipping action verb: '%s'", found_area)
                    elif found_area != actual_area_name and found_area not in all_area_names:
                        # 這是一個「幻想」的區域名稱
                        corrected = corrected.replace(found_area + device_name_pattern, 
                                                     actual_area_name + device_name_pattern)
                        _LOGGER.info("📍 Corrected hallucinated area: '%s' → '%s'", found_area, actual_area_name)
            
            if corrected != response_text:
                _LOGGER.info("✅ Area corrected: %s", corrected[:80])
            
            return corrected
            
        except Exception as err:
            _LOGGER.warning("Failed to correct area in response: %s", err)
            return response_text

    async def _get_single_device_state(self, name: str, area: str) -> tuple[str, str]:
        """v7: 查詢單一設備狀態
        
        根據設備名稱和區域查詢設備的當前狀態。
        
        Args:
            name: 設備名稱（中文）
            area: 區域名稱（中文）
        
        Returns:
            tuple: (設備狀態字串, 實際區域名稱)
            例如：(「on」, 「書房」) 或 (「cool, 26°C」, 「客廳」)
        """
        from homeassistant.helpers import label_registry as lr
        from homeassistant.helpers import entity_registry as er
        from homeassistant.helpers import area_registry as ar
        
        try:
            # 獲取 registries
            label_reg = lr.async_get(self.hass)
            entity_registry = er.async_get(self.hass)
            area_reg = ar.async_get(self.hass)
            
            # 找到 assist 標籤
            assist_label_id = None
            for label_id, label_entry in label_reg.labels.items():
                if label_entry.name.lower() == "assist":
                    assist_label_id = label_id
                    break
            
            if assist_label_id is None:
                _LOGGER.warning("Label 'assist' not found")
                return "unavailable"
            
            # 搜尋匹配的設備
            best_match = None
            best_score = 0
            
            for entity_id, entity_entry in entity_registry.entities.items():
                if assist_label_id not in entity_entry.labels:
                    continue
                
                state = self.hass.states.get(entity_id)
                if state is None:
                    continue
                
                friendly_name = state.attributes.get("friendly_name", "")
                
                # 獲取設備的 area（優先從 entity，再從 device 繼承）
                device_area = None
                if entity_entry.area_id:
                    area_entry = area_reg.async_get_area(entity_entry.area_id)
                    if area_entry:
                        device_area = area_entry.name
                
                # 如果 entity 沒有區域，嘗試從關聯的 device 獲取
                if device_area is None and entity_entry.device_id:
                    device_reg = dr.async_get(self.hass)
                    device_entry = device_reg.async_get(entity_entry.device_id)
                    if device_entry and device_entry.area_id:
                        area_entry = area_reg.async_get_area(device_entry.area_id)
                        if area_entry:
                            device_area = area_entry.name
                
                # 計算匹配分數
                score = 0
                
                # 名稱匹配
                if name in friendly_name or friendly_name in name:
                    score += 10
                elif any(char in friendly_name for char in name if char not in "的"):
                    score += 3
                
                # 區域匹配
                if device_area:
                    if area == device_area:
                        score += 10
                    elif area in device_area or device_area in area:
                        score += 5
                
                # 根據用戶問題推斷期望的 domain 類型
                # 如果用戶問「溫度」，優先匹配 sensor
                sensor_keywords = ["溫度", "濕度", "感應器", "sensor", "功率", "電量"]
                light_keywords = ["燈", "light", "亮度"]
                climate_keywords = ["冷氣", "空調", "climate", "暖氣"]
                
                expected_domain = None
                for kw in sensor_keywords:
                    if kw in name:
                        expected_domain = "sensor"
                        break
                for kw in light_keywords:
                    if kw in name:
                        expected_domain = "light"
                        break
                for kw in climate_keywords:
                    if kw in name:
                        expected_domain = "climate"
                        break
                
                # 如果 domain 匹配預期，加分
                if expected_domain and state.domain == expected_domain:
                    score += 15
                
                _LOGGER.debug("Device %s: score=%d (domain=%s, expected=%s)", 
                             entity_id, score, state.domain, expected_domain)
                
                if score > best_score:
                    best_score = score
                    best_match = state
                    best_match_area = device_area  # 保存匹配設備的實際區域
            
            if best_match is None:
                _LOGGER.warning("No matching device found for %s@%s", name, area)
                return ("unavailable", area)  # 返回 tuple
            
            _LOGGER.info("Found matching device: %s (score=%d)", best_match.entity_id, best_score)
            
            # 根據設備類型生成狀態字串
            domain = best_match.domain
            state_value = best_match.state
            actual_area = best_match_area or area  # 使用實際區域，fallback 到模型猜測的區域
            
            # 生成狀態字串
            state_str = state_value  # 默認
            
            if domain == "light":
                if state_value == "on":
                    brightness = best_match.attributes.get("brightness")
                    if brightness:
                        pct = int(brightness / 255 * 100)
                        state_str = f"on, {pct}%亮度"
                    else:
                        state_str = "on"
                else:
                    state_str = "off"
            
            elif domain == "climate":
                if state_value == "off":
                    state_str = "off"
                else:
                    mode = best_match.attributes.get("hvac_action", state_value)
                    temp = best_match.attributes.get("temperature")
                    if temp:
                        state_str = f"{mode}, {temp}°C"
                    else:
                        state_str = mode
            
            elif domain == "lock":
                state_str = "locked" if state_value == "locked" else "unlocked"
            
            elif domain == "cover":
                position = best_match.attributes.get("current_position")
                if position is not None:
                    state_str = f"{state_value}, {position}%"
            
            elif domain == "sensor":
                # 處理感應器（溫度、濕度、功率等）
                unit = best_match.attributes.get("unit_of_measurement", "")
                device_class = best_match.attributes.get("device_class", "")
                
                try:
                    value = float(state_value)
                    if device_class == "temperature":
                        state_str = f"{value}{unit}" if unit else f"{value}°C"
                    elif device_class == "humidity":
                        state_str = f"{value}{unit}" if unit else f"{value}%"
                    elif device_class == "power":
                        state_str = f"{value}{unit}" if unit else f"{value}W"
                    else:
                        state_str = f"{value}{unit}" if unit else str(value)
                except (ValueError, TypeError):
                    state_str = state_value
            
            return (state_str, actual_area)
                
        except Exception as err:
            _LOGGER.error("Error getting single device state: %s", err, exc_info=True)
            return ("unavailable", area)

    async def _call_model_with_state(
        self, 
        original_question: str, 
        state_result: str, 
        device_name: str, 
        area: str
    ) -> str:
        """v7: 調用 /process_with_state 進行二次對話
        
        當模型輸出 ACTION get_state 後，系統查詢設備狀態並返回結果。
        此方法調用模型 API 根據狀態結果生成回答。
        
        Args:
            original_question: 用戶的原始問題
            state_result: 設備狀態結果
            device_name: 設備名稱
            area: 區域名稱
        
        Returns:
            模型生成的回答文字
        """
        url = f"{self.model_url}/process_with_state"
        payload = {
            "original_question": original_question,
            "state_result": state_result,
            "device_name": device_name,
            "area": area,
        }
        
        _LOGGER.info("🔍 Calling /process_with_state with payload: %s", payload)
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get("response_text", "")
                        _LOGGER.info("✅ /process_with_state response: %s", response_text[:100])
                        return response_text
                    else:
                        error_text = await response.text()
                        _LOGGER.error("❌ /process_with_state error: %s - %s", response.status, error_text)
                        return "抱歉喵，我無法理解這個設備的狀態喵..."
                        
        except Exception as err:
            _LOGGER.error("Error calling /process_with_state: %s", err, exc_info=True)
            return "抱歉喵，查詢設備狀態時發生錯誤喵..."

    async def _call_model_api(self, text: str, devices: list[dict], history: list[dict] = None) -> dict:
        """呼叫 Qwen 模型 API
        
        Args:
            text: 用戶輸入
            devices: 設備列表
            history: 對話歷史（可選）格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        url = f"{self.model_url}/process"
        payload = {
            "text": text,
            "devices": devices,
            "language": "zh",
        }
        
        # ⭐ 如果有歷史，加入 payload
        if history:
            payload["history"] = history
        
        _LOGGER.debug("API URL: %s", url)
        _LOGGER.debug("Sending %d devices", len(devices))
        if history:
            _LOGGER.debug("Sending %d history messages", len(history))
        
        try:
            async with async_timeout.timeout(30):
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            _LOGGER.error("API Error: %s", error_text)
                            raise aiohttp.ClientError(f"HTTP {response.status}")
                        
                        result = await response.json()
                        _LOGGER.debug("API Response: %s", result)
                        return result
                        
        except Exception as err:
            _LOGGER.error("Error calling model API: %s", err, exc_info=True)
            raise

    def _save_to_history(self, conversation_id: str, user_message: str, assistant_message: str) -> None:
        """保存對話到歷史記錄
        
        Args:
            conversation_id: 對話 ID
            user_message: 用戶訊息
            assistant_message: 助手回覆
        """
        global _conversation_history
        
        if conversation_id not in _conversation_history:
            _conversation_history[conversation_id] = []
        
        # 添加用戶和助手訊息
        _conversation_history[conversation_id].append({
            "role": "user",
            "content": user_message,
        })
        _conversation_history[conversation_id].append({
            "role": "assistant",
            "content": assistant_message,
        })
        
        # 限制歷史長度（保留最近的 MAX_HISTORY_TURNS 輪對話 = 2*MAX_HISTORY_TURNS 條訊息）
        max_messages = MAX_HISTORY_TURNS * 2
        if len(_conversation_history[conversation_id]) > max_messages:
            _conversation_history[conversation_id] = _conversation_history[conversation_id][-max_messages:]
            _LOGGER.debug("Trimmed conversation history to %d messages", max_messages)
        
        _LOGGER.debug("Saved to history. Total messages in conversation %s: %d", 
                     conversation_id, len(_conversation_history[conversation_id]))

    def _find_best_match(self, target: str, candidates: list[str], threshold: float = 0.6) -> str | None:
        """使用模糊匹配找出最相似的名稱
        
        Args:
            target: 要匹配的目標字串
            candidates: 候選字串列表
            threshold: 相似度門檻（0-1），預設 0.6
            
        Returns:
            最相似的候選字串，如果沒有超過門檻則返回 None
        """
        if not target or not candidates:
            return None
        
        best_match = None
        best_ratio = 0.0
        top_matches = []  # 記錄前幾名的匹配結果
        
        target_lower = target.lower()
        
        for candidate in candidates:
            if not candidate:
                continue
                
            candidate_lower = candidate.lower()
            
            # 計算相似度
            ratio = SequenceMatcher(None, target_lower, candidate_lower).ratio()
            
            # 記錄前 5 名
            if len(top_matches) < 5 or ratio > top_matches[-1][1]:
                top_matches.append((candidate, ratio))
                top_matches.sort(key=lambda x: x[1], reverse=True)
                top_matches = top_matches[:5]
            
            # 如果是完全匹配，直接返回
            if ratio == 1.0:
                _LOGGER.debug("Perfect match found: '%s'", candidate)
                return candidate
            
            # 記錄最佳匹配
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = candidate
        
        # 顯示前 5 名匹配結果
        if top_matches:
            _LOGGER.debug("Top 5 matches for '%s':", target)
            for match, ratio in top_matches:
                _LOGGER.debug("  - '%s' (similarity: %.2f)", match, ratio)
        
        # 只有超過門檻才返回
        if best_ratio >= threshold:
            _LOGGER.info("Found match: '%s' -> '%s' (similarity: %.2f)", 
                        target, best_match, best_ratio)
            return best_match
        
        _LOGGER.warning("No match found for '%s' (best: '%s' with %.2f, threshold: %.2f)", 
                       target, best_match or "N/A", best_ratio, threshold)
        return None

    async def _correct_device_names(self, action_result: dict, devices: list[dict]) -> dict:
        """修正 action_result 中的裝置名稱和區域名稱
        
        將模型輸出的名稱與實際存在的裝置/區域名稱進行模糊匹配，
        如果找到相似的名稱就進行修正。
        """
        params = action_result.get('params', {})
        if not params:
            return action_result
        
        _LOGGER.info("🔍 Starting name correction...")
        _LOGGER.info("Original params: %s", params)
        
        corrected = False
        
        # 收集所有裝置名稱
        device_names = [d.get('friendlyName') for d in devices if d.get('friendlyName')]
        _LOGGER.info("Available device names (%d): %s", len(device_names), device_names[:10])  # 顯示前 10 個
        
        # 收集所有區域名稱
        area_names = list(set([d.get('area') for d in devices if d.get('area')]))
        _LOGGER.info("Available area names: %s", area_names)
        
        # 修正裝置名稱
        if 'name' in params:
            original_name = params['name']
            _LOGGER.info("Trying to match device name: '%s'", original_name)
            corrected_name = self._find_best_match(original_name, device_names)
            if corrected_name and corrected_name != original_name:
                _LOGGER.info("✏️ Correcting device name: '%s' -> '%s'", 
                            original_name, corrected_name)
                params['name'] = corrected_name
                corrected = True
            elif corrected_name:
                _LOGGER.info("✓ Device name already correct: '%s'", original_name)
            else:
                _LOGGER.warning("⚠️ No match found for device name: '%s'", original_name)
        
        # 修正區域名稱
        if 'area' in params:
            original_area = params['area']
            _LOGGER.info("Trying to match area name: '%s'", original_area)
            corrected_area = self._find_best_match(original_area, area_names)
            if corrected_area and corrected_area != original_area:
                _LOGGER.info("✏️ Correcting area name: '%s' -> '%s'", 
                            original_area, corrected_area)
                params['area'] = corrected_area
                corrected = True
            elif corrected_area:
                _LOGGER.info("✓ Area name already correct: '%s'", original_area)
            else:
                _LOGGER.warning("⚠️ No match found for area name: '%s'", original_area)
        
        if corrected:
            _LOGGER.info("✅ Names corrected: %s", params)
        else:
            _LOGGER.info("ℹ️ No corrections needed")
        
        return action_result

    def _is_error_response(self, response: str) -> bool:
        """檢測回應是否為錯誤訊息
        
        Home Assistant 內建 assist 的錯誤訊息通常包含特定關鍵字。
        如果檢測到這些關鍵字，說明 assist 無法處理請求。
        
        這些模式來自 Home Assistant 官方源碼的錯誤訊息。
        """
        if not response:
            return True
        
        # 錯誤關鍵字列表（來自 HA 源碼 + 繁體中文）
        error_patterns = [
            # === 繁體中文錯誤訊息 ===
            "並不存在",
            "不存在",
            "找不到",
            "無法找到",
            "沒有找到",
            "我不知道",
            "抱歉",
            
            # === Home Assistant 官方英文錯誤訊息 ===
            # "Sorry, I am not aware of any device called"
            "not aware of any device",
            "not aware of any",
            
            # "Sorry, {name} is not exposed"
            "is not exposed",
            "not exposed",
            
            # "Sorry, no device supports the required features"
            "no device supports",
            "supports the required features",
            
            # "Sorry, I couldn't find that timer"
            "couldn't find that",
            
            # Generic error patterns
            "not found",
            "doesn't exist",
            "does not exist",
            "could not find",
            "couldn't find",
            "cannot find",
            "can't find",
            "i don't know",
            "sorry,",  # 大部分 HA 錯誤訊息都以 "Sorry," 開頭
            
            # Connection/service errors
            "connection lost",
            "error communicating",
            "failed to talk",
        ]
        
        response_lower = response.lower()
        for pattern in error_patterns:
            if pattern in response_lower:
                _LOGGER.info("🚨 Detected error pattern: '%s' in response", pattern)
                return True
        
        return False

    async def _fallback_to_builtin_assist(
        self, 
        user_input: conversation.ConversationInput,
        language: str
    ) -> conversation.ConversationResult:
        """使用內建 Assist 解析器作為備援（三層保險）
        
        第一層: script.assist_ha_fallback (Home Assistant 內建)
        第二層: script.assist_gemini_fallback (Gemini AI)
        第三層: 友善錯誤訊息
        
        每層都會通過 /fallback_assistant 進行喵化處理。
        """
        _LOGGER.info("=== Fallback to Built-in Assist ===")
        _LOGGER.info("Original input: %s", user_input.text)
        
        # 記錄到調試資訊
        self._debug_info["processing_steps"].append({
            "step": "fallback_started",
            "reason": "Primary processing failed or bypassed",
        })
        
        # ✅ Trace: 開始 Fallback
        trace.async_conversation_trace_append(
            trace.ConversationTraceEventType.AGENT_DETAIL,
            {
                "step": "fallback_started",
                "total_layers": 3,
            }
        )
        
        # 第一層：嘗試 Home Assistant 內建 Assist
        try:
            service_data = {"query": user_input.text}
            _LOGGER.info("🔄 Layer 1: Calling script.assist_ha_fallback")
            
            # ✅ Trace: Layer 1 開始
            trace.async_conversation_trace_append(
                trace.ConversationTraceEventType.AGENT_DETAIL,
                {
                    "step": "fallback_layer_1",
                    "service": "script.assist_ha_fallback",
                }
            )
            
            service_response = await self.hass.services.async_call(
                "script",
                "assist_ha_fallback",
                service_data,
                blocking=True,
                return_response=True,
            )
            
            assistant_response = service_response.get('response', '')
            _LOGGER.info("Layer 1 response: %s", assistant_response[:100] if assistant_response else "N/A")
            
            # ⭐ 檢查是否為錯誤訊息
            if assistant_response and not self._is_error_response(assistant_response):
                # ✅ Trace: Layer 1 成功
                trace.async_conversation_trace_append(
                    trace.ConversationTraceEventType.AGENT_DETAIL,
                    {
                        "step": "fallback_layer_1_success",
                        "response_preview": assistant_response[:100],
                    }
                )
                
                # 成功的回應，進行喵化處理
                try:
                    catgirl_response = await self._call_fallback_assistant_api(
                        user_input.text,
                        assistant_response
                    )
                    _LOGGER.info("✅ Layer 1 succeeded with catgirl response")
                    
                    # 記錄成功
                    self._debug_info["processing_steps"].append({
                        "step": "fallback_layer_1_success",
                        "service": "script.assist_ha_fallback",
                    })
                    
                    intent_response = intent.IntentResponse(language=language)
                    intent_response.async_set_speech(catgirl_response)
                    
                    # ⭐ 保存 fallback 回應到歷史
                    self._save_to_history(user_input.conversation_id or ulid.ulid(), user_input.text, catgirl_response)
                    
                    return conversation.ConversationResult(
                        response=intent_response,
                        conversation_id=user_input.conversation_id or ulid.ulid()
                    )
                except Exception as catgirl_err:
                    _LOGGER.warning("Catgirl-ification failed, using raw response: %s", catgirl_err)
                    
                    # 記錄喵化失敗
                    self._debug_info["processing_steps"].append({
                        "step": "catgirl_api_failed",
                        "error": str(catgirl_err),
                        "fallback_to": "raw_ha_response",
                    })
                    
                    intent_response = intent.IntentResponse(language=language)
                    intent_response.async_set_speech(assistant_response)
                    
                    # ⭐ 保存 fallback 回應到歷史
                    self._save_to_history(user_input.conversation_id or ulid.ulid(), user_input.text, assistant_response)
                    
                    return conversation.ConversationResult(
                        response=intent_response,
                        conversation_id=user_input.conversation_id or ulid.ulid()
                    )
            else:
                # 錯誤回應，拋出異常觸發 Layer 2
                _LOGGER.warning("❌ Layer 1 returned error response, triggering Layer 2")
                
                # ✅ Trace: Layer 1 錯誤檢測
                trace.async_conversation_trace_append(
                    trace.ConversationTraceEventType.AGENT_DETAIL,
                    {
                        "step": "fallback_layer_1_error_detected",
                        "error_response": assistant_response[:100],
                    }
                )
                
                raise Exception(f"HA assist failed: {assistant_response}")
            
        except Exception as err:
            _LOGGER.warning("❌ Layer 1 failed: %s", err)
            _LOGGER.info("Trying Layer 2...")
        
        # 第二層：嘗試 Gemini Fallback
        try:
            service_data = {"query": user_input.text}
            _LOGGER.info("🔄 Layer 2: Calling script.assist_gemini_fallback")
            
            # ✅ Trace: Layer 2 開始
            trace.async_conversation_trace_append(
                trace.ConversationTraceEventType.AGENT_DETAIL,
                {
                    "step": "fallback_layer_2",
                    "service": "script.assist_gemini_fallback",
                }
            )
            
            service_response = await self.hass.services.async_call(
                "script",
                "assist_gemini_fallback",
                service_data,
                blocking=True,
                return_response=True,
            )
            
            gemini_response = service_response.get('response', '')
            _LOGGER.info("Layer 2 response: %s", gemini_response[:100] if gemini_response else "N/A")
            
            # Gemini 已經在提示詞中處理好貓娘風格，直接使用
            if gemini_response:
                _LOGGER.info("✅ Layer 2 succeeded (Gemini already catgirl-ified)")
                
                # 記錄成功
                self._debug_info["processing_steps"].append({
                    "step": "fallback_layer_2_success",
                    "service": "script.assist_gemini_fallback",
                })
                
                # ✅ Trace: Layer 2 成功
                trace.async_conversation_trace_append(
                    trace.ConversationTraceEventType.AGENT_DETAIL,
                    {
                        "step": "fallback_layer_2_success",
                        "response_preview": gemini_response[:100],
                    }
                )
                
                intent_response = intent.IntentResponse(language=language)
                intent_response.async_set_speech(gemini_response)
                
                # ⭐ 保存 fallback 回應到歷史
                self._save_to_history(user_input.conversation_id or ulid.ulid(), user_input.text, gemini_response)
                
                return conversation.ConversationResult(
                    response=intent_response,
                    conversation_id=user_input.conversation_id or ulid.ulid()
                )
            
        except Exception as err:
            _LOGGER.error("❌ Layer 2 also failed: %s", err)
            _LOGGER.warning("All fallback layers failed, using final backup response")
            
            # ✅ Trace: Layer 2 失敗
            trace.async_conversation_trace_append(
                trace.ConversationTraceEventType.AGENT_DETAIL,
                {
                    "step": "fallback_layer_2_failed",
                    "error": str(err),
                }
            )
        
        # 第三層：終極備援 - 友善錯誤訊息
        _LOGGER.error("💔 All fallback attempts exhausted")
        
        # 記錄失敗
        self._debug_info["processing_steps"].append({
            "step": "fallback_all_failed",
            "message": "All fallback layers exhausted",
        })
        
        # ✅ Trace: Layer 3 (最後備援)
        trace.async_conversation_trace_append(
            trace.ConversationTraceEventType.AGENT_DETAIL,
            {
                "step": "fallback_layer_3_final",
                "message": "All fallback layers exhausted",
            }
        )
        
        final_message = "抱歉，我現在無法處理這個請求喵... 請稍後再試一次。"
        intent_response = intent.IntentResponse(language=language)
        intent_response.async_set_speech(final_message)
        
        # ⭐ 保存 fallback 回應到歷史
        self._save_to_history(user_input.conversation_id or ulid.ulid(), user_input.text, final_message)
        
        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=user_input.conversation_id or ulid.ulid()
        )

    async def _call_fallback_assistant_api(self, user_question: str, assistant_response: str) -> str:
        """呼叫模型的 fallback_assistant endpoint 將內建 Assist 的回應喵化
        
        Args:
            user_question: 使用者的原始問題
            assistant_response: 內建 Assist 的回應
            
        Returns:
            喵化後的回應文字
        """
        url = f"{self.model_url}/fallback_assistant"
        payload = {
            "user_question": user_question,
            "assistant_response": assistant_response,
        }
        
        _LOGGER.debug("Fallback assistant API URL: %s", url)
        _LOGGER.debug("Payload: %s", payload)
        
        try:
            async with async_timeout.timeout(30):
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            _LOGGER.error("Fallback assistant API Error: %s", error_text)
                            raise aiohttp.ClientError(f"HTTP {response.status}")
                        
                        result = await response.json()
                        _LOGGER.debug("Fallback assistant API Response: %s", result)
                        
                        # 返回喵化後的回應
                        return result.get('response_text', assistant_response)
                        
        except Exception as err:
            _LOGGER.error("Error calling fallback_assistant API: %s", err, exc_info=True)
            raise


    async def _execute_search(
        self,
        action_result: dict,
        language: str,
        user_input: conversation.ConversationInput,
    ) -> intent.IntentResponse:
        """執行搜尋動作
        
        流程:
        1. 模型第一次輸出包含初始回應 (如: "讓我幫你查一下...")
        2. 呼叫 script.assist_search_google 獲取搜尋結果
        3. 將搜尋結果送回模型的 /search_result endpoint
        4. 模型生成最終貓娘風格回應
        5. 返回最終回應給用戶
        
        注意: 根據 HA 標準,conversation agent 只返回最終回應
        初始回應 ("讓我幫你查一下...") 僅用於日誌,不會顯示給用戶
        """
        params = action_result.get('params', {})
        query = params.get('query', '')
        initial_response = action_result.get('response_text', '')
        
        _LOGGER.info("=== Executing Search Action ===")
        _LOGGER.info("Initial response (for logging): %s", initial_response)
        _LOGGER.info("Search query: %s", query)
        
        if not query:
            _LOGGER.warning("No query provided in search action")
            response = intent.IntentResponse(language=language)
            response.async_set_speech("抱歉,我不知道要搜尋什麼喵...")
            return response
        
        try:
            # 呼叫 Home Assistant script 進行搜尋
            service_data = {"query": query}
            _LOGGER.info("Calling script.assist_search_google with query: %s", query)
            
            service_response = await self.hass.services.async_call(
                "script",
                "assist_search_google",
                service_data,
                blocking=True,
                return_response=True,
            )
            
            search_result = service_response.get('response', '')
            _LOGGER.info("Search result received: %s", search_result[:200])  # Log first 200 chars
            
            if not search_result:
                _LOGGER.warning("Empty search result from script")
                response = intent.IntentResponse(language=language)
                response.async_set_speech("抱歉,搜尋沒有返回結果喵...")
                return response
            
            # 將搜尋結果送到模型生成最終回應
            try:
                final_response = await self._call_search_result_api(query, search_result)
                _LOGGER.info("✅ Final response generated: %s", final_response[:200])
                
                response = intent.IntentResponse(language=language)
                response.async_set_speech(final_response)
                return response
                
            except Exception as err:
                _LOGGER.error("Error calling search_result API: %s", err, exc_info=True)
                # 如果模型 API 失敗，直接返回搜尋結果
                response = intent.IntentResponse(language=language)
                response.async_set_speech(search_result)
                return response
            
        except Exception as err:
            _LOGGER.error("Error executing search: %s", err, exc_info=True)
            # ⭐ 重新拋出異常，讓上層可以捕捉並觸發 fallback
            raise

    async def _call_search_result_api(self, user_question: str, search_result: str) -> str:
        """呼叫模型的 search_result endpoint 生成最終回應"""
        url = f"{self.model_url}/search_result"
        payload = {
            "user_question": user_question,
            "search_result": search_result,
        }
        
        _LOGGER.debug("Search result API URL: %s", url)
        _LOGGER.debug("Payload: %s", payload)
        
        try:
            async with async_timeout.timeout(30):
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            _LOGGER.error("Search result API Error: %s", error_text)
                            raise aiohttp.ClientError(f"HTTP {response.status}")
                        
                        result = await response.json()
                        _LOGGER.debug("Search result API Response: %s", result)
                        
                        # ⭐ 修正：模型返回的是 {"response_text": "..."}，不是 {"response": "..."}
                        catgirl_response = result.get('response_text', result.get('response', search_result))
                        _LOGGER.info("🎀 Catgirl-ified search response: %s", catgirl_response[:200])
                        return catgirl_response
                        
        except Exception as err:
            _LOGGER.error("Error calling search_result API: %s", err, exc_info=True)
            raise

    async def _execute_climate_set_mode(
        self,
        params: dict,
        response_text: str,
        language: str,
    ) -> intent.IntentResponse:
        """執行 climate_set_mode 動作
        
        呼叫 script.set_climate_mode 並傳入參數:
        - area: 區域名稱
        - mode: 模式 (auto, cool, heat, dry, fan_only 等)
        """
        area = params.get('area', '')
        mode = params.get('mode', '')
        
        _LOGGER.info("=== Executing climate_set_mode ===")
        _LOGGER.info("Area: %s", area)
        _LOGGER.info("Mode: %s", mode)
        
        if not area or not mode:
            _LOGGER.warning("Missing required parameters: area=%s, mode=%s", area, mode)
            response = intent.IntentResponse(language=language)
            response.async_set_speech("抱歉，缺少必要的參數喵...")
            return response
        
        try:
            # 呼叫 Home Assistant script
            service_data = {
                "area": area,
                "mode": mode,
            }
            _LOGGER.info("Calling script.set_climate_mode with data: %s", service_data)
            
            await self.hass.services.async_call(
                "script",
                "set_climate_mode",
                service_data,
                blocking=True,
            )
            
            _LOGGER.info("✅ climate_set_mode executed successfully")
            
            # 返回模型生成的回應
            response = intent.IntentResponse(language=language)
            response.async_set_speech(response_text)
            return response
            
        except Exception as err:
            _LOGGER.error("Error executing climate_set_mode: %s", err, exc_info=True)
            # ⭐ 重新拋出異常，讓上層可以捕捉並觸發 fallback
            raise

    async def _execute_via_intent(
        self,
        action_result: dict,
        language: str,
        user_input: conversation.ConversationInput,
    ) -> intent.IntentResponse:
        """⭐ 將 ACTION 轉換為 Home Assistant Intent 並執行（正確版本）"""
        
        action = action_result.get('action')
        params = action_result.get('params', {})
        response_text = action_result.get('response_text', '')
        
        # ⭐ 特殊處理：climate_set_mode 不使用 Intent，直接呼叫 script
        if action == "climate_set_mode":
            return await self._execute_climate_set_mode(params, response_text, language)
        
        # ACTION 到 Intent 的映射
        ACTION_TO_INTENT = {
            "turn_on": "HassTurnOn",
            "turn_off": "HassTurnOff",
            "light_set": "HassLightSet",
            "set_light": "HassLightSet",  # 模型可能輸出這個
            "get_state": "HassGetState",
            "climate_set_temp": "HassClimateSetTemperature",
            "cover_control": "HassSetPosition",
        }
        
        intent_type = ACTION_TO_INTENT.get(action)
        if not intent_type:
            _LOGGER.warning("Unsupported action: %s", action)
            response = intent.IntentResponse(language=language)
            response.async_set_speech(response_text)
            return response
        
        # ⭐ 關鍵：構建正確的 slots 格式
        # 格式必須是：{"slot_name": {"value": "slot_value"}}
        slots = {}
        
        if "name" in params:
            slots["name"] = {"value": params["name"]}
        
        if "area" in params:
            slots["area"] = {"value": params["area"]}
        
        if "domain" in params:
            slots["domain"] = {"value": params["domain"]}
        
        if "brightness" in params:
            slots["brightness"] = {"value": params["brightness"]}
        
        if "color" in params:
            slots["color"] = {"value": params["color"]}
        
        if "temperature" in params:
            slots["temperature"] = {"value": params["temperature"]}
        
        _LOGGER.info("=== Calling Home Assistant Intent ===")
        _LOGGER.info("Intent type: %s", intent_type)
        _LOGGER.info("Slots: %s", slots)
        
        try:
            # ⭐ 正確的 intent.async_handle 呼叫
            intent_response = await intent.async_handle(
                self.hass,
                "qwen_catgirl",           # platform
                intent_type,               # intent 名稱（例如 "HassTurnOff"）
                slots,                     # slots 字典
                user_input.text,           # 原始文字
                user_input.context,        # context
                language,                  # 語言
            )
            
            # ⭐ 修正回應文字：將「未分類」或錯誤區域替換為實際區域
            # 如果是 BERT 或 Template 處理的，跳過修正（它們的回應已經是正確的）
            processed_by = action_result.get('processed_by', 'qwen')
            if processed_by in ('bert', 'template'):
                _LOGGER.info("⚡ Skipping area correction for %s response", processed_by)
                corrected_response = response_text
            else:
                corrected_response = await self._correct_area_in_response(
                    response_text, intent_response.matched_states
                )
            intent_response.async_set_speech(corrected_response)
            
            _LOGGER.info("✅ Intent executed successfully")
            _LOGGER.info("Matched entities: %s", 
                        [state.entity_id for state in intent_response.matched_states or []])
            
            return intent_response
            
        except intent.IntentHandleError as err:
            _LOGGER.error("Intent handling error: %s", err)
            # ⭐ 重新拋出異常，讓上層可以捕捉並觸發 fallback
            raise
        except intent.InvalidSlotInfo as err:
            _LOGGER.error("Invalid slot info: %s", err)
            # ⭐ 重新拋出異常，讓上層可以捕捉並觸發 fallback
            raise
        except Exception as err:
            # ⭐ 檢查是否為區域相關錯誤（區域不存在或該區域沒有此設備）
            error_str = str(err)
            # INVALID_AREA: 區域不存在, MatchFailedReason.AREA: 該區域沒有此設備
            is_area_error = ("INVALID_AREA" in error_str or "MatchFailedReason.AREA" in error_str)
            if is_area_error and "area" in slots:
                _LOGGER.warning("Area mismatch detected, retrying without area constraint...")
                
                # 移除 area slot 重試
                retry_slots = {k: v for k, v in slots.items() if k != "area"}
                _LOGGER.info("Retrying with slots: %s", retry_slots)
                
                try:
                    intent_response = await intent.async_handle(
                        self.hass,
                        "qwen_catgirl",
                        intent_type,
                        retry_slots,
                        user_input.text,
                        user_input.context,
                        language,
                    )
                    
                    # ⭐ 修正回應文字：用實際設備區域替換錯誤區域
                    # 如果是 BERT 或 Template 處理的，跳過修正
                    corrected_response = response_text
                    wrong_area = slots.get("area", {}).get("value", "")
                    
                    processed_by = action_result.get('processed_by', 'qwen')
                    if processed_by in ('bert', 'template'):
                        _LOGGER.info("⚡ Skipping area correction for %s response (retry path)", processed_by)
                    elif intent_response.matched_states and wrong_area:
                        # 獲取實際設備的區域
                        from homeassistant.helpers import entity_registry as er
                        from homeassistant.helpers import area_registry as ar
                        from homeassistant.helpers import device_registry as dr
                        
                        entity_registry = er.async_get(self.hass)
                        area_reg = ar.async_get(self.hass)
                        device_reg = dr.async_get(self.hass)
                        
                        matched_entity_id = intent_response.matched_states[0].entity_id
                        entity_entry = entity_registry.async_get(matched_entity_id)
                        
                        actual_area_name = None
                        if entity_entry:
                            # 優先使用 entity 的區域
                            area_id = entity_entry.area_id
                            # 如果 entity 沒有區域，嘗試從 device 獲取
                            if not area_id and entity_entry.device_id:
                                device_entry = device_reg.async_get(entity_entry.device_id)
                                if device_entry:
                                    area_id = device_entry.area_id
                            
                            if area_id:
                                area_entry = area_reg.async_get_area(area_id)
                                if area_entry:
                                    actual_area_name = area_entry.name
                        
                        if actual_area_name and actual_area_name != wrong_area:
                            corrected_response = response_text.replace(wrong_area, actual_area_name)
                            _LOGGER.info("📍 Corrected area in response: '%s' → '%s'", wrong_area, actual_area_name)
                    
                    intent_response.async_set_speech(corrected_response)
                    _LOGGER.info("✅ Intent executed successfully (without area)")
                    _LOGGER.info("Matched entities: %s", 
                                [state.entity_id for state in intent_response.matched_states or []])
                    return intent_response
                    
                except Exception as retry_err:
                    _LOGGER.error("Retry without area also failed: %s", retry_err)
                    raise retry_err
            
            _LOGGER.error("Unexpected error executing intent: %s", err, exc_info=True)
            # ⭐ 重新拋出異常，讓上層可以捕捉並觸發 fallback
            raise
