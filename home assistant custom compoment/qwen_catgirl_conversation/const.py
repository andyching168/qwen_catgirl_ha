"""Constants for Qwen Catgirl Conversation integration."""

DOMAIN = "qwen_catgirl_conversation"
CONF_MODEL_URL = "model_url"
CONF_FALLBACK_KEYWORDS = "fallback_keywords"
CONF_REMOVABLE_KEYWORDS = "removable_keywords"
DEFAULT_MODEL_URL = "http://0.0.0.0:8123"

# 預設的 fallback 關鍵字（會直接交給 Gemini 處理）
DEFAULT_FALLBACK_KEYWORDS = [
    "播放", "暫停", "停止", "繼續", "下一首", "上一首", "下一個", "上一個",
    "play", "pause", "stop", "resume", "next", "previous",
    "音量", "聲音", "大聲", "小聲", "volume", "mute",
    "nest hub", "chromecast", "電視", "音箱",
]

# 預設的可移除關鍵字（識別後會從請求中移除再送給 Gemini）
DEFAULT_REMOVABLE_KEYWORDS = [
    "使用 Gemini",
    "用 Gemini",
    "幫我",
    "use Gemini",
    "with Gemini",
]

# ACTION 到 Home Assistant Intent 的映射
ACTION_TO_INTENT_MAP = {
    "turn_on": "HassTurnOn",
    "turn_off": "HassTurnOff",
    "light_set": "HassLightSet",
    "cover_control": "HassSetPosition",  # 或 HassTurnOn/Off
    "climate_set_temp": "HassClimateSetTemperature",
    "get_weather": "HassGetWeather",
    "get_time": "HassGetCurrentTime",
    "get_date": "HassGetCurrentDate",
    "media_pause": "HassMediaPause",
    "media_unpause": "HassMediaUnpause",
    "media_next": "HassMediaNext",
    "media_previous": "HassMediaPrevious",
    "set_volume": "HassSetVolume",
    "volume_up": "HassSetVolumeRelative",
    "volume_down": "HassSetVolumeRelative",
    "vacuum_start": "HassVacuumStart",
    "vacuum_return": "HassVacuumReturnToBase",
    "broadcast": "HassBroadcast",
    "search": "ScriptSearch",
}
