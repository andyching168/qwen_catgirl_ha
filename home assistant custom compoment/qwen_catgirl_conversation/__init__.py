"""The Qwen Catgirl Conversation integration."""
from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv

from .const import DOMAIN
from .conversation_agent import QwenCatgirlConversationAgent

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.empty_config_schema(DOMAIN)


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the Qwen Catgirl Conversation component."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Qwen Catgirl Conversation from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    
    # 建立 conversation agent 實例
    agent = QwenCatgirlConversationAgent(hass, entry)
    
    # 註冊到 conversation component
    from homeassistant.components import conversation
    conversation.async_set_agent(hass, entry, agent)
    
    # 儲存 agent 引用（供 config_flow 更新設定時使用）
    hass.data[DOMAIN][entry.entry_id] = agent
    
    _LOGGER.info("Qwen Catgirl Conversation agent registered: %s", entry.entry_id)
    
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    hass.data[DOMAIN].pop(entry.entry_id)
    
    # 取消註冊 agent - 同樣修正
    from homeassistant.components import conversation
    conversation.async_unset_agent(hass, entry)
    
    return True
