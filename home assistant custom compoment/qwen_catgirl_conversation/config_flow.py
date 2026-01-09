"""Config flow for Qwen Catgirl Conversation integration."""
from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.data_entry_flow import FlowResult
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    DOMAIN,
    CONF_MODEL_URL,
    CONF_FALLBACK_KEYWORDS,
    CONF_REMOVABLE_KEYWORDS,
    DEFAULT_MODEL_URL,
    DEFAULT_FALLBACK_KEYWORDS,
    DEFAULT_REMOVABLE_KEYWORDS,
)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_NAME, default="Qwen Catgirl"): str,
        vol.Required(CONF_MODEL_URL, default=DEFAULT_MODEL_URL): str,
        vol.Optional(
            CONF_FALLBACK_KEYWORDS,
            default=DEFAULT_FALLBACK_KEYWORDS,
        ): selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=DEFAULT_FALLBACK_KEYWORDS,
                multiple=True,
                custom_value=True,
                mode=selector.SelectSelectorMode.DROPDOWN,
            ),
        ),
        vol.Optional(
            CONF_REMOVABLE_KEYWORDS,
            default=DEFAULT_REMOVABLE_KEYWORDS,
        ): selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=DEFAULT_REMOVABLE_KEYWORDS,
                multiple=True,
                custom_value=True,
                mode=selector.SelectSelectorMode.DROPDOWN,
            ),
        ),
    }
)


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Qwen Catgirl Conversation."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user",
                data_schema=STEP_USER_DATA_SCHEMA,
            )

        # 關鍵字已經是列表格式（由 selector 處理）
        return self.async_create_entry(title=user_input[CONF_NAME], data=user_input)

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> OptionsFlow:
        """Get the options flow for this handler."""
        return OptionsFlow()  # ✅ 不傳入 config_entry


class OptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for Qwen Catgirl Conversation."""

    # ✅ 移除 __init__，parent class 已經提供 self.config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            # 關鍵字已經是列表格式（由 selector 處理）
            keywords_list = user_input.get(CONF_FALLBACK_KEYWORDS, DEFAULT_FALLBACK_KEYWORDS)
            removable_list = user_input.get(CONF_REMOVABLE_KEYWORDS, DEFAULT_REMOVABLE_KEYWORDS)

            # ⭐ 更新現有 agent 的設定（不重新載入整個 entry）
            from . import DOMAIN
            agent = self.hass.data.get(DOMAIN, {}).get(self.config_entry.entry_id)
            if agent:
                agent.fallback_keywords = keywords_list
                agent.removable_keywords = removable_list
                _LOGGER = __import__('logging').getLogger(__name__)
                _LOGGER.warning("=" * 60)
                _LOGGER.warning("✅ Fallback keywords updated: %s", keywords_list)
                _LOGGER.warning("✅ Removable keywords updated: %s", removable_list)
                _LOGGER.warning("=" * 60)
            
            return self.async_create_entry(title="", data=user_input)

        # 獲取當前設定 - self.config_entry 可以直接用
        current_keywords = self.config_entry.options.get(
            CONF_FALLBACK_KEYWORDS,
            self.config_entry.data.get(CONF_FALLBACK_KEYWORDS, DEFAULT_FALLBACK_KEYWORDS),
        )
        
        current_removable = self.config_entry.options.get(
            CONF_REMOVABLE_KEYWORDS,
            self.config_entry.data.get(CONF_REMOVABLE_KEYWORDS, DEFAULT_REMOVABLE_KEYWORDS),
        )

        options_schema = vol.Schema(
            {
                vol.Required(
                    CONF_MODEL_URL,
                    default=self.config_entry.options.get(
                        CONF_MODEL_URL,
                        self.config_entry.data.get(CONF_MODEL_URL, DEFAULT_MODEL_URL),
                    ),
                ): str,
                vol.Optional(
                    CONF_FALLBACK_KEYWORDS,
                    default=current_keywords,
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=current_keywords if isinstance(current_keywords, list) else DEFAULT_FALLBACK_KEYWORDS,
                        multiple=True,
                        custom_value=True,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    ),
                ),
                vol.Optional(
                    CONF_REMOVABLE_KEYWORDS,
                    default=current_removable,
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=current_removable if isinstance(current_removable, list) else DEFAULT_REMOVABLE_KEYWORDS,
                        multiple=True,
                        custom_value=True,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    ),
                ),
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=options_schema,
        )
