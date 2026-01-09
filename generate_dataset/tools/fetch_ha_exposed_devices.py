#!/usr/bin/env python3
"""
fetch_ha_devices_from_list.py

從手動複製的 WebUI 清單中解析實體 ID，並從 HA API 抓取詳細資訊
"""

import requests
import json
import sys
import os
from typing import Dict, List, Any, Set
from datetime import datetime
import re

# ==============================================================================
# 配置區
# ==============================================================================

HA_URL = os.getenv('HA_URL', 'http://homeassistant.local:8123')
HA_TOKEN = os.getenv('HA_TOKEN', 'your_long_lived_access_token_here')

# 從 WebUI 複製的暴露清單（放在同目錄的文字檔）
EXPOSED_LIST_FILE = "exposed_entities_list.txt"

OUTPUT_FILE = "ha_exposed_devices.json"

# ==============================================================================
# API Client
# ==============================================================================

class HomeAssistantAPI:
    def __init__(self, url: str, token: str):
        self.url = url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    def get_states(self) -> List[Dict]:
        response = requests.get(
            f"{self.url}/api/states",
            headers=self.headers,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    
    def get_config(self) -> Dict:
        response = requests.get(
            f"{self.url}/api/config",
            headers=self.headers,
            timeout=10
        )
        response.raise_for_status()
        return response.json()

# ==============================================================================
# 解析 WebUI 清單
# ==============================================================================

def parse_exposed_list(filepath: str) -> List[Dict[str, str]]:
    """
    解析從 WebUI 複製的暴露清單
    
    格式範例：
    大燈
    light.da_deng
    書房
    -
    
    Returns:
        List of {entity_id, friendly_name, area}
    """
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到檔案：{filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    
    entities = []
    i = 0
    
    while i < len(lines):
        # 跳過空行和分隔符
        if not lines[i] or lines[i] == '-':
            i += 1
            continue
        
        # 如果這行包含 . 就是 entity_id
        if '.' in lines[i]:
            entity_id = lines[i]
            
            # 往前找 friendly_name（上一個非空、非分隔符的行）
            friendly_name = None
            for j in range(i-1, -1, -1):
                if lines[j] and lines[j] != '-' and '.' not in lines[j]:
                    friendly_name = lines[j]
                    break
            
            # 往後找 area（下一個非空、非分隔符的行）
            area = None
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if next_line and next_line != '-' and '.' not in next_line:
                    area = next_line if next_line != '—' else None
            
            entities.append({
                'entity_id': entity_id,
                'friendly_name': friendly_name or entity_id,
                'area': area
            })
        
        i += 1
    
    return entities


# ==============================================================================
# 設備資訊提取
# ==============================================================================

def extract_device_info(entity: Dict, webui_info: Dict) -> Dict[str, Any]:
    """提取並合併設備資訊"""
    attributes = entity.get('attributes', {})
    entity_id = entity['entity_id']
    domain = entity_id.split('.')[0]
    
    device_info = {
        'entity_id': entity_id,
        'domain': domain,
        'friendly_name': webui_info.get('friendly_name') or attributes.get('friendly_name', entity_id),
        'webui_area': webui_info.get('area'),  # 從 WebUI 複製的區域
        'ha_area_id': attributes.get('area_id'),  # HA 實際的 area_id
        'state': entity['state'],
        'device_class': attributes.get('device_class'),
        'supported_features': attributes.get('supported_features'),
        'unit_of_measurement': attributes.get('unit_of_measurement'),
    }
    
    # 根據 domain 提取特定屬性
    if domain == 'light':
        brightness = attributes.get('brightness')
        device_info['light_info'] = {
            'brightness': brightness,
            'brightness_pct': round(brightness / 255 * 100) if brightness else None,
            'color_mode': attributes.get('color_mode'),
            'supported_color_modes': attributes.get('supported_color_modes'),
            'rgb_color': attributes.get('rgb_color'),
            'color_temp': attributes.get('color_temp'),
        }
    
    elif domain == 'climate':
        device_info['climate_info'] = {
            'temperature': attributes.get('temperature'),
            'current_temperature': attributes.get('current_temperature'),
            'hvac_mode': attributes.get('hvac_mode'),
            'hvac_modes': attributes.get('hvac_modes'),
            'fan_mode': attributes.get('fan_mode'),
        }
    
    elif domain == 'media_player':
        volume = attributes.get('volume_level')
        device_info['media_info'] = {
            'volume_level': volume,
            'volume_pct': round(volume * 100) if volume else None,
            'is_volume_muted': attributes.get('is_volume_muted'),
            'media_title': attributes.get('media_title'),
            'source': attributes.get('source'),
        }
    
    elif domain == 'cover':
        device_info['cover_info'] = {
            'current_position': attributes.get('current_position'),
            'device_class': attributes.get('device_class'),
        }
    
    elif domain == 'fan':
        device_info['fan_info'] = {
            'percentage': attributes.get('percentage'),
            'preset_mode': attributes.get('preset_mode'),
        }
    
    elif domain == 'sensor':
        device_info['sensor_info'] = {
            'state_class': attributes.get('state_class'),
            'device_class': attributes.get('device_class'),
        }
    
    elif domain == 'binary_sensor':
        device_info['binary_sensor_info'] = {
            'device_class': attributes.get('device_class'),
            'is_on': entity['state'] == 'on',
        }
    
    elif domain == 'lock':
        device_info['lock_info'] = {
            'is_locked': entity['state'] == 'locked',
        }
    
    elif domain == 'switch':
        device_info['switch_info'] = {
            'device_class': attributes.get('device_class'),
        }
    
    elif domain == 'script':
        device_info['script_info'] = {
            'mode': attributes.get('mode'),
        }
    
    elif domain == 'weather':
        device_info['weather_info'] = {
            'temperature': attributes.get('temperature'),
            'humidity': attributes.get('humidity'),
        }
    
    return device_info


def group_by_category(devices: List[Dict]) -> Dict[str, List[Dict]]:
    categories = {}
    for device in devices:
        domain = device['domain']
        if domain not in categories:
            categories[domain] = []
        categories[domain].append(device)
    return categories


def group_by_area(devices: List[Dict]) -> Dict[str, List[Dict]]:
    areas = {'未分類': []}
    for device in devices:
        # 優先使用 webui_area，其次用 ha_area_id
        area = device.get('webui_area') or device.get('ha_area_id') or '未分類'
        if area not in areas:
            areas[area] = []
        areas[area].append(device)
    return areas


def generate_device_list_format(devices: List[Dict]) -> str:
    """生成訓練資料用的設備列表格式"""
    lines = []
    for device in devices:
        entity_id = device['entity_id']
        name = device['friendly_name']
        state = device['state']
        
        line = f"{entity_id} '{name}' = {state}"
        
        # 添加額外資訊
        extra = []
        
        if device['domain'] == 'light' and device.get('light_info', {}).get('brightness_pct'):
            extra.append(f"{device['light_info']['brightness_pct']}%")
        elif device['domain'] == 'climate' and device.get('climate_info', {}).get('current_temperature'):
            extra.append(f"{device['climate_info']['current_temperature']}°C")
        elif device['domain'] == 'sensor' and device.get('unit_of_measurement'):
            extra.append(f"{device['unit_of_measurement']}")
        
        if extra:
            line += ";" + ";".join(extra)
        
        lines.append(line)
    
    return "\n".join(lines)

# ==============================================================================
# 主程式
# ==============================================================================

def main():
    print("=" * 80)
    print("Home Assistant 暴露設備爬取工具（精確版）")
    print("=" * 80)
    print()
    
    # 1. 解析 WebUI 清單
    print(f"正在解析 WebUI 清單：{EXPOSED_LIST_FILE}")
    try:
        webui_entities = parse_exposed_list(EXPOSED_LIST_FILE)
        print(f"✓ 解析成功，找到 {len(webui_entities)} 個實體")
        print()
    except FileNotFoundError as e:
        print(f"✗ {e}")
        print("\n請建立 exposed_entities_list.txt 並貼上從 WebUI 複製的內容")
        sys.exit(1)
    except Exception as e:
        print(f"✗ 解析失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 2. 連接到 Home Assistant
    print(f"正在連接到 Home Assistant: {HA_URL}")
    try:
        ha = HomeAssistantAPI(HA_URL, HA_TOKEN)
        config = ha.get_config()
        print(f"✓ 連接成功！Home Assistant 版本: {config.get('version')}")
        print()
    except Exception as e:
        print(f"✗ 連接失敗: {e}")
        sys.exit(1)
    
    # 3. 取得所有實體狀態
    print("正在取得實體詳細資訊...")
    try:
        all_entities = ha.get_states()
        entity_dict = {e['entity_id']: e for e in all_entities}
        print(f"✓ 成功取得 {len(all_entities)} 個實體")
        print()
    except Exception as e:
        print(f"✗ 取得實體失敗: {e}")
        sys.exit(1)
    
    # 4. 匹配並提取設備資訊
    print("正在匹配設備...")
    exposed_devices = []
    not_found = []
    
    webui_dict = {e['entity_id']: e for e in webui_entities}
    
    for entity_id, webui_info in webui_dict.items():
        if entity_id in entity_dict:
            device_info = extract_device_info(entity_dict[entity_id], webui_info)
            exposed_devices.append(device_info)
        else:
            not_found.append(entity_id)
    
    print(f"✓ 成功匹配 {len(exposed_devices)} 個設備")
    
    if not_found:
        print(f"  ⚠ 有 {len(not_found)} 個實體找不到（可能暫時離線）：")
        for entity_id in not_found[:5]:
            print(f"    - {entity_id}")
        if len(not_found) > 5:
            print(f"    ... 還有 {len(not_found) - 5} 個")
    
    print()
    
    # 5. 統計資訊
    print("設備統計：")
    print("-" * 80)
    
    by_category = group_by_category(exposed_devices)
    for domain, devices in sorted(by_category.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {domain:20s}: {len(devices):3d} 個")
    
    print()
    print("區域統計：")
    print("-" * 80)
    
    by_area = group_by_area(exposed_devices)
    for area, devices in sorted(by_area.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {area:20s}: {len(devices):3d} 個")
    
    print()
    
    # 6. 生成訓練資料格式
    device_list_text = generate_device_list_format(exposed_devices)
    
    # 7. 儲存結果
    output_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'ha_version': config.get('version'),
            'total_exposed': len(exposed_devices),
            'source': 'WebUI manual copy',
            'categories': {k: len(v) for k, v in by_category.items()},
            'areas': {k: len(v) for k, v in by_area.items()},
        },
        'devices': exposed_devices,
        'by_category': by_category,
        'by_area': by_area,
        'device_list_text': device_list_text,
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    device_list_file = "ha_device_list.txt"
    with open(device_list_file, 'w', encoding='utf-8') as f:
        f.write(device_list_text)
    
    print(f"✓ 資料已儲存到：")
    print(f"  - {OUTPUT_FILE} (完整 JSON)")
    print(f"  - {device_list_file} (純文字列表)")
    print()
    
    # 8. 顯示範例
    print("範例設備（前 20 個）：")
    print("-" * 80)
    for line in device_list_text.split('\n')[:20]:
        print(f"  {line}")
    
    if len(exposed_devices) > 20:
        print(f"  ... 還有 {len(exposed_devices) - 20} 個設備")
    
    print()
    print("=" * 80)
    print("完成！")
    print("=" * 80)


if __name__ == "__main__":
    if HA_TOKEN == "your_long_lived_access_token_here":
        print("請設定 HA_TOKEN：")
        print("  export HA_URL='http://your-ha-url:8123'")
        print("  export HA_TOKEN='your-token'")
        sys.exit(1)
    
    main()
