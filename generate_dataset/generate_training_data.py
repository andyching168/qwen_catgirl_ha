#!/usr/bin/env python3
"""
generate_training_data_v6.py

改進：
1. 使用英文標記替代中文全形標記
2. 隨機化設備狀態（brightness, temperature 等）
3. 更真實的訓練場景
"""

import json
import random
from typing import List, Dict, Any
from collections import defaultdict  # ← 修改 1：加這行
import yaml
from pypinyin import lazy_pinyin

# ==============================================================================
# 訓練資料生成配置（在此調整各類型資料的數量）
# ==============================================================================

# 基礎控制資料
TURN_ON_COUNT = 1200     # turn_on 開啟控制資料數量（提高比例）
TURN_OFF_COUNT = 1200    # turn_off 關閉控制資料數量（提高比例）

# Get State 資料（v7 新架構核心功能）
GET_STATE_COUNT = 1000       # get_state 查詢狀態資料數量
GET_STATE_RESPONSE_COUNT = 800  # get_state 二次對話回應資料數量

# 其他功能資料
CHAT_COUNT = 400         # 純聊天資料數量
SEARCH_COUNT = 800       # ACTION search 搜尋資料數量（降低比例）
CLIMATE_MODE_COUNT = 800 # climate_set_mode 冷氣模式資料數量

# 虛擬設備數量
VIRTUAL_DEVICES_COUNT = 150  # 生成的虛擬設備數量

# 設備採樣範圍（每條訓練資料使用的設備數量）
MIN_DEVICES_PER_SAMPLE = 25  # 最少設備數
MAX_DEVICES_PER_SAMPLE = 31  # 最多設備數

# ==============================================================================
# 拼音轉換
# ==============================================================================

def to_pinyin_entity_id(text: str) -> str:
    """將中文轉換為拼音 entity_id 格式"""
    pinyin_list = lazy_pinyin(text)
    return '_'.join(pinyin_list).lower()


# ==============================================================================
# Domain 分組格式生成
# ==============================================================================

def devices_to_domain_grouped_format(devices: List[Dict]) -> str:
    """Domain 分組格式"""
    grouped = defaultdict(list)
    for device in devices:
        domain = device.get('domain', 'unknown')
        grouped[domain].append(device)
    
    lines = []
    for domain in sorted(grouped.keys()):
        lines.append(f"{domain}:")
        
        for device in grouped[domain]:
            entity_id = device['entity_id']
            short_id = entity_id.split('.', 1)[1] if '.' in entity_id else entity_id
            name = device['friendly_name']
            state = device['state']
            
            line = f"  {short_id} '{name}' {state}"
            
            # 屬性
            domain_type = device.get('domain', '')
            if domain_type == 'light' and device.get('brightness_pct'):
                line += f" {device['brightness_pct']}%"
                if device.get('color'):
                    line += f" {device['color']}"
            elif domain_type == 'climate':
                if device.get('current_temp'):
                    line += f" curr={device['current_temp']}"
                if device.get('target_temp'):
                    line += f" target={device['target_temp']}"
            elif domain_type == 'cover' and device.get('position') is not None:
                line += f" pos={device['position']}%"
            elif domain_type == 'fan' and device.get('percentage'):
                line += f" {device['percentage']}%"
            
            lines.append(line)
    
    return "\n".join(lines)



# ==============================================================================
# 設備狀態隨機化
# ==============================================================================

def randomize_device_state(device: Dict) -> Dict:
    """
    隨機化設備狀態（讓訓練資料更多樣）
    
    - light: 隨機 on/off, brightness, color
    - climate: 隨機溫度、模式
    - sensor: 隨機數值
    - cover: 隨機位置
    """
    domain = device['domain']
    randomized = device.copy()
    
    if domain == 'light':
        randomized['state'] = random.choice(['on', 'off'])
        # 如果是開的，加上隨機亮度和顏色
        if randomized['state'] == 'on':
            randomized['brightness_pct'] = random.choice([20, 30, 50, 70, 80, 100])
            if random.random() < 0.3:  # 30% 機率有顏色
                randomized['color'] = random.choice(['暖白', '冷白', '紅色', '藍色', '黃色'])
    
    elif domain == 'climate':
        randomized['state'] = random.choice(['cool', 'heat', 'off', 'fan_only','auto','heat_cool'])
        current_temp_random = random.randint(20, 28)
        if randomized['state'] != 'off':
            randomized['current_temp'] = current_temp_random
            randomized['target_temp'] = random.randint(20, 28)
        else:
            randomized['current_temp'] = current_temp_random
    
    elif domain == 'cover':
        randomized['state'] = random.choice(['open', 'closed'])
        randomized['position'] = random.choice([0, 50, 100]) if randomized['state'] != 'closed' else 0
    
    elif domain == 'sensor':
        if '溫度' in device['friendly_name']:
            randomized['state'] = f"{random.randint(18, 30)}.{random.randint(0, 9)}"
        elif '濕度' in device['friendly_name']:
            randomized['state'] = str(random.randint(40, 80))
        else:
            randomized['state'] = str(random.randint(0, 100))
    
    elif domain == 'fan':
        randomized['state'] = random.choice(['on', 'off'])
        if randomized['state'] == 'on':
            randomized['percentage'] = random.choice([25, 50, 75, 100])
    
    elif domain in ['switch', 'lock', 'binary_sensor']:
        if domain == 'lock':
            randomized['state'] = random.choice(['locked', 'unlocked'])
        else:
            randomized['state'] = random.choice(['on', 'off'])
    
    return randomized



def generate_training_data_optimized():
    """優化後的訓練資料生成器"""
    
    # 關鍵修改：不要每條資料都用全部 209 個設備
    # 隨機選擇 30-60 個設備
    
    for _ in range(count):
        # 隨機採樣設備
        sampled_devices = random.sample(
            self.all_devices, 
            k=random.randint(30, 60)  # ← 關鍵！
        )
        
    device_list = generate_device_list_text(sampled_devices)
    
    # 現在每條資料：
    # System: 150 tokens
    # 設備列表（30-60 個）: 300-600 tokens
    # User + Assistant: 60 tokens
    # 總計：510-810 tokens
    # → 大部分在 768 以內！✅

# ==============================================================================
# 虛擬設備生成
# ==============================================================================

def generate_virtual_devices(count: int = VIRTUAL_DEVICES_COUNT) -> List[Dict]:
    """生成虛擬設備（使用真實拼音）"""
    virtual_areas = [
        '臥室', '客廳', '廚房', '浴室', '玄關',
        '陽台', '儲藏室', '車庫', '走廊', '餐廳',
        '書房', '主臥', '次臥', '兒童房', '更衣室'
    ]
    
    device_types = {
        'light': ['燈', '吸頂燈', '檯燈', '壁燈', '崁燈', '夜燈', '落地燈', '筒燈', '螢幕燈', '顯示器燈', '桌燈'],
        'switch': ['開關', '插座', '電源開關', '總開關', '螢幕開關', '顯示器開關'],
        'fan': ['風扇', '循環扇', '吊扇', '排風扇', '抽風扇'],
        'climate': ['冷氣', '空調', '暖氣', '冷暖氣'],
        'cover': ['窗簾', '捲簾', '百葉窗', '遮陽簾'],
        'lock': ['鎖', '門鎖', '智慧鎖', '電子鎖'],
        'sensor': ['溫度感應器', '濕度感應器', '人體感應器', '門窗感應器'],
        'binary_sensor': ['門', '窗戶', '人體偵測', '漏水偵測'],
    }
    
    virtual_devices = []
    
    for _ in range(count):
        area = random.choice(virtual_areas)
        domain = random.choice(list(device_types.keys()))
        device_name = random.choice(device_types[domain])
        
        full_name = f"{area}{device_name}"
        pinyin_id = to_pinyin_entity_id(full_name)
        entity_id = f"{domain}.{pinyin_id}"
        
        virtual_devices.append({
            'entity_id': entity_id,
            'friendly_name': full_name,
            'domain': domain,
            'area': area,
            'state': 'off'  # 預設狀態，會被隨機化
        })
    
    return virtual_devices

# ==============================================================================
# 改進的訓練資料生成器
# ==============================================================================

class ImprovedTrainingDataGenerator:
    """改進版訓練資料生成器 v3"""
    
    def __init__(self, real_devices: List[Dict], virtual_devices: List[Dict], action_spec: Dict):
        self.real_devices = real_devices
        self.virtual_devices = virtual_devices
        self.all_devices = real_devices + virtual_devices
        self.action_spec = action_spec
        
        # 按 domain 分組
        self.devices_by_domain = {}
        for device in self.all_devices:
            domain = device['domain']
            if domain not in self.devices_by_domain:
                self.devices_by_domain[domain] = []
            self.devices_by_domain[domain].append(device)
        
        # 按 area 分組
        self.devices_by_area = {}
        for device in self.all_devices:
            area = device['area']
            if area not in self.devices_by_area:
                self.devices_by_area[area] = []
            self.devices_by_area[area].append(device)
    # 在 ImprovedTrainingDataGenerator 類別中加入這個方法：

    def generate_chat_data(self, count: int) -> List[Dict]:
        """生成純聊天資料（不執行 ACTION）"""
        
        chat_templates = {
            "greeting": {
                "inputs": [
                    "你好", "嗨", "早安", "午安", "晚安", "在嗎",
                    "嗨嗨", "哈囉", "您好", "hi", "hello"
                ],
                "responses": [
                    "你好啊主人喵！有什麼需要幫忙的嗎～",
                    "嗨嗨主人喵～",
                    "在的喵！需要什麼服務呢？",
                    "您好喵！隨時為您服務～",
                    "早安主人喵！今天要做什麼呢？",
                    "嗨！日和喵隨時待命喵～"
                ]
            },
            "gratitude": {
                "inputs": [
                    "謝謝", "感謝", "辛苦了", "太棒了", "很好",
                    "謝謝你", "感謝你", "謝啦", "thanks", "謝了"
                ],
                "responses": [
                    "不客氣喵！",
                    "嘿嘿，主人滿意就好喵！",
                    "能幫上忙真是太好了喵～",
                    "這是我應該做的喵！",
                    "很高興能為您服務喵～",
                    "不用客氣喵！隨時找我喵～"
                ]
            },
            "chitchat": {
                "inputs": [
                    "你在幹嘛", "累不累", "你好可愛",
                    "你是誰", "你叫什麼名字", "介紹一下自己",
                    "心情好嗎", "你會做什麼", "你能幫我什麼"
                ],
                "responses": [
                    "我在待命喵～隨時準備為您服務！",
                    "不累喵！我精力充沛喵～",
                    "謝謝誇獎喵！主人也很棒喵～",
                    "我是日和喵，您的智慧助理喵！",
                    "我可以幫您控制家裡的設備喵～",
                    "我能幫您控制燈光、風扇、冷氣等設備喵～"
                ]
            },
            "complaints": {
                "inputs": [
                    "好熱", "好冷", "好暗", "好吵",
                    "太亮了", "太暗了", "溫度太高", "溫度太低"
                ],
                "responses": [
                    "需要我幫您調整什麼嗎喵？",
                    "那我來幫您處理一下喵～",
                    "讓我看看能幫什麼忙喵！",
                    "我這就幫您改善喵～要開冷氣嗎？"
                ]
            },
            "praise": {
                "inputs": [
                    "你真棒", "做得好", "厲害", "好聰明",
                    "你好厲害", "真聰明", "太棒了"
                ],
                "responses": [
                    "謝謝主人喵～您的鼓勵是我最大的動力！",
                    "嘿嘿～能幫到您我很開心喵！",
                    "太好了喵！能得到您的認可真開心～",
                    "主人過獎了喵！我會繼續努力的～"
                ]
            },
            "farewell": {
                "inputs": [
                    "再見", "拜拜", "晚安", "我要睡了", "先這樣",
                    "bye", "掰掰", "回見"
                ],
                "responses": [
                    "再見主人喵！有需要隨時叫我～",
                    "晚安喵！祝您有個好夢～",
                    "拜拜主人喵！期待下次為您服務～",
                    "好的喵！隨時待命～",
                    "祝您有愉快的一天喵～"
                ]
            },
            "noise": {  # ← 新增：無意義輸入 / 雜訊
                "inputs": [
                    "asdfgh", "123456", "...", "？？？",
                    "嗯嗯", "呃", "啊", "喔",
                    "qwerty", "test", "測試", "亂碼",
                    "   ", "@@@@", "####", "%%%%"
                ],
                "responses": [
                    "抱歉喵，我沒聽懂您的意思，可以再說一次嗎？",
                    "嗯？主人是要說什麼呢喵？",
                    "我不太明白喵～能更具體一點嗎？",
                    "這個我聽不懂喵，需要什麼幫助呢？",
                    "主人是不是不小心觸發了喵？有需要幫忙嗎？"
                ]
            },
            "unclear": {  # ← 新增：模糊 / 不清楚的請求
                "inputs": [
                    "那個", "幫我", "弄一下", "處理",
                    "調整", "改一下", "看看", "檢查",
                    "修理", "處理一下", "搞定", "用一下"
                ],
                "responses": [
                    "好的喵！但是要處理什麼呢？可以說具體一點嗎～",
                    "沒問題喵！請問是要調整哪個設備呢？",
                    "我隨時準備好了喵！是要開燈還是調整溫度呢？",
                    "好的喵～能告訴我具體要做什麼嗎？",
                    "收到喵！但要調整什麼設備呢？"
                ]
            },
            "out_of_scope": {  # ← 新增：超出能力範圍
                "inputs": [
                    "幫我煮飯", "洗衣服", "打掃", "買東西",
                    "訂外賣", "叫計程車", "算數學",
                    "播放音樂", "看電影", "玩遊戲", "講笑話",
                    "唱歌", "跳舞", "寫作業", "做報告"
                ],
                "responses": [
                    "抱歉喵，這個我做不到，但可以幫您控制家裡的設備喵！",
                    "這超出我的能力範圍了喵～我主要負責智慧家居控制喔～",
                    "這個我不會喵，但我可以幫您開燈、調溫度什麼的！",
                    "我的專長是控制家裡的設備喵～這個我幫不上忙呢～",
                    "嗯～這個有點困難喵，我還是乖乖管理家電就好～"
                ]
            }
        }
        
        training_data = []
        
        for _ in range(count):
            category = random.choice(list(chat_templates.keys()))
            user_input = random.choice(chat_templates[category]["inputs"])
            assistant_response = random.choice(chat_templates[category]["responses"])
            
            training_data.append(self._create_training_item_no_devices(
                user_input=user_input,
                assistant_text=assistant_response,
                assistant_action=""  # 無 ACTION
            ))
        
        return training_data

    def generate_basic_control_with_slots(self, action: str, count: int) -> List[Dict]:
        """
        生成基礎控制訓練資料（v3.0 - 基於 name + area Slots）
        
        核心改變：
        1. 不再輸出 entity_id
        2. 輸出中文 name + area
        3. 由 Home Assistant Intent 系統自動匹配實體
        """
        
        training_data = []
        supported_domains = ['light', 'switch', 'fan', 'cover', 'climate', 'media_player', 'vacuum']
        
        # 表達方式模板
        if action == 'turn_on':
            # 明確指定設備名稱 + 區域
            specific_with_area_templates = [
                "打開{area}的{name}", 
                "開啟{area}{name}", 
                "把{area}的{name}打開", 
                "開{area}{name}",
                "請幫我開{area}的{name}", 
                "麻煩開一下{area}{name}", 
                "幫忙開{area}的{name}",
                "能幫我開{area}{name}嗎", 
                "{area}{name}開一下",
                "打開{area}{name}",
            ]
            
            # 只有設備名稱（依賴上下文推斷區域）
            specific_only_name_templates = [
                "打開{name}", 
                "開啟{name}", 
                "把{name}打開", 
                "開{name}",
                "請幫我開{name}", 
                "麻煩開一下{name}", 
                "{name}開一下",
                "能開{name}嗎",
            ]
            
            # 區域 + 設備類型（例如：打開書房的燈）
            area_with_type_templates = [
                "打開{area}的{type}", 
                "開{area}{type}", 
                "把{area}的{type}打開",
                "幫我開{area}的{type}", 
                "{area}的{type}打開", 
                "開一下{area}的{type}",
                "可以開{area}{type}嗎",
            ]
            
            # 只有設備類型（例如：開燈）
            type_only_templates = [
                "開{type}", 
                "打開{type}", 
                "把{type}打開", 
                "幫我開{type}",
                "{type}開一下", 
                "可以開{type}嗎",
                "開一下{type}",
            ]
            
            # 回應模板（移除 entity_id，使用設備名稱）
            response_templates = [
                "好的主人，馬上幫您開啟{full_name}喵！",
                "沒問題喵！正在打開{full_name}",
                "收到！{full_name}開啟中喵～",
                "好喵！{full_name}馬上開給您",
                "明白了！開啟{full_name}囉喵",
                "{full_name}開好囉喵！",
                "來囉喵～{full_name}已經開啟",
                "好的呢！{full_name}打開了喵",
                "交給我吧！{full_name}開啟中～",
                "遵命喵！{full_name}馬上就亮",
                "好的，很樂意為您開啟{full_name}喵",
                "當然可以！{full_name}開啟囉喵",
                "沒問題的喵！{full_name}已開啟",
                "{full_name}開了喵！",
                "開好囉～{full_name}",
                "{full_name}已開喵",
                "好～{full_name}開啟",
                "喵嗚～{full_name}亮起來囉！",
                "好的說～{full_name}開開喵！"
            ]
            
        else:  # turn_off
            specific_with_area_templates = [
                "關閉{area}的{name}", 
                "關掉{area}{name}", 
                "把{area}的{name}關掉", 
                "關{area}{name}",
                "請幫我關{area}的{name}", 
                "麻煩關一下{area}{name}", 
                "幫忙關{area}的{name}",
                "能幫我關{area}{name}嗎", 
                "{area}{name}關一下",
                "關掉{area}{name}",
            ]
            
            specific_only_name_templates = [
                "關閉{name}", 
                "關掉{name}", 
                "把{name}關掉", 
                "關{name}",
                "請幫我關{name}", 
                "麻煩關一下{name}", 
                "{name}關一下",
                "能關{name}嗎",
            ]
            
            area_with_type_templates = [
                "關閉{area}的{type}", 
                "關{area}{type}", 
                "把{area}的{type}關掉",
                "幫我關{area}的{type}", 
                "{area}的{type}關掉", 
                "關一下{area}的{type}",
                "可以關{area}{type}嗎",
            ]
            
            type_only_templates = [
                "關{type}", 
                "關閉{type}", 
                "把{type}關掉", 
                "幫我關{type}",
                "{type}關一下", 
                "可以關{type}嗎",
                "關一下{type}",
            ]
            
            response_templates = [
                "好的，正在關閉{full_name}喵！",
                "沒問題喵！幫您把{full_name}關掉",
                "收到！{full_name}關閉中喵～",
                "好喵！{full_name}馬上關掉",
                "明白了！關閉{full_name}囉喵",
                "{full_name}關好囉喵！",
                "來囉喵～{full_name}已經關閉",
                "好的呢！{full_name}關掉了喵",
                "交給我吧！{full_name}關閉中～",
                "遵命喵！{full_name}馬上就暗",
                "好的，很樂意為您關閉{full_name}喵",
                "當然可以！{full_name}關閉囉喵",
                "沒問題的喵！{full_name}已關閉",
                "{full_name}關了喵！",
                "關好囉～{full_name}",
                "{full_name}已關喵",
                "好～{full_name}關閉",
                "喵嗚～{full_name}休息囉！",
                "好的說～{full_name}關關喵！"
            ]
        
        # 設備類型的中文名稱（用於模糊表達）
        type_names = {
            'light': ['燈', '電燈', '照明', '燈光', '大燈', '小燈'],
            'switch': ['開關', '插座', '電源', '總開關'],
            'fan': ['風扇', '電扇', '風扇機', '循環扇'],
            'cover': ['窗簾', '捲簾', '百葉窗'],
            'climate': ['冷氣', '空調', '暖氣'],
            'media_player': ['音響', '播放器', '喇叭'],
            'vacuum': ['掃地機器人', '掃地機', '吸塵器'],
        }
        
        # ========================================================================
        # 情況 1：明確指定 區域 + 設備名稱 (40%)
        # 輸出：ACTION turn_on\nname 大燈\narea 書房
        # ========================================================================
        situation_1_count = int(count * 0.4)
        for _ in range(situation_1_count):
            domain = random.choice(supported_domains)
            if domain not in self.devices_by_domain or not self.devices_by_domain[domain]:
                continue
            
            device = random.choice(self.devices_by_domain[domain])
            
            # 提取純設備名稱（去除區域資訊）
            # 例如：「書房大燈」→ 「大燈」
            device_name = device['friendly_name']
            area = device.get('area', '')
            
            # 如果設備名稱中包含區域，嘗試移除
            if area and area in device_name:
                device_name = device_name.replace(area, '').strip()
            
            # 用戶輸入：明確指定區域和設備名稱
            template = random.choice(specific_with_area_templates)
            user_input = template.format(area=area, name=device_name)
            
            # 助理回應（使用完整名稱）
            full_name = f"{area}{device_name}" if area else device_name
            assistant_text = random.choice(response_templates).format(full_name=full_name)
            
            # 新格式：輸出 name + area（不含 entity_id）
            assistant_action = f"ACTION {action}\nname {device_name}\narea {area}"
            
            training_data.append(self._create_training_item_no_devices(user_input, assistant_text, assistant_action))
        
        # ========================================================================
        # 情況 2：只有設備名稱 (30%)
        # 輸出：ACTION turn_on\nname 大燈
        # （依賴上下文或衛星位置推斷區域）
        # ========================================================================
        situation_2_count = int(count * 0.3)
        for _ in range(situation_2_count):
            domain = random.choice(supported_domains)
            if domain not in self.devices_by_domain or not self.devices_by_domain[domain]:
                continue
            
            device = random.choice(self.devices_by_domain[domain])
            device_name = device['friendly_name']
            area = device.get('area', '')
            
            # 移除區域資訊
            if area and area in device_name:
                device_name = device_name.replace(area, '').strip()
            
            # 用戶輸入：只提供設備名稱
            template = random.choice(specific_only_name_templates)
            user_input = template.format(name=device_name)
            
            # 助理回應
            full_name = f"{area}{device_name}" if area else device_name
            assistant_text = random.choice(response_templates).format(full_name=full_name)
            
            # 輸出：只有 name（area 省略或從上下文推斷）
            if area and random.random() > 0.5:  # 50% 機率包含 area
                assistant_action = f"ACTION {action}\nname {device_name}\narea {area}"
            else:
                assistant_action = f"ACTION {action}\nname {device_name}"
            
            training_data.append(self._create_training_item_no_devices(user_input, assistant_text, assistant_action))
        
        # ========================================================================
        # 情況 3：區域 + 設備類型 (20%)
        # 輸出：ACTION turn_on\nname 燈\narea 書房
        # （用於「打開書房的燈」這類模糊表達）
        # ========================================================================
        situation_3_count = int(count * 0.2)
        for _ in range(situation_3_count):
            domain = random.choice(supported_domains)
            if domain not in self.devices_by_domain or not self.devices_by_domain[domain]:
                continue
            
            device = random.choice(self.devices_by_domain[domain])
            area = device.get('area', '')
            
            # 選擇設備類型的中文名稱
            type_name = random.choice(type_names.get(domain, [domain]))
            
            # 用戶輸入：區域 + 設備類型
            template = random.choice(area_with_type_templates)
            user_input = template.format(area=area, type=type_name)
            
            # 助理回應
            full_name = f"{area}的{type_name}" if area else type_name
            assistant_text = random.choice(response_templates).format(full_name=full_name)
            
            # 輸出：通用類型名稱 + 區域
            assistant_action = f"ACTION {action}\nname {type_name}\narea {area}"
            
            training_data.append(self._create_training_item_no_devices(user_input, assistant_text, assistant_action))
        
        # ========================================================================
        # 情況 4：只有設備類型 (10%)
        # 輸出：ACTION turn_on\nname 燈
        # （用於「開燈」這類極簡表達）
        # ========================================================================
        situation_4_count = count - situation_1_count - situation_2_count - situation_3_count
        for _ in range(situation_4_count):
            domain = random.choice(supported_domains)
            if domain not in self.devices_by_domain or not self.devices_by_domain[domain]:
                continue
            
            device = random.choice(self.devices_by_domain[domain])
            
            # 選擇設備類型的中文名稱
            type_name = random.choice(type_names.get(domain, [domain]))
            
            # 用戶輸入：只有設備類型
            template = random.choice(type_only_templates)
            user_input = template.format(type=type_name)
            
            # 助理回應（可能需要反問區域）
            if random.random() > 0.7:  # 30% 機率詢問區域
                assistant_text = f"好的喵！請問要開啟哪個區域的{type_name}呢？"
                assistant_action = ""  # 不執行 ACTION，等待用戶澄清
            else:
                # 假設從上下文或衛星位置推斷區域
                area = device.get('area', '')
                full_name = f"{area}的{type_name}" if area else type_name
                assistant_text = random.choice(response_templates).format(full_name=full_name)
                assistant_action = f"ACTION {action}\nname {type_name}"
            
            training_data.append(self._create_training_item_no_devices(user_input, assistant_text, assistant_action))
        
        return training_data
    
    def generate_search_action_data(self, count: int) -> List[Dict]:
        """
        生成 ACTION search 的訓練資料
        
        使用時機：
        - 使用者詢問需要即時性資訊（天氣、新聞、股票等）
        - 需要查詢外部資訊才能正確回答的問題
        
        範例：
        User: 中和區週五會下雨嗎
        AI: 讓我幫你查一下中和區週五的天氣喵！
            ACTION search
            query 中和區週五天氣預報
        """
        
        search_templates = {
    "weather": {
      "patterns": [
        "{location}{time}會{weather}嗎",
        "{location}{time}的天氣{how}",
        "{time}{location}會不會{weather}",
        "{location}{time}天氣預報",
        "查{location}{time}天氣",
        "{location}{time}氣溫{how}",
        "{time}{location}會下雨嗎",
        "幫我查{location}{time}天氣",
        "{location}{time}需要帶傘嗎",
        "{location}{time}適合出門嗎",
        "{time}{location}天氣{how}樣",
        "{location}{time}冷不冷",
        "{location}{time}熱不熱",
        "跟我說{location}{time}天氣",
        "{location}{time}詳細天氣",
        "查一下{location}{time}的氣象",
        "{location}{time}氣溫幾度",
        "{location}{time}會不會很{weather_adj}"
      ],
      "locations": [
        "台北",
        "新北",
        "桃園",
        "台中",
        "台南",
        "高雄",
        "中和區",
        "板橋區",
        "新莊區",
        "三重區",
        "永和區",
        "信義區",
        "大安區",
        "中山區",
        "松山區",
        "內湖區",
        "台灣",
        "北部",
        "南部",
        "中部",
        "東部",
        "基隆",
        "新竹",
        "宜蘭",
        "花蓮",
        "台東",
        "日本東京",
        "美國紐約"
      ],
      "times": [
        "今天",
        "明天",
        "後天",
        "週末",
        "週五",
        "週六",
        "週日",
        "這週",
        "下週",
        "這個禮拜",
        "下個禮拜",
        "早上",
        "中午",
        "下午",
        "晚上",
        "今晚",
        "待會",
        "傍晚",
        "凌晨",
        "接下來幾個小時"
      ],
      "weathers": [
        "下雨",
        "出太陽",
        "颳風",
        "打雷",
        "下雪",
        "晴天",
        "陰天",
        "雷陣雨"
      ],
      "weather_adj": [
        "冷",
        "熱",
        "濕",
        "悶"
      ],
      "hows": [
        "怎麼樣",
        "如何",
        "好嗎",
        "多少",
        "概況"
      ],
      "responses": [
        "讓我幫你查一下{location}{time}的天氣喵！",
        "好喵！我來幫你查看看{location}{time}的天氣～",
        "沒問題喵！查詢{location}{time}天氣中～",
        "好的呢！馬上幫您查{location}{time}的天氣預報喵！",
        "收到喵！我來看看{location}{time}天氣怎麼樣～",
        "喵！{location}{time}的天氣是嗎？我馬上來查喵！",
        "好的喵～幫你看看{location}{time}的天氣預報～"
      ]
    },
    "news": {
      "patterns": [
        "查{topic}的新聞",
        "有關{topic}的新聞",
        "幫我看{topic}新聞",
        "{topic}最新消息",
        "{topic}有什麼新聞",
        "搜尋{topic}相關新聞",
        "找{topic}的報導",
        "{topic}新聞",
        "查一下{topic}",
        "看看{topic}的消息",
        "最近有什麼{topic}的新聞嗎",
        "跟我講一下{topic}的近況",
        "我想知道{topic}的最新發展",
        "有沒有{topic}的八卦",
        "搜尋{topic}頭條"
      ],
      "topics": [
        "政府普發10000元",
        "台積電",
        "AI",
        "ChatGPT",
        "地震",
        "颱風",
        "疫情",
        "選舉",
        "房價",
        "股市",
        "油價",
        "匯率",
        "太魯閣事故",
        "捷運",
        "高鐵",
        "總統",
        "立法院",
        "行政院",
        "Nvidia",
        "AMD",
        "烏俄戰爭",
        "以巴衝突",
        "通膨",
        "升息",
        "科技",
        "財經",
        "體育",
        "娛樂",
        "Sora",
        "Llama 3",
        "Qwen"
      ],
      "responses": [
        "好喵！我來幫你查看看喵！",
        "沒問題喵！馬上幫您搜尋相關新聞～",
        "收到！我來找找{topic}的最新消息喵～",
        "好的呢！幫您查詢{topic}的新聞～",
        "讓我幫你找找{topic}的相關報導喵！",
        "喵！{topic}的新聞嗎？我來找找看有沒有什麼新消息喵～",
        "了解喵！幫你彙整一下{topic}的相關新聞！"
      ]
    },
    "information": {
      "patterns": [
        "{query}是什麼",
        "什麼是{query}",
        "{query}怎麼用",
        "如何{query}",
        "{query}的方法",
        "幫我查{query}",
        "搜尋{query}",
        "找{query}資料",
        "{query}在哪裡",
        "{query}怎麼做",
        "{query}的資訊",
        "幫我查{query}並介紹給我聽",
        "我想知道{query}是什麼，可以說明嗎",
        "查一下{query}，然後跟我解釋",
        "搜尋{query}的定義",
        "跟我說明一下{query}",
        "介紹一下{query}",
        "我想學{query}",
        "{query}的教學",
        "{query}的背景資料",
        "{query}的評價",
        "{query}是什麼意思",
        "{query}的優缺點",
        "能幫我查看看 {query} 並且介紹給我聽嗎"
      ],
      "queries": [
        "Home Assistant",
        "智慧家居設定",
        "Python教學",
        "如何煮飯",
        "蛋糕食譜",
        "咖啡怎麼泡",
        "台北101",
        "故宮博物院",
        "日月潭",
        "健保卡",
        "報稅",
        "勞保",
        "信用卡推薦",
        "手機推薦",
        "筆電推薦",
        "qwen 模型",
        "Claude 3",
        "Llama 3",
        "Gemini 1.5 Pro",
        "大型語言模型",
        "Transformer架構",
        "RAG",
        "Fine-tuning",
        "LoRA",
        "相對論",
        "光合作用",
        "區塊鏈",
        "NFT",
        "元宇宙",
        "台灣黑熊",
        "玉山",
        "如何架設網站",
        "SQL語法",
        "九二共識",
        "ECFA"
      ],
      "responses": [
        "好喵！我來幫你查一下～",
        "沒問題喵！讓我搜尋看看～",
        "收到！馬上幫您查詢喵～",
        "好的呢！我來找找相關資料～",
        "讓我幫你搜尋一下喵！",
        "喵！是想知道{query}的資訊嗎？我馬上幫你查資料然後整理給你喵！",
        "收到喵！幫你搜尋{query}，然後再跟你解釋喵～",
        "沒問題！查詢{query}的資料中，等我一下喵！"
      ]
    },
    "stock_finance": {
      "patterns": [
        "{stock}股價多少",
        "查{stock}股價",
        "{stock}今天漲還是跌",
        "{stock}現在多少",
        "幫我看{stock}",
        "{currency}匯率",
        "查{currency}匯率",
        "{currency}現在多少",
        "看一下{stock}的K線圖",
        "{stock}的財報如何",
        "{currency}兌台幣匯率",
        "台幣換{currency}怎麼算",
        "{stock}的目標價"
      ],
      "stocks": [
        "台積電",
        "聯發科",
        "鴻海",
        "台達電",
        "特斯拉",
        "蘋果",
        "微軟",
        "NVIDIA",  # 改用英文，避免與設備名稱混淆
        "AMD",
        "加權指數",
        "美股",
        "道瓊",
        "那斯達克",
        "廣達",
        "緯創",
        "長榮",
        "陽明",
        "0050",
        "00878",
        "QQQ",
        "VT",
        "比特幣",
        "乙太幣"
      ],
      "currencies": [
        "美元",
        "日幣",
        "歐元",
        "人民幣",
        "港幣",
        "韓元",
        "英鎊",
        "泰銖",
        "澳幣"
      ],
      "responses": [
        "好喵！我來幫你查一下目前的價格～",
        "沒問題喵！馬上查詢最新資訊～",
        "收到！讓我看看現在的行情喵～",
        "好的呢！幫您查詢即時資訊～",
        "喵！馬上幫您查詢金融資訊！",
        "好的喵！我來看看{stock}的最新股價～",
        "匯率查詢是嗎？沒問題喵！"
      ]
    },
    "location_business": {
      "patterns": [
        "附近有{place}嗎",
        "哪裡有{place}",
        "找{place}",
        "{place}推薦",
        "好吃的{place}",
        "{location}有什麼{place}",
        "{location}{place}推薦",
        "幫我找{location}的{place}",
        "我想去{place}，{location}有推薦的嗎",
        "最近的{place}在哪",
        "{place}營業時間",
        "幫我導航到{place}"
      ],
      "places": [
        "餐廳",
        "咖啡廳",
        "早餐店",
        "便利商店",
        "停車場",
        "加油站",
        "銀行",
        "ATM",
        "醫院",
        "診所",
        "藥局",
        "健身房",
        "景點",
        "夜市",
        "電影院",
        "KTV",
        "百貨公司",
        "捷運站",
        "火車站",
        "公車站",
        "郵局"
      ],
      "locations": [
        "附近",
        "這裡",
        "台北",
        "信義區",
        "大安區",
        "公司附近",
        "我家附近",
        "台北車站",
        "西門町",
        "東區",
        "市政府"
      ],
      "responses": [
        "好喵！我來幫你找找看～",
        "沒問題喵！馬上搜尋附近的地點～",
        "收到！讓我查查看喵～",
        "好的呢！幫您搜尋一下～",
        "喵！在找{place}嗎？我來幫你看看{location}有哪些喵～",
        "收到喵！搜尋{location}的{place}中～"
      ]
    },
    "realtime_info": {
      "patterns": [
        "現在{info}",
        "今天{info}",
        "查{info}",
        "幫我看{info}",
        "幫我查{info}",
        "我想知道{info}",
        "{info}查詢"
      ],
      "patterns_with_value": [
        "{info}是多少",
        "查一下{info}是多少",
        "今天的{info}是多少",
        "幫我看一下{info}是多少"
      ],
      "infos": [
        "幾點",
        "時間",
        "日期",
        "星期幾"
      ],
      "infos_with_value": [
        "油價",
        "空氣品質",
        "PM2.5",
        "紫外線指數",
        "溫度",
        "濕度",
        "日出時間",
        "日落時間",
        "農曆日期",
        "潮汐時間"
      ],
      "responses": [
        "好喵！我來幫你查一下～",
        "沒問題喵！馬上查詢～",
        "收到！讓我看看喵～",
        "好的喵！即時資訊查詢... 喵！我來了～",
        "馬上幫您查詢{info}喵！"
      ]
    }
  }
        
        training_data = []
        
        for _ in range(count):
            # 隨機選擇查詢類別
            category = random.choice(list(search_templates.keys()))
            template_data = search_templates[category]
            
            # 特殊處理 realtime_info：需要根據 info 類型選擇 pattern
            if category == "realtime_info":
                # 隨機選擇是否使用「是多少」類型的 pattern
                if random.random() < 0.5:
                    # 使用一般 pattern + 一般 info（時間、日期等）
                    pattern = random.choice(template_data["patterns"])
                    info = random.choice(template_data["infos"])
                else:
                    # 使用「是多少」pattern + 數值型 info（油價、溫度等）
                    pattern = random.choice(template_data["patterns_with_value"])
                    info = random.choice(template_data["infos_with_value"])
            else:
                # 其他類別正常選擇 pattern
                pattern = random.choice(template_data["patterns"])
            
            # 根據不同類別填充參數
            if category == "weather":
                location = random.choice(template_data["locations"])
                time = random.choice(template_data["times"])
                user_input = pattern.format(
                    location=location,
                    time=time,
                    weather=random.choice(template_data.get("weathers", [""])),
                    weather_adj=random.choice(template_data.get("weather_adj", [""])),
                    how=random.choice(template_data.get("hows", [""]))
                )
                query = f"{location}{time}天氣預報"
                response_text = random.choice(template_data["responses"]).format(
                    location=location, time=time
                )
                
            elif category == "news":
                topic = random.choice(template_data["topics"])
                user_input = pattern.format(topic=topic)
                query = f"{topic} 新聞"
                response_text = random.choice(template_data["responses"])
                # 處理 response 中的變數
                if "{topic}" in response_text:
                    response_text = response_text.format(topic=topic)
                    
            elif category == "information":
                query_term = random.choice(template_data["queries"])
                user_input = pattern.format(query=query_term)
                query = query_term
                response_text = random.choice(template_data["responses"])
                # 處理 response 中的變數
                if "{query}" in response_text:
                    response_text = response_text.format(query=query_term)
                
            elif category == "stock_finance":
                if "currency" in pattern:
                    currency = random.choice(template_data["currencies"])
                    user_input = pattern.format(currency=currency)
                    query = f"{currency}匯率"
                    item_name = currency
                else:
                    stock = random.choice(template_data["stocks"])
                    user_input = pattern.format(stock=stock)
                    query = f"{stock}股價"
                    item_name = stock
                response_text = random.choice(template_data["responses"])
                # 處理 response 中的變數
                if "{stock}" in response_text:
                    response_text = response_text.format(stock=item_name)
                
            elif category == "location_business":
                place = random.choice(template_data["places"])
                location = random.choice(template_data.get("locations", [""]))
                user_input = pattern.format(place=place, location=location)
                query = f"{location}{place}" if location else place
                response_text = random.choice(template_data["responses"])
                # 處理 response 中的變數
                if "{place}" in response_text or "{location}" in response_text:
                    response_text = response_text.format(place=place, location=location)
                
            elif category == "realtime_info":
                # info 和 pattern 已經在外層根據類型選好了
                user_input = pattern.format(info=info)
                query = info
                response_text = random.choice(template_data["responses"])
                # 處理 response 中的變數
                if "{info}" in response_text:
                    response_text = response_text.format(info=info)
            
            # 組合 Assistant 回應
            assistant_action = f"ACTION search\nquery {query}"
            
            training_data.append(self._create_training_item_no_devices(
                user_input=user_input,
                assistant_text=response_text,
                assistant_action=assistant_action
            ))
        
        return training_data

    def generate_climate_set_mode_data(self, count: int) -> List[Dict]:
        """生成 climate_set_mode 訓練資料"""
        
        training_data = []
        
        # 取得所有 climate 設備
        climate_devices = self.devices_by_domain.get('climate', [])
        if not climate_devices:
            print("  ⚠️  警告：沒有 climate 設備，跳過 climate_set_mode 資料生成")
            return []
        
        # 定義模板資料
        modes_data = {
            "cool": {
                "zh_names": ["冷氣模式", "製冷", "製冷模式"],
                "verbs": ["切換到", "改成", "設定成", "設為", "調成", "換成"]
            },
            "heat": {
                "zh_names": ["暖氣模式", "加熱", "加熱模式"],
                "verbs": ["切換到", "改成", "設定成", "設為", "調成", "換成"]
            },
            "fan_only": {
                "zh_names": ["送風模式", "風扇模式", "通風"],
                "verbs": ["切換到", "改成", "設定成", "設為", "調成", "換成"]
            },
            "dry": {
                "zh_names": ["除濕模式", "乾燥模式", "除濕"],
                "verbs": ["切換到", "改成", "設定成", "設為", "調成", "換成"]
            },
            "auto": {
                "zh_names": ["自動模式", "自動", "智能模式"],
                "verbs": ["切換到", "改成", "設定成", "設為", "調成", "換成"]
            },
            "off": {
                "zh_names": ["關閉", "停止"],
                "verbs": ["關閉", "關掉", "停止"]
            }
        }
        
        # 用戶輸入模板
        patterns = [
            # 原有模板
            "{verb}{area}冷氣{mode}",
            "把{area}冷氣{verb}{mode}",
            "{area}冷氣{verb}{mode}",
            "{area}{verb}{mode}",
            "幫我把{area}冷氣{verb}{mode}",
            "請把{area}的冷氣{verb}{mode}",
            "{area}空調{verb}{mode}",
            "把{area}空調{verb}{mode}",
            
            # 新增：簡化模板
            "把{area}{verb}{mode}",
            "{area}{verb}{mode}吧",
            "幫我{verb}{area}{mode}",
            "{area}開{mode}",
            "把{area}開成{mode}",
            "讓{area}{verb}{mode}",
            "{area}要{mode}",
            
            # ✅ 新增：更多口語化模板（提高多樣性）
            "{area}{mode}",              # 超簡化
            "{area}模式{verb}{mode}",
            "把{area}調{mode}",
            "{area}設{mode}",
            "{area}冷氣開{mode}",
            "麻煩{verb}{area}{mode}",
            "能{verb}{area}{mode}嗎"
        ]
        
        # 回應模板（大幅增加「喵」的使用）
        response_templates = [
            "好的喵，正在切換{area}冷氣到{mode_zh}",
            "好的，正在設定{area}冷氣為{mode_zh}喵",
            "收到喵～正在調整{area}冷氣模式為{mode_zh}",
            "收到喵，切換{area}冷氣到{mode_zh}",
            "馬上幫您設定{area}冷氣為{mode_zh}喵",
            "好的喵，{area}冷氣切換到{mode_zh}囉",
            "了解喵，正在設定{area}冷氣{mode_zh}",
            "好的喵，{area}冷氣改成{mode_zh}了",
            "收到喵！馬上把{area}設定成{mode_zh}",
            "好的主人喵～{area}切換到{mode_zh}囉",
            "沒問題喵，{area}冷氣正在切換到{mode_zh}",
            "好喵～立刻幫您設定{area}{mode_zh}",
            "明白了喵，{area}冷氣設定為{mode_zh}",
            "好的喵！正在調整{area}為{mode_zh}",
            "收到喵～{area}改成{mode_zh}囉",
            "馬上處理喵，{area}切換{mode_zh}中",
            "好的主人，{area}冷氣設定{mode_zh}喵～",
            "了解喵！{area}正在調整為{mode_zh}",
            # ✅ 新增更多回應變化
            "好喵，幫您把{area}調成{mode_zh}",
            "沒問題喵～{area}正在切換到{mode_zh}",
            "馬上為您設定{area}{mode_zh}喵",
            "好的，{area}要{mode_zh}喵"
        ]
        
        for _ in range(count):
            # 隨機選擇設備
            device = random.choice(climate_devices)
            area = device['area']
            
            # 隨機選擇模式
            mode = random.choice(list(modes_data.keys()))
            mode_info = modes_data[mode]
            mode_zh = random.choice(mode_info["zh_names"])
            verb = random.choice(mode_info["verbs"])
            
            # 生成用戶輸入
            # 隨機選擇模板，有些模板可能不需要 verb
            pattern = random.choice(patterns)
            
            # 如果模板包含 {verb}，才需要 verb，否則只用 mode
            if "{verb}" in pattern:
                user_input = pattern.format(area=area, mode=mode_zh, verb=verb)
            else:
                user_input = pattern.format(area=area, mode=mode_zh)
            
            # 生成回應
            response_text = random.choice(response_templates).format(
                area=area,
                mode_zh=mode_zh
            )
            
            # 生成 ACTION
            assistant_action = f"ACTION climate_set_mode\narea {area}\nmode {mode}"
            
            training_data.append(self._create_training_item_no_devices(
                user_input=user_input,
                assistant_text=response_text,
                assistant_action=assistant_action
            ))
        
        return training_data

    def _create_training_item(self, user_input: str, assistant_text: str, assistant_action: str) -> Dict:
        """建立訓練資料項目（Domain 分組格式 + 設備採樣）"""
        
        # ← 修改 3：這裡是核心修改
        # 隨機採樣設備（避免截斷）
        num_devices = random.randint(MIN_DEVICES_PER_SAMPLE, MAX_DEVICES_PER_SAMPLE)
        sampled_devices = random.sample(self.all_devices, k=min(num_devices, len(self.all_devices)))
        
        # 隨機化狀態
        randomized_devices = [randomize_device_state(d) for d in sampled_devices]
        
        # 使用 Domain 分組格式（省 12% token）
        device_list_text = devices_to_domain_grouped_format(randomized_devices)
        
        user_message = f"""Available devices:
{device_list_text}

User request:
{user_input}"""
        
        return {
            "messages": [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": f"{assistant_text}\n{assistant_action}"}
            ]
        }
    
    def _get_system_prompt(self) -> str:
        """取得 System Prompt"""
        # 也改用英文標記
        original = self.action_spec.get('system_prompt', '')
        
        # 替換中文標記為英文
        modified = original.replace('【當前設備】', 'Available devices:')
        modified = modified.replace('【使用者請求】', 'User request:')
        
        return modified
    
    def _create_training_item_no_devices(self, user_input: str, assistant_text: str, assistant_action: str) -> Dict:
        """建立訓練資料項目（不含設備列表 - v7 新架構）
        
        用於：
        - get_state 查詢（模型需要主動調用 ACTION get_state）
        - 純聊天
        - 搜尋（ACTION search）
        """
        user_message = f"User request:\n{user_input}"
        
        assistant_content = assistant_text
        if assistant_action:
            assistant_content += f"\n{assistant_action}"
        
        return {
            "messages": [
                {"role": "system", "content": self._get_system_prompt_v7()},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_content}
            ]
        }
    
    def _get_system_prompt_v7(self) -> str:
        """取得 v7 版本的 System Prompt（不包含設備列表）"""
        return """你是日和喵，可愛的貓娘智慧家居助理喵！

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
3. name 是設備名稱（例如：大燈、風扇、冷氣）
4. area 是區域名稱（例如：書房、客廳、臥室），如果使用者沒有說區域的話，預設用書房
5. 純聊天時不輸出 ACTION
6. 回應要親切可愛，適當加入「喵」

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

用戶：你好
助理：你好呀主人！有什麼需要幫忙的嗎喵？"""
    
    def generate_get_state_data(self, count: int) -> List[Dict]:
        """生成 get_state 查詢的訓練資料
        
        讓模型學會在用戶詢問狀態時主動調用 ACTION get_state
        """
        
        # 查詢類型模板
        query_templates = {
            "on_off_check": {
                "patterns": [
                    "{area}{name}開著嗎",
                    "{area}{name}是開的嗎",
                    "{area}的{name}有開嗎",
                    "{name}有開嗎",
                    "{area}{name}現在是開的還是關的",
                    "{name}是不是開著",
                    "{area}{name}亮著嗎",
                    "{name}開了沒",
                ],
                "responses": [
                    "我來幫你看一下喵～",
                    "讓我查查看喵！",
                    "我來確認一下喵～",
                    "讓我看看喵",
                    "查詢中喵！",
                    "我幫你看看喵～",
                ]
            },
            "state_check": {
                "patterns": [
                    "{area}{name}現在是什麼狀態",
                    "{area}的{name}狀態如何",
                    "{name}目前的狀態",
                    "查一下{area}{name}的狀態",
                    "看看{name}現在怎樣",
                    "幫我看一下{area}{name}",
                ],
                "responses": [
                    "讓我查一下喵！",
                    "好的，我來看看喵～",
                    "查詢中喵！",
                    "我幫你確認一下喵～",
                ]
            },
            "temperature_check": {
                "patterns": [
                    "{area}冷氣現在幾度",
                    "{area}現在溫度多少",
                    "{area}冷氣設定幾度",
                    "{area}冷氣溫度是多少",
                    "冷氣現在溫度",
                ],
                "responses": [
                    "讓我查查看喵！",
                    "我來看看溫度喵～",
                    "好的，查一下喵！",
                ]
            },
            "brightness_check": {
                "patterns": [
                    "{area}燈的亮度多少",
                    "{area}燈現在亮度",
                    "{name}現在亮度是多少",
                ],
                "responses": [
                    "讓我看看喵～",
                    "查一下亮度喵！",
                ]
            },
            "lock_check": {
                "patterns": [
                    "大門有鎖嗎",
                    "門鎖好了嗎",
                    "門有鎖上嗎",
                    "大門是鎖著的嗎",
                ],
                "responses": [
                    "讓我看看喵！",
                    "我來確認一下喵～",
                    "查詢中喵！安全很重要喵～",
                ]
            },
            "cover_check": {
                "patterns": [
                    "{area}窗簾開了嗎",
                    "{area}窗簾是開的嗎",
                    "窗簾現在的位置",
                ],
                "responses": [
                    "我來看看喵～",
                    "讓我查一下喵！",
                ]
            },
        }
        
        # 設備類型映射
        device_areas = ['書房', '客廳', '臥室', '兒童房', '廚房', '餐廳', '玄關', '主臥']
        device_names = {
            'light': ['燈', '大燈', '小燈', '吸頂燈', '檯燈'],
            'switch': ['開關', '插座'],
            'fan': ['風扇', '電扇', '循環扇'],
            'climate': ['冷氣', '空調'],
            'cover': ['窗簾'],
            'lock': ['門鎖', '大門鎖'],
        }
        
        training_data = []
        
        for _ in range(count):
            # 隨機選擇查詢類型
            query_type = random.choice(list(query_templates.keys()))
            template_data = query_templates[query_type]
            
            # 選擇適合的設備
            if query_type == "temperature_check":
                device_type = 'climate'
                name = random.choice(device_names['climate'])
            elif query_type == "brightness_check":
                device_type = 'light'
                name = random.choice(device_names['light'])
            elif query_type == "lock_check":
                device_type = 'lock'
                name = random.choice(device_names['lock'])
            elif query_type == "cover_check":
                device_type = 'cover'
                name = random.choice(device_names['cover'])
            else:
                device_type = random.choice(['light', 'switch', 'fan', 'climate'])
                name = random.choice(device_names[device_type])
            
            area = random.choice(device_areas)
            
            # 生成用戶輸入
            pattern = random.choice(template_data["patterns"])
            user_input = pattern.format(area=area, name=name)
            
            # 生成助理回應
            response_text = random.choice(template_data["responses"])
            
            # 生成 ACTION
            assistant_action = f"ACTION get_state\nname {name}\narea {area}"
            
            training_data.append(self._create_training_item_no_devices(
                user_input=user_input,
                assistant_text=response_text,
                assistant_action=assistant_action
            ))
        
        return training_data
    
    def generate_get_state_response_data(self, count: int) -> List[Dict]:
        """生成 get_state 二次對話的訓練資料
        
        模擬：用戶問狀態 → 模型調用 get_state → 收到狀態結果 → 模型回答
        這是多輪對話的訓練資料
        """
        
        # 狀態結果模板
        state_templates = {
            "on": {
                "states": ["on", "開啟", "亮著"],
                "responses": [
                    "{full_name}現在是開著的喵～",
                    "{full_name}是開的喔喵！",
                    "目前{full_name}是開著的狀態喵～",
                    "{full_name}開著呢喵！",
                ]
            },
            "off": {
                "states": ["off", "關閉", "關著"],
                "responses": [
                    "{full_name}現在是關著的喵～",
                    "{full_name}目前是關的喔喵！",
                    "目前{full_name}是關閉的狀態喵～",
                    "{full_name}沒開喵！",
                ]
            },
            "climate_cool": {
                "states": ["cool, {temp}°C", "冷氣模式, 設定{temp}度"],
                "responses": [
                    "{full_name}現在是冷氣模式，設定{temp}度喵～",
                    "{full_name}目前在{temp}度的冷氣模式喔喵！",
                    "目前{full_name}設定{temp}度，冷氣模式喵～",
                ]
            },
            "climate_heat": {
                "states": ["heat, {temp}°C", "暖氣模式, 設定{temp}度"],
                "responses": [
                    "{full_name}現在是暖氣模式，設定{temp}度喵～",
                    "{full_name}目前在{temp}度的暖氣模式喔喵！",
                ]
            },
            "climate_off": {
                "states": ["off"],
                "responses": [
                    "{full_name}目前是關閉的喵～",
                    "{full_name}沒有開啟喔喵！",
                ]
            },
            "locked": {
                "states": ["locked", "已鎖定"],
                "responses": [
                    "{full_name}是鎖好的喵！安心喵～",
                    "{full_name}有鎖上喔喵！",
                    "放心喵！{full_name}是鎖著的～",
                ]
            },
            "unlocked": {
                "states": ["unlocked", "未鎖定"],
                "responses": [
                    "{full_name}沒有鎖喵！需要幫你鎖上嗎？",
                    "{full_name}是開著的喔喵！可能需要鎖一下～",
                ]
            },
            "unavailable": {
                "states": ["unavailable", "無法使用"],
                "responses": [
                    "抱歉喵，{full_name}目前無法讀取狀態喵...",
                    "嗯...{full_name}好像連不上喵，可能需要檢查一下～",
                ]
            },
            "brightness": {
                "states": ["{brightness}%"],
                "responses": [
                    "{full_name}現在亮度是{brightness}%喵～",
                    "{full_name}目前設定在{brightness}%亮度喔喵！",
                ]
            },
        }
        
        device_areas = ['書房', '客廳', '臥室', '兒童房']
        device_names = {
            'light': ['大燈', '燈', '吸頂燈'],
            'fan': ['風扇', '循環扇'],
            'climate': ['冷氣', '空調'],
            'lock': ['門鎖', '大門鎖'],
        }
        
        training_data = []
        
        for _ in range(count):
            # 選擇狀態類型
            state_type = random.choice(list(state_templates.keys()))
            template_data = state_templates[state_type]
            
            # 選擇適合的設備
            if state_type in ["climate_cool", "climate_heat", "climate_off"]:
                name = random.choice(device_names['climate'])
                temp = random.randint(20, 28)
            elif state_type in ["locked", "unlocked"]:
                name = random.choice(device_names['lock'])
                temp = None
            elif state_type == "brightness":
                name = random.choice(device_names['light'])
                brightness = random.choice([20, 30, 50, 70, 80, 100])
                temp = None
            else:
                device_type = random.choice(['light', 'fan'])
                name = random.choice(device_names[device_type])
                temp = None
            
            area = random.choice(device_areas)
            full_name = f"{area}{name}"
            
            # 生成原始問題
            user_questions = [
                f"{full_name}開著嗎",
                f"{area}的{name}是什麼狀態",
                f"{full_name}現在怎樣",
                f"看看{full_name}",
            ]
            original_question = random.choice(user_questions)
            
            # 生成狀態結果（模擬 get_state 返回值）
            state_template = random.choice(template_data["states"])
            if temp:
                state_result = state_template.format(temp=temp)
            elif state_type == "brightness":
                state_result = state_template.format(brightness=brightness)
            else:
                state_result = state_template
            
            # 生成助理回應
            response_template = random.choice(template_data["responses"])
            if temp:
                response_text = response_template.format(full_name=full_name, temp=temp)
            elif state_type == "brightness":
                response_text = response_template.format(full_name=full_name, brightness=brightness)
            else:
                response_text = response_template.format(full_name=full_name)
            
            # 建立多輪對話格式
            # 模擬：用戶問 -> 模型查詢 -> 系統返回結果 -> 模型回答
            training_item = {
                "messages": [
                    {"role": "system", "content": self._get_system_prompt_v7()},
                    {"role": "user", "content": f"User request:\n{original_question}"},
                    {"role": "assistant", "content": f"我來幫你看一下喵～\nACTION get_state\nname {name}\narea {area}"},
                    {"role": "user", "content": f"State result:\n{full_name}: {state_result}"},
                    {"role": "assistant", "content": response_text}
                ]
            }
            
            training_data.append(training_item)
        
        return training_data

# ==============================================================================
# 主程式
# ==============================================================================

def main():
    print("=" * 80)
    print("訓練資料生成器 v3（英文標記 + 隨機狀態）")
    print("=" * 80)
    print()
    
    # 載入真實設備
    print("載入真實設備...")
    with open("tools/ha_exposed_devices.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    real_devices = []
    for device in data.get('devices', []):
        real_devices.append({
            'entity_id': device['entity_id'],
            'friendly_name': device['friendly_name'],
            'domain': device['domain'],
            'area': device.get('webui_area') or device.get('ha_area_id') or '未分類',
            'state': device['state']
        })
    
    print(f"✓ 載入 {len(real_devices)} 個真實設備")
    
    # 生成虛擬設備
    print("生成虛擬設備...")
    virtual_devices = generate_virtual_devices(count=VIRTUAL_DEVICES_COUNT)
    print(f"✓ 生成 {len(virtual_devices)} 個虛擬設備")
    print()
    
    # 載入 ACTION 定義
    with open("ACTIONS.yaml", 'r', encoding='utf-8') as f:
        action_spec = yaml.safe_load(f)
    
    # 生成訓練資料
    print("生成訓練資料...")
    generator = ImprovedTrainingDataGenerator(real_devices, virtual_devices, action_spec)
    
    all_training_data = []
    
    # 使用配置變數來控制各類型資料數量
    actions_to_generate = [
        ('turn_on', TURN_ON_COUNT),
        ('turn_off', TURN_OFF_COUNT)
    ]
    
    for action_name, count in actions_to_generate:
        data = generator.generate_basic_control_with_slots(action_name, count)
        all_training_data.extend(data)
        print(f"  ✓ {action_name}: {len(data)} 條")
    
    # 加入純聊天資料
    print("  ✓ 生成純聊天資料...")
    chat_data = generator.generate_chat_data(CHAT_COUNT)
    all_training_data.extend(chat_data)
    print(f"  ✓ 純聊天: {len(chat_data)} 條")
    
    # 加入 get_state 資料（v7 新架構核心功能）
    print("  ✓ 生成 get_state 資料...")
    get_state_data = generator.generate_get_state_data(GET_STATE_COUNT)
    all_training_data.extend(get_state_data)
    print(f"  ✓ get_state: {len(get_state_data)} 條")
    
    # 加入 get_state 二次對話資料
    print("  ✓ 生成 get_state 二次對話資料...")
    get_state_response_data = generator.generate_get_state_response_data(GET_STATE_RESPONSE_COUNT)
    all_training_data.extend(get_state_response_data)
    print(f"  ✓ get_state 二次對話: {len(get_state_response_data)} 條")
    
    # 加入 ACTION search 資料
    print("  ✓ 生成 ACTION search 資料...")
    search_data = generator.generate_search_action_data(SEARCH_COUNT)
    all_training_data.extend(search_data)
    print(f"  ✓ ACTION search: {len(search_data)} 條")
    
    # 加入 climate_set_mode 資料
    print("  ✓ 生成 climate_set_mode 資料...")
    climate_mode_data = generator.generate_climate_set_mode_data(CLIMATE_MODE_COUNT)
    all_training_data.extend(climate_mode_data)
    print(f"  ✓ climate_set_mode: {len(climate_mode_data)} 條")
    print()
    
    # 計算總數和比例
    total_count = len(all_training_data)
    control_count = TURN_ON_COUNT + TURN_OFF_COUNT
    get_state_total = GET_STATE_COUNT + GET_STATE_RESPONSE_COUNT
    
    print(f"總計：{total_count} 條訓練資料")
    print(f"  - 控制資料 (turn_on/off)：{control_count} 條 ({int(control_count/total_count*100)}%)")
    print(f"  - get_state（查詢+回應）：{get_state_total} 條 ({int(get_state_total/total_count*100)}%)")
    print(f"  - 純聊天：{CHAT_COUNT} 條 ({int(CHAT_COUNT/total_count*100)}%)")
    print(f"  - ACTION search：{SEARCH_COUNT} 條 ({int(SEARCH_COUNT/total_count*100)}%)")
    print(f"  - climate_set_mode：{CLIMATE_MODE_COUNT} 條 ({int(CLIMATE_MODE_COUNT/total_count*100)}%)")
    print()
    
    # 顯示範例
    print("範例訓練資料：")
    print("-" * 80)
    sample = random.choice(all_training_data)
    print(json.dumps(sample, ensure_ascii=False, indent=2))
    print("-" * 80)
    print()
    
    # 儲存
    output_file = "training_data_v7_get_state.jsonl"  # v7: 新增 get_state 架構
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✓ 已儲存到：{output_file}")
    print()
    print("=" * 80)
    print("完成！")
    print("=" * 80)


if __name__ == "__main__":
    try:
        import pypinyin
    except ImportError:
        print("請先安裝：pip install pypinyin")
        exit(1)
    
    main()
