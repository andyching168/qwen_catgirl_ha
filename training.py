#!/usr/bin/env python3
"""
Qwen 貓娘訓練腳本（最簡化版本）
"""

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct"
TRAIN_FILE = "training_data_v7_get_state.jsonl"
OUTPUT_DIR = "./qwen-catgirl-ha-switch-v2"
MAX_SEQ_LENGTH = 1280

print("🦥 Unsloth 訓練")
print("=" * 80)

# 載入模型
print("載入模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
print("✓ 模型載入完成\n")

# 配置 LoRA
print("配置 LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
print("✓ LoRA 配置完成\n")

# 載入資料
print("載入訓練資料...")
dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
print(f"✓ 訓練資料：{len(dataset)} 條\n")

# 預處理：直接轉成文字
print("預處理資料...")
def convert_to_text(example):
    """直接轉成 ChatML 格式的文字"""
    messages = example["messages"]
    text = ""
    for msg in messages:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    return {"text": text}

dataset = dataset.map(convert_to_text, remove_columns=["messages"])
print("✓ 預處理完成\n")

# 顯示範例
print("範例資料（前 300 字元）：")
print("-" * 80)
print(dataset[0]["text"][:300])
print("...")
print("-" * 80)
print()

# 訓練
print("開始訓練...")
print("=" * 80)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        optim="adamw_8bit",
        report_to="none",
        seed=42,
    ),
)

trainer.train()

# 儲存
print("\n儲存模型...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✓ 模型已儲存到：{OUTPUT_DIR}")

# 合併模型
print("合併模型...")
try:
    model.save_pretrained_merged(f"{OUTPUT_DIR}_merged", tokenizer, save_method="merged_16bit")
    print(f"✓ 合併模型已儲存到：{OUTPUT_DIR}_merged")
except:
    print("⚠ 合併失敗（可跳過）")

print("\n" + "=" * 80)
print("✓ 訓練完成！")
print("=" * 80)

