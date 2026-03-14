from transformers import BitsAndBytesConfig, AutoTokenizer
import torch
import random
from datasets import load_dataset

data_files = {
    "train": "data/train.json",
}
dataset = load_dataset("json", data_files=data_files)
train_data = dataset["train"]

model_name = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_prompt(instruction) -> str:
    examples = train_data.shuffle(seed=random.randint(0, len(train_data))).select(range(4))
    examples = [x for x in examples]

    examples_text = ""
    for i, ex in enumerate(examples, 1):
        examples_text += (
            f"範例{i}：\n"
            f"指令：{ex['instruction']}\n"
            f"輸出：{ex['output']}\n\n"
        )

    prompt = (
            "請根據以下範例進行文言文與白話文互譯：\n\n"
            f"{examples_text}"
            "請翻譯以下內容：\n"
            f"{instruction}\n\n"
            "請輸出翻譯結果："
        )
    return prompt

# def get_prompt(instruction) -> str:
#     prompt = (
#             "角色：你是精通繁體中文的文言文與白話文雙向翻譯的大學中文系教授。\n"
#             "任務：若輸入為文言文，翻譯成白話文；若輸入為白話文，翻譯為文言文。\n"
#             f"{instruction}\n"
#         )
#     return prompt

def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
