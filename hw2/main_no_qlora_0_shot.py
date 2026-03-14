import gc, torch
import os, json, subprocess
import random
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, 
)
from utils_0_shot import get_prompt, get_bnb_config


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


offload_dir = os.path.normpath(r"offload")
os.makedirs(offload_dir, exist_ok=True)

model_name = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder=offload_dir,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16, 
    load_in_4bit=False
)

model.config.pad_token_id = tokenizer.pad_token_id

model.config.use_cache = False
model.enable_input_require_grads() 
model.gradient_checkpointing_enable()



data_files = {
    "test": "data/private_test.json",
}
dataset = load_dataset("json", data_files=data_files)
 
test_dataset = dataset["test"]


# random sampling
# train_size = 8
# test_size = 4

# train_dataset = train_dataset.shuffle(seed = random.randint(0, len(train_dataset))).select(range(train_size))
# train_dataset = [x for x in train_dataset]

# test_dataset = test_dataset.shuffle(seed=random.randint(0, len(test_dataset))).select(range(test_size))
# test_dataset = [x for x in test_dataset]


# copy ids and instructions of test data
test_ids = [x["id"] for x in test_dataset]
test_instructions = [x["instruction"] for x in test_dataset]


test_prompts = [get_prompt(x["instruction"]) for x in test_dataset]



model.eval()
predictions = []

with torch.no_grad():
    for i, prompt in enumerate(tqdm(test_prompts)):
        model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            use_cache=True,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        content = tokenizer.decode(output_ids, skip_special_tokens=True)

        predictions.append({
            "id": test_ids[i],
            "instruction": test_instructions[i],
            "output": content
        })



output_dir = os.path.normpath(r"output_no_qlora_0_shot")
os.makedirs(output_dir, exist_ok=True)

filepath = os.path.join(output_dir, "r14922112_output.json")

with open(filepath, "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)