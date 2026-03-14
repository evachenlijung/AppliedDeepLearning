import gc, torch
import os, json, subprocess
import random
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel
from utils import get_prompt, get_bnb_config
import argparse


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path_to_model_checkpoint_folder",
    type=str,
    default="Qwen/Qwen3-4B",
    help="The path to the base model.",
)
parser.add_argument(
    "--path_to_adapter_checkpoint_folder",
    type=str,
    required=True,
    help="Path to the saved PEFT checkpoint folder.",
)
parser.add_argument(
    "--test_data_path",
    type=str,
    default="",
    required=True,
    help="Path to test data.",
)
parser.add_argument(
    "--prediction_output_path",
    type=str,
    default="",
    required=True,
    help="Path to test data.",
)
args = parser.parse_args()

offload_dir = os.path.normpath(r"offload")
os.makedirs(offload_dir, exist_ok=True)
model_name = "Qwen/Qwen3-4B"
bnb_config = get_bnb_config()

if args.path_to_model_checkpoint_folder:
    model = AutoModelForCausalLM.from_pretrained(
        args.path_to_model_checkpoint_folder,
        quantization_config=bnb_config,
        device_map="auto",
        offload_folder=offload_dir,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        offload_folder=offload_dir,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

# Load LoRA
model = PeftModel.from_pretrained(model, args.path_to_adapter_checkpoint_folder)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.enable_input_require_grads() 
model.gradient_checkpointing_enable()



data_files = {
    "test": f"{os.path.normpath(args.test_data_path)}",
}
dataset = load_dataset("json", data_files=data_files)
test_dataset = dataset["test"]


# # random sampling
# test_size = 4

# test_dataset = test_dataset.shuffle(seed=random.randint(0, len(test_dataset))).select(range(test_size))
# test_dataset = [x for x in test_dataset]


# copy ids and instructions of test data
test_ids = [x["id"] for x in test_dataset]
test_instructions = [x["instruction"] for x in test_dataset]

test_prompts = [get_prompt(x["instruction"]) for x in test_dataset]



output_path = os.path.normpath(f"{args.prediction_output_path}")

# inference
model.eval()
predictions = []

with torch.no_grad():
    for i, prompt in enumerate(tqdm(test_prompts)):
        model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=64,
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

output_dir = os.path.dirname(output_path)
os.makedirs(output_dir, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)