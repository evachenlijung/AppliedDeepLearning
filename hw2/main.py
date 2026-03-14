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
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import get_prompt, get_bnb_config


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


offload_dir = os.path.normpath(r"offload")
os.makedirs(offload_dir, exist_ok=True)

bnb_config = get_bnb_config()

model_name = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    offload_folder=offload_dir,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

model.config.pad_token_id = tokenizer.pad_token_id

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

model.config.use_cache = False
model.enable_input_require_grads() 
model.gradient_checkpointing_enable()



data_files = {
    "train": "data/train.json",
    "validation": "data/public_test.json",
    "test": "data/private_test.json",
}
dataset = load_dataset("json", data_files=data_files)



# train/val/test
train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]


# # random sampling
# train_size = 16
# test_size = 4

# train_dataset = train_dataset.shuffle(seed = random.randint(0, len(train_dataset))).select(range(train_size))
# train_dataset = [x for x in train_dataset]

# test_dataset = test_dataset.shuffle(seed=random.randint(0, len(test_dataset))).select(range(test_size))
# test_dataset = [x for x in test_dataset]


# copy ids and instructions of test data
test_ids = [x["id"] for x in test_dataset]
test_instructions = [x["instruction"] for x in test_dataset]

# 生成 prompt
train_prompts = [get_prompt(x["instruction"]) for x in train_dataset]
train_outputs = [x["output"] for x in train_dataset]

val_prompts = [get_prompt(x["instruction"]) for x in val_dataset]
val_outputs = [x["output"] for x in val_dataset]

test_prompts = [get_prompt(x["instruction"]) for x in test_dataset]



def prepare_lm_input(prompts, outputs, max_length=512):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    for p, o in zip(prompts, outputs):
        prompt_tokens = tokenizer(p, add_special_tokens=False)["input_ids"]
        output_tokens = tokenizer(o, add_special_tokens=False)["input_ids"]

        # 拼接
        input_ids = prompt_tokens + output_tokens
        input_ids = input_ids[:max_length]
        attention_mask = [1] * len(input_ids)

        # label: prompt 部分設成 -100
        labels = [-100] * len(prompt_tokens) + output_tokens
        labels = labels[:max_length]

        # padding
        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            labels += [-100] * pad_len

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": torch.tensor(input_ids_list),
        "attention_mask": torch.tensor(attention_mask_list),
        "labels": torch.tensor(labels_list),
    }


train_encodings = prepare_lm_input(train_prompts, train_outputs)
val_encodings = prepare_lm_input(val_prompts, val_outputs)



class LMDataset(Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = encodings["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }
    
train_data = LMDataset(train_encodings)
val_data = LMDataset(val_encodings)



example_idx = 0
decoded_input = tokenizer.decode(train_data.input_ids[example_idx], skip_special_tokens=True)
decoded_label = tokenizer.decode(
    [x for x in train_data.labels[example_idx] if x != -100],
    skip_special_tokens=True
)

output_dir = os.path.normpath(r"output")
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    do_train=True, 
    do_eval=True,
    do_predict=True,

    # batch setting
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,
    bf16=True,
    gradient_checkpointing=True,

    # training setting 
    num_train_epochs=1,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_ratio=0.03,

    # validation and saving
    eval_strategy="steps",
    eval_steps=10,
    save_strategy="steps",
    save_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False, 

    # log
    logging_strategy="steps",
    logging_steps=10,

    # CPU / dataloader
    dataloader_num_workers=12, 
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()



# plot learning curve on public testing set
logs = trainer.state.log_history

train_steps = [x["step"] for x in logs if "loss" in x]
train_loss = [x["loss"] for x in logs if "loss" in x]

eval_steps = [x["step"] for x in logs if "eval_loss" in x]
eval_loss = [x["eval_loss"] for x in logs if "eval_loss" in x]

plt.figure(figsize=(8, 5))
plt.plot(train_steps, train_loss, label="Training Loss")
plt.plot(eval_steps, eval_loss, label="Validation (Public Test) Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Learning Curve on Public Testing Set")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "learning_curve.png"))
  


# save model
final_model_path = os.path.normpath(r"final_model")
os.makedirs(final_model_path, exist_ok=True)
trainer.save_model(final_model_path)



# inference
model.eval()
predictions = []

with torch.no_grad():
    for i, prompt in enumerate(tqdm(test_prompts)):
        model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
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



filepath = os.path.join(output_dir, "r14922112_output.json")

with open(filepath, "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)


# final performance of the model
metrics = trainer.evaluate(eval_dataset=val_data)
print(f"Final Evaluation Loss: {metrics['eval_loss']:g}")

# test perplexity
checkpoints = [d for d in os.listdir("output") if "checkpoint" in d]
checkpoints.sort(key=lambda x: int(x.split("-")[1]))
last_checkpoint = checkpoints[-1]

cmd = [
    "python",
    "ppl.py",
    "--base_model_path", final_model_path,
    "--peft_path", f'{os.path.join(output_dir, last_checkpoint)}',
    "--test_data_path", f'{filepath}'
]

result = subprocess.run(cmd, capture_output=True, text=True)

print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)