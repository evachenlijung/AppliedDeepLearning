import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import json, logging
import traceback
import torch, numpy as np, random, json
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.util import mine_hard_negatives
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--retriever_dir", type=str, default="retriever")
argparser.add_argument("--reranker_dir", type=str, default="reranker")
argparser.add_argument("--n_epoch", type=int, default=1)
args = argparser.parse_args()

retriever_dir = os.path.normpath(args.retriever_dir)
reranker_dir = os.path.normpath(args.reranker_dir)
n_epoch = args.n_epoch

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model_name = "cross-encoder/ms-marco-MiniLM-L12-v2"
train_batch_size = 128
num_epochs = 1
num_rand_negatives = 4

device = "cuda" if torch.cuda.is_available() else "cpu"
retriever = SentenceTransformer(retriever_dir, device=device)
model = CrossEncoder(model_name)

# path setting
train_data_path = os.path.normpath(r"data/train.json")
figure_dir = os.path.normpath(r"figure")

os.makedirs(reranker_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)

# data preparation
with open(train_data_path, 'r', encoding="utf-8") as f:
    train_data = json.load(f)

# train_data = random.sample(train_data, 512)

pairs = []
for qa in train_data:
    q = qa["rewrite"].strip()
    pos_passages = [p for p, l in zip(qa["evidences"], qa["retrieval_labels"]) if l == 1]
    if len(pos_passages) == 0:
        continue
    for pos in pos_passages:
        pairs.append({
            "query": q,
            "answer": pos,
        })

train_dataset = Dataset.from_list(pairs)

dataset = mine_hard_negatives(
    dataset=train_dataset,
    model=retriever,
    range_min=10,
    range_max=50,
    max_score=0.8,
    relative_margin=0.05,
    num_negatives=num_rand_negatives,
    sampling_strategy="random",
    batch_size=256,
    use_faiss=True,
)

loss = CachedMultipleNegativesRankingLoss(
    model=model,
    num_negatives=num_rand_negatives,
    mini_batch_size=64,
)

args = CrossEncoderTrainingArguments(
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,
    bf16=True,
    eval_strategy="no",
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=50,
    logging_first_step=True,
)

trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    loss=loss,
)
trainer.train()

model.save_pretrained(reranker_dir)



# figure
logs = trainer.state.log_history
losses = [line["loss"] for line in logs if "loss" in line]
steps = [line["step"] for line in logs if "loss" in line]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.figure(figsize=(8,5))
plt.plot(losses, label="Train Loss", color="b", alpha=0.6)

plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
figure_path = os.path.join(figure_dir, f"reranker_loss_{timestamp}.png")
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
print(f"📈 Loss curve saved to {figure_path}")