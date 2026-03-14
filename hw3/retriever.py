import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch, numpy as np, random, json
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from datasets import Dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer,
    losses,
)
from sentence_transformers.training_args import BatchSamplers
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--retriever_dir", type=str, default="retriever")
argparser.add_argument("--n_epoch", type=int, default=1)
args = argparser.parse_args()

retriever_dir = os.path.normpath(args.retriever_dir)
n_epoch = args.n_epoch

# model setting
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "intfloat/multilingual-e5-small"
model = SentenceTransformer(model_name)

# path setting
train_data_path = os.path.normpath(r"data/train.json")
figure_dir = os.path.normpath(r"figure")

os.makedirs(retriever_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)

# data preparation
with open(train_data_path, 'r', encoding="utf-8") as f:
    train_data = json.load(f)

# debug
# train_data = random.sample(train_data, 512)

anchors = []
positives = []
for qa in train_data:
    q = qa["rewrite"].strip()
    pos_passages = [p for p, l in zip(qa["evidences"], qa["retrieval_labels"]) if l == 1]
    if len(pos_passages) == 0:
        continue
    for pos in pos_passages:
        anchors.append(q)
        positives.append(pos)

# MNR loss
train_dataset = Dataset.from_dict({
    "anchor": anchors,
    "positive": positives,
})
loss = losses.MultipleNegativesRankingLoss(model)

args = SentenceTransformerTrainingArguments(
    num_train_epochs=n_epoch,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    bf16=True, 
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    logging_steps=50,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
)
trainer.train()
trainer.save_model(output_dir=retriever_dir)

mnr_logs = trainer.state.log_history
mnr_losses = [line["loss"] for line in mnr_logs if "loss" in line]
mnr_steps = [line["step"] for line in mnr_logs if "loss" in line]


# figure
plt.plot(
    mnr_steps, mnr_losses, 
    label="MNR Loss", 
    color='orange', 
    alpha=0.8
)

plt.title("Retriever Training Loss: MNR")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
figure_name = os.path.join(figure_dir, f"retriever_loss_{timestamp}")
plt.savefig(figure_name, dpi=300, bbox_inches='tight')
print(f"📈 Loss curve saved to {figure_name}")