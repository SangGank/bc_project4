from tqdm import tqdm, trange
import argparse
import random
import torch
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, AdamW, TrainingArguments, get_linear_schedule_with_warmup
import numpy as np
import os
import json

torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
np.random.seed(2023)
random.seed(2023)

# Use subset (10 example) of original training dataset
sample_idx = np.random.choice(range(len(dataset['train'])), 20)
training_dataset = dataset['train'][sample_idx]
data_path = "../data/"
context_path = "wikipedia_documents.json"
with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
    wiki = json.load(f)
contexts = list(
    dict.fromkeys([v["text"] for v in wiki.values()])
    )   