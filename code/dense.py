from tqdm import tqdm, trange
import argparse
import random
import torch
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, AdamW, TrainingArguments, get_linear_schedule_with_warmup
import numpy as np
import os
import json
from datasets import DatasetDict, load_from_disk, load_metric
from transformers import AutoTokenizer
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset, SequentialSampler)
import faiss
import wandb

wandb.init(project="project4", entity="sanggang", name='base')

torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
np.random.seed(2023)
random.seed(2023)



class BertEncoder(BertPreTrainedModel):
  def __init__(self, config):
    super(BertEncoder, self).__init__(config)

    self.bert = BertModel(config)
    self.init_weights()

  def forward(self, input_ids,
              attention_mask=None, token_type_ids=None):
      outputs = self.bert(input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)
      pooled_output = outputs[1]
      return pooled_output


def train(args, dataset, p_model, q_model):
  
  wandb.config = {
      "learning_rate": args.learning_rate,
      "epochs": args.num_train_epochs,
      "batch_size": args.per_device_train_batch_size,
      "weight_decay": args.weight_decay,
    }

  # Dataloader
  train_sampler = RandomSampler(dataset)
  train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)

  # Optimizer
  no_decay = ['bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
        {'params': [p for n, p in p_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in p_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in q_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in q_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
  t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

  # Start training!
  global_step = 0

  p_model.zero_grad()
  q_model.zero_grad()
  torch.cuda.empty_cache()

  train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

  for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")

    for step, batch in enumerate(epoch_iterator):
      q_encoder.train()
      p_encoder.train()

      if torch.cuda.is_available():
        batch = tuple(t.cuda() for t in batch)

      p_inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2]
                  }

      q_inputs = {'input_ids': batch[3],
                  'attention_mask': batch[4],
                  'token_type_ids': batch[5]}

      p_outputs = p_model(**p_inputs)  # (batch_size, emb_dim)
      q_outputs = q_model(**q_inputs)  # (batch_size, emb_dim)


      # Calculate similarity score & loss
      sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))  # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)

      # target: position of positive samples = diagonal element
      targets = torch.arange(0, args.per_device_train_batch_size).long()
      if torch.cuda.is_available():
        targets = targets.to('cuda')

      sim_scores = F.log_softmax(sim_scores, dim=1)

      loss = F.nll_loss(sim_scores, targets)

      wandb.log({"loss": loss.item(), "sim_scores_mean": sim_scores, "step": global_step})

      loss.backward()
      optimizer.step()
      scheduler.step()
      q_model.zero_grad()
      p_model.zero_grad()
      global_step += 1

      torch.cuda.empty_cache()
  return p_model, q_model

args = TrainingArguments(
    output_dir="dense_retireval",
    evaluation_strategy="epoch",
    learning_rate=0.0003,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.0478
)


model_checkpoint = 'klue/bert-base'

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

datasets = load_from_disk('./data/train_dataset')
training_dataset = datasets['train']

q_seqs = tokenizer(training_dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
p_seqs = tokenizer(training_dataset['context'], padding="max_length", truncation=True, return_tensors='pt')

train_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],
                        q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])


# load pre-trained model on cuda (if available)
p_encoder = BertEncoder.from_pretrained(model_checkpoint)
q_encoder = BertEncoder.from_pretrained(model_checkpoint)

if torch.cuda.is_available():
  p_encoder.cuda()
  q_encoder.cuda()

p_encoder, q_encoder = train(args, train_dataset, p_encoder, q_encoder)
p_encoder.save_pretrained('./output/p_model')
q_encoder.save_pretrained('./output/q_model')

# search_corpus = list(set([example['context'] for example in datasets['validation']]))
# eval_batch_size = 8

# # Construt dataloader
# valid_p_seqs = tokenizer(search_corpus, padding="max_length", truncation=True, return_tensors='pt')
# valid_dataset = TensorDataset(valid_p_seqs['input_ids'], valid_p_seqs['attention_mask'], valid_p_seqs['token_type_ids'])
# valid_sampler = SequentialSampler(valid_dataset)
# valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=eval_batch_size)

# # Inference using the passage encoder to get dense embeddeings
# p_embs = []

# with torch.no_grad():

#   epoch_iterator = tqdm(valid_dataloader, desc="Iteration", position=0, leave=True)
#   p_encoder.eval()

#   for _, batch in enumerate(epoch_iterator):
#     batch = tuple(t.cuda() for t in batch)

#     p_inputs = {'input_ids': batch[0],
#                 'attention_mask': batch[1],
#                 'token_type_ids': batch[2]
#                 }

#     outputs = p_encoder(**p_inputs).to('cpu').numpy()
#     p_embs.extend(outputs)

# p_embs = np.array(p_embs)
# p_embs.shape  # (num_passage, emb_dim)

# valid_q_seqs = tokenizer(query, padding="max_length", truncation=True, return_tensors='pt').to('cuda')

# with torch.no_grad():
#   q_encoder.eval()
#   q_embs = q_encoder(**valid_q_seqs).to('cpu').numpy()

# torch.cuda.empty_cache()


# num_clusters = 16
# niter = 5
# k = 5

# # 1. Clustering
# emb_dim = p_embs.shape[-1]
# index_flat = faiss.IndexFlatL2(emb_dim)

# clus = faiss.Clustering(emb_dim, num_clusters)
# clus.verbose = True
# clus.niter = niter
# clus.train(p_embs, index_flat)
# centroids = faiss.vector_float_to_array(clus.centroids)
# centroids = centroids.reshape(num_clusters, emb_dim)

# quantizer = faiss.IndexFlatL2(emb_dim)
# quantizer.add(centroids)

# # 2. SQ8 + IVF indexer (IndexIVFScalarQuantizer)
# indexer = faiss.IndexIVFScalarQuantizer(quantizer, quantizer.d, quantizer.ntotal, faiss.METRIC_L2)
# indexer.train(p_embs)
# indexer.add(p_embs)

# D, I = indexer.search(q_embs, k)

# for i, q in enumerate(query[:1]):
#   print("[Search query]\n", q, "\n")
#   print("[Ground truth passage]")
#   print(ground_truth[i], "\n")

#   d = D[i]
#   i = I[i]
#   for j in range(k):
#     print("Top-%d passage with distance %.4f" % (j+1, d[j]))
#     print(search_corpus[i[j]])
#   print('\n')
