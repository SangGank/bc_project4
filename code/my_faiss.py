import faiss
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
import pickle


from torch.utils.data import (DataLoader, RandomSampler, TensorDataset, SequentialSampler)
from tqdm import tqdm, trange
 

class BertEncoder(BertPreTrainedModel):

    def __init__(self, config):
        super(BertEncoder, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()
      
      
    def forward(
            self,
            input_ids, 
            attention_mask=None,
            token_type_ids=None
        ): 
  
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]
        return pooled_output
    
class FaissRetrieval:
    def __init__(self, p_embs, num_clusters=16):

        """
        Arguments:
            p_embs (torch.Tensor):
                위에서 사용한 Passage Encoder로 구한
                전체 Passage들의 Dense Representation을 받아옵니다.
                
        Summary:
            초기화하는 부분
            `build_faiss` 메소드도 여기서 수행하면 좋을 것 같습니다.
        """
        
        self.p_embs = p_embs
        self.build_faiss(num_clusters=num_clusters)

    def build_faiss(self, num_clusters=16):

        """
        Note:
            위에서 Faiss를 사용했던 기억을 떠올려보면,
            Indexer를 구성해서 .search() 메소드를 활용했습니다.
            여기서는 Indexer 구성을 해주도록 합시다.
        """

        emb_dim = self.p_embs.shape[-1]

        quantizer = faiss.IndexFlatL2(emb_dim)
        self.indexer = faiss.IndexIVFScalarQuantizer(
            quantizer,
            quantizer.d,
            num_clusters,
            faiss.METRIC_L2,
        )
        self.indexer.train(self.p_embs)
        self.indexer.add(self.p_embs)

    def get_relevant_doc(self, q_emb, k=1):
        """
        Arguments:
            query (torch.Tensor):
                Dense Representation으로 표현된 query를 받습니다.
                문자열이 아님에 주의합시다.
            k (int, default=1):
                상위 몇 개의 유사한 passage를 뽑을 것인지 결정합니다.

        Note:
            받은 query를 이 객체에 저장된 indexer를 활용해서
            유사한 문서를 찾아봅시다.
        """
        
        q_emb = q_emb.astype(np.float32)
        D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]
    

data_path = "./data/"
context_path = "remove_punc2.json"
with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
    dataset = json.load(f)


search_corpus = list(set([example['text'] for example in dataset.values()]))
print(f'총 {len(search_corpus)}개의 지문이 있습니다.')







model_checkpoint = 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


p_encoder = BertEncoder.from_pretrained(model_checkpoint).to("cuda")
q_encoder = BertEncoder.from_pretrained(model_checkpoint).to("cuda")


# eval_batch_size = 8

# # Construt dataloader
# valid_p_seqs = tokenizer(
#     search_corpus,
#     padding="max_length",
#     truncation=True,
#     return_tensors="pt"
# )
# valid_dataset = TensorDataset(
#     valid_p_seqs["input_ids"],
#     valid_p_seqs["attention_mask"],
#     valid_p_seqs["token_type_ids"]
# )
# valid_sampler = SequentialSampler(valid_dataset)
# valid_dataloader = DataLoader(
#     valid_dataset,
#     sampler=valid_sampler,
#     batch_size=eval_batch_size
# )

# # Inference using the passage encoder to get dense embeddeings
# p_embs = []
# with torch.no_grad():

#     epoch_iterator = tqdm(
#         valid_dataloader,
#         desc="Iteration",
#         position=0,
#         leave=True
#     )
#     p_encoder.eval()

#     for _, batch in enumerate(epoch_iterator):
#         batch = tuple(t.cuda() for t in batch)

#         p_inputs = {
#             "input_ids": batch[0],
#             "attention_mask": batch[1],
#             "token_type_ids": batch[2]
#         }
        
#         outputs = p_encoder(**p_inputs).to("cpu").numpy()
#         p_embs.extend(outputs)

# p_embs = np.array(p_embs)
# np.save('p_emb_mult2.npy', p_embs)
p_embs = np.load('p_emb_mult2.npy')

# with open('p_emb.pkl', 'wb') as f:
#     pickle.dump(p_embs, f)
# with open('./data/p_emb.pkl', 'rb') as f:
#     p_embs = pickle.load(f)


query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

valid_q_seqs = tokenizer(query, padding="max_length", truncation=True, return_tensors="pt").to("cuda")

with torch.no_grad():
    q_encoder.eval()
    q_embs = q_encoder(**valid_q_seqs).to("cpu").numpy()

# p_embs는 처음에 만든 embedding을 이용합시다.
retriever = FaissRetrieval(p_embs,num_clusters = 1400)
retriever.build_faiss()
results = retriever.get_relevant_doc(q_embs, k=10)

torch.cuda.empty_cache()

print("[Search query]\n", query, "\n")

print(f"Top-{len(results[0])} passages")
print(results[1])
for d, i in zip(*results):
    print(d)
    # if d>20:
    #     break
    print(f"Distance {d:.5f} | Passage {i}")
    print(search_corpus[i])
print('\n')