import pickle
import numpy as np
from collections import Counter
from model import SkipGramModel
import torch.optim as optim
from torch import nn
import torch

dataset = "NCI1"

with open('./'+dataset+'/graph_voc_2.json', 'rb') as f:
    graph_enc = pickle.load(f)

sub_graph_voc = []
for g in list(graph_enc.keys()):
    sub_graph_voc.extend(graph_enc[g])

min_cnt = 5

sub_graph_vocab = dict(Counter(sub_graph_voc))
sub_graph_vocab = {i:sub_graph_vocab[i] for i in list(sub_graph_vocab.keys()) if sub_graph_vocab[i]>=min_cnt}

for g in list(graph_enc.keys()):
    graph_enc[g] = [x for x in graph_enc[g] if x in list(sub_graph_vocab.keys())]

id_to_sub_graph = {i:list(sub_graph_vocab.keys())[i] for i in range(len(sub_graph_vocab))}
sub_graph_to_id = {id_to_sub_graph[i]:i for i in list(id_to_sub_graph.keys())}

model_1 = SkipGramModel(len(graph_enc), len(id_to_sub_graph), 512)

def init_sample_table():
    sample_table = []
    sample_table_size = 1e8
    pow_frequency = np.array(list(sub_graph_vocab.values())) ** 0.75
    words_pow = sum(pow_frequency)
    ratio = pow_frequency / words_pow
    count = np.round(ratio * sample_table_size)
    for wid, c in enumerate(count):
        sample_table += [sub_graph_to_id[list(sub_graph_vocab.keys())[wid]]] * int(c)
    sample_table = np.array(sample_table)
    return sample_table

sample_table = init_sample_table()
neg_count = 2
epoch = 20000000

opt = optim.SGD(model_1.parameters(), lr=0.005)
model_1.train()

for i in range(epoch):
    opt.zero_grad()

    doc_id = np.random.randint(1, len(graph_enc))
    if len(graph_enc[doc_id])==0:
        continue
    doc_u = torch.tensor([doc_id], dtype=torch.long, requires_grad=False)

    pos_v = [sub_graph_to_id[x] for x in graph_enc[doc_id]]

    for p in pos_v:

        while(True):
            neg_v = np.random.choice(sample_table, size=(neg_count)).tolist()
            if p not in neg_v:
                break

        pos = torch.tensor([p], dtype=torch.long, requires_grad=False)
        neg_v = torch.tensor(neg_v, dtype=torch.long, requires_grad=False)

        loss_val = model_1(doc_u, pos, neg_v)

        print(loss_val)
        loss_val.backward()
        opt.step()

print('Completed')