import pickle
import numpy as np
from collections import Counter
from model import SkipGramModel
import torch.optim as optim
import torch

dataset = "NCI1"

with open('./'+dataset+'/graph_voc_2.json', 'rb') as f:
    graph_enc = pickle.load(f)

sub_graph_vocab = []
for g in list(graph_enc.keys()):
    sub_graph_vocab.extend(graph_enc[g])

sub_graph_vocab = dict(Counter(sub_graph_vocab))
id_to_sub_graph = {i+1:list(sub_graph_vocab.keys())[i] for i in range(len(sub_graph_vocab))}
sub_graph_to_id = {id_to_sub_graph[i]:i for i in list(id_to_sub_graph.keys())}

model_1 = SkipGramModel(len(graph_enc), len(id_to_sub_graph), 512)
#
# doc_u = torch.tensor([1], dtype=torch.long, requires_grad=False)
# pos_v = torch.tensor([2], dtype=torch.long, requires_grad=False)
# neg_v = torch.tensor([6,7], dtype=torch.long, requires_grad=False)
#
# loss = model_1(doc_u, pos_v, neg_v)

def init_sample_table():
    sample_table = []
    sample_table_size = 1e8
    pow_frequency = np.array(list(sub_graph_vocab.values())) ** 0.75
    words_pow = sum(pow_frequency)
    ratio = pow_frequency / words_pow
    count = np.round(ratio * sample_table_size)
    for wid, c in enumerate(count):
        sample_table += [wid] * int(c)
    sample_table = np.array(sample_table)
    return sample_table

sample_table = init_sample_table()
neg_count = 4
epoch = 20000

opt = optim.SGD(model_1.parameters(), lr=0.01)
model_1.train()

for i in range(epoch):
    opt.zero_grad()

    doc_id = np.random.randint(1, len(graph_enc))
    doc_u = torch.tensor([doc_id], dtype=torch.long, requires_grad=False)

    pos_v = [np.random.randint(1, len(graph_enc[doc_id]))]
    pos_v = torch.tensor(pos_v, dtype=torch.long, requires_grad=False)

    neg = np.random.choice(sample_table, size=(neg_count)).tolist()
    neg_v = torch.tensor(neg, dtype=torch.long, requires_grad=False)

    loss = model_1(doc_u, pos_v, neg_v)
    print(loss)
    loss.backward()
    opt.step()

print('Completed')