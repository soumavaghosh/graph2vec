import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):

    def __init__(self, doc_size, voc_size, emb_dimension):

        super(SkipGramModel, self).__init__()
        self.voc_size = voc_size
        self.doc_size = doc_size
        self.emb_dimension = emb_dimension
        self.d_embeddings = nn.Embedding(doc_size, emb_dimension, sparse=True)
        self.u_embeddings = nn.Embedding(voc_size, emb_dimension, sparse=True)
        self.init_emb()

    def init_emb(self):

        initrange = 0.5 / self.emb_dimension * 10
        self.d_embeddings.weight.data.uniform_(-initrange, initrange)
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, doc_u, pos_v):

        emb_d = self.d_embeddings(doc_u)
        emb_v = self.u_embeddings(pos_v)
        score = torch.matmul(emb_d, torch.transpose(emb_v, 0, 1))
        score = F.log_softmax(score, dim=1)

        return score