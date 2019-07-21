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
        self.v_embeddings = nn.Embedding(voc_size, emb_dimension, sparse=True)
        self.init_emb()

    def init_emb(self):

        initrange = 0.5 / self.emb_dimension
        self.d_embeddings.weight.data.uniform_(-initrange, initrange)
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, doc_u, pos_v, neg_v):

        emb_u = self.u_embeddings(doc_u)
        emb_v = self.v_embeddings(pos_v)
        score = torch.matmul(emb_u, torch.transpose(emb_v, 0, 1))
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        neg_emb_v = self.v_embeddings(neg_v)
        #emb_u = emb_u.squeeze(0).repeat(neg_v.shape[0],1)
        neg_score = torch.matmul(neg_emb_v, torch.transpose(emb_u, 0, 1))
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)
        return -1 * (torch.sum(score)+torch.sum(neg_score))