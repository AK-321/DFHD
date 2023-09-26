from .SetTransformer import SAB
from .gnn import GNNGraph

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import GraphConvolution

class Attention(torch.nn.Module):
    def __init__(self, Qdim, Kdim, mid_dim, *args, **kwargs):
        super(Attention, self).__init__(*args, **kwargs)
        self.model_dim = mid_dim
        self.Qdense = torch.nn.Linear(Qdim, mid_dim)
        self.Kdense = torch.nn.Linear(Kdim, mid_dim)

    def forward(self, main_feat, other_feat, fix_feat, mask=None):
        Q = self.Qdense(main_feat)
        K = self.Kdense(other_feat)
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)

        if mask is not None:
            Attn = torch.masked_fill(Attn, mask, -(1 << 32))

        Attn = torch.softmax(Attn, dim=-1)
        fix_feat = torch.diag(fix_feat)
        other_feat = torch.matmul(fix_feat, other_feat)
        O = torch.matmul(Attn, other_feat)

        return O

class AdjAttenAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, mid_dim, *args, **kwargs):
        super(AdjAttenAgger, self).__init__(*args, **kwargs)
        self.model_dim = mid_dim
        self.Qdense = torch.nn.Linear(Qdim, mid_dim)
        self.Kdense = torch.nn.Linear(Kdim, mid_dim)

    def forward(self, main_feat, other_feat, fix_feat, mask=None):
        Q = self.Qdense(main_feat)
        K = self.Kdense(other_feat)
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)

        if mask is not None:
            Attn = torch.masked_fill(Attn, mask, -(1 << 32))

        Attn = torch.softmax(Attn, dim=-1)
        fix_feat = torch.diag(fix_feat)
        other_feat = torch.matmul(fix_feat, other_feat)
        O = torch.matmul(Attn, other_feat)

        return O

class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, ehr_adj, ddi_adj, device=torch.device('cpu')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        ehr_adj = self.normalize(ehr_adj + np.eye(ehr_adj.shape[0]))
        ddi_adj = self.normalize(ddi_adj + np.eye(ddi_adj.shape[0]))

        self.ehr_adj = torch.FloatTensor(ehr_adj).to(device)
        self.ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)
        self.gcn3 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        ehr_node_embedding = self.gcn1(self.x, self.ehr_adj)
        ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)

        ehr_node_embedding = F.relu(ehr_node_embedding)
        ddi_node_embedding = F.relu(ddi_node_embedding)
        ehr_node_embedding = self.dropout(ehr_node_embedding)
        ddi_node_embedding = self.dropout(ddi_node_embedding)

        ehr_node_embedding = self.gcn2(ehr_node_embedding, self.ehr_adj)
        ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
        return ehr_node_embedding, ddi_node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class DFHDModel(torch.nn.Module):
    def __init__(
        self, global_para, substruct_para, emb_dim, voc_size, ehr_adj, ddi_adj,
        substruct_num, global_dim, substruct_dim, use_embedding=False,
        device=torch.device('cpu'), dropout=0.5, *args, **kwargs
    ):
        super(DFHDModel, self).__init__(*args, **kwargs)
        self.device = device
        self.use_embedding = use_embedding

        if self.use_embedding:
            self.substruct_emb = torch.nn.Parameter(
                torch.zeros(substruct_num, emb_dim)
            )
        else:
            self.substruct_encoder = GNNGraph(**substruct_para)

        self.global_encoder = GNNGraph(**global_para)

        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], emb_dim),
            torch.nn.Embedding(voc_size[1], emb_dim),
            torch.nn.Embedding(voc_size[2], emb_dim)
        ])
        self.seq_encoders = torch.nn.ModuleList([
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True)
        ])
        if dropout > 0 and dropout < 1:
            self.rnn_dropout = torch.nn.Dropout(p=dropout)
        else:
            self.rnn_dropout = torch.nn.Sequential()
        self.sab = SAB(substruct_dim, substruct_dim, 2, use_ln=True)
        self.query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 4, emb_dim)
        )
        self.query2 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(128, 131)
        )
        self.query3 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64)
        )
        self.substruct_rela = torch.nn.Linear(emb_dim, substruct_num)
        self.aggregator = AdjAttenAgger(
            global_dim, substruct_dim, max(global_dim, substruct_dim)
        )
        self.aggregator2 = Attention(
            global_dim, substruct_dim, max(global_dim, substruct_dim)
        )
        score_extractor = [
            torch.nn.Linear(substruct_dim, substruct_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(substruct_dim // 2, 1)
        ]
        self.score_extractor = torch.nn.Sequential(*score_extractor)
        self.init_weights()
        self.gcn = GCN(voc_size=voc_size[2], emb_dim=emb_dim, ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device)

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
        if self.use_embedding:
            torch.nn.init.xavier_uniform_(self.substruct_emb)

    def forward(
        self, substruct_data, mol_data, patient_data,
        ddi_mask_H, tensor_ddi_adj, average_projection
    ):
        seq1, seq2 = [], []
        seq3 = []
        for adm in patient_data:
            Idx1 = torch.LongTensor([adm[0]]).to(self.device)
            Idx2 = torch.LongTensor([adm[1]]).to(self.device)
            Idx3 = torch.LongTensor([adm[3]]).to(self.device)
            repr1 = self.rnn_dropout(self.embeddings[0](Idx1))
            repr2 = self.rnn_dropout(self.embeddings[1](Idx2))
            repr3 = self.rnn_dropout(self.embeddings[2](Idx3))
            seq1.append(torch.sum(repr1, keepdim=True, dim=1))
            seq2.append(torch.sum(repr2, keepdim=True, dim=1))
            seq3.append(torch.sum(repr3, keepdim=True, dim=1))
        seq1 = torch.cat(seq1, dim=1)
        seq2 = torch.cat(seq2, dim=1)
        seq3 = torch.cat(seq3, dim=1)
        output1, hidden1 = self.seq_encoders[0](seq1)
        output2, hidden2 = self.seq_encoders[1](seq2)
        output3, hidden3 = self.seq_encoders[2](seq3)
        seq_repr = torch.cat([hidden1, hidden2], dim=-1)
        last_repr = torch.cat([output1[:, -1],  output2[:, -1]], dim=-1)
        patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])
        medicinerep = torch.cat([output3[:,-1].flatten(), hidden3.flatten()])
        medicines = self.query2(medicinerep)
        medicines_embedding = medicines.unsqueeze(1).expand(-1, 64)

        query = self.query(patient_repr)
        substruct_weight = torch.sigmoid(self.substruct_rela(query))

        global_embeddings = self.global_encoder(**mol_data)
        global_embeddings = torch.mm(average_projection, global_embeddings)
        global_embeddings = torch.add(global_embeddings,medicines_embedding)

        substruct_embeddings = self.sab(
            self.substruct_emb.unsqueeze(0) if self.use_embedding else
            self.substruct_encoder(**substruct_data).unsqueeze(0)
        ).squeeze(0)
        molecule_embeddings = self.aggregator(
            global_embeddings, substruct_embeddings,
            substruct_weight, mask=torch.logical_not(ddi_mask_H > 0)
        )

        ehr_embedding, ddi_embedding = self.gcn()
        drug_memory = ehr_embedding - ddi_embedding
        molecule_embeddings = torch.add(molecule_embeddings, drug_memory)
        molecule_embeddings = torch.add(molecule_embeddings, medicines_embedding)

        score = self.score_extractor(molecule_embeddings).t()
        neg_pred_prob = torch.sigmoid(score)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(tensor_ddi_adj).sum()
        return score, batch_neg
