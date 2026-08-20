import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from rewritten_GCN import GCNConv
from torch_geometric.nn import GINConv,GATConv
from torch_geometric.utils import degree


def reset_parameters(w):
    stdv = 1. / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv) 

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        
        self.linear1 = nn.Linear(in_channels, 2 * out_channels)
        self.linear2 = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x

"""def func_k(curve, weight_fct='exp'):
    k = [1, 2, 3, 4, 5, 6,7, 8, 9,10]

    for i in range(len(k)):
        if weight_fct == 'exp':
            cur = (1+torch.exp(-k[i]*curve))/2
        elif weight_fct == 'lin':
            cur = torch.sigmoid(-k[i]*curve)
        if i == 0:
            Multi_curve = cur.unsqueeze(0).T
        else:
            Multi_curve = torch.cat((Multi_curve, cur.unsqueeze(0).T), 1)
        #Multi_curve = curve.unsqueeze(0).T
    #print(Multi_curve.shape)
    return Multi_curve"""

def func_k(curve):
    return curve.unsqueeze(1)


class CGCN(nn.Module):
    def __init__(self, aggregator, feature, hidden1, hidden2, decoder1, dropout, num_layers=1):
        super(CGCN, self).__init__()

        assert num_layers in (1, 2), f"num_layers must be 1 or 2, got {num_layers}"
        self.num_layers = num_layers

        if aggregator == 'GCN':
            if num_layers == 1:
                self.encoder_o1 = GCNConv(feature, hidden2)
            else:
                self.encoder_o1 = GCNConv(feature, hidden1)
                self.encoder_o2 = GCNConv(hidden1, hidden2)

        self.decoder1 = nn.Linear(hidden2 * 4, decoder1)
        self.decoder2 = nn.Linear(decoder1, 1)

        self.lin1 = nn.Linear(1, 1)
        self.lin2 = nn.Linear(1, 1)
        self.lin3 = nn.Linear(1, 1)

        self.dropout = dropout
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def forward(self, data_o, idx, use_curvature=True, weight_fct='exp'):

        x_o, adj = data_o.x, data_o.edge_index

        if use_curvature:
            curvature = data_o.curva
            curva1 = self.lin1(func_k(curvature))
            curva1 = F.dropout(curva1, self.dropout, training=self.training)
            edge_weight = curva1
        else:
            edge_weight = torch.ones((adj.size(1), 1), device=x_o.device)

        x1_o = F.relu(self.encoder_o1(x_o, adj, edge_weight))

        if self.num_layers == 2:
            x1_o = F.dropout(x1_o, self.dropout, training=self.training)
            x1_o = F.relu(self.encoder_o2(x1_o, adj, edge_weight))

        idx0 = idx[0].long()
        idx1 = idx[1].long()

        entity1 = x1_o[idx0]
        entity2 = x1_o[idx1]

        add = entity1 + entity2
        product = entity1 * entity2
        concatenate = torch.cat((entity1, entity2), dim=1)
        feature = torch.cat((add, product, concatenate), dim=1)

        log = F.relu(self.decoder1(feature))
        log = self.decoder2(log)

        return log