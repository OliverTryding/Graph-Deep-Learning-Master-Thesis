import torch
import torch.nn as nn
import torch.nn.functional as F

import dhg
from dhg import Hypergraph
from dhg.nn import HGNNConv
    
class action_network(torch.nn.Module):
    """Classify a node feature vector into {S,L,B,I}."""
    def __init__(self, num_features: int, G: Hypergraph, aggregation: str = "mean", dropout: float = 0.3):
        super().__init__()
        self.lin_message = torch.nn.Linear(num_features, 4, bias=False)
        self.lin_update = nn.Linear(num_features, 4, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.G = G
        self.aggregation = aggregation

    def forward(self, x, edge_weight=(None,None)):
        # Compute the messages
        m_ji = self.lin_message(x)
        m_ji = self.dropout(m_ji)
        m_ji = torch.nn.GELU()(m_ji)
        # Aggregate the messages
        m_i = self.G.v2v(m_ji, self.aggregation, v2e_weight=edge_weight[0], e2v_weight=edge_weight[1])

        h_i = torch.nn.GELU()(self.lin_update(x) + m_i)
        h_i = F.log_softmax(h_i, dim=1)
        return h_i
    
# class action_network(torch.nn.Module):
#     """Classify a node feature vector into {S,L,B,I}."""
#     def __init__(self, num_features: int, G: Hypergraph, aggregation: str = "mean", dropout: float = 0.3):
#         super().__init__()
#         self.lin_update = nn.Linear(num_features, 4, bias=True)
#         self.G = G
#         self.aggregation = aggregation
#         self.conv = HGNNConv(num_features, 4, bias=False, drop_rate=dropout)

#     def forward(self, x, edge_weight=(None,None)):
#         # Aggregate the messages
#         m_i = self.conv(x, self.G)

#         h_i = torch.nn.GELU()(self.lin_update(x) + m_i)
#         h_i = F.log_softmax(h_i, dim=1)
#         return h_i