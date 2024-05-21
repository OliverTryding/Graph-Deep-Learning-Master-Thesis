import torch
import torch.nn as nn
import torch.nn.functional as F

import dhg
from dhg import Hypergraph
from dhg.nn import HGNNConv
    
class environment_network(nn.Module):
    def __init__(self, num_features: int, G: Hypergraph, aggregation: str = "mean", dropout: float = 0.3):
        super().__init__()
        self.G = G
        self.lin_message = nn.Linear(num_features, num_features, bias=False)
        self.lin_update = nn.Linear(num_features, num_features, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.aggregation = aggregation

    def forward(self, x, action, edge_weight=(None,None)):
        # Unless the action is S or B, the sent message is zero
        send = action[:, 0] + action[:, 2]
        # Unless the action is S or L, the received message is zero
        receive = action[:, 0] + action[:, 1]
        
        # Compute the messages
        m_ji = self.lin_message(x)
        m_ji = self.dropout(m_ji)
        m_ji = torch.nn.GELU()(m_ji)
        m_ji = m_ji * send.view(-1, 1) # mask the messages

        # Aggregate the messages
        m_i = self.G.v2v(m_ji, self.aggregation, v2e_weight=edge_weight[0], e2v_weight=edge_weight[1])
        m_i = m_i * receive.view(-1, 1) # mask the messages

        h_i = torch.nn.GELU()(self.lin_update(x) + m_i)
        return h_i
    
# class environment_network(nn.Module):
#     def __init__(self, num_features: int, G: Hypergraph, aggregation: str = "mean", dropout: float = 0.3):
#         super().__init__()
#         self.G = G
#         self.lin_update = nn.Linear(num_features, num_features, bias=True)
#         self.aggregation = aggregation
#         self.conv = HGNNConv(num_features, num_features, bias=False, drop_rate=dropout)

#     def forward(self, x, action, edge_weight=(None,None)):
#         # Unless the action is S or B, the sent message is zero
#         send = action[:, 0] + action[:, 2]
#         # Unless the action is S or L, the received message is zero
#         receive = action[:, 0] + action[:, 1]

#         # Zero out features where send is zero
#         x_masked = x * send.view(-1, 1)

#         # Aggregate the messages
#         m_i = self.conv(x_masked, self.G)
#         m_i = m_i * receive.view(-1, 1) # mask the messages

#         h_i = torch.nn.GELU()(self.lin_update(x) + m_i)
#         return h_i