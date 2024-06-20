import torch
import torch.nn as nn
import torch.nn.functional as F

import dhg
from dhg import Hypergraph
from dhg.nn import HGNNConv
    
class action_network(torch.nn.Module):
    """Classify a node feature vector into {S,L,B,I}."""
    def __init__(self, num_features: int, G: Hypergraph, aggregation: str = "mean", activation = nn.ReLU(), layer_dims: list = [], dropout: float = 0.3):
        super().__init__()

        self.layers = nn.ModuleList()
        self.activation = activation
        layer_dims = [num_features] + layer_dims + [num_features]
        for i in range(len(layer_dims)-1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            self.layers.append(activation)
        self.lin_update = nn.Linear(num_features, num_features, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.G = G
        self.aggregation = aggregation

    def forward(self, x, edge_weight=(None,None)):
        x = self.update(x, edge_weight)
        h_i = F.log_softmax(x, dim=1)
        return h_i
    
    def update(self, x, edge_weight=(None,None)):
        # Compute the messages
        m_ji = x
        for layer in self.layers:
            m_ji = layer(m_ji)
        m_ji = self.dropout(m_ji)

        # Aggregate the messages
        m_i = self.G.v2v(m_ji, self.aggregation, v2e_weight=edge_weight[0], e2v_weight=edge_weight[1])
        h_i = self.lin_update(x) + m_i
        h_i = self.activation(h_i)

        return h_i