import torch
import torch.nn as nn
import torch.nn.functional as F

import dhg
from dhg import Hypergraph
from dhg.nn import HGNNConv
    
class action_network(torch.nn.Module):
    """Classify a node feature vector into {S,L,B,I}."""
    def __init__(self, num_features: int, aggregation: str = "mean", activation = nn.ReLU(), layer_dims: list = [], depth: int = 1, dropout: float = 0.3):
        super().__init__()

        self.depth = depth
        self.layers = nn.ModuleList()
        self.activation = activation
        layer_dims = [num_features] + layer_dims + [num_features]
        for i in range(len(layer_dims)-1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            self.layers.append(activation)
        self.lin_update = nn.Linear(num_features, num_features, bias=True)
        self.action_layer = nn.Linear(num_features, 4)
        self.dropout = nn.Dropout(dropout)
        self.aggregation = aggregation

    def forward(self, x, G: Hypergraph):
        if self.depth == 0:
            action = torch.zeros(x.shape[0], 4).to(x.device)
            action[:, 0] = 1
            return action
        for _ in range(self.depth):
            x = self.update(x, G)
        a_i = self.action_layer(x)

        # Compute the action probabilities
        p_i = F.log_softmax(a_i, dim=1) # probabilities for action selection {S, L, B, I}

        return p_i
    
    def update(self, x, G: Hypergraph):
        # Compute the messages
        m_ji = x
        for layer in self.layers:
            m_ji = layer(m_ji)
        m_ji = self.dropout(m_ji)

        # Aggregate the messages
        m_i = G.v2v(m_ji, self.aggregation)
        h_i = self.lin_update(x) + m_i
        h_i = self.activation(h_i)

        return h_i