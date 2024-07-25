import torch
import torch.nn as nn
import torch.nn.functional as F

import dhg
from dhg import Hypergraph
from dhg.nn import HGNNConv
    
class environment_network(nn.Module):
    def __init__(self, num_features: int, aggregation: str = "mean", activation = nn.ReLU(), layer_dims: list = [], depth: int = 1, dropout: float = 0.3):
        super().__init__()

        self.depth = depth
        self.layers = nn.ModuleList()
        self.activation = activation
        layer_dims = [num_features] + layer_dims + [num_features]
        for i in range(len(layer_dims)-1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            self.layers.append(activation)
        #self.lin_update = nn.Linear(num_features, num_features, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.aggregation = aggregation

    def forward(self, x, action, G: Hypergraph):
        # Unless the action is S or B, the sent message is zero
        send = action[:, 1]
        # Unless the action is S or L, the received message is zero
        receive = action[:, 0]

        for _ in range(self.depth):
            x = self.update(x, G, send, receive)
            x = self.dropout(x)
        return x
    
    def update(self, x, G: Hypergraph, send=None, receive=None):
        # Compute the messages
        m_ji = x
        m_ji = m_ji * send.view(-1, 1) # mask the messages
        for layer in self.layers:
            m_ji = layer(m_ji)
        # Aggregate the messages
        m_i = G.v2v(m_ji, self.aggregation)
        m_i = m_i * receive.view(-1, 1) # mask the messages
        h_i = m_i #+ self.lin_update(x)
        h_i = self.activation(h_i)
        return h_i