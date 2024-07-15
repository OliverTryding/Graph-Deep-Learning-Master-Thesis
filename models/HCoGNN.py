import torch
import torch.nn as nn
import torch.nn.functional as F

import dhg
from dhg import Graph
from dhg import Hypergraph

from models.ActionNetwork import action_network
from models.EnvironmentNetwork import environment_network
from models.Encoders import PosEncoder

class HCoGNN_node_classifier(nn.Module):
    def __init__(self, num_features: int, num_classes: int, num_iterations: int = 1, activation = nn.ReLU(), 
                 action_net: action_network = None, environment_net: environment_network = None, classifier_layers: list = [], 
                 tau: float = 0.1, dropout: float = 0.5, layerNorm: bool = True, pos_enc: bool = False, k: int = 10):
        super().__init__()
        self.pos_enc = pos_enc
        self.encoder = PosEncoder(k=k)
        self.classifier = nn.ModuleList()
        dim = num_features
        for hidden_dim in classifier_layers:
            assert isinstance(hidden_dim, int), "All elements of classifier_layers should be integers"
            self.classifier.append(nn.Linear(dim, hidden_dim))
            self.classifier.append(activation)
            dim = hidden_dim
        self.classifier.append(nn.Linear(dim, num_classes))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_features) if layerNorm else nn.Identity()
        self.num_iterations = num_iterations
        self.activation = activation
        self.action_net = action_net
        self.environment_net = environment_net
        self.tau = tau
        self.action_history = []

    def forward(self, x, G: Hypergraph):
        if self.pos_enc:
            x = self.encoder(x, G)

        # Pass the input through the MPNN for a number of iterations
        for i in range(self.num_iterations):
            # Apply layer normalization
            x = self.layer_norm(x)

            # Determine the actions
            if i == 0:
                action = torch.zeros(x.shape[0], 4).to(x.device)
                action[:, 0] = 1
                if self.eval():
                    self.action_history.append(torch.argmax(action, dim=1))
            else:
                p_i = self.action_net(x, G)
                # Sample an action using the straight-through Gumbel-softmax estimator
                action = F.gumbel_softmax(p_i, self.tau, hard=True)
                if self.eval():
                    self.action_history.append(torch.argmax(action, dim=1))

            # Update the node features
            x = self.environment_net(x, action, G)

        # Pass the output through the classifier
        for layer in self.classifier:
            x = layer(x)
        out = self.dropout(x)
        # Apply softmax to get probabilities
        out = F.softmax(out, dim=1)
        return out