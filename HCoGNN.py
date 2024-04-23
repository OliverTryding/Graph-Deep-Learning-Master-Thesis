import torch
import torch.nn as nn
import torch.nn.functional as F

import dhg
from dhg import Graph
from dhg import Hypergraph

from ActionNetwork import action_network
from EnvironmentNetwork import environment_network

class HCoGNN_node_classifier(nn.Module):
    def __init__(self, num_features: int, num_classes: int, G: Graph, num_iterations: int):
        super().__init__()
        self.classifier = nn.Linear(num_features, num_classes)
        self.G = G
        self.num_iterations = num_iterations
        self.action_net = action_network(num_features, G)
        self.environment_net = environment_network(num_features, G)
        self.action_history = []

    def forward(self, x, edge_weight=(None,None)):          
        # Pass the input through the MPNN for a number of iterations
        for i in range(self.num_iterations):
            # Determine the actions
            p_i = self.action_net(x, edge_weight) # probabilities for action selection {S, L, B, I}
            # Sample an action using the straight-through Gumbel-softmax estimator
            action = F.gumbel_softmax(p_i, hard=True)
            if self.eval():
                self.action_history.append(torch.argmax(action, dim=1))
            # Update the node features
            x = self.environment_net(x, action, edge_weight)
        # Pass the output through the classifier
        out = self.classifier(x)
        # Apply softmax to get probabilities
        out = F.softmax(out, dim=1)
        return out