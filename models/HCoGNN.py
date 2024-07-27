import torch
import torch.nn as nn
import torch.nn.functional as F

import dhg
from dhg import Graph
from dhg import Hypergraph

from models.ActionNetwork import action_network
from models.EnvironmentNetwork import environment_network

class HCoGNN_node_classifier(nn.Module):
    def __init__(self, num_features: int, num_classes: int, activation = nn.ReLU(), 
                 action_net_receive: action_network = None, action_net_send: action_network = None, environment_nets: list = [], 
                 classifier_layers: list = [], tau: float = 0.1, dropout: float = 0.5, layerNorm: bool = True, skip_connection: bool = False):
        super().__init__()

        self.layer_norm = nn.LayerNorm(num_features) if layerNorm else nn.Identity()
        self.skip_connection = skip_connection
        self.skip_connections = []

        self.linear = nn.Linear(num_features, num_features)

        self.classifier = nn.ModuleList()
        dim = num_features
        for hidden_dim in classifier_layers:
            assert isinstance(hidden_dim, int), "All elements of classifier_layers should be integers"
            self.classifier.append(nn.Linear(dim, hidden_dim))
            self.classifier.append(activation)
            dim = hidden_dim
        self.classifier.append(nn.Linear(dim, num_classes))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.action_net_send = action_net_send
        self.action_net_receive = action_net_receive
        self.environment_networks = nn.ModuleList(environment_nets)
        self.num_iterations = len(environment_nets)
        self.tau = tau
        self.save_action_history = False
        self.action_history = []

    def forward(self, x, G: Hypergraph):
        # Pass the input through the linear encoding layer
        x = self.linear(x)

        # Pass the input through the MPNN for a number of iterations
        for i, environment_net in enumerate(self.environment_networks):
            # Apply layer normalization
            x = self.layer_norm(x)

            # Determine the actions
            zero_pad = torch.zeros(x.shape[0], 1).to(x.device)
            p_i_send = self.action_net_send(x, G)
            p_i_send = torch.cat([p_i_send, zero_pad], dim=1)
            p_i_receive = self.action_net_receive(x, G)
            p_i_receive = torch.cat([p_i_receive, zero_pad], dim=1)

            # Sample an action using the straight-through Gumbel-softmax estimator
            send_action = F.gumbel_softmax(p_i_send, self.tau, hard=True)[:,0]
            receive_action = F.gumbel_softmax(p_i_receive, self.tau, hard=True)[:,0]
            action = torch.stack([receive_action, send_action], dim=1)

            if self.save_action_history:
                self.action_history.append((3 - action[:, 0] * 2 - action[:, 1]).to(torch.int8))

            if i == 0:
                initial_action = action
                action = torch.ones(x.shape[0], 2).to(x.device)

            # Update the node features
            x = environment_net(x, action, G)

        # Pass the output through the classifier
        for layer in self.classifier:
            x = layer(x)
        out = self.dropout(x)

        # Apply softmax to get probabilities
        #out = F.softmax(out, dim=1)

        return (out, initial_action) if self.training else out