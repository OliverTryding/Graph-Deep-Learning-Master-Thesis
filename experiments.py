import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

from utils import EarlyStopping
from utils import fix_seeds
from utils import train, validate
from utils import test, visualize_results

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import networkx as nx

from dataloader import load_data

import dhg
from dhg import Graph
from dhg import Hypergraph
from dhg.models import GCN
from dhg.random import set_seed
from dhg.metrics import GraphVertexClassificationEvaluator as Evaluator

from models.HCoGNN import HCoGNN_node_classifier
from models.ActionNetwork import action_network
from models.EnvironmentNetwork import environment_network

# Fix the seeds for reproducibility
fix_seeds(255)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load the data
X, labels, G, num_classes, num_nodes, num_node_features, num_edges, max_edge_size, train_mask, val_mask, test_mask = load_data('cocitation_cora', train_percentage=0.5)
X = X.to(device)
labels = labels.to(device)
G = G.to(device)
train_mask = train_mask.to(device)
val_mask = val_mask.to(device)
test_mask = test_mask.to(device)

action_net = action_network(num_node_features, "mean", nn.ReLU(), [32], depth=0, dropout=0.5).to(device)
environment_net = environment_network(num_node_features, "mean", nn.ReLU(), [128], depth=1, dropout=0.5).to(device)
model = HCoGNN_node_classifier(num_node_features, num_classes, 2, nn.ReLU(), action_net, environment_net, [64], tau=0.01, dropout=0.5, layerNorm=True).to(device)

params = [{'params': model.classifier.parameters(), 'lr': 0.01, 'weight_decay': 1e-5}, 
          {'params': model.action_net.parameters(), 'lr': 0.01, 'weight_decay': 1e-5}, 
          {'params': model.environment_net.parameters(), 'lr': 0.01, 'weight_decay': 1e-5}]

# Adam optimizer
optimizer = torch.optim.Adam(params)

# L-BFGS optimizer
# optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01, max_iter=20, max_eval=25, history_size=100, line_search_fn='strong_wolfe')

#edge_weight = random_walk_matrix(G)
edge_weight = (None,None)

# Run the training
early_stopper = EarlyStopping(patience=100, mode='max', delta=-0.01)
print('')
print("Training...")
for epoch in range(2000):
    loss = train(model, optimizer, X, G, labels, train_mask)

    _, val_accuracy = validate(model, X, G, labels, train_mask, val_mask)
    if early_stopper(model, val_accuracy):
        # print("Early stopping")
        model = early_stopper.best_model

    if epoch % 100 == 0:
        train_accuracy, val_accuracy = validate(model, X, G, labels, train_mask, val_mask)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
        print(f'Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

model = early_stopper.best_model

# Test the model
train_accuracy, val_accuracy = validate(model, X, G, labels, train_mask, val_mask)
accuracy, predictions = test(model, X, G, labels, test_mask)
print(f'Test Accuracy: {accuracy:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

visualize_results(model, X, G, labels, test_mask)