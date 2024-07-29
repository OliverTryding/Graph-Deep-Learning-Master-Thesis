import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

from utils import EarlyStopping
from utils import fix_seeds
from utils import train, validate
from utils import test, visualize_results
from utils import get_edges_train_mask
from dataloader import train_mask_for_classes_flat


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import networkx as nx

from dataloader import load_data

from datasets.synthetic.generate_synth_data import generate_hypergraph_dataset
from datasets.synthetic.generate_synth_data import generate_dataset

import dhg
from dhg import Graph
from dhg import Hypergraph
from dhg.models import GCN
from dhg.random import set_seed
from dhg.metrics import GraphVertexClassificationEvaluator as Evaluator

from models.HCoGNN import HCoGNN_node_classifier
from models.ActionNetwork import action_network
from models.EnvironmentNetwork import environment_network
from models.Encoders import PosEncoder

from dhg.models import HyperGCN
from dhg.models import HGNN

# Fix the seeds for reproducibility
fix_seeds(420)

# Dataset Parameters
num_vertices = 100
num_hyperedges = 150
num_classes = 2
homophily = 0.9
feature_dim = 1
colors = ['red', 'blue', 'green']

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Generate the dataset
print('Generating the dataset...')

dataset = generate_dataset('minesweeper', 10, 10, 20)

if len(dataset) == 1:
    G, labels, features = dataset[0]

    color_list = [colors[l] for l in labels]

    # Load the data
    X = torch.tensor(features, dtype=torch.float32).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    G.to(device)

    # train_mask = get_edges_train_mask(G, 0.4).to(device)
    # val_mask = ~train_mask
    # test_mask = ~train_mask

    train_mask, test_mask = train_mask_for_classes_flat(labels, num_classes, 20)
    val_mask = test_mask.clone()

    #edge_weight = random_walk_matrix(G)
    edge_weight = (None,None)
else:
    perm = torch.randperm(len(dataset))
    train_mask = perm[:int(0.7*len(dataset))]
    val_mask = perm[int(0.7*len(dataset)):int(0.8*len(dataset))]
    test_mask = perm[int(0.8*len(dataset)):]

print('Dataset generated.')


# Add positional encoding
print('Adding positional encoding...')
Encoder = PosEncoder()
encoded_features = []
for idx, data in enumerate(dataset):
    G, labels, features = data
    features = Encoder(torch.tensor(features, dtype=torch.float32).to(device), G)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    G.to(device)
    dataset[idx] = (G, labels, features)

num_encoded_features = feature_dim + num_vertices - 1

# Define models
print('Defining the models...')

action_net_send = action_network(num_encoded_features, "mean", nn.GELU(), [], depth=1, dropout=0).to(device)
action_net_recieve = action_network(num_encoded_features, "mean", nn.GELU(), [], depth=1, dropout=0).to(device)
#action_net_send = HGNN(num_encoded_features, 64, 1, True).to(device)
#action_net_recieve = HGNN(num_encoded_features, 64, 1, True).to(device)
#environment_net = environment_network(num_encoded_features, "mean", nn.ReLU(), [128], depth=1, dropout=0).to(device)
environment_nets = []
for _ in range(10):
    environment_net = environment_network(num_encoded_features, "mean", nn.GELU(), [128], depth=1, dropout=0).to(device)
    environment_nets.append(environment_net)
model = HCoGNN_node_classifier(num_encoded_features, num_classes, nn.GELU(), action_net_send, action_net_recieve, environment_nets, [256], tau=0.001, dropout=0.5, layerNorm=True, skip_connection=True).to(device)

params = [{'params': model.classifier.parameters(), 'lr': 0.01, 'weight_decay': 1e-5}, 
          {'params': model.action_net_send.parameters(), 'lr': 0.001, 'weight_decay': 1e-5}, 
          {'params': model.action_net_receive.parameters(), 'lr': 0.001, 'weight_decay': 1e-5}, 
          {'params': model.environment_networks.parameters(), 'lr': 0.001, 'weight_decay': 1e-5}]

# Adam optimizer
optimizer = torch.optim.Adam(params)

# Run the training
early_stopper = EarlyStopping(patience=200, mode='min', delta=0, break_training=True)
delay = True
delay_patience = 200
print("Training...")
print('')
for epoch in range(2000):
    for idx, data in enumerate(dataset):
        if idx in train_mask:
            # Load the data
            G, labels, X = data

            color_list = [colors[l] for l in labels]

            loss = train(model, optimizer, X, G, labels, train_mask, delay=delay)
            if idx == 0:
                delay_patience = delay_patience - 1
                if delay_patience == 0:
                    delay = False
                    print("Delay is off")
                if not delay:
                    #_, val_accuracy = validate(model, X, G, labels, train_mask, val_mask)
                    if early_stopper(model, loss):
                        model = early_stopper.best_model
                        if early_stopper.break_training:
                            print("Early stopping")
                            break

    if epoch % 100 == 0:
        avg_val_accuracy = 0
        for idx, data in enumerate(dataset):
            if idx in val_mask:
                # Load the data
                G, labels, X = data

                color_list = [colors[l] for l in labels]

                loss = train(model, optimizer, X, G, labels, train_mask, delay=delay)

                if not delay:
                    # val_accuracy = validate(model, X, G, labels, train_mask, val_mask)
                    if early_stopper(model, loss):
                        model = early_stopper.best_model
                        if early_stopper.break_training:
                            print("Early stopping")
                            break
            
                avg_val_accuracy += validate(model, X, G, labels, val_mask, delay=delay)

        val_accuracy = avg_val_accuracy / len(val_mask)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}')

model = early_stopper.best_model

# Test the model
print('')
print('Testing...')
print('')

for idx, data in enumerate(dataset):
    if idx in test_mask:
        G, labels, features = data

        color_list = [colors[l] for l in labels]

        # Load the data
        X = torch.tensor(features, dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)
        G.to(device)

        accuracy, predictions = test(model, X, G, labels, test_mask)
        print(f'Test Accuracy: {accuracy:.4f}')

        # visualize_results(model, X, G, labels, test_mask, show_graphs=False)

# train_accuracy, val_accuracy = validate(model, X, G, labels, train_mask, val_mask)
# accuracy, predictions = test(model, X, G, labels, test_mask)
# print(f'Test Accuracy: {accuracy:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# visualize_results(model, X, G, labels, test_mask, show_graphs=False)
