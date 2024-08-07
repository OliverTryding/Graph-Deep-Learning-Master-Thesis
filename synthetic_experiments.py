import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from itertools import product

from utils import EarlyStopping
from utils import fix_seeds
from utils import train, validate
from utils import test, visualize_results, visualize_ms
from utils import get_edges_train_mask
from dataloader import train_mask_for_classes_flat


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import networkx as nx

from dataloader import load_data

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
num_vertices = 25
num_hyperedges = 150
num_classes = 2
homophily = 0.9
feature_dim = 9
colors = ['red', 'blue', 'green']
positional_encoding = False

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Generate the dataset
print('Generating the dataset...')

dataset = generate_dataset('minesweeper', 20, 7, 5)
#final_ms = dataset[-1]

if len(dataset) == 1:
    G, labels, features = dataset[0]()

    color_list = [colors[l] for l in labels]

    # Load the data
    X = features.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    G.to(device)

    # train_mask = get_edges_train_mask(G, 0.4).to(device)
    # val_mask = ~train_mask
    # test_mask = ~train_mask

    train_mask, test_mask = train_mask_for_classes_flat(labels, num_classes, 20)
    val_mask = test_mask.clone()

    #edge_weight = random_walk_matrix(G)
    edge_weight = (None,None)
else:
    # Minesweeper
    perm = torch.randperm(len(dataset))
    full_mask = torch.ones(num_vertices, dtype=torch.bool).to(device)
    train_mask = perm[:int(0.7*len(dataset))]
    val_mask = perm[int(0.7*len(dataset)):int(0.8*len(dataset))]
    test_mask = perm[int(0.8*len(dataset)):]
    final_test = max(test_mask)
    final_ms = dataset[final_test]

    # # Root neighbors
    # train_mask = torch.arange(0, 100).to(device)
    # val_mask = torch.arange(100, 200).to(device)
    # test_mask = torch.arange(200, 300).to(device)
    # full_mask = torch.tensor([0]).to(device)

print('Dataset generated.')

if positional_encoding:
    # Add positional encoding
    print('Adding positional encoding...')
    Encoder = PosEncoder()
    encoded_features = []
    for idx, data in enumerate(dataset):
        G, labels, features = data
        features = Encoder(torch.tensor(features, dtype=torch.float32).to(device), G)
        if idx in train_mask:
            labels = torch.tensor(labels, dtype=torch.float32).to(device)
        else:
            labels = torch.tensor(labels, dtype=torch.float32).to(device)[:,0]
        G.to(device)
        dataset[idx] = (G, labels, features)

    num_encoded_features = feature_dim + num_vertices - 1
else:
    # No positional encoding
    for idx, data in enumerate(dataset):
        G, labels, features = data
        features = torch.tensor(features, dtype=torch.float32).to(device)
        if idx in train_mask:
            labels = torch.tensor(labels, dtype=torch.float32).to(device)#[:,0]
        else:
            labels = torch.tensor(labels, dtype=torch.float32).to(device)[:,1]
        G.to(device)
        dataset[idx] = (G, labels, features)
    num_encoded_features = feature_dim

# Define models
print('Defining the models...')

action_net_send = action_network(num_encoded_features, "sum", nn.GELU(), [4], depth=1, dropout=0).to(device)
action_net_recieve = action_network(num_encoded_features, "sum", nn.GELU(), [4], depth=1, dropout=0).to(device)
#action_net_send = HGNN(num_encoded_features, 64, 1, True).to(device)
#action_net_recieve = HGNN(num_encoded_features, 64, 1, True).to(device)
#environment_net = environment_network(num_encoded_features, "mean", nn.ReLU(), [128], depth=1, dropout=0).to(device)
environment_nets = []
#environment_net = environment_network(num_encoded_features, "mean", nn.GELU(), [16], depth=1, dropout=0).to(device)
for _ in range(2):
    environment_net = environment_network(num_encoded_features, "sum", nn.GELU(), [], depth=1, dropout=0).to(device)
    environment_nets.append(environment_net)
model = HCoGNN_node_classifier(num_encoded_features, num_classes, nn.GELU(), action_net_send, action_net_recieve, environment_nets, [16], tau=0.001, dropout=0, layerNorm=False, skip_connection=True).to(device)

params = [{'params': model.classifier.parameters(), 'lr': 0.01, 'weight_decay': 1e-5}, 
          {'params': model.action_net_send.parameters(), 'lr': 0.001, 'weight_decay': 1e-5}, 
          {'params': model.action_net_receive.parameters(), 'lr': 0.001, 'weight_decay': 1e-5}, 
          {'params': model.environment_networks.parameters(), 'lr': 0.001, 'weight_decay': 1e-5}]

# Adam optimizer
optimizer = torch.optim.Adam(params)

# loss function
#loss_f = nn.BCEWithLogitsLoss()
loss_f = nn.CrossEntropyLoss()
#loss_f = nn.MSELoss()

# Run the training
early_stopper = EarlyStopping(patience=100, mode='min', delta=0, break_training=True)
delay = False
delay_patience = 1
print("Training...")
print('')
for epoch in range(1001):
    avg_loss = 0
    for idx, data in enumerate(dataset):
        if idx in train_mask:
            # Load the data
            G, labels, X = data

            loss = train(model, optimizer, loss_f, X, G, labels, train_mask=None, delay=delay)
            avg_loss += loss / len(train_mask)
            if idx == 0:
                delay_patience = delay_patience - 1
                if delay_patience <= 0 and delay:
                    delay = False
                    print("Delay is off")
                if not delay:
                    #_, val_accuracy = validate(model, X, G, labels, train_mask, val_mask)
                    if early_stopper(model, loss):
                        model = early_stopper.best_model
                        if early_stopper.break_training:
                            print("Early stopping")
                            print(f"Best model at epoch {epoch} with score {early_stopper.best_score}")
                            break
    else:
        if epoch % 100 == 0:
            avg_val_accuracy = 0
            for idx, data in enumerate(dataset):
                if idx in val_mask:
                    # Load the data
                    G, labels, X = data
                    avg_val_accuracy += validate(model, X, G, labels, categorical=True, val_mask=full_mask, delay=delay)

            val_accuracy = avg_val_accuracy / len(val_mask)
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
            print(f'Validation Accuracy: {val_accuracy:.4f}')
        continue
    break

if early_stopper.best_model is not None:
    model = early_stopper.best_model

# Test the model
print('')
print('Testing...')
print('')

avg_accuracy = 0
for idx, data in enumerate(dataset):
    if idx in test_mask:
        G, labels, features = data

        # Load the data
        X = features.clone().detach().to(device)
        labels = labels.clone().detach().to(device)
        G.to(device)

        accuracy, predictions = test(model, X, G, labels, categorical=True, test_mask=full_mask)
        avg_accuracy += accuracy / len(test_mask)

        # visualize_results(model, X, G, labels, test_mask, show_graphs=False)

print(f'Test Accuracy: {avg_accuracy:.4f}')

# train_accuracy, val_accuracy = validate(model, X, G, labels, train_mask, val_mask)
# accuracy, predictions = test(model, X, G, labels, test_mask)
# print(f'Test Accuracy: {accuracy:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

visualize_ms(model, X, G, labels, final_ms)
visualize_results(model, X, G, labels, True, show_graphs=False)
