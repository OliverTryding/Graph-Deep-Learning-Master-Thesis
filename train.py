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
from models.Encoders import PosEncoder

import wandb
import argparse

def main(dataset='cocitation_cora', 
         model='HCoGNN',
         train_percentage=0.5, 
         activation_fun=nn.ReLU(), 
         action_net_depth=0, 
         environment_net_depth=1, 
         action_net_hidden=[], 
         environment_net_hidden=[], 
         hidden=[], 
         num_layers=1,
         tau=0.01, 
         do_act=0.3,
         do_env=0.3,
         dropout=0.5, 
         layerNorm=True, 
         pos_enc=False,
         classifier_lr=0.01, 
         action_net_lr=0.01, 
         environment_net_lr=0.01, 
         weight_decay=1e-5,
         batch_size=0,
         seed=255):

    # Initialize a Weights & Biases run
    wandb.init(project='HCoGNN', config={
        'dataset': dataset,
        'model': model,
        'train_percentage': train_percentage,
        'activation_fun': activation_fun,
        'action_net_depth': action_net_depth,
        'do_act': do_act,
        'environment_net_depth': environment_net_depth,
        'action_net_hidden': action_net_hidden,
        'environment_net_hidden': environment_net_hidden,
        'do_env': do_env,
        'num_layers': num_layers,
        'hidden': hidden,
        'tau': tau,
        'dropout': dropout,
        'layerNorm': layerNorm,
        'pos_enc': pos_enc,
        'classifier_lr': classifier_lr,
        'action_net_lr': action_net_lr,
        'environment_net_lr': environment_net_lr,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'seed': seed
    })

    # Fix the seeds for reproducibility
    fix_seeds(seed)

    # Load the data
    X, labels, G, num_classes, num_nodes, num_node_features, num_edges, max_edge_size, train_mask, val_mask, test_mask = load_data(dataset, train_percentage=train_percentage)
    X = X.to(device)
    labels = labels.to(device)
    G = G.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # Add positional encoding
    if pos_enc:
        Encoder = PosEncoder()
        X = Encoder(X, G)
    num_encoded_features = X.shape[1]

    # Define the model
    action_net_send = action_network(num_encoded_features, "mean", activation_fun, action_net_hidden, depth=action_net_depth, dropout=do_act).to(device)
    action_net_receive = action_network(num_encoded_features, "mean", activation_fun, action_net_hidden, depth=action_net_depth, dropout=do_act).to(device)
    environment_net = environment_network(num_encoded_features, "mean", activation_fun, environment_net_hidden, depth=environment_net_depth, dropout=do_env).to(device)
    model = HCoGNN_node_classifier(num_encoded_features, num_classes, num_layers, activation_fun, action_net_send, action_net_receive, environment_net, hidden, tau=tau, dropout=dropout, layerNorm=layerNorm).to(device)

    params = [{'params': model.classifier.parameters(), 'lr': classifier_lr, 'weight_decay': weight_decay}, 
              {'params': model.action_net_send.parameters(), 'lr': action_net_lr, 'weight_decay': weight_decay}, 
              {'params': model.action_net_receive.parameters(), 'lr': action_net_lr, 'weight_decay': weight_decay}, 
              {'params': model.environment_net.parameters(), 'lr': environment_net_lr, 'weight_decay': weight_decay}]

    # Adam optimizer
    optimizer = torch.optim.Adam(params)

    # L-BFGS optimizer
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01, max_iter=20, max_eval=25, history_size=100, line_search_fn='strong_wolfe')

    #edge_weight = random_walk_matrix(G)
    edge_weight = (None,None)

    # Run the training
    early_stopper = EarlyStopping(patience=200, mode='min', delta=0.0, break_training=True)
    print('')
    print("Training...")
    for epoch in range(2000):
        loss = train(model, optimizer, X, G, labels, train_mask)

        #_, val_accuracy = validate(model, X, G, labels, train_mask, val_mask)
        if early_stopper(model, loss):
            print(f"Best validation accuracy: {early_stopper.best_score:.4f}")
            print(f"Current validation accuracy: {val_accuracy:.4f}")
            model = early_stopper.best_model
            if early_stopper.break_training:
                print("Early stopping")
                break

        if epoch % 100 == 0:
            train_accuracy, val_accuracy = validate(model, X, G, labels, train_mask, val_mask)
            print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
            print(f'Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    model = early_stopper.best_model

    # Test the model
    train_accuracy, val_accuracy = validate(model, X, G, labels, train_mask, val_mask)
    accuracy, predictions = test(model, X, G, labels, test_mask)
    print(f'Test Accuracy: {accuracy:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    visualize_results(model, X, G, labels, test_mask, show_graphs=False)
    wandb.log({"Test Accuracy": accuracy, "Training Accuracy": train_accuracy, "Validation Accuracy": val_accuracy})

    # Finish the wandb run
    wandb.finish()

if __name__ == '__main__':
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    parser = argparse.ArgumentParser()

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset name')
    parser.add_argument('--model', type=str, default='HCoGNN', help='Model name')
    parser.add_argument('--train_percentage', type=float, default=0.6, help='Percentage of data used for training')
    parser.add_argument('--activation_fun', type=str, default='ReLU', help='Activation function')
    parser.add_argument('--action_net_depth', type=int, default=0, help='Depth of action network')
    parser.add_argument('--environment_net_depth', type=int, default=0, help='Depth of environment network')
    parser.add_argument('--action_net_hidden', nargs='+', type=int, default=[], help='Hidden units in action network')
    parser.add_argument('--environment_net_hidden', nargs='+', type=int, default=[], help='Hidden units in environment network')
    parser.add_argument('--hidden', nargs='+', type=int, default=[128], help='Hidden units in classifier')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in classifier')
    parser.add_argument('--tau', type=float, default=0.01, help='Temperature parameter')
    parser.add_argument('--do_act', type=float, default=0, help='Dropout rate for action network')
    parser.add_argument('--do_env', type=float, default=0.2, help='Dropout rate for environment network')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate for classifier')
    parser.add_argument('--layerNorm', type=bool, default=True, help='Whether to use layer normalization')
    parser.add_argument('--classifier_lr', type=float, default=0.01, help='Learning rate for classifier')
    parser.add_argument('--action_net_lr', type=float, default=0.01, help='Learning rate for action network')
    parser.add_argument('--environment_net_lr', type=float, default=0.00005, help='Learning rate for environment network')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=0, help='Batch size')
    parser.add_argument('--seed', type=int, default=420, help='Random seed')

    args = parser.parse_args()

    main(dataset='cora',
         model='HCoGNN',
         train_percentage=0.6,
         activation_fun=nn.ReLU(),
         action_net_depth=0,
         environment_net_depth=1,
         action_net_hidden=[16],
         environment_net_hidden=[32],
         hidden=[64],
         num_layers=3,
         tau=0.001,
         do_act=0,
         do_env=0,
         dropout=0.5,
         layerNorm=True,
         pos_enc=True,
         classifier_lr=0.01,
         action_net_lr=0.001,
         environment_net_lr=0.0005,
         weight_decay=5e-5,
         seed=420)