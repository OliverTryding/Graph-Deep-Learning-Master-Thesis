import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import copy
from sklearn.metrics import confusion_matrix

import dhg
from dhg import Graph
from dhg import Hypergraph
from dhg.random import set_seed

def fix_seeds(seed):
    """
    Fix seeds for reproducibility the same way as the authors did
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def random_walk_matrix(G):
    deg = G.deg_v
    v2e_weight = torch.ones_like(G.v2e_weight, dtype=torch.float32)
    e2v_weight = torch.ones_like(G.e2v_weight, dtype=torch.float32)
    for i, v in enumerate(G.v2e_src):
        v2e_weight[i] = 1.0 / deg[v]
    return v2e_weight, e2v_weight

bfgs_loss = torch.inf

# BFGS Closure
def closure(model, optimizer, X, G, labels, train_mask):
    optimizer.zero_grad()
    out = model(X, G)
    loss = F.cross_entropy(out[train_mask], labels[train_mask])
    loss.backward()
    global bfgs_loss 
    bfgs_loss = loss.item()
    return loss

# Get train mask from edges
def get_edges_train_mask(G, train_percentage):
    # Get the edges
    edges = G.e[0]
    num_edges = len(edges)

    # Get train mask
    train_mask = torch.zeros(G.num_v, dtype=torch.bool)

    # Shuffle the edges
    permutation = torch.randperm(num_edges)
    edges = [edges[i] for i in permutation]

    # Get the number of training edges
    num_train_edges = int(train_percentage * num_edges)

    # Get the training edges
    train_edges = edges[:num_train_edges]

    # Get the training mask
    for e in train_edges:
        for v in e:
            train_mask[int(v)] = True

    return train_mask

# Add S prediction bias to the model
def initial_action_loss(initial_action):
    return torch.nn.MSELoss()(initial_action, torch.ones_like(initial_action))

# Training function
def train(model, optimizer, X, G, labels, train_mask):
    model.train()
    if optimizer.__class__.__name__ == 'LBFGS':
        optimizer.step(closure=lambda: closure(model, optimizer, X, G, labels, train_mask))
        return bfgs_loss
    else:
        if model.__class__.__name__ == 'HCoGNN_node_classifier':
            optimizer.zero_grad()
            out, initial_action = model(X, G)
            loss = F.cross_entropy(out[train_mask], labels[train_mask]) #+ torch.nn.MSELoss()(initial_action, torch.ones_like(initial_action))
            loss.backward()
            optimizer.step()
            return loss.item()
        else:
            optimizer.zero_grad()
            out = model(X, G)
            #loss = F.nll_loss(out[train_mask], labels[train_mask])
            loss = F.cross_entropy(out[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()
            return loss.item()

def validate(model, X, G, labels, train_mask, val_mask):
    model.eval()
    with torch.no_grad():
        logits = model(X, G) # Log probabilities

        # Train accuracy
        train_logits = logits[train_mask] # Log probabilities of train nodes
        train_labels = labels[train_mask] # True labels of train nodes
        train_pred = train_logits.max(1)[1] # Predicted labels
        train_correct = train_pred.eq(train_labels).sum().item() # Number of correctly classified nodes
        train_accuracy = train_correct / train_mask.sum().item() # Accuracy

        # Validation accuracy
        val_logits = logits[val_mask] # Log probabilities of validation nodes
        val_labels = labels[val_mask] # True labels of validation nodes
        val_pred = val_logits.max(1)[1] # Predicted labels
        val_correct = val_pred.eq(val_labels).sum().item() # Number of correctly classified nodes
        val_accuracy = val_correct / val_mask.sum().item() # Accuracy
            
    return train_accuracy, val_accuracy

# Testing function
def test(model, X, G, labels, test_mask):
    model.eval()
    with torch.no_grad():
        logits = model(X, G)
        test_logits = logits[test_mask] # Log probabilities of test nodes
        test_labels = labels[test_mask] # True labels of test nodes
        pred = test_logits.max(1)[1] # Predicted labels
        correct = pred.eq(test_labels).sum().item() # Number of correctly classified nodes
        accuracy = correct / test_mask.sum().item() # Accuracy
                        
    return accuracy, pred

def visualize_results(model, X, G, labels, test_mask, show_graphs=False):
    model.eval()
    model.save_action_history = True
    with torch.no_grad():
        logits = model(X, G)
        test_logits = logits[test_mask] # Log probabilities of test nodes
        test_labels = labels[test_mask] # True labels of test nodes
        pred = test_logits.max(1)[1] # Predicted labels

        cm = confusion_matrix(test_labels.cpu(), pred.cpu())

        # Normalize confusion matrix
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(1, 2, figsize=(20,10))

        # Plot confusion matrix
        im = ax[0].matshow(cm, cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax[0])
        ax[0].set_title('Confusion matrix')
        ax[0].set_ylabel('True label')
        ax[0].set_xlabel('Predicted label')

        # Plot action history
        if hasattr(model, 'action_history'):
            print(f"Action history: {len(model.action_history)} layers")
            ratios = []
            for layer in range(model.num_iterations):
                actions = model.action_history[layer].cpu()

                # Convert list to numpy array
                actions_array = np.array(actions)

                # Compute the ratio
                ratio = np.bincount(actions_array, minlength=4) / len(actions_array)
                ratio = np.round(ratio, 4)
                print(f"Ratio of actions in layer {layer}: {ratio} (S, L, B, I)")
                ratios.append(ratio)

            # Plot the ratios as stackplot
            x_axis = np.array(range(model.num_iterations))
            y_axis = np.array(ratios).T
            ax[1].stackplot(x_axis, y_axis, labels=['Standard', 'Listen', 'Broadcast', 'Isolate'])
            ax[1].set_xticks(range(model.num_iterations))
            ax[1].set_xlim(0, model.num_iterations-1)
            ax[1].set_title('Action history')
            ax[1].set_ylabel('Ratio')
            ax[1].set_xlabel('Iteration')
            ax[1].legend(loc='upper right')

        if show_graphs:
            v_index = np.arange(G.num_v)

            # Plot the training and test nodes
            colors = ['blue', 'pink', 'orange']
            vertex_colors = [colors[c] for c in labels]
            G.draw(v_label=v_index, v_color=vertex_colors, e_color='black')

            # Plot the accuracy
            accuracy_mask = [1 if p == t else 0 for p, t in zip(pred, test_labels)] # 1 if correct, 0 if incorrect for each test node
            accuracy_colors = ['blue' if t else 'gray' for _, t in enumerate(test_mask)] # blue if test node, gray if not
            j = 0
            for i, c in enumerate(accuracy_colors):
                if c == 'gray':
                    accuracy_colors[i] = 'gray'
                else:
                    accuracy_colors[i] = 'red' if accuracy_mask[j] == 0 else 'green'
                    j += 1

            G.draw(v_label=v_index, v_color=accuracy_colors, e_color='black')

            # Plot node actions
            if hasattr(model, 'action_history'):
                for layer in range(model.num_iterations):
                    actions = model.action_history[layer].cpu()
                    colors = ['blue', 'pink', 'orange', 'gray'] # Standard, Listen, Broadcast, Isolate
                    vertex_colors = [colors[a] for a in actions]
                    G.draw(v_label=v_index, v_color=vertex_colors, e_color='black')

        plt.show()

    model.save_action_history = False

class EarlyStopping:
    def __init__(self, patience=5, mode='min', delta=0.0, break_training=False):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.best_model = None
        self.break_training = break_training

        if self.mode == 'min':
            self.best_score = np.Inf
        else:
            self.best_score = -np.Inf

    def __call__(self, model, score):
        if self.mode == 'min':
            if score < self.best_score:
                self.best_score = score
                self.best_model = copy.deepcopy(model)
            if score < self.best_score + self.delta:
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score:
                self.best_score = score
                self.best_model = copy.deepcopy(model)
            if score > self.best_score + self.delta:
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.stop = True

        return self.stop