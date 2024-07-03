import torch

import dhg
from dhg import Graph
from dhg import Hypergraph
from dhg.data import BaseData
from dhg.data import Cora
from dhg.data import Citeseer
from dhg.data import Pubmed
from dhg.data import CoauthorshipCora
from dhg.data import CocitationCora
from dhg.data import YelpRestaurant
from dhg.data import Yelp3k
from dhg.data import IMDB4k
from dhg.data import Cooking200
import dhg.datapipe as dd

def load_dataset(dataset_name: str):
    if dataset_name == 'cora':
        return Cora()
    elif dataset_name == 'citeseer':
        return Citeseer()
    elif dataset_name == 'pubmed':
        return Pubmed()
    elif dataset_name == 'coauthorship_cora':
        return CoauthorshipCora()
    elif dataset_name == 'cocitation_cora':
        return CocitationCora()
    elif dataset_name == 'yelp_restaurant':
        return YelpRestaurant()
    elif dataset_name == 'yelp3k':
        return Yelp3k()
    elif dataset_name == 'imdb4k':
        return IMDB4k()
    elif dataset_name == 'cooking200':
        return Cooking200()
    else:
        raise ValueError(f"Dataset {dataset_name} not found.")
    
def load_data(dataset_name: str, train_percentage: float = 0.1):
    dataset = load_dataset(dataset_name)
    X = dataset['features']
    labels = dataset['labels']
    G = Hypergraph(dataset['num_vertices'], dataset['edge_list'])

    # Print the data object
    print("Data object:", dataset)

    # Print the number of classes
    num_classes = dataset['num_classes']
    print("Number of classes:", num_classes)

    # Print the number of nodes
    num_nodes = dataset['num_vertices']
    print("Number of nodes:", num_nodes)

    # Print the number of node features
    num_node_features = dataset['dim_features']
    print("Number of node features:", num_node_features)

    # Print number of edges
    num_edges = dataset['num_edges']
    print("Number of edges:", num_edges)

    # Print maximum edge size
    max_edge_size = max([len(edge) for edge in dataset['edge_list']])
    print("Maximum edge size:", max_edge_size)

    # Print the node features
    print("Node features:", dataset['features'].shape)

    # Print the labels
    print("Labels:", dataset['labels'].shape)

    # Get the minimum train masks
    min_train_mask = dataset['train_mask']

    # Compute new masks
    num_train = int(train_percentage * num_nodes)
    num_test = num_nodes - num_train
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:num_train] = True
    test_mask[num_train:] = True
    perm = torch.randperm(num_nodes)
    train_mask = train_mask[perm]
    test_mask = test_mask[perm]
    val_mask = test_mask.clone()

    # Set the minimum train mask
    train_mask[min_train_mask] = True
    test_mask[min_train_mask] = False
    val_mask[min_train_mask] = False

    return X, labels, G, num_classes, num_nodes, num_node_features, num_edges, max_edge_size, train_mask, val_mask, test_mask