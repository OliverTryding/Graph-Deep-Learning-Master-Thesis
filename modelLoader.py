import torch

import dhg
from dhg import Graph
from dhg import Hypergraph

def load_model(model_name: str, args: dict, device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    if model_name == 'HGNN':
        num_features = args['num_features']
        num_classes = args['num_classes']
        model = dhg.models.GCN(num_features, num_classes).to(device)
    elif model_name == 'HCoGNN':
        action_net = dhg.models.action_network(num_features, "mean", torch.nn.ReLU(), [32], depth=0, dropout=0.5).to(device)
        environment_net = dhg.models.environment_network(num_features, "mean", torch.nn.ReLU(), [128], depth=1, dropout=0.5).to(device)
        model = dhg.models.HCoGNN_node_classifier(num_features, num_classes, 2, torch.nn.ReLU(), action_net, environment_net, [64], tau=0.01, dropout=0.5, layerNorm=True).to(device)
    else:
        raise ValueError(f"Model {model_name} not found.")
    return model