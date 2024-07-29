import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('C:\\Users\\lolis\\Documents\\UNI\\Kurser\\master_thesis\\dhg')

import dhg
from dhg.data import BaseData
from dhg import Hypergraph

def generate_dataset(data, num, *args):
    dataset = []
    if data == 'synth':
        for i in range(num):
            hg, labels, features = generate_hypergraph_dataset(*args)
            dataset.append((hg, labels, features))
    elif data == 'minesweeper':
        for i in range(num):
            hg, labels, features = generate_minesweeper(*args)
            dataset.append((hg, labels, features))
    return dataset

def generate_hypergraph_dataset(num_vertices, num_hyperedges, num_classes, homophily, feature_dim=None):
    vertices = np.arange(num_vertices)
    labels = np.random.randint(0, num_classes, size=num_vertices)
    p = 0.3

    features = None
    if feature_dim:
        if feature_dim == 1:
            # Easy features
            features = np.zeros((num_vertices, feature_dim))
            for i in range(num_vertices):
                # Set features to be the same as the label
                features[i] = np.array([labels[i]])
        else:
            # Rand features
            features = np.random.rand(num_vertices, feature_dim)
    
    hyperedges = []
    for _ in range(num_hyperedges):
        possible_edge_sizes = np.arange(2, num_vertices // 2)
        edge_size_probs = (1-p) ** (possible_edge_sizes-1) * p
        edge_size_probs /= edge_size_probs.sum()
        edge_size = np.random.choice(possible_edge_sizes, p=edge_size_probs)
        hyperedge_vertices = np.random.choice(vertices, size=edge_size, replace=False)
        
        if np.random.rand() < homophily:
            # edge_label = np.random.choice(labels[hyperedge_vertices])
            edge_label = labels[hyperedge_vertices[0]]
            labels[hyperedge_vertices] = edge_label
        
        hyperedges.append(hyperedge_vertices)

    hg = Hypergraph(num_vertices, hyperedges)

    features = None
    if feature_dim:
        if feature_dim == 1:
            # Easy features
            features = np.zeros((num_vertices, feature_dim))
            for i in range(num_vertices):
                # Set features to be the same as the label
                if np.random.rand() < 0.5:
                    features[i] = np.array([labels[i]])
                else:
                    features[i] = np.array([1-labels[i]])
        else:
            # Rand features
            features = np.random.rand(num_vertices, feature_dim)
    
    return hg, labels, features

def generate_action_test_dataset():
    vertices = np.arange(10)
    labels = np.array([2, 1, 0, 0, 0, 0, 0, 0, 1, 2])
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5, 6, 7], [7,8], [8, 9]]
    hg = Hypergraph(10, edges)
    features = np.random.rand(10, 3)

def generate_minesweeper(board_dim: int = 10, num_mines: int = 10):
    # Generate a minesweeper board
    board = np.zeros((board_dim, board_dim))
    num_mines = num_mines
    mine_positions = np.random.choice(board_dim**2, num_mines, replace=False)
    mine_positions = np.unravel_index(mine_positions, (board_dim, board_dim))
    board[mine_positions] = 1

    # Generate the size 2 hyperedges for adjacent vertices. Vertices are adjacent if they are horizontally, vertically or diagonally adjacent.
    adjacent_vertices = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            for i in range(board_dim):
                for j in range(board_dim):
                    x = i + dx
                    y = j + dy
                    if x >= 0 and x < board_dim and y >= 0 and y < board_dim:
                        vertex1 = i * board_dim + j
                        vertex2 = x * board_dim + y
                        adjacent_vertices.append([vertex1, vertex2])

    hyperedges = adjacent_vertices

    hg = Hypergraph(board_dim**2, hyperedges)
    labels = board.flatten()
    features = np.zeros((board_dim**2, 1))
    return hg, labels, features

def draw_minesweeper(hg, labels):
    board_dim = int(np.sqrt(hg.num_v))
    color_list = ['red' if l == 1 else 'blue' for l in labels]
    hg.draw(v_color=color_list, e_color='black')

if __name__ == '__main__':

    # First dataset
    # # Parameters
    # num_vertices = 20
    # num_hyperedges = 10
    # num_classes = 3
    # homophily = 0.2
    # feature_dim = 3
    # colors = ['red', 'blue', 'green']

    # # Generate the dataset
    # print('Generating the dataset...')

    # hg, labels, features = generate_hypergraph_dataset(
    #     num_vertices, num_hyperedges, num_classes, homophily, feature_dim)

    # color_list = [colors[l] for l in labels]

    # print('Dataset generated.')

    # print(hg.e)
    # print(features)

    # print('Drawing the hypergraph...')

    # hg.draw(v_color=color_list, e_color='black')
    # plt.show()

    # Minesweeper dataset
    # Parameters
    num_mines = 5
    board_dim = 4
    colors = ['blue', 'red']

    # Generate the dataset
    print('Generating the dataset...')
    hg, labels, features = generate_minesweeper(board_dim, num_mines)

    color_list = [colors[int(l)] for l in labels]

    print('Dataset generated.')

    print(hg.e)
    print(features)

    print('Drawing the hypergraph...')
    hg.draw(v_color=color_list, e_color='black')
    plt.show()
