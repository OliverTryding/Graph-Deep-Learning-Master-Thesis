import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors

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
            ms = minesweeper_data(*args)
            dataset.append(ms)
    elif data == 'rootneighbors':
        num_train = num // 3
        num_val = num // 3
        num_test = num - num_train - num_val
        for i in range(num_train):
            rn = RootNeighbors(train=True)
            hg, labels, features = rn()
            dataset.append((hg, labels, features))
        for i in range(num_val):
            rn = RootNeighbors(train=False)
            hg, labels, features = rn()
            dataset.append((hg, labels, features))
        for i in range(num_test):
            rn = RootNeighbors(train=False)
            hg, labels, features = rn()
            dataset.append((hg, labels, features))
    else:
        raise ValueError(f"Invalid dataset: {data}")
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

class minesweeper_data():
    def __init__(self, num_mines, board_dim):
        self.num_mines = num_mines
        self.board_dim = board_dim
        self.colors = ['blue', 'red']
        self.hg, self.labels, self.features, self.board = self.set_minesweeper()
        #self.color_list = [self.colors[int(l)] for l in self.labels]

    def __getitem__(self, index):
        return self.hg, self.labels, self.features

    def __len__(self):
        return 1

    def __str__(self):
        return f'Minesweeper dataset with {self.num_mines} mines and board dimensions {self.board_dim}x{self.board_dim}'

    def __repr__(self):
        return f'Minesweeper dataset with {self.num_mines} mines and board dimensions {self.board_dim}x{self.board_dim}'

    def __call__(self):
        return self.hg, self.labels, self.features

    def __iter__(self):
        return iter([self.hg, self.labels, self.features])

    def __next__(self):
        return self.hg, self.labels, self.features

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.hg, self.labels, self.features

    def __str__(self):
        return f'Minesweeper dataset with {self.num_mines} mines and board dimensions {self.board_dim}x{self.board_dim}'

    def __repr__(self):
        return f'Minesweeper dataset with {self.num_mines} mines and board dimensions {self.board_dim}x{self.board_dim}'

    def __call__(self):
        return self.hg, self.labels, self.features

    def __iter__(self):
        return iter([self.hg, self.labels, self.features])

    def __next__(self):
        return self.hg, self.labels, self.features

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.hg, self.labels, self.features

    def __str__(self):
        return f'Minesweeper dataset with {self.num_mines} mines and board dimensions {self.board_dim}x{self.board_dim}'

    def set_minesweeper(self):
        # Generate a minesweeper board
        board_dim = self.board_dim
        num_mines = self.num_mines
        board = np.zeros((board_dim, board_dim))
        mine_positions = np.random.choice(board_dim**2, num_mines, replace=False)
        mine_positions = np.unravel_index(mine_positions, (board_dim, board_dim))
        board[mine_positions] = 1

        # Set features and edges for minesweeper
        adjacent_vertices = []
        features = np.zeros((board_dim**2, 9))
        for i in range(board_dim):
            for j in range(board_dim):
                vertex = i * board_dim + j
                adjacent_mines = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        x = i + dx
                        y = j + dy
                        if x >= 0 and x < board_dim and y >= 0 and y < board_dim:
                            vertex1 = i * board_dim + j
                            vertex2 = x * board_dim + y
                            adjacent_vertices.append([vertex1, vertex2])
                            if board[x, y] == 1:
                                adjacent_mines += 1
                features[vertex,adjacent_mines] = 1

        hg = Hypergraph(board_dim**2, adjacent_vertices)

        labels = board.flatten()
        anti_labels = 1 - labels
        labels = np.stack((labels, anti_labels), axis=1)

        return hg, labels, features, board

    def draw(self, custom_colors=None, ax=None, savefig=False):
        """
        Visualizes the Minesweeper board using Matplotlib with optional custom colors.
        
        Parameters:
            board (list of list of int): The Minesweeper board, where -1 represents a mine and other integers represent the count of adjacent mines.
            custom_colors (dict): Optional dictionary mapping each unique label to a color.
            labels (list of list of int): Optional labels for custom coloring.
        """

        board_dim = self.board_dim
        labels = self.labels
        features = self.features

        # Convert the board to a NumPy array for easier manipulation
        mines = labels[:,0].reshape((board_dim, board_dim))
        board = features.argmax(axis=1).reshape((board_dim, board_dim)) * (mines == 0) - mines

        # Create a figure and axis
        if ax is None:
            fig_dim = max(board_dim // 10, 10)
            fig, ax = plt.subplots(figsize=(fig_dim, fig_dim))
            ax.set_aspect('equal')

        # Define default color map
        cmap = mcolors.ListedColormap(['white', 'blue', 'green', 'pink', 'red', 'purple', 'maroon', 'turquoise', 'black', 'gray'])
        bounds = [-1.5, -0.5] + [i + 0.5 for i in range(9)]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Calculate dynamic font size based on board size
        num_rows, num_cols = board.shape
        font_size = max(min(100 / max(num_rows, num_cols), 15), 6)  # Adjust the scaling factor and limits as needed

        if custom_colors is None:       
            # Plot the mines and numbers
            for (i, j), value in np.ndenumerate(board):
                if value == -1:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color='white'))
                    ax.text(j, i, 'M', ha='center', va='center', color='red', fontsize=font_size)
                elif value == 0:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color='lightgrey'))
                    pass
                else:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color='white'))
                    ax.text(j, i, str(int(value)), ha='center', va='center', color='black', fontsize=font_size)
            
            # Plot the background colors
            colored_board = np.where(board == -1, -1, board)
            ax.imshow(colored_board, cmap=cmap, origin='upper')
            
        else:
            # Plot the mines and numbers with custom colors
            for (i, j), value in np.ndenumerate(board):
                if value == -1:
                    ax.text(j, i, 'M', ha='center', va='center', color='black', fontsize=font_size)
                elif value == 0:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color='lightgrey'))
                else:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color='white'))
                    ax.text(j, i, str(value), ha='center', va='center', color='black', fontsize=font_size)

                v_idx = int(i*board_dim+j)
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color=custom_colors[v_idx]))


        # Set the grid lines
        ax.set_xticks(np.arange(-.5, board.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, board.shape[0], 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
        
        # Remove axis labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        if savefig:
            with PdfPages('minesweeper.pdf') as pdf:
                pdf.savefig()  # saves the current figure
                plt.close()  # close the figure

        return ax

class RootNeighbors():
    """Given a rooted tree, predict the average of the features of root-neighbors of degree 6"""
    def __init__(self, train=True):
        self.train = train
        self.level_1 = None
        self.level_2 = None
        self.feature_dim = 5
        self.hg, self.labels, self.features = self.set_rooted_tree()

    def __call__(self):
        return self.hg, self.labels, self.features

    def set_rooted_tree(self):
        hyper_edges = []
        
        # The number of nodes in the first level of each tree in the train, validation, and test set is sampled from a uniform distribution U[3, 10], U[5, 12], and U[5, 12] respectively.
        num_nodes_1 = np.random.randint(3, 11) if self.train else np.random.randint(5, 13)
        self.level_1 = num_nodes_1
        for i in range(num_nodes_1):
            hyper_edges.append([0, i+1])

        # The number of level-1 nodes with a degree of 6 is sampled independently from a uniform distribution U[1, 3], U[3, 5], U[3, 5] for the train, validation, and test set, respectively.
        num_degree_6 = np.random.randint(1, 4) if self.train else np.random.randint(3, 6)
        self.level_2 = num_degree_6
        degree_6_nodes = np.random.choice(np.arange(1, num_nodes_1+1), num_degree_6, replace=False).tolist()
        num_nodes = num_nodes_1 + 1
        for i in degree_6_nodes:
            for j in range(6):
                hyper_edges.append([i, num_nodes + j])
            num_nodes += 6

        # The degree of the remaining level-1 nodes are sampled from the uniform distribution U[2, 3].
        remaining_nodes = [i for i in range(1, num_nodes_1+1) if i not in degree_6_nodes]
        for i in remaining_nodes:
            degree = np.random.randint(2, 4)
            for j in range(degree):
                hyper_edges.append([i, num_nodes + j])
            num_nodes += degree

        self.num_vertices = num_nodes

        # Each feature is independently sampled from a uniform distribution U[−2, 2]
        features = np.random.uniform(-2, 2, (self.num_vertices, self.feature_dim))

        hg = Hypergraph(num_nodes, hyper_edges)

        # The labels are the average of the features of the root-neighbors of the degree-6 nodes.
        labels = features[degree_6_nodes].mean(axis=0)

        return hg, labels, features

    def draw(self, custom_colors=None):
        if custom_colors:
            pass
        else:
            # Set colors for each level
            color_list = ['' for _ in range(self.hg.num_v)]
            color_list[0] = 'red'
            for i in range(1, self.level_1+1):
                color_list[i] = 'blue'
            for i in range(self.level_1+1, len(color_list)):
                color_list[i] = 'green'
            fig = plt.figure(figsize=(10, 10))
            hg.draw(v_color=color_list, e_color='black')
            plt.show()

        with PdfPages('rootneighbor.pdf') as pdf:
            pdf.savefig()  # saves the current figure
            plt.close()  # close the figure

        

if __name__ == '__main__':

    # First dataset -------------------------------
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

    # Minesweeper dataset -------------------------------
    # Parameters
    num_mines = 10
    board_dim = 4
    colors = ['blue', 'red']

    # Generate the dataset
    print('Generating the dataset...')
    ms = minesweeper_data(num_mines, board_dim)
    hg, labels, features = ms()

    print('Dataset generated.')
    custom_colors = ['lightgrey', 'green'] * 8

    ms.draw(custom_colors, savefig=True)

    # RootNeighbors dataset -------------------------------
    # # Parameters

    # # Generate the dataset
    # print('Generating the dataset...')
    # rn = RootNeighbors(train=False)
    # hg, labels, features = rn()

    # print(f"Labels: {labels}")

    # print('Dataset generated.')

    # rn.draw()
