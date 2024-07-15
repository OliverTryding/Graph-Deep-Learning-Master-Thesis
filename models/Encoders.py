import numpy as np

import torch

import dhg
from dhg import Graph
from dhg import Hypergraph

import torch.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def torch_sparse_to_scipy(sparse_tensor):
    # Ensure the tensor is in COO format
    sparse_tensor = sparse_tensor.coalesce()
    
    # Extract the indices and values
    indices = sparse_tensor.indices().numpy()
    values = sparse_tensor.values().numpy()
    shape = sparse_tensor.size()
    
    # Create a SciPy sparse matrix
    scipy_sparse_matrix = sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)
    
    return scipy_sparse_matrix

def laplacian_encoder(G: Hypergraph, k: int):
    """
        A function to encode the graph as a Laplacian matrix
    """
    L_sym = torch_sparse_to_scipy(G.L_sym.cpu())

    L_sym = L_sym.tocsc()  # Convert to CSC format

    # Compute the eigenvalues and eigenvectors of the Laplacian matrix
    eigenvals, eigenvecs = spla.eigsh(L_sym, k=k, which='SM')  # smallest magnitude eigenvalues

    return eigenvecs

class PosEncoder():
    """
        An Object for positional encoders
    """

    def __init__(self, k: int = 10):
        """
            Constructor for the PosEncoder class
        """
        self.k = k

    def __call__(self, x, G):
        """
            Call method for the PosEncoder class
        """
        eigenvecs = torch.tensor(laplacian_encoder(G, self.k), dtype=torch.float32).to(x.device)
        return torch.cat((x, eigenvecs), dim=1)
    
