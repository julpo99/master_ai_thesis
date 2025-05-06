from collections import Counter

import kgbench as kg
import torch
import torch.nn.functional as F
from torch import nn


def enrich(triples: torch.Tensor, n: int, r: int) -> torch.Tensor:
    """
    Enrich the triples with their inverses and self-loops

    :param triples (torch.Tensor): 2D tensor of triples (shape: [num_edges, 3])
    :param n (int): Number of entities (nodes in the graph)
    :param r (int): Number of relations
    :return (torch.Tensor): Enriched triples (original + inverses + self-loops)
    """

    # Get the device of the input triples (CPU or CUDA)
    device = triples.device

    # Create the inverse triples: (object, relation + r, subject)
    inverses = torch.cat([
        triples[:, 2:],
        triples[:, 1:2] + r,
        triples[:, :1]
    ], dim=1)

    # Create self-loops: (node, self-loop relation id, node)
    node_indices = torch.arange(n, dtype=torch.long, device=device)[:, None]
    selfloops = torch.cat([
        node_indices,
        torch.full_like(node_indices, fill_value=2 * r, dtype=torch.long, device=device),
        node_indices
    ], dim=1)

    # Concatenate original, inverse, and self-loop triples
    return torch.cat([triples, inverses, selfloops], dim=0)


def sum_sparse(indices: torch.Tensor, values: torch.Tensor, size: tuple, row: bool = True) -> torch.Tensor:
    """
    Sum the rows or columns of a sparse matrix, and redistribute the results back to the non-sparse row/column entries.

    :param indices (torch.Tensor): 2D tensor of indices
    :param values (torch.Tensor): 1D tensor of values
    :param size (tuple): Size of the sparse matrix (rows, columns)
    :param row (bool): Whether to sum the rows or columns, default is True (sum the rows)
    :return (torch.Tensor): Tensor containing the sums of the rows or columns
    """

    # Ensure indices are on the correct device
    device = indices.device

    # Choose the appropriate sparse tensor type (torch.sparse_coo_tensor replaces deprecated torch.sparse.FloatTensor)
    ST = torch.sparse_coo_tensor

    # Ensure indices have the correct shape
    assert indices.dim() == 2

    k, _ = indices.shape

    # Transpose indices if summing columns instead of rows
    if not row:
        indices = torch.cat([indices[:, 1:2], indices[:, 0:1]], dim=1)
        size = (size[1], size[0])  # Swap matrix dimensions

    # Create a tensor of ones for summation
    ones = torch.ones((size[1], 1), device=device)

    # Create sparse matrix with proper CUDA support
    smatrix = ST(indices.t(), values, size=size, device=device)

    # Compute row/column sums using matrix multiplication
    sums = torch.mm(smatrix, ones)  # Row/column sums

    # Extract sums corresponding to indices
    sums = sums[indices[:, 0]]

    # Ensure output shape matches expectation
    assert sums.size() == (k, 1)

    return sums.view(k)


def adj(triples: torch.Tensor, num_nodes: int, num_rels: int, cuda: bool = False, vertical: bool = True) -> tuple:
    """
    Computes a sparse adjacency matrix for the given graph (the adjacency matrices of all relations are stacked vertically or horizontally).


    :param triples (torch.Tensor): 2D tensor of triples (subject, relation, object)
    :param num_nodes (int): Number of nodes
    :param num_rels (int): Number of relations
    :param cuda (bool): Whether to use CUDA
    :param vertical (bool): If True, stack adjacency matrices vertically; otherwise, horizontally.
    :return tuple: Tuple containing the adjacency matrix indices and its size
    """

    # Get adjacency matrix size
    size = (num_rels * num_nodes, num_nodes) if vertical else (num_nodes, num_rels * num_nodes)

    # Extract subjects (from), relations, and objects (to)
    fr, rel, to = triples[:, 0].clone(), triples[:, 1].clone(), triples[:, 2].clone()

    # Compute offsets based on stacking mode
    offset = rel * num_nodes

    if vertical:
        fr += offset
    else:
        to += offset

    # Stack adjacency indices (no need for Python lists)
    # indices = torch.stack([fr, to], dim=0)
    indices = torch.cat([fr.unsqueeze(0), to.unsqueeze(0)], dim=0)

    # Validate adjacency matrix indices
    assert indices.size(1) == triples.size(0), "Mismatch in number of triples and indices"
    assert indices[0].max() < size[0], f"Max index {indices[0].max()} exceeds matrix size {size[0]}"
    assert indices[1].max() < size[1], f"Max index {indices[1].max()} exceeds matrix size {size[1]}"

    return indices.t(), size


class RGCN(nn.Module):
    """
    Relational Graph Convolutional Network (R-GCN)
    """

    def __init__(self, triples: torch.Tensor, num_nodes: int, num_rels: int, num_classes: int, emb_dim: int = 16,
                 bases: int = None):
        """
        Initialize the R-GCN model
        :param triples (torch.Tensor): 2D tensor of triples
        :param num_nodes (int): Number of entities
        :param num_rels (int): Number of relations
        :param num_classes (int): Number of classes
        :param emb_dim (int): Embedding size
        :param bases (int): Number of bases
        """

        super(RGCN, self).__init__()

        self.emb_dim = emb_dim
        self.bases = bases
        self.num_classes = num_classes

        kg.tic()

        # Enrich triples with inverses and self-loops
        self.triples = enrich(triples, num_nodes, num_rels)

        print(f'Triples enriched in {kg.toc():.2}s')

        kg.tic()

        # Compute adjacency matrices
        hor_indices, hor_size = adj(self.triples, num_nodes, num_rels * 2 + 1, vertical=False)
        ver_indices, ver_size = adj(self.triples, num_nodes, num_rels * 2 + 1, vertical=True)

        print(f'Adjacency matrices computed in {kg.toc():.2}s')

        _, rel_nodes = hor_size
        num_rels = rel_nodes // num_nodes

        kg.tic()

        # Initialize adjacency matrix values
        values = torch.ones(ver_indices.size(0), dtype=torch.float, device=triples.device)
        values /= sum_sparse(ver_indices, values, ver_size)
        print(f'Sum sparse computed in {kg.toc():.2}s')

        kg.tic()
        # Create sparse adjacency matrices
        self.register_buffer('hor_graph', torch.sparse_coo_tensor(hor_indices.T, values, size=hor_size,
                                                                  dtype=torch.float32))
        self.register_buffer('ver_graph', torch.sparse_coo_tensor(ver_indices.T, values, size=ver_size,
                                                                  dtype=torch.float32))
        print(f'Sparse tensors created in {kg.toc():.3}s')

        # Initialize weight matrices
        self.weights1, self.bases1 = self._init_layer(num_rels, num_nodes, emb_dim, bases)  # First transformation layer
        self.weights2, self.bases2 = self._init_layer(num_rels, emb_dim, num_classes,
                                                      bases)  # Second transformation layer

        # Initialize biases
        self.bias1 = nn.Parameter(torch.zeros(emb_dim))
        self.bias2 = nn.Parameter(torch.zeros(num_classes))

    def _init_layer(self, num_rels, in_dim, out_dim, bases):
        """
        Helper function to initialize weight layers
        """
        if bases is None:
            weights = nn.Parameter(torch.empty(num_rels, in_dim, out_dim))
            nn.init.xavier_uniform_(weights, gain=nn.init.calculate_gain('relu'))
            return weights, None
        else:
            comps = nn.Parameter(torch.empty(num_rels, bases))
            bases = nn.Parameter(torch.empty(bases, in_dim, out_dim))
            nn.init.xavier_uniform_(comps, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(bases, gain=nn.init.calculate_gain('relu'))
            return comps, bases

    def forward(self):
        """
        Forward pass of R-GCN
        """

        ## Layer 1

        num_nodes, rel_nodes = self.hor_graph.shape
        num_rels = rel_nodes // num_nodes

        if self.bases1 is not None:
            weights = torch.matmul(self.weights1, self.bases1.view(self.bases, -1)).view(num_rels, num_nodes,
                                                                                         self.emb_dim)
            # q: what exactly is the line above doing? explain we in every detail please
            # a: The line above is reshaping the bases tensor to a 2D tensor with the number of rows equal to the number
            # of bases and the number of columns equal to the product of the number of input dimensions and the number of
            # output dimensions. Then, it is multiplying the weights tensor with the reshaped bases tensor and reshaping
            # the result to a 3D tensor with the number of dimensions equal to the number of relations, the number of nodes,
            # and the embedding dimension. This is done to apply the bases to the weights in the first layer of the R-GCN model.
            # q: but is see a torch.matmul, what does it do?
            # a: torch.matmul is a function that performs matrix multiplication between two tensors. In this case, it is multiplying the weights tensor with the reshaped bases tensor.
        else:
            weights = self.weights1

        assert weights.size() == (num_rels, num_nodes, self.emb_dim)

        # Apply weights and sum over relations
        h = torch.matmul(self.hor_graph, weights.view(num_rels * num_nodes, self.emb_dim))
        # q: what exactly is the line above doing? explain we in every detail please
        # a: The line above is multiplying the horizontal graph tensor with the reshaped weights tensor. The horizontal graph tensor
        # is a sparse tensor that represents the adjacency matrix of the graph, and the weights tensor is a 3D tensor that contains
        # the weights of the first layer of the R-GCN model. The result of the multiplication is a 2D tensor that contains the
        # weighted sum of the embeddings of the nodes in the graph.

        assert h.size() == (num_nodes, self.emb_dim)

        h = F.relu(h + self.bias1)

        # Layer 2
        h = torch.matmul(self.ver_graph, h)  # Sparse matrix multiplication
        h = h.view(num_rels, num_nodes, self.emb_dim)  # New dimension for relations

        if self.bases2 is not None:
            weights = torch.matmul(self.weights2, self.bases2.view(self.bases, -1)).view(num_rels, self.emb_dim,
                                                                                         self.num_classes)
            # q: what exactly is the line above doing? explain we in every detail please
            # a: The line above is reshaping the bases tensor to a 2D tensor with the number of rows equal to the number of bases
            # and the number of columns equal to the product of the number of input dimensions and the number of output dimensions.
            # Then, it is multiplying the weights tensor with the reshaped bases tensor and reshaping the result to a 3D tensor with
            # the number of dimensions equal to the number of relations, the embedding dimension, and the number of classes. This is
            # done to apply the bases to the weights in the second layer of the R-GCN model.

        else:
            weights = self.weights2

        assert weights.size() == (num_rels, self.emb_dim, self.num_classes)

        # Apply weights and sum over relations
        h = torch.bmm(h, weights).sum(dim=0)
        # q: what exactly is the line above doing? explain we in every detail please
        # a: The line above is performing batch matrix multiplication between the embeddings tensor and the weights tensor.
        # The embeddings tensor contains the embeddings of the nodes in the graph, and the weights tensor contains the weights
        # of the second layer of the R-GCN model. The result of the batch matrix multiplication is a 3D tensor that contains the
        # weighted sum of the embeddings of the nodes in the graph.
        # q: what is sum(dim=0) doing?
        # a: The sum(dim=0) function is summing the embeddings of the nodes in the graph along the first dimension of the tensor.
        # This operation is performed to aggregate the embeddings of the nodes in the graph into a single tensor that contains the
        # weighted sum of the embeddings of the nodes.
        # q: h is 67x1153679x10, weights is 67x10x8, what is the result of the multiplication?
        # a: The result of the multiplication is a 67x1153679x8 tensor. This tensor contains the weighted sum of the embeddings of
        # the nodes in the graph for each relation and class.
        # q: and how is it after the sum(dim=0)?
        # a: After the sum(dim=0) operation, the tensor is reduced along the first dimension, resulting in a 1153679x8 tensor that
        # contains the weighted sum of the embeddings of the nodes in the graph for each class.
        # q: explain me how sum(dim=0) works and how it changes the shape of the tensor
        # a: The sum(dim=0) function sums the elements of the tensor along the first dimension. This operation reduces the size of
        # the tensor along the first dimension and returns a tensor with the same shape as the input tensor, except for the first
        # dimension, which is removed. In this case, the sum(dim=0) operation reduces the size of the tensor along the first dimension,
        # resulting in a tensor with a shape of 1153679x8.

        assert h.size() == (num_nodes, self.num_classes)

        return h + self.bias2  # softmax is applied in the loss function

    def penalty(self, p=2):
        """
        Compute L2 penalty for regularization.
        """
        assert p == 2, "Only L2 penalty is supported"

        if self.bases is None:
            return self.weights1.pow(p).sum()

        return self.weights1.pow(p).sum() + self.bases1.pow(p).sum()


def go(name='amplus', lr=0.01, wd=0.0, l2=0.0, epochs=50, prune=False, optimizer='adam', final=False, emb_dim=16,
       bases=None, printnorms=None):
    # Load dataset
    data = kg.load(name, torch=True, prune_dist=2 if prune else None, final=final)

    print(f'Loaded {data.triples.size(0)} triples, {data.num_entities} entities, {data.num_relations} relations')

    kg.tic()

    # Initialize R-GCN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"):
        print('Using CUDA')

    rgcn = RGCN(data.triples, num_nodes=data.num_entities, num_rels=data.num_relations, num_classes=data.num_classes,
                emb_dim=emb_dim, bases=bases).to(device)

    print(f'Model created in {kg.toc():.3}s')

    # Move data to the same device as the model
    data.training = data.training.to(device)
    data.withheld = data.withheld.to(device)

    # Select optimizer
    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW
    }
    if optimizer not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    opt = optimizers[optimizer](rgcn.parameters(), lr=lr, weight_decay=wd)

    kg.tic()

    for e in range(epochs):
        kg.tic()

        # Zero gradients
        opt.zero_grad()

        # Forward pass
        out = rgcn()

        # Extract indices for training & withheld sets
        idxt, clst = data.training[:, 0], data.training[:, 1].to(torch.int64)
        idxw, clsw = data.withheld[:, 0], data.withheld[:, 1].to(torch.int64)

        # Compute loss
        out_train = out[idxt, :]
        loss = F.cross_entropy(out_train, clst, reduction='mean')

        if l2 > 0.0:
            loss += l2 * rgcn.penalty()

        # Backward pass (compute  gradients)
        loss.backward()

        # Update weights
        opt.step()

        # Compute performance metrics
        with torch.no_grad():
            preds_train = out[idxt].argmax(dim=1)
            preds_withheld = out[idxw].argmax(dim=1)

            training_acc = torch.sum(preds_train == clst).item() / idxt.size(0)
            withheld_acc = torch.sum(preds_withheld == clsw).item() / idxw.size(0)

        # Print epoch statistics
        print(
            f'Epoch {e:02}: \t\t loss {loss:.4f}, \t\t train acc {training_acc:.2f}, \t\t withheld acc'
            f' {withheld_acc:.2f}, '
            f'\t\t ({kg.toc():.3}s)')

        # Print relation norms if requested
        if printnorms is not None:
            nr = data.num_relations

            def print_norms(weights, layer_num):
                ctr = Counter()

                for r in range(nr):
                    ctr[data.i2r[r]] = weights[r].norm().item()
                    ctr['inv_' + data.i2r[r]] = weights[r + nr].norm().item()  # Handle inverse relations

                print(f'Relations with largest weight norms in layer {layer_num}.')
                for rel, w in ctr.most_common(printnorms):
                    print(f'    norm {w:.4f} for {rel}')

            print_norms(rgcn.weights1, 1)
            print_norms(rgcn.weights2, 2)

    print(f'\nTraining complete! (total time: {kg.toc() / 60:.2f}m)')


if __name__ == '__main__':
    go(name='amplus', lr=0.01, wd=0.0, l2=0.0005, epochs=50, prune=True, optimizer='adam', final=False, emb_dim=10,
       bases=40, printnorms=None)
