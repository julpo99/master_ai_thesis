import kgbench as kg
import torch
import torch.nn.functional as F
from torch import nn

from utils import enrich, sum_sparse, adj, sum_sparse2, spmm


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


class LGCN(nn.Module):
    """
    Latent Graph Convolutional Networks (L-GCN)
    """

    def __init__(self, triples: torch.Tensor, num_nodes: int, num_rels: int, num_classes: int, emb_dim: int = 1600,
                 weights_size: int = 16):
        """
        Initialize the L-GCN model
        :param triples (torch.Tensor): 2D tensor of triples
        :param num_nodes (int): Number of entities
        :param num_rels (int): Number of relations
        :param num_classes (int): Number of classes
        :param emb_dim (int): Embedding size
        """

        super(LGCN, self).__init__()

        self.emb_dim = emb_dim
        self.weights_size = weights_size
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

        # Trainable Node Embeddings
        self.node_embeddings = nn.Parameter(torch.FloatTensor(num_nodes, emb_dim))  # Learnable embeddings for nodes

        # Kaiming initialization for node embeddings
        nn.init.kaiming_normal_(self.node_embeddings, mode='fan_in')

        # Initialize weight matrices
        self.weights1 = self._init_layer(emb_dim, weights_size, num_rels)
        self.weights2 = self._init_layer(weights_size, weights_size, num_classes)

        # Initialize biases
        self.bias1 = nn.Parameter(torch.zeros(weights_size))
        self.bias2 = nn.Parameter(torch.zeros(num_classes))

    def _init_layer(self, rows, columns, layers):
        """
        Helper function to initialize weight layers
        """
        weights = nn.Parameter(torch.empty(rows, columns, layers))
        nn.init.xavier_uniform_(weights, gain=nn.init.calculate_gain('relu'))
        return weights

    def forward(self):
        """
        Forward pass of L-GCN
        """

        num_nodes, rel_nodes = self.hor_graph.shape
        num_rels = rel_nodes // num_nodes

        # Embedding layer
        weights = self.weights1

        assert self.node_embeddings.size() == (num_nodes, self.emb_dim)
        assert weights.size() == (self.emb_dim, self.weights_size, num_rels)

        h = torch.einsum('ni,ior->rno', self.node_embeddings, weights)

        assert h.size() == (num_rels, num_nodes, self.weights_size)

        h = torch.reshape(h, (num_rels * num_nodes, self.weights_size))

        assert h.size() == (num_rels * num_nodes, self.weights_size)

        assert self.hor_graph.size() == (num_nodes, num_rels * num_nodes)

        # Layer 1 L-GCN
        h = torch.matmul(self.hor_graph, h)

        assert h.size() == (num_nodes, self.weights_size)

        assert self.bias1.size() == (self.weights_size,)

        h = F.relu(h + self.bias1)

        # Layer 2 L-GCN

        weights = self.weights2

        assert weights.size() == (self.weights_size, self.weights_size, self.num_classes)

        h = torch.einsum('ni,ioc->cno', h, weights)

        assert h.size() == (self.num_classes, num_nodes, self.weights_size)

        h = torch.reshape(h, (self.weights_size, num_nodes, self.num_classes))

        assert h.size() == (self.weights_size, num_nodes, self.num_classes)

        h = torch.sum(h, dim=0)

        assert h.size() == (num_nodes, self.num_classes)

        return h + self.bias2  # softmax is applied in the loss function

    def penalty(self, p=2):
        """
        Compute L2 penalty for regularization.
        """
        assert p == 2, "Only L2 penalty is supported"

        return self.weights1.pow(p).sum()


class LGCN2(nn.Module):
    """
    Latent Graph Convolutional Networks (L-GCN)
    """

    def __init__(self, triples: torch.Tensor, num_nodes: int, num_rels: int, num_classes: int, emb_dim: int = 16,
                 weights_size: int = 16, rp=16, ldepth=0, lwidth=64):
        """
        Initialize the L-GCN model
        :param triples (torch.Tensor): 2D tensor of triples
        :param num_nodes (int): Number of entities
        :param num_rels (int): Number of relations
        :param num_classes (int): Number of classes
        :param emb_dim (int): Embedding size
        """

        super(LGCN2, self).__init__()

        self.emb_dim = emb_dim
        self.weights_size = weights_size
        self.num_classes = num_classes

        r = len(triples)

        # Compute the (non-relational) index pairs of connected edges, and a dense matrix of n-hot encodings of the relations
        indices = list({(s, o) for (s, _, o) in triples})
        p2i = {(s, o): i for i, (s, o) in enumerate(indices)}
        indices = torch.tensor(indices, dtype=torch.long)
        nt, _ = indices.size()

        # Compute indices for the horizontally and vertically stacked adjacency matrices.
        # -- All edges have all relations, so we just take the indices above and repeat them a bunch of times
        s, o = indices[:, 0][None, :], indices[:, 1][None, :]
        rm = torch.arange(rp)[:, None]  # relation multiplier
        se, oe = (s * rm).reshape(-1, 1), (o * rm).reshape(-1, 1)
        # -- indices multiplied by relation
        s, o = s.expand(rp, nt).reshape(-1, 1), o.expand(rp, nt).reshape(-1, 1)


        self.register_buffer('hindices', torch.cat([s, oe], dim=1))
        self.register_buffer('vindices', torch.cat([se, o], dim=1))

        self.register_buffer('nhots', torch.zeros(nt, r, dtype=torch.uint8))
        for s, p, o in triples:
            self.nhots[p2i[(s, o)], p] = 1
        # -- filling a torch tensor this way is pretty slow. Might be better to start with a python list


        # maps relations to latent relations (one per layer)
        to_latent1 = [nn.Linear(r, lwidth)]
        to_latent2 = [nn.Linear(r, lwidth)]
        for _ in range(ldepth - 1):
            to_latent1.append(nn.ReLU())
            to_latent2.append(nn.ReLU())

            to_latent1.append(nn.Linear(lwidth, lwidth))
            to_latent2.append(nn.Linear(lwidth, lwidth))

        to_latent1.append(nn.ReLU())
        to_latent2.append(nn.ReLU())

        to_latent1.append(nn.Linear(lwidth, rp))
        to_latent2.append(nn.Linear(lwidth, rp))

        self.to_latent1 = nn.Sequential(*to_latent1)
        self.to_latent2 = nn.Sequential(*to_latent2)

        self.rp, self.r, self.num_nodes, self.nt = rp, r, num_nodes, nt


        # layer 1 weights
        self.weights1 = self._init_layer(rp, num_nodes, emb_dim)

        # layer 2 weights
        self.weights2 = self._init_layer(rp, emb_dim, num_classes)

        self.bias1 = nn.Parameter(torch.FloatTensor(emb_dim).zero_())
        self.bias2 = nn.Parameter(torch.FloatTensor(num_classes).zero_())

    def _init_layer(self, rows, columns, layers):
        """
        Helper function to initialize weight layers
        """
        weights = nn.Parameter(torch.empty(rows, columns, layers))
        nn.init.xavier_uniform_(weights, gain=nn.init.calculate_gain('relu'))
        return weights

    def forward(self):
        """
        Forward pass of L-GCN
        """
        LACT = torch.relu

        rp, r, n, nt = self.rp, self.r, self.n, self.nt

        latents1 = self.to_latent1(self.nhots)
        assert latents1.size() == (nt, rp)
        latents1 = torch.softmax(latents1, dim=1)
        latents1 = latents1.t().reshape(-1)

        # column normalize
        latents1 = latents1 / sum_sparse2(self.hindices, latents1, (n, n * rp), row=False)

        assert self.hindices.size(0) == latents1.size(0), f'{self.indices.size()} {latents1.size()}'

        ## Layer 1
        e = self.emb
        b, c = self.bases, self.numcls

        weights = self.weights1

        assert weights.size() == (rp, n, e)

        # Apply weights and sum over relations
        # h = torch.mm(hor_graph, )
        h = spmm(indices=self.hindices, values=latents1, size=(n, n * rp), xmatrix=weights.view(rp * n, e))
        assert h.size() == (n, e)

        h = F.relu(h + self.bias1)

        ## Layer 2

        latents2 = self.to_latent2(self.nhots)
        assert latents2.size() == (nt, rp)
        latents2 = torch.softmax(latents2, dim=1)
        latents2 = latents2.t().reshape(-1)
        # latents2 = LACT(latents2)

        # row normalize
        latents2 = latents2 / sum_sparse2(self.vindices, latents2, (n * rp, n), row=True)

        # Multiply adjacencies by hidden
        # h = torch.mm(ver_graph, h) # sparse mm
        h = spmm(indices=self.vindices, values=latents2, size=(n * rp, n), xmatrix=h)

        h = h.view(rp, n, e)  # new dim for the relations

        weights = self.weights2

        # Apply weights, sum over relations
        h = torch.einsum('rhc, rnh -> nc', weights, h)
        # h = torch.bmm(h, weights).sum(dim=0)

        assert h.size() == (n, c)

        return h + self.bias2  # -- softmax is applied in the loss

    def penalty(self, p=2):
        """
        Compute L2 penalty for regularization.
        """
        assert p == 2, "Only L2 penalty is supported"

        return self.weights1.pow(p).sum()
