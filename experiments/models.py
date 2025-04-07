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
        else:
            weights = self.weights1

        assert weights.size() == (num_rels, num_nodes, self.emb_dim)

        # Apply weights and sum over relations
        h = torch.matmul(self.hor_graph, weights.view(num_rels * num_nodes, self.emb_dim))

        assert h.size() == (num_nodes, self.emb_dim)

        h = F.relu(h + self.bias1)

        # Layer 2
        h = torch.matmul(self.ver_graph, h)  # Sparse matrix multiplication
        h = h.view(num_rels, num_nodes, self.emb_dim)  # New dimension for relations

        if self.bases2 is not None:
            weights = torch.matmul(self.weights2, self.bases2.view(self.bases, -1)).view(num_rels, self.emb_dim,
                                                                                         self.num_classes)
        else:
            weights = self.weights2

        assert weights.size() == (num_rels, self.emb_dim, self.num_classes)

        # Apply weights and sum over relations
        h = torch.bmm(h, weights).sum(dim=0)

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
                 weights_size: int = 16, bases: int = None):
        """
        Initialize the L-GCN model
        :param triples (torch.Tensor): 2D tensor of triples
        :param num_nodes (int): Number of entities
        :param num_rels (int): Number of relations
        :param num_classes (int): Number of classes
        :param emb_dim (int): Embedding size
        :param weights_size (int): Size of the weight matrix
        :param bases (int): Number of bases
        """

        super(LGCN, self).__init__()

        self.emb_dim = emb_dim
        self.weights_size = weights_size
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
        # self.register_buffer('ver_graph', torch.sparse_coo_tensor(ver_indices.T, values, size=ver_size,
        #                                                           dtype=torch.float32))
        print(f'Sparse tensors created in {kg.toc():.3}s')

        # Trainable Node Embeddings
        self.node_embeddings = nn.Parameter(torch.FloatTensor(num_nodes, emb_dim))  # Learnable embeddings for nodes
        # Kaiming initialization for node embeddings
        nn.init.kaiming_normal_(self.node_embeddings, mode='fan_in')

        # Initialize weight matrices
        self.weights1, self.bases1 = self._init_layer(num_rels, emb_dim, weights_size, bases)
        self.weights2, self.bases2 = self._init_layer(num_rels, weights_size, num_classes, bases)

        # Initialize biases
        self.bias1 = nn.Parameter(torch.zeros(weights_size))
        self.bias2 = nn.Parameter(torch.zeros(num_classes))

    def _init_layer(self, num_rels, in_dim, out_dim, bases):
        """
        Helper function to initialize weight layers
        """
        if bases is None:
            weights = nn.Parameter(torch.empty(num_rels, in_dim, out_dim))
            nn.init.xavier_uniform_(weights, gain=nn.init.calculate_gain('relu'))
            return weights
        else:
            comps = nn.Parameter(torch.empty(num_rels, bases))
            bases = nn.Parameter(torch.empty(bases, in_dim, out_dim))
            nn.init.xavier_uniform_(comps, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(bases, gain=nn.init.calculate_gain('relu'))
            return comps, bases

    def forward(self):
        """
        Forward pass of L-GCN
        """

        num_nodes, rel_nodes = self.hor_graph.shape
        num_rels = rel_nodes // num_nodes

        # Layer 1 L-GCN

        if self.bases1 is not None:
            weights = torch.matmul(self.weights1, self.bases1.view(self.bases, -1)).view(num_rels, self.emb_dim,
                                                                                         self.weights_size)
            assert weights.shape == (num_rels, self.emb_dim, self.weights_size)

            weights = torch.transpose(weights, 0, 1)
            weights = torch.transpose(weights, 1, 2)

            assert weights.shape == (self.emb_dim, self.weights_size, num_rels), f'weights.shape: {weights.shape}'

        else:
            print(f'No bases1')
            weights = torch.transpose(self.weights, 0, 1)
            weights = torch.transpose(weights, 1, 2)

            assert weights.shape == (self.emb_dim, self.weights_size, num_rels), f'weights.shape: {weights.shape}'

        assert self.node_embeddings.size() == (num_nodes, self.emb_dim)

        assert weights.size() == (self.emb_dim, self.weights_size, num_rels)

        h = torch.einsum('ni,ior->rno', self.node_embeddings, weights)

        assert h.size() == (num_rels, num_nodes, self.weights_size)

        h = torch.reshape(h, (num_rels * num_nodes, self.weights_size))

        assert h.size() == (num_rels * num_nodes, self.weights_size)

        assert self.hor_graph.size() == (num_nodes, num_rels * num_nodes)

        h = torch.matmul(self.hor_graph, h)

        assert h.size() == (num_nodes, self.weights_size)

        assert self.bias1.size() == (self.weights_size,)

        h = F.relu(h + self.bias1)

        # Layer 2 L-GCN

        if self.bases2 is not None:
            weights = torch.matmul(self.weights2, self.bases2.view(self.bases, -1)).view(num_rels, self.weights_size,
                                                                                         self.num_classes)
            assert weights.shape == (num_rels, self.weights_size, self.num_classes)

            weights = torch.transpose(weights, 0, 1)
            weights = torch.transpose(weights, 1, 2)

        else:
            print(f'No bases2')
            weights = torch.transpose(self.weights2, 0, 1)
            weights = torch.transpose(weights, 1, 2)

        assert weights.size() == (self.weights_size, self.num_classes, num_rels), f'weights.shape: {weights.shape}'

        h = torch.einsum('ni,icr->rnc', h, weights)

        assert h.size() == (num_rels, num_nodes, self.num_classes)

        h = torch.reshape(h, (num_rels * num_nodes, self.num_classes))

        assert h.size() == (num_rels * num_nodes, self.num_classes)

        assert self.hor_graph.size() == (num_nodes, num_rels * num_nodes)

        h = torch.matmul(self.hor_graph, h)

        assert h.size() == (num_nodes, self.num_classes)

        assert self.bias2.size() == (self.num_classes,)

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

        r = len(set(triples[:, 1].tolist()))

        # Compute the (non-relational) index pairs of connected edges, and a dense matrix of n-hot encodings of the relations
        indices = list(set(map(tuple, triples[:, [0, 2]].tolist())))

        # triples = triples.tolist()
        # ctr = Counter((s, o) for (s, _, o) in triples)
        # print(ctr.most_common(250))
        # exit()

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

        # triples_list = triples.tolist()
        self.register_buffer('nhots', torch.zeros(nt, r))
        # for s, p, o in triples_list:
        #     self.nhots[p2i[(s, o)], p] = 1

        s, p, o = triples[:, 0], triples[:, 1], triples[:, 2]
        rows = torch.tensor([p2i[(int(s_), int(o_))] for s_, o_ in zip(s, o)])
        self.nhots[rows, p] = 1

        # -- filling a torch tensor this way is pretty slow. Might be better to start with a python list

        print(self.nhots.sum(dim=1).mean())

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
