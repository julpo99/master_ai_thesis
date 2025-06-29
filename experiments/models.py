import kgbench as kg
import torch
import torch.nn.functional as F
from torch import nn

from experiments.utils import enrich, sum_sparse_rgcn, adj, sum_sparse, spmm


class RGCN(nn.Module):
    """
    Relational Graph Convolutional Network (R-GCN)
    """

    def __init__(self, triples: torch.Tensor, num_nodes: int, num_rels: int, num_classes: int, emb_dim: int = 16,
                 bases: int = None, enrich_flag=False):
        """
        Initialize the R-GCN model
        :param triples (torch.Tensor): 2D tensor of triples
        :param num_nodes (int): Number of entities
        :param num_rels (int): Number of relations
        :param num_classes (int): Number of classes
        :param emb_dim (int): Embedding size
        :param bases (int): Number of bases
        """

        super().__init__()

        self.emb_dim = emb_dim
        self.bases = bases
        self.num_classes = num_classes

        if enrich_flag:
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
        values /= sum_sparse_rgcn(ver_indices, values, ver_size)
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


class RGCN_EMB(nn.Module):
    """
    Relational Graph Convolutional Network (R-GCN) with embeddings
    """

    def __init__(self, triples: torch.Tensor, num_nodes: int, num_rels: int, num_classes: int, emb_dim: int = 1600,
                 weights_size: int = 16, bases: int = None, enrich_flag=False):
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

        super().__init__()

        self.emb_dim = emb_dim
        self.weights_size = weights_size
        self.bases = bases
        self.num_classes = num_classes

        if enrich_flag:
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
        # rel_nodes, _ = ver_size
        num_rels = rel_nodes // num_nodes

        kg.tic()

        # Initialize adjacency matrix values
        values = torch.ones(ver_indices.size(0), dtype=torch.float, device=triples.device)
        values /= sum_sparse_rgcn(ver_indices, values, ver_size)
        print(f'Sum sparse computed in {kg.toc():.2}s')

        kg.tic()
        # Create sparse adjacency matrices
        self.register_buffer('hor_graph', torch.sparse_coo_tensor(hor_indices.T, values, size=hor_size,
                                                                  dtype=torch.float32))
        # self.register_buffer('ver_graph', torch.sparse_coo_tensor(ver_indices.T, values, size=ver_size,
        #                                                           dtype=torch.float32))
        print(f'Sparse tensors created in {kg.toc():.3}s')

        kg.tic()

        # Trainable Node Embeddings
        self.node_embeddings = nn.Parameter(torch.FloatTensor(num_nodes, emb_dim))  # Learnable embeddings for nodes
        # Kaiming initialization for node embeddings
        nn.init.kaiming_normal_(self.node_embeddings, mode='fan_in')
        # The initialisation below is faster but not as good in our experience
        # nn.init.xavier_uniform_(self.node_embeddings)

        # Initialize weight matrices
        self.weights1, self.bases1 = self._init_layer(num_rels, emb_dim, weights_size, bases)
        self.weights2, self.bases2 = self._init_layer(num_rels, weights_size, num_classes, bases)

        kg.tic()
        # Initialize biases
        self.bias1 = nn.Parameter(torch.zeros(weights_size))
        self.bias2 = nn.Parameter(torch.zeros(num_classes))

        print(f'Weights initialized in {kg.toc():.3}s')

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
            weights = torch.transpose(self.weights1, 0, 1)
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


class LGCN(nn.Module):
    """
    Latent Graph Convolutional Networks (L-GCN)
    """

    def __init__(self, triples: torch.Tensor, num_nodes: int, num_rels: int, num_classes: int, emb_dim: int = 16,
                 rp=16, ldepth=0, lwidth=64, dropout=0.0, enrich_flag=False):
        """
        Initialize the L-GCN model
        :param triples (torch.Tensor): 2D tensor of triples
        :param num_nodes (int): Number of entities
        :param num_rels (int): Number of relations
        :param num_classes (int): Number of classes
        :param emb_dim (int): Embedding size
        """

        super().__init__()

        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.dropout = nn.Dropout(p=dropout)

        if enrich_flag:
            kg.tic()

            # Enrich triples with inverses and self-loops
            triples = enrich(triples, num_nodes, num_rels)

            print(f'Triples enriched in {kg.toc():.2}s')
            num_rels = num_rels * 2 + 1  # number of relations (including inverses and self-loops)

        # Compute the (non-relational) index pairs of connected edges, and a dense matrix of n-hot encodings of the relations
        pairs = triples[:, [0, 2]]
        unique_pairs, inverse_indices = torch.unique(pairs, dim=0, return_inverse=True)
        nt = unique_pairs.size(0)  # Number of unique node pairs

        # Create a mapping from (subject, object) to index
        pair_to_index = {tuple(pair.tolist()): idx for idx, pair in enumerate(unique_pairs)}

        # Compute indices for the horizontally and vertically stacked adjacency matrices.
        # -- All edges have all relations, so we just take the indices above and repeat them a bunch of times
        s, o = unique_pairs[:, 0][None, :], unique_pairs[:, 1][None, :]
        rm = torch.arange(rp)[:, None]  # relation multiplier
        se, oe = (s * rm).reshape(-1, 1), (o * rm).reshape(-1, 1)
        # -- indices multiplied by relation
        s, o = s.expand(rp, nt).reshape(-1, 1), o.expand(rp, nt).reshape(-1, 1)

        self.register_buffer('hindices', torch.cat([s, oe], dim=1))
        self.register_buffer('vindices', torch.cat([se, o], dim=1))

        self.register_buffer('nhots', torch.zeros(nt, num_rels))
        s, p, o = triples[:, 0], triples[:, 1], triples[:, 2]
        rows = torch.tensor([pair_to_index[(int(s_), int(o_))] for s_, o_ in zip(s, o)])
        self.nhots[rows, p] = 1

        # maps relations to latent relations (one per layer)
        if ldepth == 0:
            to_latent1 = [nn.Linear(num_rels, rp)]
            to_latent2 = [nn.Linear(num_rels, rp)]
        else:
            to_latent1 = [nn.Linear(num_rels, lwidth)]
            to_latent2 = [nn.Linear(num_rels, lwidth)]
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

        self.rp, self.r, self.num_nodes, self.nt = rp, num_rels, num_nodes, nt

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

        rp, r, n, nt = self.rp, self.r, self.num_nodes, self.nt

        latents1 = self.to_latent1(self.nhots)
        latents1 = self.dropout(latents1)
        assert latents1.size() == (nt, rp)
        latents1 = torch.softmax(latents1, dim=1)
        latents1 = latents1.t().reshape(-1)
        assert latents1.size() == (nt * rp,)

        # column normalize
        latents1 = latents1 / sum_sparse(self.hindices, latents1, (n, n * rp), row=False).clamp(min=1e-6)

        assert self.hindices.size(0) == latents1.size(0), f'{self.indices.size()} {latents1.size()}'

        ## Layer 1
        e = self.emb_dim
        c = self.num_classes

        weights = self.weights1

        assert weights.size() == (rp, n, e)

        # Apply weights and sum over relations
        # h = torch.mm(hor_graph, )

        h = spmm(indices=self.hindices, values=latents1, size=(n, n * rp), xmatrix=weights.view(n * rp, e))
        h = self.dropout(h)
        assert h.size() == (n, e)

        h = F.relu(h + self.bias1)

        ## Layer 2

        latents2 = self.to_latent2(self.nhots)
        assert latents2.size() == (nt, rp)
        latents2 = torch.softmax(latents2, dim=1)
        latents2 = latents2.t().reshape(-1)
        assert latents2.size() == (nt * rp,)
        latents2 = LACT(latents2)

        # row normalize
        latents2 = latents2 / sum_sparse(self.vindices, latents2, (n * rp, n), row=True).clamp(min=1e-6)

        # Multiply adjacencies by hidden
        # h = torch.mm(ver_graph, h) # sparse mm
        h = spmm(indices=self.vindices, values=latents2, size=(n * rp, n), xmatrix=h)
        assert h.size() == (n * rp, e)

        h = h.view(rp, n, e)  # new dim for the relations

        weights = self.weights2

        assert weights.size() == (rp, e, c)

        # Apply weights, sum over relations
        h = torch.einsum('rhc, rnh -> nc', weights, h)
        # h = torch.bmm(h, weights).sum(dim=0)

        assert h.size() == (n, c)

        # return h + self.bias2  # -- softmax is applied in the loss
        logits = h + self.bias2
        if torch.isnan(logits).any():
            print("NaN detected in logits!", logits.max().item(), logits.min().item())
        return logits

    def penalty(self, p=2):
        """
        Compute L2 penalty for regularization.
        """
        assert p == 2, "Only L2 penalty is supported"

        return self.weights1.pow(p).sum()


class LGCN_REL_EMB(nn.Module):
    """
    Latent Graph Convolutional Network with explicit relation embeddings.
    """

    def __init__(self, triples: torch.Tensor, num_nodes: int, num_rels: int, num_classes: int, emb_dim: int = 16,
                 rp=16, dropout=0.0, enrich_flag=False):
        """
        Initialize the LGCN model with relation embeddings.

        Args:
            triples (torch.Tensor): Tensor of shape (num_triples, 3) containing (subject, relation, object) triples.
            num_nodes (int): Number of nodes/entities in the graph.
            num_rels (int): Number of unique relations.
            num_classes (int): Number of output classes.
            emb_dim (int): Dimension of node embeddings.
            rp (int): Relation projection dimension.
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.rp = rp
        self.dropout = nn.Dropout(p=dropout)

        if enrich_flag:
            kg.tic()

            # Enrich triples with inverses and self-loops
            triples = enrich(triples, num_nodes, num_rels)

            print(f'Triples enriched in {kg.toc():.2}s')
            self.num_rels = self.num_rels * 2 + 1  # number of relations (including inverses and self-loops)

        # Extract unique (subject, object) pairs
        pairs = triples[:, [0, 2]]
        unique_pairs, inverse_indices = torch.unique(pairs, dim=0, return_inverse=True)
        nt = unique_pairs.size(0)  # Number of unique node pairs

        # Create a mapping from (subject, object) to index
        pair_to_index = {tuple(pair.tolist()): idx for idx, pair in enumerate(unique_pairs)}

        # Construct the binary relation matrix (nt x num_rels) (Old dense matrix approach)
        # relation_matrix = torch.zeros(nt, num_rels)
        # for idx, (s, r, o) in enumerate(triples.tolist()):
        #     pair_idx = pair_to_index[(s, o)]
        #     relation_matrix[pair_idx, r] = 1.0
        # self.register_buffer('relation_matrix_old', relation_matrix)

        # New sparse matrix approach
        row_indices = []
        col_indices = []

        for s, r, o in triples.tolist():
            pair_idx = pair_to_index[(s, o)]
            row_indices.append(pair_idx)
            col_indices.append(r)

        indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
        values = torch.ones(len(row_indices), dtype=torch.float)
        relation_matrix = torch.sparse_coo_tensor(indices, values, size=(nt, self.num_rels))
        self.register_buffer('relation_matrix', relation_matrix.coalesce())

        # Define learnable relation embeddings (num_rels x rp)
        self.relation_embeddings = nn.Parameter(torch.randn(self.num_rels, rp))

        # Compute indices for the horizontally and vertically stacked adjacency matrices.
        # -- All edges have all relations, so we just take the indices above and repeat them a bunch of times
        s, o = unique_pairs[:, 0][None, :], unique_pairs[:, 1][None, :]
        rm = torch.arange(rp)[:, None]  # relation multiplier
        se, oe = (s * rm).reshape(-1, 1), (o * rm).reshape(-1, 1)
        # -- indices multiplied by relation
        s, o = s.expand(rp, nt).reshape(-1, 1), o.expand(rp, nt).reshape(-1, 1)

        self.register_buffer('hindices', torch.cat([s, oe], dim=1))
        self.register_buffer('vindices', torch.cat([se, o], dim=1))

        self.rp, self.r, self.num_nodes, self.nt = rp, self.num_rels, num_nodes, nt

        # Initialize weights for two layers
        self.weights1 = nn.Parameter(torch.empty(rp, num_nodes, emb_dim))
        self.weights2 = nn.Parameter(torch.empty(rp, emb_dim, num_classes))
        nn.init.xavier_uniform_(self.weights1, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weights2, gain=nn.init.calculate_gain('relu'))

        # Bias terms
        self.bias1 = nn.Parameter(torch.zeros(emb_dim))
        self.bias2 = nn.Parameter(torch.zeros(num_classes))

    def forward(self):
        """
        Forward pass of the LGCN model.
        """

        rp, r, n, nt = self.rp, self.num_rels, self.num_nodes, self.nt

        # Compute latent representations for each node pair
        # latents = self.relation_matrix @ self.relation_embeddings  # Shape: (nt, rp) (Old dense matrix approach)
        latents = torch.sparse.mm(self.relation_matrix, self.relation_embeddings)
        latents = torch.softmax(latents, dim=1)  # Normalize across relations
        latents_flat = latents.t().reshape(-1)  # Shape: (nt * rp,)

        # Normalize latents for horizontal adjacency
        latents_norm = latents_flat / sum_sparse(self.hindices, latents_flat,
                                                 (n, n * rp), row=False)

        # First layer: Apply weights and aggregate
        e = self.emb_dim
        c = self.num_classes

        weights1_flat = self.weights1.view(rp * n, e)
        h = spmm(self.hindices, latents_norm, (n, n * rp), weights1_flat)
        assert h.size() == (n, e)
        h = F.relu(h + self.bias1)

        h = self.dropout(h)
        # Normalize latents for vertical adjacency
        latents_norm = latents_flat / sum_sparse(self.vindices, latents_flat,
                                                 (n * rp, n), row=True)

        # Second layer: Aggregate and apply weights
        h = spmm(self.vindices, latents_norm, (n * rp, n), h)
        h = h.view(rp, n, e)
        h = torch.einsum('rhc,rnh->nc', self.weights2, h)

        return h + self.bias2  # Output logits

    def penalty(self, p=2):
        """
        Compute L2 penalty for regularization.

        Args:
            p (int): Norm degree (only supports L2 norm).

        Returns:
            torch.Tensor: L2 penalty.
        """
        assert p == 2, "Only L2 penalty is supported."
        return self.weights1.pow(p).sum()
