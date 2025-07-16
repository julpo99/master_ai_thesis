import torch
import torch.nn.functional as F
from torch import nn

import kgbench as kg
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

        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.bases = bases

        # Enrich triples with inverses and self-loops
        enrich_flag and kg.tic()
        triples = enrich(triples, self.num_nodes, self.num_rels) if enrich_flag else triples
        enrich_flag and print(f"Triples enriched in {kg.toc():.2f}s")
        self.num_rels = self.num_rels * 2 + 1 if enrich_flag else self.num_rels

        # Compute adjacency matrices
        kg.tic()
        hor_indices, hor_size = adj(triples, self.num_nodes, self.num_rels, vertical=False)
        ver_indices, ver_size = adj(triples, self.num_nodes, self.num_rels, vertical=True)

        print(f'Adjacency matrices computed in {kg.toc():.2}s')

        _, self.rel_nodes = hor_size
        self.num_rels = self.rel_nodes // self.num_nodes

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
        self.weights1, self.bases1 = self._init_layer(self.num_rels, self.num_nodes, self.emb_dim,
                                                      self.bases)  # First transformation layer
        self.weights2, self.bases2 = self._init_layer(self.num_rels, self.emb_dim, self.num_classes,
                                                      self.bases)  # Second transformation layer

        # Initialize biases
        self.bias1 = nn.Parameter(torch.zeros(self.emb_dim))
        self.bias2 = nn.Parameter(torch.zeros(self.num_classes))


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
        r, n, e, c = self.num_rels, self.num_nodes, self.emb_dim, self.num_classes

        n, rel_nodes = self.hor_graph.shape
        r = rel_nodes // n

        # Layer 1

        if self.bases1 is not None:
            weights = torch.matmul(self.weights1, self.bases1.view(self.bases, -1)).view(r, n, e)
        else:
            weights = self.weights1

        assert weights.size() == (r, n, e)

        # Apply weights and sum over relations
        h = torch.matmul(self.hor_graph, weights.view(r * n, e))

        assert h.size() == (n, e)

        h = F.relu(h + self.bias1)

        # Layer 2
        h = torch.matmul(self.ver_graph, h)  # Sparse matrix multiplication
        h = h.view(r, n, e)  # New dimension for relations

        if self.bases2 is not None:
            weights = torch.matmul(self.weights2, self.bases2.view(self.bases, -1)).view(r, e, c)
        else:
            weights = self.weights2

        assert weights.size() == (r, e, c)

        # Apply weights and sum over relations
        h = torch.bmm(h, weights).sum(dim=0)

        assert h.size() == (n, c)

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

        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.weights_size = weights_size
        self.bases = bases

        # Enrich triples with inverses and self-loops
        enrich_flag and kg.tic()
        triples = enrich(triples, self.num_nodes, self.num_rels) if enrich_flag else triples
        enrich_flag and print(f"Triples enriched in {kg.toc():.2f}s")
        self.num_rels = self.num_rels * 2 + 1 if enrich_flag else self.num_rels

        # Compute adjacency matrices
        kg.tic()
        hor_indices, hor_size = adj(triples, self.num_nodes, self.num_rels, vertical=False)
        ver_indices, ver_size = adj(triples, self.num_nodes, self.num_rels, vertical=True)

        print(f'Adjacency matrices computed in {kg.toc():.2}s')

        _, self.rel_nodes = hor_size
        self.num_rels = self.rel_nodes // self.num_nodes

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

        kg.tic()

        # Trainable Node Embeddings
        self.node_embeddings = nn.Parameter(torch.FloatTensor(self.num_nodes, self.emb_dim))  # Learnable embeddings
        # for nodes
        nn.init.kaiming_normal_(self.node_embeddings, mode='fan_in')

        # Initialize weight matrices
        self.weights1, self.bases1 = self._init_layer(self.num_rels, self.emb_dim, self.weights_size, self.bases)
        self.weights2, self.bases2 = self._init_layer(self.num_rels, self.weights_size, self.num_classes, self.bases)

        kg.tic()
        # Initialize biases
        self.bias1 = nn.Parameter(torch.zeros(self.weights_size))
        self.bias2 = nn.Parameter(torch.zeros(self.num_classes))


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
        r, n, e, w, c = self.num_rels, self.num_nodes, self.emb_dim, self.weights_size, self.num_classes

        n, rel_nodes = self.hor_graph.shape
        r = rel_nodes // n

        # Layer 1 L-GCN

        if self.bases1 is not None:
            weights = torch.matmul(self.weights1, self.bases1.view(self.bases, -1)).view(r, e, w)
            assert weights.shape == (r, e, w)

            weights = torch.transpose(weights, 0, 1)
            weights = torch.transpose(weights, 1, 2)

            assert weights.shape == (e, w, r), f'weights.shape: {weights.shape}'

        else:
            weights = torch.transpose(self.weights1, 0, 1)
            weights = torch.transpose(weights, 1, 2)

            assert weights.shape == (e, w, r), f'weights.shape: {weights.shape}'

        assert self.node_embeddings.size() == (n, e)

        assert weights.size() == (e, w, r)

        h = torch.einsum('ni,ior->rno', self.node_embeddings, weights)

        assert h.size() == (r, n, w)

        h = torch.reshape(h, (r * n, w))

        assert h.size() == (r * n, w)

        assert self.hor_graph.size() == (n, r * n)

        h = torch.matmul(self.hor_graph, h)

        assert h.size() == (n, w)

        assert self.bias1.size() == (w,)

        h = F.relu(h + self.bias1)

        # Layer 2 L-GCN

        if self.bases2 is not None:
            weights = torch.matmul(self.weights2, self.bases2.view(self.bases, -1)).view(r, w,
                                                                                         n)
            assert weights.shape == (r, w, n)

            weights = torch.transpose(weights, 0, 1)
            weights = torch.transpose(weights, 1, 2)

        else:
            weights = torch.transpose(self.weights2, 0, 1)
            weights = torch.transpose(weights, 1, 2)

        assert weights.size() == (w, n, r), f'weights.shape: {weights.shape}'

        h = torch.einsum('ni,icr->rnc', h, weights)

        assert h.size() == (r, n, n)

        h = torch.reshape(h, (r * n, n))

        assert h.size() == (r * n, n)

        assert self.hor_graph.size() == (n, r * n)

        h = torch.matmul(self.hor_graph, h)

        assert h.size() == (n, n)

        assert self.bias2.size() == (n,)

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

        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.rp = rp
        self.dropout = nn.Dropout(p=dropout)

        # Enrich triples with inverses and self-loops
        enrich_flag and kg.tic()
        triples = enrich(triples, self.num_nodes, self.num_rels) if enrich_flag else triples
        enrich_flag and print(f"Triples enriched in {kg.toc():.2f}s")
        self.num_rels = self.num_rels * 2 + 1 if enrich_flag else self.num_rels

        # 1. Extract unique (subject, object) pairs from triples
        pairs = triples[:, [0, 2]]
        unique_pairs, inverse_indices = torch.unique(pairs, dim=0, return_inverse=True)
        self.nt = int(unique_pairs.size(0))  # Number of unique node pairs

        # 2. Create the sparse (n-hot style) relation matrix: [nt, num_rels]
        row_indices = inverse_indices  # shape (num_triples,)
        col_indices = triples[:, 1]  # relation indices (num_triples,)

        indices = torch.stack([row_indices, col_indices], dim=0).long()  # shape (2, num_triples)
        values = torch.ones(row_indices.size(0), dtype=torch.float)

        relation_matrix = torch.sparse_coo_tensor(indices, values, size=(self.nt, self.num_rels)).coalesce()
        self.register_buffer('relation_matrix', relation_matrix)

        # 3. Compute horizontally and vertically stacked indices for latent graph edges
        s, o = unique_pairs[:, 0][None, :], unique_pairs[:, 1][None, :]
        rm = torch.arange(self.rp)[:, None]  # relation multiplier
        se, oe = (s * rm).reshape(-1, 1), (o * rm).reshape(-1, 1)
        s, o = s.expand(self.rp, self.nt).reshape(-1, 1), o.expand(self.rp, self.nt).reshape(-1, 1)

        self.register_buffer('hindices', torch.cat([s, oe], dim=1))
        self.register_buffer('vindices', torch.cat([se, o], dim=1))

        # maps relations to latent relations (one per layer)
        if ldepth == 0:
            to_latent1 = [nn.Linear(self.num_rels, self.rp)]
            to_latent2 = [nn.Linear(self.num_rels, self.rp)]
        else:
            to_latent1 = [nn.Linear(self.num_rels, lwidth)]
            to_latent2 = [nn.Linear(self.num_rels, lwidth)]
            for _ in range(ldepth - 1):
                to_latent1.append(nn.ReLU())
                to_latent2.append(nn.ReLU())

                to_latent1.append(nn.Linear(lwidth, lwidth))
                to_latent2.append(nn.Linear(lwidth, lwidth))

            to_latent1.append(nn.ReLU())
            to_latent2.append(nn.ReLU())

            to_latent1.append(nn.Linear(lwidth, self.rp))
            to_latent2.append(nn.Linear(lwidth, self.rp))

        self.to_latent1 = nn.Sequential(*to_latent1)
        self.to_latent2 = nn.Sequential(*to_latent2)

        # Initialize weights for two layers
        self.weights1 = nn.Parameter(torch.empty(self.rp, self.num_nodes, self.emb_dim))
        self.weights2 = nn.Parameter(torch.empty(self.rp, self.emb_dim, self.num_classes))
        nn.init.xavier_uniform_(self.weights1, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weights2, gain=nn.init.calculate_gain('relu'))

        # Bias terms
        self.bias1 = nn.Parameter(torch.zeros(emb_dim))
        self.bias2 = nn.Parameter(torch.zeros(num_classes))

    def forward(self):
        """
        Forward pass of L-GCN
        """
        LACT = torch.relu

        rp, r, n, nt = self.rp, self.num_rels, self.num_nodes, self.nt

        latents1 = self.to_latent1(self.relation_matrix)
        latents1 = self.dropout(latents1)
        assert latents1.size() == (nt, rp)
        # latents1 = torch.softmax(latents1, dim=1)
        latents1 = latents1.t().reshape(-1)
        assert latents1.size() == (nt * rp,)

        # column normalize
        latents1 = latents1 / sum_sparse(self.hindices, latents1, (n, n * rp), row=False)

        assert self.hindices.size(0) == latents1.size(0), f'{self.indices.size()} {latents1.size()}'

        ## Layer 1
        e = self.emb_dim
        c = self.num_classes

        weights = self.weights1

        assert weights.size() == (rp, n, e)

        weights_flat = weights.view(rp * n, e)

        # Check starts here
        A_check = torch.sparse_coo_tensor(self.hindices.t(), latents1, size=(n, n * rp)).coalesce()
        h_check = torch.sparse.mm(A_check, weights_flat)  # shape: (n, e)
        # Check ends here

        # Apply weights and sum over relations
        # h = torch.mm(hor_graph, )

        h = spmm(indices=self.hindices, values=latents1, size=(n, n * rp), xmatrix=weights.view(n * rp, e))
        h = self.dropout(h)
        assert h.size() == (n, e)

        h = F.relu(h + self.bias1)

        ## Layer 2

        latents2 = self.to_latent2(self.relation_matrix)
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
            dropout (float): Dropout probability.
            enrich_flag (bool): If True, enrich the relation embeddings.
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.rp = rp
        self.dropout = nn.Dropout(p=dropout)

        # Enrich triples with inverses and self-loops
        enrich_flag and kg.tic()
        triples = enrich(triples, self.num_nodes, self.num_rels) if enrich_flag else triples
        enrich_flag and print(f"Triples enriched in {kg.toc():.2f}s")
        self.num_rels = self.num_rels * 2 + 1 if enrich_flag else self.num_rels

        # 1. Extract unique (subject, object) pairs from triples
        pairs = triples[:, [0, 2]]
        unique_pairs, inverse_indices = torch.unique_consecutive(pairs, dim=0,
                                                                 return_inverse=True)  # unique_consecutive does not order the pairs, unique does
        self.nt = int(unique_pairs.size(0))  # Number of unique node pairs

        # 2. Create the sparse (n-hot style) relation matrix: [nt, num_rels]
        row_indices = inverse_indices  # shape (num_triples,)
        col_indices = triples[:, 1]  # relation indices (num_triples,)

        indices = torch.stack([row_indices, col_indices], dim=0).long()  # shape (2, num_triples)
        values = torch.ones(row_indices.size(0), dtype=torch.float)

        relation_matrix = torch.sparse_coo_tensor(indices, values, size=(self.nt, self.num_rels)).coalesce()
        self.register_buffer('relation_matrix', relation_matrix)

        # 3. Compute horizontally and vertically stacked indices for latent graph edges
        fr, to = unique_pairs[:, 0][None, :], unique_pairs[:, 1][None, :]
        rm = torch.arange(self.rp)[:, None]  # relation multiplier

        # fr_offset = (fr * rm).reshape(-1, 1) # LGCN Style
        # to_offset = (to * rm).reshape(-1, 1) # LGCN Style
        fr_offset = (fr + rm * num_nodes).reshape(-1, 1)  # RGCN Style
        to_offset = (to + rm * num_nodes).reshape(-1, 1)  # RGCN Style

        fr, to = fr.expand(self.rp, self.nt).reshape(-1, 1), to.expand(self.rp, self.nt).reshape(-1, 1)

        self.register_buffer('hor_indices', torch.cat([fr, to_offset], dim=1))
        self.register_buffer('ver_indices', torch.cat([fr_offset, to], dim=1))

        # Relation embedding matrix is now an identity matrix, non-trainable
        assert self.rp == self.num_rels, "To use identity relation embeddings, self.rp must equal self.num_rels"
        identity = torch.eye(self.num_rels)
        self.register_buffer('relation_embeddings', identity)

        # Initialize weights for two layers
        self.weights1 = nn.Parameter(torch.empty(self.rp, self.num_nodes, self.emb_dim))
        self.weights2 = nn.Parameter(torch.empty(self.rp, self.emb_dim, self.num_classes))
        nn.init.xavier_uniform_(self.weights1, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weights2, gain=nn.init.calculate_gain('relu'))

        # Bias terms
        self.bias1 = nn.Parameter(torch.zeros(emb_dim))
        self.bias2 = nn.Parameter(torch.zeros(num_classes))

    def forward(self):
        """
        Forward pass of the LGCN model.
        """

        rp, r, n, nt, e, c = self.rp, self.num_rels, self.num_nodes, self.nt, self.emb_dim, self.num_classes

        # Compute latent representations for each node pair
        # latents = self.relation_matrix @ self.relation_embeddings  # Shape: (nt, rp) (Old dense matrix approach)
        latents = torch.sparse.mm(self.relation_matrix, self.relation_embeddings)
        # latents = torch.softmax(latents, dim=1)  # Normalize across relations
        latents_flat = latents.t().reshape(-1)
        assert latents_flat.size() == (nt * rp,)

        # Normalize latents for horizontal adjacency
        latents_norm = latents_flat / sum_sparse(self.ver_indices, latents_flat,
                                                 (n * rp, n), row=True)
        latents_norm = torch.nan_to_num(latents_norm, nan=0.0)

        # First layer: Apply weights and aggregate
        weights1_flat = self.weights1.view(rp * n, e)

        # hor_graph check
        hor_graph = torch.sparse_coo_tensor(self.hor_indices.t(), latents_norm, size=(n, n * rp)).coalesce()
        hor_graph = hor_graph.to_dense()
        h_check = torch.mm(hor_graph, weights1_flat)  # shape: (n, e)

        # More efficient sparse matrix multiplication
        h = spmm(indices=self.hor_indices, values=latents_norm, size=(n, n * rp), xmatrix=weights1_flat)
        assert h.size() == (n, e)
        h = F.relu(h + self.bias1)

        # h = self.dropout(h)
        # Normalize latents for vertical adjacency
        latents_norm = latents_flat / sum_sparse(self.ver_indices, latents_flat,
                                                 (n * rp, n), row=True)
        latents_norm = torch.nan_to_num(latents_norm, nan=0.0)

        # ver_graph check
        ver_graph = torch.sparse_coo_tensor(self.ver_indices.t(), latents_norm, size=(n * rp, n)).coalesce()
        ver_graph = ver_graph.to_dense()
        h_check = torch.mm(ver_graph, h)

        # Second layer: Aggregate and apply weights
        h = spmm(self.ver_indices, latents_norm, (n * rp, n), h)
        h = h.view(rp, n, e)
        h = torch.einsum('rne,rec->nc', h, self.weights2)

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
