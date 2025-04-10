import torch
from torch.autograd import Variable


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


def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if type(tensor) == bool:
        return 'cuda' if tensor else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'


def spmm(indices, values, size, xmatrix):
    cuda = indices.is_cuda

    sm = sparsemm(cuda)
    return sm(indices.t(), values, size, xmatrix)


def sum_sparse2(indices, values, size, row=True):
    """
    Sum the rows or columns of a sparse matrix, and redistribute the
    results back to the non-sparse row/column entries

    Arguments are interpreted as defining sparse matrix. Any extra dimensions
    are treated as batch.

    :return:
    """

    assert len(indices.size()) == len(values.size()) + 1

    if len(indices.size()) == 2:
        # add batch dim
        indices = indices[None, :, :]
        values = values[None, :]
        bdims = None
    else:
        # fold up batch dim
        bdims = indices.size()[:-2]
        k, r = indices.size()[-2:]
        assert bdims == values.size()[:-1]
        assert values.size()[-1] == k

        indices = indices.view(-1, k, r)
        values = values.view(-1, k)

    b, k, r = indices.size()

    if not row:
        # transpose the matrix
        indices = torch.cat([indices[:, :, 1:2], indices[:, :, 0:1]], dim=2)
        size = size[1], size[0]

    ones = torch.ones((size[1], 1), device=d(indices))

    s, _ = ones.size()
    ones = ones[None, :, :].expand(b, s, 1).contiguous()

    # print(indices.size(), values.size(), size, ones.size())
    # sys.exit()

    sums = batchmm(indices, values, size, ones)  # row/column sums
    bindex = torch.arange(b, device=d(indices))[:, None].expand(b, indices.size(1))
    sums = sums[bindex, indices[:, :, 0], 0]

    if bdims is None:
        return sums.view(k)

    return sums.view(*bdims + (k,))


def batchmm(indices, values, size, xmatrix, cuda=None):
    """
    Multiply a batch of sparse matrices (indices, values, size) with a batch of dense matrices (xmatrix)

    :param indices:
    :param values:
    :param size:
    :param xmatrix:
    :return:
    """

    if cuda is None:
        cuda = indices.is_cuda

    b, n, r = indices.size()
    dv = 'cuda' if cuda else 'cpu'

    height, width = size

    size = torch.tensor(size, device=dv, dtype=torch.long)

    bmult = size[None, None, :].expand(b, n, 2)
    m = torch.arange(b, device=dv, dtype=torch.long)[:, None, None].expand(b, n, 2)

    bindices = (m * bmult).view(b * n, r) + indices.view(b * n, r)

    bfsize = Variable(size * b)
    bvalues = values.contiguous().view(-1)

    b, w, z = xmatrix.size()
    bxmatrix = xmatrix.view(-1, z)

    sm = sparsemm(cuda)

    result = sm(bindices.t(), bvalues, bfsize, bxmatrix)

    return result.view(b, height, -1)


def sparsemm(use_cuda):
    """
    :param use_cuda:
    :return:
    """

    return SparseMMGPU.apply if use_cuda else SparseMMCPU.apply


class SparseMMCPU(torch.autograd.Function):
    """
    Sparse matrix multiplication with gradients over the value-vector

    Does not work with batch dim.
    """

    @staticmethod
    def forward(ctx, indices, values, size, xmatrix):
        # print(type(size), size, list(size), intlist(size))
        # print(indices.size(), values.size(), torch.Size(intlist(size)))

        matrix = torch.sparse.FloatTensor(indices, values, torch.Size(intlist(size)))

        ctx.indices, ctx.matrix, ctx.xmatrix = indices, matrix, xmatrix

        return torch.mm(matrix, xmatrix)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.data

        # -- this will break recursive autograd, but it's the only way to get grad over sparse matrices

        i_ixs = ctx.indices[0, :]
        j_ixs = ctx.indices[1, :]
        output_select = grad_output[i_ixs, :]
        xmatrix_select = ctx.xmatrix[j_ixs, :]

        grad_values = (output_select * xmatrix_select).sum(dim=1)

        grad_xmatrix = torch.mm(ctx.matrix.t(), grad_output)
        return None, Variable(grad_values), None, Variable(grad_xmatrix)


class SparseMMGPU(torch.autograd.Function):
    """
    Sparse matrix multiplication with gradients over the value-vector

    Does not work with batch dim.
    """

    @staticmethod
    def forward(ctx, indices, values, size, xmatrix):
        # print(type(size), size, list(size), intlist(size))

        matrix = torch.cuda.sparse.FloatTensor(indices, values, torch.Size(intlist(size)))

        ctx.indices, ctx.matrix, ctx.xmatrix = indices, matrix, xmatrix

        return torch.mm(matrix, xmatrix)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.data

        # -- this will break recursive autograd, but it's the only way to get grad over sparse matrices

        i_ixs = ctx.indices[0, :]
        j_ixs = ctx.indices[1, :]
        output_select = grad_output[i_ixs]
        xmatrix_select = ctx.xmatrix[j_ixs]

        grad_values = (output_select * xmatrix_select).sum(dim=1)

        grad_xmatrix = torch.mm(ctx.matrix.t(), grad_output)
        return None, Variable(grad_values), None, Variable(grad_xmatrix)


def intlist(tensor):
    """
    A slow and stupid way to turn a tensor into an iterable over ints
    :param tensor:
    :return:
    """
    if type(tensor) is list or type(tensor) is tuple:
        return tensor

    tensor = tensor.squeeze()

    assert len(tensor.size()) == 1

    s = tensor.size()[0]

    l = [None] * s
    for i in range(s):
        l[i] = int(tensor[i])

    return l


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

    # Stack adjacency indices
    indices = torch.cat([fr.unsqueeze(0), to.unsqueeze(0)], dim=0)

    # Validate adjacency matrix indices
    assert indices.size(1) == triples.size(0), "Mismatch in number of triples and indices"
    assert indices[0].max() < size[0], f"Max index {indices[0].max()} exceeds matrix size {size[0]}"
    assert indices[1].max() < size[1], f"Max index {indices[1].max()} exceeds matrix size {size[1]}"

    return indices.t(), size
