from typing import Tuple

import torch
from torch import Tensor


def enrich(triples: Tensor, n: int, r: int) -> Tensor:
    """
    Enrich triples by adding inverse relations and self-loop edges.

    Args:
        triples (Tensor): A 2D tensor of shape (num_triples, 3), where each row is (subject, relation, object).
        n (int): The number of entities (nodes) in the graph.
        r (int): The number of distinct relations (before adding inverses and self-loops).

    Returns:
        Tensor: A 2D tensor of enriched triples with shape (num_triples * 2 + n, 3),
                including original triples, inverse relations, and self-loops.
    """
    device = triples.device

    inverses = torch.cat([
        triples[:, 2:],
        triples[:, 1:2] + r,
        triples[:, :1]
    ], dim=1)

    node_indices = torch.arange(n, dtype=torch.long, device=device)[:, None]
    selfloops = torch.cat([
        node_indices,
        torch.full_like(node_indices, fill_value=2 * r, dtype=torch.long, device=device),
        node_indices
    ], dim=1)

    return torch.cat([triples, inverses, selfloops], dim=0)


def sum_sparse_rgcn(indices: torch.Tensor, values: torch.Tensor, size: tuple, row: bool = True) -> torch.Tensor:
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


def sum_sparse(
        indices: torch.Tensor,
        values: torch.Tensor,
        size: Tuple[int, int],
        row: bool = True
) -> torch.Tensor:
    """
    Sums the rows or columns of a sparse matrix defined by (indices, values) and redistributes
    the sums back to each corresponding entry.

    Supports optional batching by prepending batch dimensions.

    Args:
        indices (torch.Tensor): Shape (B?, nnz, 2). Sparse indices of the matrix.
        values (torch.Tensor): Shape (B?, nnz). Non-zero values of the matrix.
        size (Tuple[int, int]): Shape of the sparse matrix (rows, columns).
        row (bool): If True, sum rows; else sum columns.

    Returns:
        torch.Tensor: Tensor of shape (B?, nnz), sum per entry's row/column.
    """

    assert len(indices.shape) == len(values.shape) + 1, "Mismatch in index/value tensor dimensions"

    if indices.dim() == 2:
        # Add batch dimension
        indices = indices.unsqueeze(0)  # (1, nnz, 2)
        values = values.unsqueeze(0)  # (1, nnz)
        bdims = None
    else:
        # Fold up batch dimension
        bdims = indices.shape[:-2]
        k, r = indices.shape[-2:]
        assert bdims == values.shape[:-1], "Batch dimensions must match"
        assert values.shape[-1] == k, "Number of values must match number of indices"

        indices = indices.view(-1, k, r)
        values = values.view(-1, k)

    b, k, r = indices.shape

    if not row:
        # Transpose index pairs
        indices = torch.cat([indices[:, :, 1:2], indices[:, :, 0:1]], dim=2)
        size = (size[1], size[0])

    ones = torch.ones((size[1], 1), device=indices.device)  # (cols, 1)
    ones = ones.expand(b, -1, -1).contiguous()  # (b, cols, 1)

    # Sparse batch matrix multiplication
    sums = batchmm(indices, values, size, ones)  # shape: (b, rows, 1)

    bindex = torch.arange(b, device=indices.device)[:, None].expand(b, k)
    result = sums[bindex, indices[:, :, 0], 0]  # gather sums for each original index

    return result.view(*bdims, k) if bdims else result.view(k)


def batchmm(
        indices: Tensor,
        values: Tensor,
        size: Tuple[int, int],
        xmatrix: Tensor,
) -> Tensor:
    """
    Performs batch multiplication of sparse matrices (in COO format) with a batch of dense matrices.

    Each sparse matrix is defined by its `indices`, `values`, and `size`, and is multiplied with
    the corresponding dense matrix from `xmatrix`.

    Args:
        indices (Tensor): Tensor of shape (B, N, 2) containing indices for the sparse matrices in batch,
                          where B is the batch size and N is the number of non-zero entries.
        values (Tensor): Tensor of shape (B, N) containing the non-zero values for each sparse matrix in the batch.
        size (Tuple[int, int]): (height, width) of each sparse matrix.
        xmatrix (Tensor): Dense matrix of shape (B, width, D) to be multiplied.

    Returns:
        Tensor: Resulting tensor of shape (B, height, D) from the sparse-dense matrix multiplication.
    """

    device = indices.device
    B, N, _ = indices.size()
    height, width = size

    size_tensor = torch.tensor(size, device=device, dtype=torch.long)

    # Construct batched indices by offsetting each batch appropriately
    bmult = size_tensor[None, None, :].expand(B, N, 2)
    batch_ids = torch.arange(B, device=device, dtype=torch.long)[:, None, None].expand(B, N, 2)

    bindices = (batch_ids * bmult + indices).view(B * N, 2)

    bvalues = values.contiguous().view(-1)

    _, _, D = xmatrix.size()
    bxmatrix = xmatrix.view(-1, D)

    assert bindices.device == bvalues.device == bxmatrix.device

    # Perform sparse matrix multiplication
    result = spmm(bindices.t(), bvalues, (height * B, width), bxmatrix)

    return result.view(B, height, D)


class SparseMMCPU(torch.autograd.Function):
    """
    Sparse matrix multiplication with autograd support for sparse values and xmatrix.

    NOTE: This implementation assumes the sparse matrix is given in COO format (indices, values).
    """

    @staticmethod
    def forward(
            ctx,
            indices: Tensor,
            values: Tensor,
            size: Tuple[int, int],
            xmatrix: Tensor
    ) -> Tensor:
        """
        Forward pass: Computes Y = A @ X where A is a sparse matrix defined by (indices, values, size).
        """
        print(f'Using CPU for sparse matrix multiplication (forward)')
        assert indices.shape[0] == 2, "Indices must be of shape (2, nnz)"
        A = torch.sparse_coo_tensor(indices, values, size, device='cpu')
        A = A.coalesce()  # Ensure no duplicate indices

        ctx.save_for_backward(indices, values, xmatrix)
        ctx.shape = size

        return torch.sparse.mm(A, xmatrix)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[None, Tensor, None, Tensor]:
        """
        Backward pass for sparse values and dense xmatrix.

        Returns:
            (grad_indices=None, grad_values, grad_size=None, grad_xmatrix)
        """
        print(f'Using CPU for sparse matrix multiplication (backward)')
        indices, values, xmatrix = ctx.saved_tensors
        rows, cols = ctx.shape

        i = indices[0, :]
        j = indices[1, :]

        # dL/dvalues = sum over D dims of (dL/dY[i] * X[j])
        grad_output_select = grad_output[i]  # shape: (nnz, D)
        xmatrix_select = xmatrix[j]  # shape: (nnz, D)
        grad_values = (grad_output_select * xmatrix_select).sum(dim=1)

        # dL/dX = Aᵗ @ dL/dY
        A = torch.sparse_coo_tensor(indices, values, (rows, cols), device='cpu').coalesce()
        grad_xmatrix = torch.sparse.mm(A.transpose(0, 1), grad_output)

        return None, grad_values, None, grad_xmatrix


class SparseMMGPU(torch.autograd.Function):
    """
    Sparse matrix multiplication with autograd support for sparse values and xmatrix.

    NOTE: This implementation assumes the sparse matrix is given in COO format (indices, values).
    """

    @staticmethod
    def forward(
            ctx,
            indices: Tensor,
            values: Tensor,
            size: Tuple[int, int],
            xmatrix: Tensor
    ) -> Tensor:
        """
        Forward pass: Computes Y = A @ X where A is a sparse matrix defined by (indices, values, size).
        """
        print(f'Using GPU for sparse matrix multiplication (forward)')
        assert indices.shape[0] == 2, "Indices must be of shape (2, nnz)"
        A = torch.sparse_coo_tensor(indices, values, size, device='cuda')
        A = A.coalesce()  # Ensure no duplicate indices

        ctx.save_for_backward(indices, values, xmatrix)
        ctx.shape = size

        return torch.sparse.mm(A, xmatrix)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[None, Tensor, None, Tensor]:
        """
        Backward pass for sparse values and dense xmatrix.

        Returns:
            (grad_indices=None, grad_values, grad_size=None, grad_xmatrix)
        """
        print(f'Using GPU for sparse matrix multiplication (backward)')
        print(f'ctx.saved_tensors: {ctx.saved_tensors}')
        indices, values, xmatrix = ctx.saved_tensors
        rows, cols = ctx.shape

        i = indices[0, :]
        j = indices[1, :]

        # dL/dvalues = sum over D dims of (dL/dY[i] * X[j])
        grad_output_select = grad_output[i]  # shape: (nnz, D)
        xmatrix_select = xmatrix[j]  # shape: (nnz, D)
        grad_values = (grad_output_select * xmatrix_select).sum(dim=1)

        # dL/dX = Aᵗ @ dL/dY
        A = torch.sparse_coo_tensor(indices, values, (rows, cols), device='cuda').coalesce()
        grad_xmatrix = torch.sparse.mm(A.transpose(0, 1), grad_output)

        return None, grad_values, None, grad_xmatrix


def spmm(
        indices: torch.Tensor,
        values: torch.Tensor,
        size: Tuple[int, int],
        xmatrix: torch.Tensor
) -> torch.Tensor:
    """
    Performs sparse matrix multiplication (SPMM) with autograd support for values and xmatrix.

    Args:
        indices (Tensor): (nnz, 2) or (2, nnz) sparse indices in COO format.
        values (Tensor): (nnz,) sparse values.
        size (Tuple[int, int]): Shape of the sparse matrix.
        xmatrix (Tensor): Dense matrix to multiply with.

    Returns:
        Tensor: Result of shape (size[0], xmatrix.shape[1])
    """
    assert indices.device == values.device == xmatrix.device, \
        "All tensors must be on the same device."

    if indices.shape[0] != 2:
        indices = indices.t()

    if indices.is_cuda:
        print("Using GPU for sparse matrix multiplication")
        return SparseMMGPU.apply(indices, values, size, xmatrix)
    else:
        print("Using CPU for sparse matrix multiplication")
        return SparseMMCPU.apply(indices, values, size, xmatrix)


def adj(
        triples: torch.Tensor,
        num_nodes: int,
        num_rels: int,
        vertical: bool = True
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Computes a sparse adjacency matrix for the given graph by stacking
    adjacency matrices of all relations vertically or horizontally.

    Args:
        triples (torch.Tensor): Tensor of shape (num_edges, 3), each row is (subject, relation, object).
        num_nodes (int): Total number of nodes in the graph.
        num_rels (int): Total number of relation types.
        vertical (bool): If True, stack adjacency matrices vertically (shape: [r*n, n]),
                         else horizontally (shape: [n, r*n]).

    Returns:
        Tuple[torch.Tensor, Tuple[int, int]]: A tuple where the first element is a 2D tensor of indices
        for the sparse matrix (shape: [num_edges, 2]), and the second element is the shape of the matrix.
    """
    device = triples.device

    # Define matrix shape
    size = (num_rels * num_nodes, num_nodes) if vertical else (num_nodes, num_rels * num_nodes)

    # Extract subject, relation, object
    fr = triples[:, 0].clone()
    rel = triples[:, 1].clone()
    to = triples[:, 2].clone()

    offset = rel * num_nodes
    fr_offset = fr + offset if vertical else fr
    to_offset = to + offset if not vertical else to

    # Combine indices
    indices = torch.cat([
        fr_offset.unsqueeze(0),
        to_offset.unsqueeze(0)
    ], dim=0)

    # Validate bounds
    assert indices.size(1) == triples.size(0), "Mismatch in number of triples and indices"
    assert indices[0].max() < size[0], f"Max index {indices[0].max()} exceeds matrix size {size[0]}"
    assert indices[1].max() < size[1], f"Max index {indices[1].max()} exceeds matrix size {size[1]}"

    return indices.t().to(device), size

# def intlist(
#         tensor: Union[torch.Tensor,
#         List[int],
#         Tuple[int, ...]]
# ) -> List[int]:
#     """
#     Converts a 1D tensor to a list of ints.
#     If the input is already a list or tuple, it is returned unchanged.
#
#     Args:
#         tensor (torch.Tensor | list | tuple): Input to be converted.
#
#     Returns:
#         List[int]: List of ints with the same values as the input.
#     """
#     if isinstance(tensor, (list, tuple)):
#         return list(tensor)
#
#     tensor = tensor.squeeze()
#     assert tensor.ndim == 1, f"Expected 1D tensor after squeeze, got shape {tensor.shape}"
#
#     return [int(x) for x in tensor]
