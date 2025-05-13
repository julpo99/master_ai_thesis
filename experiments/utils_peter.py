def intlist_peter(tensor):
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


def sparsemm_peter(use_cuda):
    """
    :param use_cuda:
    :return:
    """

    return SparseMMGPU_peter.apply if use_cuda else SparseMMCPU_peter.apply


class SparseMMCPU_peter(torch.autograd.Function):
    """
    Sparse matrix multiplication with gradients over the value-vector

    Does not work with batch dim.
    """

    @staticmethod
    def forward(ctx, indices, values, size, xmatrix):
        # print(type(size), size, list(size), intlist(size))
        # print(indices.size(), values.size(), torch.Size(intlist(size)))

        # matrix: full sparse A
        # xmatrix: the X
        # returns Y = AX

        matrix = torch.sparse_coo_tensor(indices, values, torch.Size(intlist(size)))

        ctx.indices, ctx.matrix, ctx.xmatrix = indices, matrix, xmatrix

        return torch.mm(matrix, xmatrix)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.data

        # -- this will break recursive autograd, but it's the only way to get grad over sparse matrices

        i_ixs = ctx.indices[0, :]
        j_ixs = ctx.indices[1, :]
        output_select = grad_output[i_ixs, :]  # ∂L/∂Y[i_k:]
        xmatrix_select = ctx.xmatrix[j_ixs, :]  # X[j_k:]

        grad_values = (output_select * xmatrix_select).sum(dim=1)  # ∂L/∂values = ∂L/∂Y[i_k] * X[j_k]

        grad_xmatrix = torch.mm(ctx.matrix.t(), grad_output)  # ∂L/∂X = Aᵗ · ∂L/∂Y
        return None, Variable(grad_values), None, Variable(grad_xmatrix)


class SparseMMGPU_peter(torch.autograd.Function):
    """
    Sparse matrix multiplication with gradients over the value-vector

    Does not work with batch dim.
    """

    @staticmethod
    def forward(ctx, indices, values, size, xmatrix):
        # print(type(size), size, list(size), intlist(size))

        # matrix: full sparse A
        # xmatrix: the X
        # returns Y = AX

        matrix = torch.sparse_coo_tensor(indices, values, torch.Size(intlist(size)), device='cuda')

        ctx.indices, ctx.matrix, ctx.xmatrix = indices, matrix, xmatrix

        return torch.mm(matrix, xmatrix)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.data

        # -- this will break recursive autograd, but it's the only way to get grad over sparse matrices

        i_ixs = ctx.indices[0, :]
        j_ixs = ctx.indices[1, :]
        output_select = grad_output[i_ixs]  # ∂L/∂Y[i_k]
        xmatrix_select = ctx.xmatrix[j_ixs]  # X[j_k]

        grad_values = (output_select * xmatrix_select).sum(dim=1)  # ∂L/∂values = ∂L/∂Y[i_k] * X[j_k]

        grad_xmatrix = torch.mm(ctx.matrix.t(), grad_output)  # ∂L/∂X = Aᵗ · ∂L/∂Y
        return None, Variable(grad_values), None, Variable(grad_xmatrix)


def sum_sparse_peter(indices, values, size, row=True):
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

    ones = torch.ones((size[1], 1), device=indices.device)

    s, _ = ones.size()
    ones = ones[None, :, :].expand(b, s, 1).contiguous()

    # print(indices.size(), values.size(), size, ones.size())
    # sys.exit()

    sums = batchmm(indices, values, size, ones)  # row/column sums
    bindex = torch.arange(b, device=indices.device)[:, None].expand(b, indices.size(1))
    sums = sums[bindex, indices[:, :, 0], 0]

    if bdims is None:
        return sums.view(k)

    return sums.view(*bdims + (k,))
