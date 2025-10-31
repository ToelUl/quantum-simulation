from typing import List, Union, Tuple, Optional
import torch
from torch import Tensor


def dagger(a: Tensor) -> Tensor:
    """Computes the conjugate transpose (dagger) of a tensor.

    This function swaps the last two dimensions of a tensor and takes the complex
    conjugate, which is equivalent to the Hermitian conjugate for a matrix.

    Args:
        a (Tensor): A tensor of shape (..., M, N).

    Returns:
        Tensor: The conjugate transpose of the input tensor, with shape (..., N, M).
    """
    return a.transpose(-2, -1).conj()


def nested_kronecker_product(tensors: List[Tensor]) -> Tensor:
    """Computes the Kronecker product of a list of tensors recursively.

    Args:
        tensors (List[Tensor]): A list of tensors to be multiplied.

    Returns:
        Tensor: The result of the nested Kronecker product.
    """
    if len(tensors) == 2:
        return torch.kron(tensors[0].contiguous(), tensors[1].contiguous())
    else:
        return torch.kron(tensors[0].contiguous(), nested_kronecker_product(tensors[1:]))


def commutator(a: Tensor, b: Tensor) -> Tensor:
    """Calculates the commutator of two tensors, [A, B] = AB - BA.

    The operation is performed as a batch matrix multiplication.

    Args:
        a (Tensor): The first tensor of shape (..., M, N).
        b (Tensor): The second tensor of shape (..., N, M).

    Returns:
        Tensor: The commutator of a and b, with shape (..., M, M).

    Raises:
        ValueError: If the shapes of a and b are not compatible for
            matrix multiplication.
    """
    if a.shape != b.shape:
        raise ValueError("The shapes of a and b should be the same.")
    return torch.matmul(a, b) - torch.matmul(b, a)


def anti_commutator(a: Tensor, b: Tensor) -> Tensor:
    """Calculates the anti-commutator of two tensors, {A, B} = AB + BA.

    The operation is performed as a batch matrix multiplication.

    Args:
        a (Tensor): The first tensor of shape (..., M, N).
        b (Tensor): The second tensor of shape (..., N, M).

    Returns:
        Tensor: The anti-commutator of a and b, with shape (..., M, M).

    Raises:
        ValueError: If the shapes of a and b are not the same.
    """
    if a.shape != b.shape:
        raise ValueError("The shapes of a and b should be the same.")
    return torch.matmul(a, b) + torch.matmul(b, a)


def bra_o_ket(
        bra: Tensor,
        o: Tensor,
        ket: Tensor,
        real: bool = True
) -> Tensor:
    """Calculates the expectation value <bra|O|ket>.

    This function computes the expectation value of an operator `o` between
    a `bra` and a `ket` state using tensor contraction.

    Args:
        bra (Tensor): The 'bra' tensor of shape (..., D_bra, D_o).
        o (Tensor): The operator tensor of shape (..., D_o, D_ket).
        ket (Tensor): The 'ket' tensor of shape (..., D_ket, D_out).
        real (bool): If True, returns only the real part of the result.
            Defaults to True.

    Returns:
        Tensor: The calculated expectation value, a scalar or batch of scalars.

    Raises:
        ValueError: If the tensor dimensions are not compatible for contraction.
    """
    if bra.shape[-1] != o.shape[-2] or o.shape[-1] != ket.shape[-2]:
        raise ValueError("The input shapes are not valid for contraction.")

    # Using torch.einsum is also a direct and clear replacement
    result = torch.einsum('...ij,...jk,...kl->...il', bra, o, ket)

    return torch.real(result) if real else result


def complex_einsum(
    pattern: str,
    a: Tensor,
    b: Tensor = None,
    conj_a: int = 1,
    conj_b: int = 1
) -> Tensor:
    """Perform a complex Einstein summation on tensors with separated real and imaginary parts.

    This function applies the Einstein summation (einsum) operation on the real and imaginary
    components of the input tensor(s) separately. The parameters `conj_a` and `conj_b`
    determine whether to conjugate the corresponding tensor by multiplying its imaginary part
    by -1.

    Args:
        pattern (str): The einsum string rule (e.g., 'ij,jk->ik').
        a (Tensor): Input tensor with shape [..., 2], where the last dimension represents [real, imag].
        b (Tensor, optional): Second input tensor with shape [..., 2]. Defaults to None.
        conj_a (int, optional): Factor for conjugating `a` (1 for no conjugation, -1 for conjugation). Defaults to 1.
        conj_b (int, optional): Factor for conjugating `b` (1 for no conjugation, -1 for conjugation). Defaults to 1.

    Returns:
        Tensor: A tensor with shape [..., 2] representing the complex result.

    Example:
        >>> import torch
        >>> # a and b each has shape (1,1,2) => interpret as (batch=1, i=1, real/imag=2).
        >>> a = torch.tensor([[[1.0, 2.0]]])  # (1,1,2)
        >>> b = torch.tensor([[[3.0, 4.0]]])  # (1,1,2)
        >>> # We'll do a very simple sum over the 'i' index => pattern 'bi,bi->b'
        >>> # but to keep dimension naming consistent, let's do 'ijk,ijk->ij' if shapes matched.
        >>> # For demonstration, let's do 'bij,bij->b' ignoring that we only have i=1 dimension:
        >>> # Real part = (1*3 - 2*4) = -5
        >>> # Imag part = (1*4 + 2*3) = 10
        >>> out = complex_einsum('bi,bi->b', a, b)
        >>> out
        tensor([[-5., 10.]])
    """
    a_real, a_imag = a[..., 0], conj_a * a[..., 1]

    if b is not None:
        b_real, b_imag = b[..., 0], conj_b * b[..., 1]
        out_real = torch.einsum(pattern, a_real, b_real) - torch.einsum(pattern, a_imag, b_imag)
        out_imag = torch.einsum(pattern, a_real, b_imag) + torch.einsum(pattern, a_imag, b_real)
    else:
        out_real = torch.einsum(pattern, a_real)
        out_imag = torch.einsum(pattern, a_imag)

    return torch.stack((out_real, out_imag), dim=-1)
