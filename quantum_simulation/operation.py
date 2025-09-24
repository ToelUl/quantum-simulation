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