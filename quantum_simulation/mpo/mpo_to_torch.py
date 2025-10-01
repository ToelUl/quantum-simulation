from typing import List, Optional, Any
import numpy as np
import torch


def build_mpo_torch(
    mpo_np: List[np.ndarray],
    dtype: Optional[torch.dtype] = None,
    device: Optional[Any] = None
) -> List[torch.Tensor]:
    """Converts a list of NumPy MPO tensors to a list of PyTorch tensors.

    Args:
        mpo_np (List[np.ndarray]): The list of MPO tensors in NumPy format.
        dtype (Optional[torch.dtype]): The desired data type of the output
            tensors. Defaults to torch.complex128 if None.
        device (Optional[Any]): The desired device of the output tensors.
            Defaults to the current device if None.

    Returns:
        List[torch.Tensor]: The list of MPO tensors as PyTorch tensors.
    """
    if dtype is None:
        dtype = torch.complex128
    return [torch.from_numpy(M).to(device=device, dtype=dtype) for M in mpo_np]