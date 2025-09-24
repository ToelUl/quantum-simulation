import numpy as np


def generate_k_space(
    lattice_length: int,
    anti_periodic_bc: bool = False,
    positive_only: bool = False
) -> np.ndarray:
    r"""Generates a 1D grid of wave vectors (k-space) for a given lattice.

    This function computes the allowed k-vectors for a 1D lattice of a specified
    length, supporting both periodic (PBC) and anti-periodic boundary
    conditions (ABC).

    For Periodic Boundary Conditions (PBC), the wave vectors are given by:
    $$ k = \frac{2\pi n}{L} $$
    For Anti-Periodic Boundary Conditions (ABC), they are given by:
    $$ k = \frac{\pi (2n - 1)}{L} $$

    Args:
        lattice_length (int): The total number of sites in the lattice (L).
        anti_periodic_bc (bool, optional): If True, computes k-vectors for
            anti-periodic boundary conditions. Defaults to False (periodic).
        positive_only (bool, optional): If True, returns only the strictly
            positive k-vectors. Defaults to False.

    Returns:
        np.ndarray: An array of k-vectors. The array is sorted in ascending
            order.
    """
    lattice_length = int(lattice_length)
    if lattice_length % 2 != 0:
        raise ValueError("Lattice length must be even.")
    if anti_periodic_bc:
        # For ABC, k = pi * (2n - 1) / L, where n = 1, 2, ..., L/2
        # This generates odd integer multiples of pi/L.
        n = np.arange(1, lattice_length // 2 + 1, dtype=int)
        k_positive = (2 * n - 1) * np.pi / lattice_length
        if positive_only:
            return k_positive
        else:
            # The full domain is symmetric, including negative counterparts.
            return np.concatenate((-k_positive[::-1], k_positive))
    else:  # Periodic Boundary Conditions
        # For PBC, k = 2 * pi * n / L
        if positive_only:
            # n ranges from 1 to L/2 - 1 to get positive, non-zero k.
            n = np.arange(1, lattice_length // 2, dtype=int)
            return 2 * n * np.pi / lattice_length
        else:
            # n ranges from -L/2 + 1 to L/2 for the full Brillouin zone.
            n = np.arange(np.ceil(-lattice_length // 2) + 1,
                          np.floor(lattice_length // 2) + 1, dtype=int)
            return 2 * n * np.pi / lattice_length