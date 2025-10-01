#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A class to construct the t-t'-J model MPO, extending the t-J builder.

This module provides the TTPrimeJModelMPOBuilder class, which adds the
next-nearest-neighbor (NNN) hopping terms (t') to the standard t-J model.
"""

from typing import Optional, Any
import torch
from .tj_model_mpo_builder import TJModelMPOBuilder


class TTPrimeJModelMPOBuilder(TJModelMPOBuilder):
    """Builds and manages the MPO for the 2D t-t'-J model as a PyTorch Module.

    Extends TJModelMPOBuilder by incorporating an additional hopping term
    between next-nearest-neighbor (NNN) sites, controlled by `t_prime`.

    The Hamiltonian is:
    H = -t * Σ_{<ij>,σ} c†_{iσ}c_{jσ} - t' * Σ_{<<ij>>,σ} c†_{iσ}c_{jσ}
        + J * Σ_{<ij>} (S_i·S_j - n_i*n_j/4)

    Attributes:
        t_prime (float): The next-nearest-neighbor hopping parameter.
    """

    def __init__(self,
                 nx: int,
                 ny: int,
                 t: float,
                 j: float,
                 t_prime: float,
                 periodic_y: bool = True,
                 device: Optional[Any] = None,
                 dtype: Optional[torch.dtype] = None):
        """Initializes the TTPrimeJModelMPOBuilder.

        Args:
            nx (int): Number of sites along the x-dimension.
            ny (int): Number of sites along the y-dimension.
            t (float): Nearest-neighbor hopping coefficient.
            j (float): Heisenberg exchange coefficient.
            t_prime (float): Next-nearest-neighbor hopping coefficient.
            periodic_y (bool): Specifies y-axis periodicity. Defaults to True.
            device (Optional[Any]): PyTorch device for the MPO tensors.
            dtype (Optional[torch.dtype]): PyTorch data type for the MPO tensors.
        """
        self.t_prime = t_prime
        # Initialize the parent class. This will handle initialization of
        # operators and trigger the full `build_fsm` chain, culminating
        # in the creation of the final PyTorch MPO.
        super().__init__(nx, ny, t, j, periodic_y, device=device, dtype=dtype)

    def build_fsm(self):
        """Constructs the FSM, including NNN hopping terms.

        This method first calls the parent's `build_fsm` to handle the
        standard t-J terms, then adds the new t' hopping terms.
        """
        # 1. Build the FSM with the standard t-J terms.
        super().build_fsm()

        # 2. Add the next-nearest-neighbor (t') hopping terms if non-zero.
        if abs(self.t_prime) < 1e-14:
            return
        print("\nAdding next-nearest-neighbor (t') hopping terms...")

        for x in range(self.nx):
            for y in range(self.ny):
                i = x * self.ny + y
                # Diagonal NNN pair 1: (x, y) -> (x+1, y+1)
                if x < self.nx - 1 and self.ny > 1:
                    is_pbc = self.periodic_y and (y == self.ny - 1)
                    is_obc = (not self.periodic_y) and (y < self.ny - 1)
                    if is_pbc or is_obc:
                        k = (x + 1) * self.ny + (y + 1) % self.ny
                        self._add_t_prime_terms(i, k)

                # Diagonal NNN pair 2: (x, y) -> (x+1, y-1)
                if x < self.nx - 1 and self.ny > 1:
                    is_pbc = self.periodic_y and (y == 0)
                    is_obc = (not self.periodic_y) and (y > 0)
                    if is_pbc or is_obc:
                        y_neighbor = (y - 1 + self.ny) % self.ny
                        k = (x + 1) * self.ny + y_neighbor
                        self._add_t_prime_terms(i, k)

    def _add_t_prime_terms(self, i: int, k: int):
        """Adds the t' hopping terms between two NNN sites i and k.

        Args:
            i (int): The index of the first site.
            k (int): The index of the second (NNN) site.
        """
        insert_ops = [self.identity, self.fermi_string, self.identity]
        self.fsm.add_term(-self.t_prime, [self.adag_up, self.a_up], [i, k], insert_ops)
        self.fsm.add_term(-self.t_prime, [self.adag_down, self.a_down], [i, k], insert_ops)
        self.fsm.add_term(-self.t_prime, [self.a_up, self.adag_up], [i, k], insert_ops)
        self.fsm.add_term(-self.t_prime, [self.a_down, self.adag_down], [i, k], insert_ops)


if __name__ == '__main__':
    params = {'nx': 3, 'ny': 3, 't': 1.0, 'j': 0.4, 't_prime': -0.2, 'periodic_y': True}
    print(f"Building MPO for a {params['nx']}x{params['ny']} t-t'-J model...")

    builder = TTPrimeJModelMPOBuilder(**params, device='cpu')
    print("Build complete.")
    builder.display_bond_dimensions()

    mpo = builder.get_mpo()
    print("\n--- Verification of PyTorch MPO ---")
    print(f"Shape of site 0 MPO tensor: {mpo[0].shape}")
