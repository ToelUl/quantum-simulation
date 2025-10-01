#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A class to construct the long-range t1-t2-J1-J2 model MPO.

This module provides the T1T2J1J2ModelMPOBuilder class, which extends the
t-t'-J model to include next-nearest-neighbor (NNN) Heisenberg exchange
interactions (J2).
"""

from typing import Optional, Any
import torch
# We build upon the t-t'-J model builder
from .ttprime_j_model_mpo_builder import TTPrimeJModelMPOBuilder


class T1T2J1J2ModelMPOBuilder(TTPrimeJModelMPOBuilder):
    """Builds the MPO for the 2D t1-t2-J1-J2 model as a PyTorch Module.

    Extends the t-t'-J model by incorporating an additional Heisenberg exchange
    term between next-nearest-neighbor (NNN) sites, controlled by `j2`.

    The Hamiltonian is:
    H = -t1 * Σ_{<ij>,σ} (c†_{iσ}c_{jσ} + h.c.)
        -t2 * Σ_{<<ij>>,σ} (c†_{iσ}c_{jσ} + h.c.)
        + J1 * Σ_{<ij>} (S_i·S_j - n_i*n_j/4)
        + J2 * Σ_{<<ij>>} (S_i·S_j - n_i*n_j/4)

    Attributes:
        j2 (float): The next-nearest-neighbor Heisenberg exchange parameter.
    """

    def __init__(self,
                 nx: int,
                 ny: int,
                 t1: float,
                 j1: float,
                 t2: float,
                 j2: float,
                 periodic_y: bool = True,
                 device: Optional[Any] = None,
                 dtype: Optional[torch.dtype] = None):
        """Initializes the T1T2J1J2ModelMPOBuilder.

        Args:
            nx (int): Number of sites along the x-dimension.
            ny (int): Number of sites along the y-dimension.
            t1 (float): Nearest-neighbor hopping coefficient.
            j1 (float): Nearest-neighbor Heisenberg exchange coefficient.
            t2 (float): Next-nearest-neighbor hopping coefficient.
            j2 (float): Next-nearest-neighbor Heisenberg exchange coefficient.
            periodic_y (bool): Specifies y-axis periodicity. Defaults to True.
            device (Optional[Any]): PyTorch device for the MPO tensors.
            dtype (Optional[torch.dtype]): PyTorch data type for the MPO tensors.
        """
        self.j2 = j2
        # Initialize the parent class (TTPrimeJModelMPOBuilder).
        # This triggers the full build chain for t1(t), j1(j), and t2(t_prime),
        # eventually calling this class's `build_fsm` method.
        super().__init__(
            nx=nx, ny=ny, t=t1, j=j1, t_prime=t2, periodic_y=periodic_y,
            device=device, dtype=dtype
        )

    def build_fsm(self):
        """Constructs the FSM, including NNN J2 exchange terms.

        This method first calls the parent's `build_fsm` to handle the
        t1, J1, and t2 terms, and then adds the new J2 exchange terms.
        """
        # 1. Build the FSM with t1, J1, and t2 terms.
        super().build_fsm()

        # 2. Add the next-nearest-neighbor (J2) exchange terms if non-zero.
        if abs(self.j2) < 1e-14:
            return
        print("\nAdding next-nearest-neighbor (J2) exchange terms...")

        for x in range(self.nx):
            for y in range(self.ny):
                i = x * self.ny + y
                # Diagonal NNN pair 1: (x, y) -> (x+1, y+1)
                if x < self.nx - 1 and self.ny > 1:
                    is_pbc = self.periodic_y and (y == self.ny - 1)
                    is_obc = (not self.periodic_y) and (y < self.ny - 1)
                    if is_pbc or is_obc:
                        k = (x + 1) * self.ny + (y + 1) % self.ny
                        self._add_j2_terms(i, k)

                # Diagonal NNN pair 2: (x, y) -> (x+1, y-1)
                if x < self.nx - 1 and self.ny > 1:
                    is_pbc = self.periodic_y and (y == 0)
                    is_obc = (not self.periodic_y) and (y > 0)
                    if is_pbc or is_obc:
                        y_neighbor = (y - 1 + self.ny) % self.ny
                        k = (x + 1) * self.ny + y_neighbor
                        self._add_j2_terms(i, k)

    def _add_j2_terms(self, i: int, k: int):
        """Adds the J2 Heisenberg exchange terms between two NNN sites.

        Args:
            i (int): The index of the first site.
            k (int): The index of the second (NNN) site.
        """
        self.fsm.add_term(self.j2, [self.s_z, self.s_z], [i, k])
        self.fsm.add_term(self.j2 / 2, [self.s_plus, self.s_minus], [i, k])
        self.fsm.add_term(self.j2 / 2, [self.s_minus, self.s_plus], [i, k])
        self.fsm.add_term(-self.j2 / 4, [self.n_tot, self.n_tot], [i, k])


if __name__ == '__main__':
    params = {'nx': 3, 'ny': 3, 't1': 1.0, 'j1': 1.0, 't2': -0.2, 'j2': 0.3, 'periodic_y': True}
    print(f"Building MPO for a {params['nx']}x{params['ny']} t1-t2-J1-J2 model...")

    builder = T1T2J1J2ModelMPOBuilder(**params, device='cpu')
    print("Build complete.")
    builder.display_bond_dimensions()

    mpo = builder.get_mpo()
    print("\n--- Verification of PyTorch MPO ---")
    print(f"Number of MPO tensors: {len(mpo)}")
    print(f"Shape of site 0 MPO tensor: {mpo[0].shape}")
    print(f"Shape of central site (4) MPO tensor: {mpo[4].shape}")
