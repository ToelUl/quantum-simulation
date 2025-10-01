#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A dedicated class for constructing the t-J model MPO using the FSM library.

This module provides the TJModelMPOBuilder class, which encapsulates the logic
for building the Matrix Product Operator (MPO) representation of the
two-dimensional t-J model Hamiltonian on a square lattice. It leverages the
provided `finite_state_machine` module to automatically generate and compress
the MPO.
"""

from typing import List, Optional, Any
import torch
import torch.nn as nn
from .auto_mpo import FSM, NamedData, generate_mpo_hardcore_boson_operators
from .mpo_to_torch import build_mpo_torch


class TJModelMPOBuilder(nn.Module):
    """Builds the MPO for the 2D t-J model as a PyTorch Module.

    Constructs the Hamiltonian for the t-J model on an Nx x Ny square lattice.
    It handles the 2D to 1D mapping, defines operators, and uses a Finite-State
    Machine (FSM) to generate the MPO, which is stored as PyTorch tensors.

    Attributes:
        nx (int): The number of sites in the x-direction.
        ny (int): The number of sites in the y-direction.
        t (float): The hopping parameter coefficient.
        j (float): The Heisenberg exchange parameter coefficient.
        periodic_y (bool): If True, applies periodic boundary conditions in y.
        num_sites (int): The total number of lattice sites (nx * ny).
        fsm (FSM): The FSM instance holding the Hamiltonian's graph.
        mpo (nn.ParameterList): The list of MPO tensors as PyTorch Parameters.
    """

    def __init__(self,
                 nx: int,
                 ny: int,
                 t: float,
                 j: float,
                 periodic_y: bool = True,
                 device: Optional[Any] = None,
                 dtype: Optional[torch.dtype] = None):
        """Initializes the builder with model parameters and constructs the MPO.

        Args:
            nx (int): Number of sites along the x-dimension.
            ny (int): Number of sites along the y-dimension.
            t (float): Coefficient for the hopping terms.
            j (float): Coefficient for the Heisenberg exchange terms.
            periodic_y (bool): Specifies y-axis periodicity. Defaults to True.
            device (Optional[Any]): PyTorch device for the MPO tensors.
            dtype (Optional[torch.dtype]): PyTorch data type for the MPO tensors.
        """
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.t = t
        self.j = j
        self.periodic_y = periodic_y
        self.num_sites = nx * ny
        self.fsm: Optional[FSM] = None

        # Step 1: Initialize all required operators.
        self._initialize_operators()

        # Step 2: Build the FSM. This will call the 'build_fsm' method of
        # this class or any subclass that overrides it, ensuring all terms
        # are added before the MPO is generated.
        self.build_fsm()

        # Step 3: Generate the MPO from the completed FSM.
        mpo_np = self.fsm.to_mpo()

        # Step 4: Convert to PyTorch tensors and register as parameters.
        mpo_torch = build_mpo_torch(mpo_np, dtype=dtype, device=device)
        self.mpo = nn.ParameterList([nn.Parameter(tensor, requires_grad=False) for tensor in mpo_torch])

    def _initialize_operators(self):
        """Generates and stores all necessary operators for the t-J model."""
        ops = generate_mpo_hardcore_boson_operators()
        self.adag_up = NamedData('adag_up', ops['adag_up'])
        self.a_up = NamedData('a_up', ops['a_up'])
        self.adag_down = NamedData('adag_down', ops['adag_down'])
        self.a_down = NamedData('a_down', ops['a_down'])
        self.fermi_string = NamedData('F', ops['F'])
        self.identity = NamedData('Id', ops['Id'])
        self.s_z = NamedData('Sz', ops['Sz'])
        self.s_plus = NamedData('Sp', ops['Sp'])
        self.s_minus = NamedData('Sm', ops['Sm'])
        self.n_tot = NamedData('n_tot', ops['n_tot'])

    def build_fsm(self):
        """Constructs the FSM for the nearest-neighbor t-J Hamiltonian.

        This method can be extended by subclasses to add more terms.
        """
        self.fsm = FSM(self.num_sites)
        for x in range(self.nx):
            for y in range(self.ny):
                i = x * self.ny + y
                # Horizontal bonds
                if x < self.nx - 1:
                    j = (x + 1) * self.ny + y
                    self._add_bond_terms(i, j)
                # Vertical bonds
                if self.ny > 1:
                    is_periodic = self.periodic_y and (y == self.ny - 1)
                    is_open = (not self.periodic_y) and (y < self.ny - 1)
                    if is_periodic or is_open:
                        j = x * self.ny + (y + 1) % self.ny
                        self._add_bond_terms(i, j)

    def _add_bond_terms(self, i: int, j: int):
        """Adds all Hamiltonian terms for a bond between sites i and j.

        Args:
            i (int): The index of the first site.
            j (int): The index of the second site.
        """
        # Hopping terms (t) with Fermi string
        insert_ops = [self.identity, self.fermi_string, self.identity]
        self.fsm.add_term(-self.t, [self.adag_up, self.a_up], [i, j], insert_ops)
        self.fsm.add_term(-self.t, [self.adag_down, self.a_down], [i, j], insert_ops)
        self.fsm.add_term(-self.t, [self.a_up, self.adag_up], [i, j], insert_ops)
        self.fsm.add_term(-self.t, [self.a_down, self.adag_down], [i, j], insert_ops)

        # Heisenberg exchange terms (J)
        self.fsm.add_term(self.j, [self.s_z, self.s_z], [i, j])
        self.fsm.add_term(self.j / 2, [self.s_plus, self.s_minus], [i, j])
        self.fsm.add_term(self.j / 2, [self.s_minus, self.s_plus], [i, j])
        self.fsm.add_term(-self.j / 4, [self.n_tot, self.n_tot], [i, j])

    def forward(self) -> List[torch.Tensor]:
        """Returns the constructed MPO as a list of PyTorch tensors."""
        return list(self.mpo)

    def get_mpo(self) -> List[torch.Tensor]:
        """Returns MPO tensors. Kept for API compatibility."""
        return self.forward()

    def display_bond_dimensions(self):
        """Prints the bond dimension of the MPO at each virtual bond."""
        print("=" * 50)
        print("MPO Bond Dimensions")
        print("=" * 50)
        if self.fsm:
            self.fsm.print_bond_dimensions()


if __name__ == '__main__':
    model_params = {'nx': 3, 'ny': 2, 't': 3.0, 'j': 1.0, 'periodic_y': False}
    print(f"Building MPO for a {model_params['nx']}x{model_params['ny']} t-J ladder...")

    tj_builder = TJModelMPOBuilder(**model_params, device='cpu')
    print("Build complete.")
    tj_builder.display_bond_dimensions()

    mpo_tensors = tj_builder.get_mpo()
    print("\n--- Verification of PyTorch MPO ---")
    print(f"Number of MPO tensors: {len(mpo_tensors)}")
    print(f"Shape of site 0 MPO tensor: {mpo_tensors[0].shape}")
    print(f"Device of site 0 MPO tensor: {mpo_tensors[0].device}")
