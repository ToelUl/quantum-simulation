#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A class to construct the MPO for the 2D transverse-field Ising model.

The Hamiltonian is defined as:
H = -J * Σ_{<ij>} S^x_i S^x_j - g * Σ_{i} S^z_i
where <ij> denotes nearest neighbors on a square lattice.
"""

from typing import List, Optional, Any
import torch
import torch.nn as nn
from .auto_mpo import FSM, NamedData, generate_mpo_spin_operators
from .mpo_to_torch import build_mpo_torch


class Ising2DMPOBuilder(nn.Module):
    """Builds the MPO for the 2D transverse-field Ising model as a PyTorch Module.

    This class uses the 'auto_mpo' Finite-State Machine (FSM) to construct the
    Matrix Product Operator (MPO) for the 2D transverse-field Ising model on a
    square lattice. The resulting MPO tensors are stored as a
    `torch.nn.ParameterList`.

    The Hamiltonian is:
    H = -J * Σ_{<ij>} S^x_i S^x_j - g * Σ_{i} S^z_i

    Attributes:
        nx (int): The number of sites in the x-direction.
        ny (int): The number of sites in the y-direction.
        num_sites (int): Total number of sites (nx * ny).
        j_coupling (float): The nearest-neighbor coupling strength (J).
        g_field (float): The transverse field strength (g).
        fsm (FSM): The FSM instance for MPO construction.
        mpo (nn.ParameterList): The list of MPO tensors as PyTorch Parameters.
    """

    def __init__(self,
                 nx: int,
                 ny: int,
                 j_coupling: float,
                 g_field: float,
                 device: Optional[Any] = None,
                 dtype: Optional[torch.dtype] = None):
        """Initializes the builder with model parameters and constructs the MPO.

        Args:
            nx (int): The number of sites in the x-direction (columns).
            ny (int): The number of sites in the y-direction (rows).
            j_coupling (float): The coefficient for the S^x S^x interaction.
            g_field (float): The coefficient for the S^z transverse field.
            device (Optional[Any]): The PyTorch device to store the MPO tensors
                on (e.g., 'cpu', 'cuda:0').
            dtype (Optional[torch.dtype]): The PyTorch data type for the MPO
                tensors (e.g., torch.complex128).
        """
        super().__init__()

        self.nx = nx
        self.ny = ny
        self.num_sites = nx * ny
        self.j_coupling = j_coupling
        self.g_field = g_field

        # Step 1: Initialize NumPy-based operators for the FSM builder.
        self._initialize_operators()

        # Step 2: Build the FSM graph for the 2D Hamiltonian.
        self.fsm = self._build_fsm()

        # Step 3: Generate the MPO as a list of NumPy arrays from the FSM.
        mpo_np = self.fsm.to_mpo()

        # Step 4: Convert NumPy MPO to PyTorch tensors and register them.
        mpo_torch = build_mpo_torch(mpo_np, dtype=dtype, device=device)
        self.mpo = nn.ParameterList([nn.Parameter(tensor, requires_grad=False) for tensor in mpo_torch])

    def _initialize_operators(self):
        """Generates and stores the required spin-1/2 operators as NamedData."""
        ops = generate_mpo_spin_operators(spin_dim=2)
        self.s_x = NamedData('Sx', ops['Sx'])
        self.s_z = NamedData('Sz', ops['Sz'])

    def _build_fsm(self) -> FSM:
        """Constructs and returns the FSM for the 2D Ising Hamiltonian.

        The 2D lattice sites are mapped to a 1D chain using row-major ordering:
        site_index = x * ny + y.

        Returns:
            FSM: The populated FSM instance representing the 2D Ising model.
        """
        fsm = FSM(site_num=self.num_sites)

        # Iterate over each site in the 2D lattice.
        for x in range(self.nx):
            for y in range(self.ny):
                # Map 2D coordinate to 1D index
                i = x * self.ny + y

                # Add the on-site transverse field term for every site
                fsm.add_term(-self.g_field, [self.s_z], [i])

                # Add horizontal bond term -J * S^x_i S^x_j
                if x < self.nx - 1:
                    j = (x + 1) * self.ny + y
                    fsm.add_term(-self.j_coupling, [self.s_x, self.s_x], [i, j])

                # Add vertical bond term -J * S^x_i S^x_j
                if y < self.ny - 1:
                    j = x * self.ny + (y + 1)
                    fsm.add_term(-self.j_coupling, [self.s_x, self.s_x], [i, j])
        return fsm

    def forward(self) -> List[torch.Tensor]:
        """Returns the constructed MPO as a list of PyTorch tensors.

        Returns:
            List[torch.Tensor]: The list containing the MPO tensors.
        """
        return list(self.mpo)

    def get_mpo(self) -> List[torch.Tensor]:
        """Returns the numerical MPO tensors as a list of torch.Tensors.

        This method is kept for compatibility with the original API.

        Returns:
            List[torch.Tensor]: The list containing the MPO tensors.
        """
        return self.forward()

    def display_bond_dimensions(self):
        """Prints the bond dimensions of the MPO."""
        print("=" * 60)
        print("2D Transverse-Field Ising Model MPO")
        print(f"Lattice: {self.nx}x{self.ny}, J={self.j_coupling}, g={self.g_field}")
        print("=" * 60)
        self.fsm.print_bond_dimensions()


if __name__ == '__main__':
    # --- Example Usage ---
    NX = 3
    NY = 3
    J = 1.0
    g = 0.5

    # Instantiate the builder. The MPO is created upon initialization.
    ising_2d_builder = Ising2DMPOBuilder(
        nx=NX,
        ny=NY,
        j_coupling=J,
        g_field=g,
        device='cpu'
    )

    # Display the bond dimensions derived from the FSM.
    ising_2d_builder.display_bond_dimensions()

    # Retrieve the MPO as a list of PyTorch tensors.
    mpo_tensors = ising_2d_builder.get_mpo()

    # --- Verification ---
    print("\n--- Verification of PyTorch MPO ---")
    print(f"Number of MPO tensors: {len(mpo_tensors)}")
    if mpo_tensors:
        first_tensor = mpo_tensors[0]
        print(f"Type of a tensor: {type(first_tensor)}")
        print(f"Data type of a tensor: {first_tensor.dtype}")
        print(f"Device of a tensor: {first_tensor.device}")
