#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A class to construct the MPO for the 2D Heisenberg model.

The Hamiltonian is defined as:
H = J * Σ_{<ij>} (S^x_i S^x_j + S^y_i S^y_j + S^z_i S^z_j)
where <ij> denotes nearest neighbors on a square lattice.
"""

from typing import List, Optional, Any
import torch
import torch.nn as nn
from .auto_mpo import FSM, NamedData, generate_mpo_spin_operators
from .mpo_to_torch import build_mpo_torch


class Heisenberg2DMPOBuilder(nn.Module):
    """Builds the MPO for the 2D Heisenberg model as a PyTorch Module.

    This class uses the 'auto_mpo' Finite-State Machine (FSM) to construct the
    Matrix Product Operator (MPO) for the 2D Heisenberg model on a square
    lattice. The lattice is mapped to a 1D chain in row-major order. The
    resulting MPO tensors are stored as a `torch.nn.ParameterList`.

    The Hamiltonian is:
    H = Jx * Σ_{<ij>} S^x_i S^x_j + Jy * Σ_{<ij>} S^y_i S^y_j + Jz * Σ_{<ij>} S^z_i S^z_j

    Attributes:
        nx (int): The number of sites in the x-direction.
        ny (int): The number of sites in the y-direction.
        num_sites (int): Total number of sites (nx * ny).
        j_coupling_x (float): Nearest-neighbor coupling strength for SxSx.
        j_coupling_y (float): Nearest-neighbor coupling strength for SySy.
        j_coupling_z (float): Nearest-neighbor coupling strength for SzSz.
        fsm (FSM): The FSM instance used for the MPO construction.
        mpo (nn.ParameterList): The list of MPO tensors as PyTorch Parameters.
    """

    def __init__(self,
                 nx: int,
                 ny: int,
                 j_coupling_x: float,
                 j_coupling_y: float,
                 j_coupling_z: float,
                 device: Optional[Any] = None,
                 dtype: Optional[torch.dtype] = None):
        """Initializes the builder with model parameters and constructs the MPO.

        Args:
            nx (int): The number of sites in the x-direction (columns).
            ny (int): The number of sites in the y-direction (rows).
            j_coupling_x (float): The nearest-neighbor coupling strength for
                the S^x S^x interaction.
            j_coupling_y (float): The nearest-neighbor coupling strength for
                the S^y S^y interaction.
            j_coupling_z (float): The nearest-neighbor coupling strength for
                the S^z S^z interaction.
            device (Optional[Any]): The PyTorch device to store the MPO tensors
                on (e.g., 'cpu', 'cuda:0').
            dtype (Optional[torch.dtype]): The PyTorch data type for the MPO
                tensors (e.g., torch.complex128).
        """
        super().__init__()

        self.nx = nx
        self.ny = ny
        self.num_sites = nx * ny
        self.j_coupling_x = j_coupling_x
        self.j_coupling_y = j_coupling_y
        self.j_coupling_z = j_coupling_z

        # Step 1: Initialize the required spin-1/2 operators using NumPy.
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
        self.s_y = NamedData('Sy', ops['Sy'])
        self.s_z = NamedData('Sz', ops['Sz'])

    def _build_fsm(self) -> FSM:
        """Constructs and returns the FSM for the 2D Hamiltonian.

        The 2D lattice sites are mapped to a 1D chain using row-major ordering:
        site_index = x * ny + y.

        Returns:
            FSM: The populated FSM instance representing the 2D Heisenberg model.
        """
        fsm = FSM(site_num=self.num_sites)

        # Iterate over each site in the 2D lattice.
        for x in range(self.nx):
            for y in range(self.ny):
                # Map 2D coordinate (x, y) to 1D site index 'i'.
                i = x * self.ny + y

                # --- Add horizontal bond terms (connections to the right) ---
                if x < self.nx - 1:
                    j = (x + 1) * self.ny + y
                    self._add_bond_terms(fsm, i, j)

                # --- Add vertical bond terms (connections below) ---
                if y < self.ny - 1:
                    j = x * self.ny + (y + 1)
                    self._add_bond_terms(fsm, i, j)
        return fsm

    def _add_bond_terms(self, fsm: FSM, i: int, j: int):
        """Adds the full Heisenberg interaction for a single bond (i, j).

        Args:
            fsm (FSM): The Finite-State Machine instance to add terms to.
            i (int): The index of the first site in the bond.
            j (int): The index of the second site in the bond.
        """
        fsm.add_term(self.j_coupling_x, [self.s_x, self.s_x], [i, j])
        fsm.add_term(self.j_coupling_y, [self.s_y, self.s_y], [i, j])
        fsm.add_term(self.j_coupling_z, [self.s_z, self.s_z], [i, j])

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
        print("2D Heisenberg Model MPO")
        print(f"Lattice: {self.nx}x{self.ny}, Jx={self.j_coupling_x}, "
              f"Jy={self.j_coupling_y}, Jz={self.j_coupling_z}")
        print("=" * 60)
        self.fsm.print_bond_dimensions()


if __name__ == '__main__':
    # --- Example Usage ---
    NX = 3
    NY = 3
    J = 1.0

    # Instantiate the builder. The MPO is created upon initialization.
    heisenberg_2d_builder = Heisenberg2DMPOBuilder(
        nx=NX,
        ny=NY,
        j_coupling_x=J,
        j_coupling_y=J,
        j_coupling_z=J,
        device='cpu'  # Can be changed to 'cuda' if a GPU is available
    )

    # Display the bond dimensions derived from the FSM.
    heisenberg_2d_builder.display_bond_dimensions()

    # Retrieve the MPO as a list of PyTorch tensors.
    mpo_tensors = heisenberg_2d_builder.get_mpo()

    # --- Verification ---
    print("\n--- Verification of PyTorch MPO ---")
    print(f"Number of MPO tensors: {len(mpo_tensors)}")
    if mpo_tensors:
        first_tensor = mpo_tensors[0]
        last_tensor = mpo_tensors[-1]
        middle_tensor_idx = len(mpo_tensors) // 2
        middle_tensor = mpo_tensors[middle_tensor_idx]

        print(f"Type of a tensor: {type(first_tensor)}")
        print(f"Data type of a tensor: {first_tensor.dtype}")
        print(f"Device of a tensor: {first_tensor.device}")
        print("-" * 35)
        print(f"Shape of the first tensor (site 0): {first_tensor.shape}")
        print(f"Shape of a middle tensor (site {middle_tensor_idx}): {middle_tensor.shape}")
        print(f"Shape of the last tensor (site {NX*NY-1}): {last_tensor.shape}")
