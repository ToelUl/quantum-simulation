#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A class to construct the MPO for the 1D Heisenberg model.

The Hamiltonian is defined as:
H = J * Σ_{i} (S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + S^z_i S^z_{i+1})
"""

from typing import List, Optional, Any
import torch
import torch.nn as nn
from .auto_mpo import FSM, NamedData, generate_mpo_spin_operators
from .mpo_to_torch import build_mpo_torch


class Heisenberg1DMPOBuilder(nn.Module):
    """Builds the MPO for the 1D Heisenberg chain as a PyTorch Module.

    This class uses the 'auto_mpo' Finite-State Machine (FSM) to construct the
    Matrix Product Operator (MPO) for the 1D Heisenberg model. The resulting
    MPO tensors are stored as a `torch.nn.ParameterList`, making them part of
    the PyTorch module.

    The Hamiltonian is:
    H = Jx * Σ_{i} S^x_i S^x_{i+1} + Jy * Σ_{i} S^y_i S^y_{i+1} + Jz * Σ_{i} S^z_i S^z_{i+1}

    Attributes:
        num_sites (int): The number of sites in the chain.
        j_coupling_x (float): Nearest-neighbor coupling strength for SxSx.
        j_coupling_y (float): Nearest-neighbor coupling strength for SySy.
        j_coupling_z (float): Nearest-neighbor coupling strength for SzSz.
        fsm (FSM): The FSM instance used for the MPO construction.
        mpo (nn.ParameterList): The list of MPO tensors as PyTorch Parameters.
    """

    def __init__(self,
                 num_sites: int,
                 j_coupling_x: float,
                 j_coupling_y: float,
                 j_coupling_z: float,
                 device: Optional[Any] = None,
                 dtype: Optional[torch.dtype] = None):
        """Initializes the builder with model parameters and constructs the MPO.

        Args:
            num_sites (int): The length of the spin chain.
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

        self.num_sites = num_sites
        self.j_coupling_x = j_coupling_x
        self.j_coupling_y = j_coupling_y
        self.j_coupling_z = j_coupling_z

        # Step 1: Initialize the required spin-1/2 operators using NumPy.
        # The FSM builder operates on NumPy arrays.
        self._initialize_operators()

        # Step 2: Build the FSM graph for the Hamiltonian.
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
        """Constructs and returns the Finite-State Machine for the Hamiltonian.

        Returns:
            FSM: The populated FSM instance representing the Heisenberg model.
        """
        fsm = FSM(site_num=self.num_sites)

        # Iterate over all nearest-neighbor pairs to add interaction terms.
        for i in range(self.num_sites - 1):
            # Add the S^x S^x term
            fsm.add_term(self.j_coupling_x, [self.s_x, self.s_x], [i, i + 1])
            # Add the S^y S^y term
            fsm.add_term(self.j_coupling_y, [self.s_y, self.s_y], [i, i + 1])
            # Add the S^z S^z term
            fsm.add_term(self.j_coupling_z, [self.s_z, self.s_z], [i, i + 1])
        return fsm

    def forward(self) -> List[torch.Tensor]:
        """Returns the constructed MPO as a list of PyTorch tensors.

        This is the standard way to retrieve the primary output of an nn.Module.

        Returns:
            List[torch.Tensor]: The list containing the MPO tensors.
        """
        return list(self.mpo)

    def get_mpo(self) -> List[torch.Tensor]:
        """Generates and returns the numerical MPO tensors.

        This method is kept for compatibility with the original API.

        Returns:
            List[torch.Tensor]: The list containing the MPO tensors.
        """
        return self.forward()

    def display_bond_dimensions(self):
        """Prints the bond dimensions of the MPO."""
        print("=" * 50)
        print("1D Heisenberg Model MPO")
        print(f"N={self.num_sites}, Jx={self.j_coupling_x}, "
              f"Jy={self.j_coupling_y}, Jz={self.j_coupling_z}")
        print("=" * 50)
        self.fsm.print_bond_dimensions()


if __name__ == '__main__':
    # --- Example Usage ---
    N = 10
    J = 1.0

    # Instantiate the builder. The MPO is created upon initialization.
    heisenberg_1d_builder = Heisenberg1DMPOBuilder(
        num_sites=N,
        j_coupling_x=J,
        j_coupling_y=J,
        j_coupling_z=J,
        device='cpu'  # Can be changed to 'cuda' if a GPU is available
    )

    # Display the bond dimensions derived from the FSM.
    heisenberg_1d_builder.display_bond_dimensions()
    # The expected bond dimensions should be [1, 4, 5, 5, ..., 5, 4, 1].

    # Retrieve the MPO as a list of PyTorch tensors.
    mpo_tensors = heisenberg_1d_builder.get_mpo()

    # --- Verification ---
    print("--- Verification of PyTorch MPO ---")
    print(f"Number of MPO tensors: {len(mpo_tensors)}")
    if mpo_tensors:
        first_tensor = mpo_tensors[0]
        print(f"Type of the first tensor: {type(first_tensor)}")
        print(f"Shape of the first tensor: {first_tensor.shape}")
        print(f"Data type of the first tensor: {first_tensor.dtype}")
        print(f"Device of the first tensor: {first_tensor.device}")