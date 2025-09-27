#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A class to construct the MPO for the 1D Heisenberg model.

The Hamiltonian is defined as:
H = J * Σ_{i} (S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + S^z_i S^z_{i+1})
"""

from auto_mpo import FSM, NamedData, generate_mpo_spin_operators

class Heisenberg1DMPOBuilder:
    """Builds the MPO for the 1D Heisenberg chain.

    Attributes:
        num_sites (int): The number of sites in the chain.
        j_coupling_x (float): The nearest-neighbor coupling strength (J) for S^x S^x interaction.
        j_coupling_y (float): The nearest-neighbor coupling strength (J) for S^y S^y interaction.
        j_coupling_z (float): The nearest-neighbor coupling strength (J) for S^z S^z interaction.
        fsm (FSM): The Finite-State Machine instance for MPO construction.
    """

    def __init__(self, num_sites: int, j_coupling_x: float, j_coupling_y: float, j_coupling_z: float):
        """Initializes the builder with model parameters.

        Args:
            num_sites (int): The length of the spin chain.
            j_coupling_x (float): The nearest-neighbor coupling strength (J) for S^x S^x interaction.
            j_coupling_y (float): The nearest-neighbor coupling strength (J) for S^y S^y interaction.
            j_coupling_z (float): The nearest-neighbor coupling strength (J) for S^z S^z interaction.

        """
        self.num_sites = num_sites
        self.j_coupling_x = j_coupling_x
        self.j_coupling_y = j_coupling_y
        self.j_coupling_z = j_coupling_z
        self.fsm = None

        self._initialize_operators()
        self.build_fsm()

    def _initialize_operators(self):
        """Generates and stores the required spin-1/2 operators."""
        ops = generate_mpo_spin_operators(spin_dim=2)
        self.s_x = NamedData('Sx', ops['Sx'])
        self.s_y = NamedData('Sy', ops['Sy'])
        self.s_z = NamedData('Sz', ops['Sz'])
        self.identity = NamedData('Id', ops['Id'])

    def build_fsm(self):
        """Constructs the Finite-State Machine for the Hamiltonian."""
        self.fsm = FSM(self.num_sites)

        for i in range(self.num_sites - 1):
            # Add the S^x S^x term
            self.fsm.add_term(self.j_coupling_x, [self.s_x, self.s_x], [i, i + 1])
            # Add the S^y S^y term
            self.fsm.add_term(self.j_coupling_y, [self.s_y, self.s_y], [i, i + 1])
            # Add the S^z S^z term
            self.fsm.add_term(self.j_coupling_z, [self.s_z, self.s_z], [i, i + 1])

    def get_mpo(self):
        """Generates and returns the numerical MPO tensors."""
        return self.fsm.to_mpo()

    def display_bond_dimensions(self):
        """Prints the bond dimensions of the MPO."""
        print("=" * 40)
        print("1D Heisenberg Model MPO")
        print(f"N={self.num_sites}, Jx={self.j_coupling_x}, Jy={self.j_coupling_y}, Jz={self.j_coupling_z}")
        print("=" * 40)
        self.fsm.print_bond_dimensions()


if __name__ == '__main__':
    # --- Example Usage ---
    N = 10
    J = 1.0

    heisenberg_1d_builder = Heisenberg1DMPOBuilder(num_sites=N, j_coupling_x=J, j_coupling_y=J, j_coupling_z=J)
    heisenberg_1d_builder.display_bond_dimensions()

    # The bond dimensions should be [1, 4, 5, 5, ..., 5, 4, 1]
    # It's larger than Ising because there are more interaction terms.
