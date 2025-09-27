#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A class to construct the MPO for the 1D transverse-field Ising model.

The Hamiltonian is defined as:
H = -J * Σ_{i} S^x_i S^x_{i+1} - g * Σ_{i} S^z_i
"""

from auto_mpo import FSM, NamedData, generate_mpo_spin_operators

class Ising1DMPOBuilder:
    """Builds the MPO for the 1D transverse-field Ising chain.

    Attributes:
        num_sites (int): The number of sites in the chain.
        j_coupling (float): The nearest-neighbor coupling strength (J).
        g_field (float): The transverse field strength (g).
        fsm (FSM): The Finite-State Machine instance for MPO construction.
    """

    def __init__(self, num_sites: int, j_coupling: float, g_field: float):
        """Initializes the builder with model parameters.

        Args:
            num_sites (int): The length of the spin chain.
            j_coupling (float): The coefficient for the S^x S^x interaction.
            g_field (float): The coefficient for the S^z transverse field.
        """
        self.num_sites = num_sites
        self.j_coupling = j_coupling
        self.g_field = g_field
        self.fsm = None

        self._initialize_operators()
        self.build_fsm()

    def _initialize_operators(self):
        """Generates and stores the required spin-1/2 operators."""
        # For spin-1/2, the local Hilbert space dimension is 2.
        ops = generate_mpo_spin_operators(spin_dim=2)
        self.s_x = NamedData('Sx', ops['Sx'])
        self.s_z = NamedData('Sz', ops['Sz'])
        self.identity = NamedData('Id', ops['Id'])

    def build_fsm(self):
        """Constructs the Finite-State Machine for the Hamiltonian."""
        self.fsm = FSM(self.num_sites)

        for i in range(self.num_sites):
            # Add the on-site transverse field term: -g * S^z_i
            self.fsm.add_term(-self.g_field, [self.s_z], [i])

            # Add the nearest-neighbor interaction term: -J * S^x_i * S^x_{i+1}
            # This is only added for sites i where a neighbor i+1 exists.
            if i < self.num_sites - 1:
                self.fsm.add_term(-self.j_coupling, [self.s_x, self.s_x], [i, i + 1])

    def get_mpo(self):
        """Generates and returns the numerical MPO tensors."""
        return self.fsm.to_mpo()

    def display_bond_dimensions(self):
        """Prints the bond dimensions of the MPO."""
        print("=" * 40)
        print("1D Transverse-Field Ising Model MPO")
        print(f"N={self.num_sites}, J={self.j_coupling}, g={self.g_field}")
        print("=" * 40)
        self.fsm.print_bond_dimensions()


if __name__ == '__main__':
    # --- Example Usage ---
    N = 10
    J = 1.0
    g = 0.5

    ising_1d_builder = Ising1DMPOBuilder(num_sites=N, j_coupling=J, g_field=g)
    ising_1d_builder.display_bond_dimensions()

    # The bond dimensions should be [1, 3, 3, ..., 3, 1]
    # This reflects the simple nearest-neighbor structure.
