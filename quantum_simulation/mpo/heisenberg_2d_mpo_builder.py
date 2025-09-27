#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A class to construct the MPO for the 2D Heisenberg model.

The Hamiltonian is defined as:
H = J * Σ_{<ij>} (S^x_i S^x_j + S^y_i S^y_j + S^z_i S^z_j)
where <ij> denotes nearest neighbors on a square lattice.
"""

from auto_mpo import FSM, NamedData, generate_mpo_spin_operators

class Heisenberg2DMPOBuilder:
    """Builds the MPO for the 2D Heisenberg model.

    Attributes:
        nx (int): The number of sites in the x-direction.
        ny (int): The number of sites in the y-direction.
        j_coupling (float): The nearest-neighbor coupling strength (J).
        num_sites (int): Total number of sites (nx * ny).
        fsm (FSM): The FSM instance for MPO construction.
    """

    def __init__(self, nx: int, ny: int, j_coupling: float):
        self.nx = nx
        self.ny = ny
        self.j_coupling = j_coupling
        self.num_sites = nx * ny
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
        """Constructs the FSM for the 2D Hamiltonian."""
        self.fsm = FSM(self.num_sites)

        # Iterate over each site in the 2D lattice.
        for x in range(self.nx):
            for y in range(self.ny):
                i = x * self.ny + y

                # --- Add horizontal bond terms ---
                if x < self.nx - 1:
                    j = (x + 1) * self.ny + y
                    self._add_bond_terms(i, j)

                # --- Add vertical bond terms ---
                if y < self.ny - 1:
                    j = x * self.ny + (y + 1)
                    self._add_bond_terms(i, j)

    def _add_bond_terms(self, i: int, j: int):
        """Adds the full Heisenberg interaction for a single bond."""
        self.fsm.add_term(self.j_coupling, [self.s_x, self.s_x], [i, j])
        self.fsm.add_term(self.j_coupling, [self.s_y, self.s_y], [i, j])
        self.fsm.add_term(self.j_coupling, [self.s_z, self.s_z], [i, j])

    def get_mpo(self):
        """Generates and returns the numerical MPO tensors."""
        return self.fsm.to_mpo()

    def display_bond_dimensions(self):
        """Prints the bond dimensions of the MPO."""
        print("=" * 40)
        print("2D Heisenberg Model MPO")
        print(f"{self.nx}x{self.ny} lattice, J={self.j_coupling}")
        print("=" * 40)
        self.fsm.print_bond_dimensions()


if __name__ == '__main__':
    # --- Example Usage ---
    NX = 3
    NY = 3
    J = 1.0

    heisenberg_2d_builder = Heisenberg2DMPOBuilder(nx=NX, ny=NY, j_coupling=J)
    heisenberg_2d_builder.display_bond_dimensions()

    # The bond dimensions will be asymmetric and larger than the 2D Ising model.
