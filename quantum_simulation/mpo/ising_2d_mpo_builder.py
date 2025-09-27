#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A class to construct the MPO for the 2D transverse-field Ising model.

The Hamiltonian is defined as:
H = -J * Σ_{<ij>} S^x_i S^x_j - g * Σ_{i} S^z_i
where <ij> denotes nearest neighbors on a square lattice.
"""

from auto_mpo import FSM, NamedData, generate_mpo_spin_operators

class Ising2DMPOBuilder:
    """Builds the MPO for the 2D transverse-field Ising model.

    Attributes:
        nx (int): The number of sites in the x-direction.
        ny (int): The number of sites in the y-direction.
        j_coupling (float): The nearest-neighbor coupling strength (J).
        g_field (float): The transverse field strength (g).
        num_sites (int): Total number of sites (nx * ny).
        fsm (FSM): The FSM instance for MPO construction.
    """

    def __init__(self, nx: int, ny: int, j_coupling: float, g_field: float):
        self.nx = nx
        self.ny = ny
        self.j_coupling = j_coupling
        self.g_field = g_field
        self.num_sites = nx * ny
        self.fsm = None

        self._initialize_operators()
        self.build_fsm()

    def _initialize_operators(self):
        """Generates and stores the required spin-1/2 operators."""
        ops = generate_mpo_spin_operators(spin_dim=2)
        self.s_x = NamedData('Sx', ops['Sx'])
        self.s_z = NamedData('Sz', ops['Sz'])
        self.identity = NamedData('Id', ops['Id'])

    def build_fsm(self):
        """Constructs the FSM for the 2D Hamiltonian."""
        self.fsm = FSM(self.num_sites)

        # Iterate over each site in the 2D lattice.
        for x in range(self.nx):
            for y in range(self.ny):
                # Map 2D coordinate to 1D index
                i = x * self.ny + y

                # Add the on-site transverse field term for every site
                self.fsm.add_term(-self.g_field, [self.s_z], [i])

                # Add horizontal bond term S^x_i S^x_j
                if x < self.nx - 1:
                    j = (x + 1) * self.ny + y
                    self.fsm.add_term(-self.j_coupling, [self.s_x, self.s_x], [i, j])

                # Add vertical bond term S^x_i S^x_j
                if y < self.ny - 1:
                    j = x * self.ny + (y + 1)
                    self.fsm.add_term(-self.j_coupling, [self.s_x, self.s_x], [i, j])

    def get_mpo(self):
        """Generates and returns the numerical MPO tensors."""
        return self.fsm.to_mpo()

    def display_bond_dimensions(self):
        """Prints the bond dimensions of the MPO."""
        print("=" * 40)
        print("2D Transverse-Field Ising Model MPO")
        print(f"{self.nx}x{self.ny} lattice, J={self.j_coupling}, g={self.g_field}")
        print("=" * 40)
        self.fsm.print_bond_dimensions()


if __name__ == '__main__':
    # --- Example Usage ---
    NX = 3
    NY = 3
    J = 1.0
    g = 0.5

    ising_2d_builder = Ising2DMPOBuilder(nx=NX, ny=NY, j_coupling=J, g_field=g)
    ising_2d_builder.display_bond_dimensions()

    # The bond dimensions will be asymmetric due to the 2D->1D mapping.
