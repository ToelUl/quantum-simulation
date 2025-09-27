#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A class to construct the long-range t1-t2-J1-J2 model MPO.

This module provides the T1T2J1J2ModelMPOBuilder class, which extends the
t-t'-J model to include next-nearest-neighbor (NNN) Heisenberg exchange
interactions (J2).
"""

# We build upon the t-t'-J model builder
from ttprime_j_model_mpo_builder import TTPrimeJModelMPOBuilder

class T1T2J1J2ModelMPOBuilder(TTPrimeJModelMPOBuilder):
    """Builds the MPO for the 2D t1-t2-J1-J2 model.

    This class extends the t-t'-J model by incorporating an additional
    Heisenberg exchange term between next-nearest-neighbor (NNN) sites,
    controlled by the parameter `j2`.

    The Hamiltonian is:
    H = -t1 * Σ_{<ij>,σ} (c†_{iσ}c_{jσ} + h.c.)
        -t2 * Σ_{<<ij>>,σ} (c†_{iσ}c_{jσ} + h.c.)
        + J1 * Σ_{<ij>} (S_i·S_j - n_i*n_j/4)
        + J2 * Σ_{<<ij>>} (S_i·S_j - n_i*n_j/4)

    Attributes:
        j2 (float): The next-nearest-neighbor Heisenberg exchange parameter.
    """

    def __init__(self, nx: int, ny: int, t1: float, j1: float, t2: float, j2: float, periodic_y: bool = True):
        """Initializes the T1T2J1J2ModelMPOBuilder.

        Args:
            nx (int): The number of sites along the x-dimension.
            ny (int): The number of sites along the y-dimension.
            t1 (float): The nearest-neighbor hopping coefficient.
            j1 (float): The nearest-neighbor Heisenberg exchange coefficient.
            t2 (float): The next-nearest-neighbor hopping coefficient.
            j2 (float): The next-nearest-neighbor Heisenberg exchange coefficient.
            periodic_y (bool): Specifies y-axis periodicity. Defaults to True.
        """
        # Store the new J2 parameter.
        self.j2 = j2

        # Initialize the parent class (TTPrimeJModelMPOBuilder).
        # We map t1->t, j1->j, t2->t_prime. The parent class will handle
        # the construction of all t1, j1, and t2 terms.
        super().__init__(nx=nx, ny=ny, t=t1, j=j1, t_prime=t2, periodic_y=periodic_y)

    def build_fsm(self):
        """Constructs the FSM, including NNN J2 terms.

        This method first calls the parent's `build_fsm` to handle the
        t1, J1, and t2 terms. It then adds the new J2 exchange terms for
        the same diagonal NNN pairs.
        """
        # 1. Call the parent method. This will build the FSM with all
        #    t1 (NN hopping), J1 (NN exchange), and t2 (NNN hopping) terms.
        super().build_fsm()

        # 2. Now, add the next-nearest-neighbor (J2) exchange terms.
        #    We proceed only if J2 is non-zero.
        if abs(self.j2) < 1e-14:
            return

        print("\nAdding next-nearest-neighbor (J2) exchange terms...")

        # The logic for finding NNN pairs is identical to the one used for t2
        # in the parent class. We iterate and add the J2 terms.
        for x in range(self.nx):
            for y in range(self.ny):
                i = x * self.ny + y

                # --- Find diagonal NNN pairs and add J2 terms ---

                # Pair 1: (x, y) and (x+1, y+1)
                if x < self.nx - 1 and self.ny > 1:
                    is_periodic_bond = self.periodic_y and (y == self.ny - 1)
                    is_open_bond = (not self.periodic_y) and (y < self.ny - 1)
                    if is_periodic_bond or is_open_bond:
                        k = (x + 1) * self.ny + (y + 1) % self.ny
                        self._add_j2_terms(i, k)

                # Pair 2: (x, y) and (x+1, y-1)
                if x < self.nx - 1 and self.ny > 1:
                    is_periodic_bond = self.periodic_y and (y == 0)
                    is_open_bond = (not self.periodic_y) and (y > 0)
                    if is_periodic_bond or is_open_bond:
                        y_neighbor = (y - 1 + self.ny) % self.ny
                        k = (x + 1) * self.ny + y_neighbor
                        self._add_j2_terms(i, k)

    def _add_j2_terms(self, i: int, k: int):
        """Adds the J2 Heisenberg exchange terms between two NNN sites i and k.

        Args:
            i (int): The index of the first site.
            k (int): The index of the second (NNN) site.
        """
        # The structure of the Heisenberg term is the same, just with the
        # J2 coefficient and applied to the NNN site pair (i, k).
        self.fsm.add_term(self.j2, [self.s_z, self.s_z], [i, k])
        self.fsm.add_term(self.j2 / 2, [self.s_plus, self.s_minus], [i, k])
        self.fsm.add_term(self.j2 / 2, [self.s_minus, self.s_plus], [i, k])
        self.fsm.add_term(-self.j2 / 4, [self.n_tot, self.n_tot], [i, k])


if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Define model parameters for a 3x3 system with long-range interactions.
    model_params = {
        'nx': 3,
        'ny': 3,
        't1': 1.0,
        'j1': 1.0,  # Often set as the energy scale
        't2': -0.2,
        'j2': 0.3,   # A non-zero J2 for NNN exchange
        'periodic_y': True
    }

    # 2. Instantiate the final builder.
    print(f"Building MPO for a {model_params['nx']}x{model_params['ny']} "
          f"t1-t2-J1-J2 model...")
    long_range_builder = T1T2J1J2ModelMPOBuilder(**model_params)
    print("Build complete.")

    # 3. Display the bond dimensions. We expect them to be even larger
    #    than the t-t'-J model because more long-range terms are being added.
    long_range_builder.display_bond_dimensions()

    # 4. Generate the numerical MPO.
    final_mpo = long_range_builder.get_mpo()
    if final_mpo:
        print("\nSuccessfully generated numerical MPO for the t1-t2-J1-J2 model.")
        print(f"Shape of site 0 MPO tensor: {final_mpo[0].shape}")
        print(f"Shape of central site (4) MPO tensor: {final_mpo[4].shape}")
