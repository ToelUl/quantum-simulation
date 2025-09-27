#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A class to construct the t-t'-J model MPO, extending the t-J builder.

This module provides the TTPrimeJModelMPOBuilder class, which adds the
next-nearest-neighbor (NNN) hopping terms (t') to the standard t-J model.
"""

from tj_model_mpo_builder import TJModelMPOBuilder

class TTPrimeJModelMPOBuilder(TJModelMPOBuilder):
    """Builds and manages the MPO for the 2D t-t'-J model.

    This class extends the TJModelMPOBuilder by incorporating an additional
    hopping term between next-nearest-neighbor (NNN) sites, controlled by
    the parameter `t_prime`.

    The Hamiltonian is:
    H = -t * Σ_{<ij>,σ} c†_{iσ}c_{jσ} - t' * Σ_{<<ij>>,σ} c†_{iσ}c_{jσ}
        + J * Σ_{<ij>} (S_i·S_j - n_i*n_j/4)

    Attributes:
        t_prime (float): The next-nearest-neighbor hopping parameter.
    """

    def __init__(self, nx: int, ny: int, t: float, j: float, t_prime: float, periodic_y: bool = True):
        """Initializes the TTPrimeJModelMPOBuilder.

        Args:
            nx (int): The number of sites along the x-dimension.
            ny (int): The number of sites along the y-dimension.
            t (float): The nearest-neighbor hopping coefficient.
            j (float): The Heisenberg exchange coefficient.
            t_prime (float): The next-nearest-neighbor hopping coefficient.
            periodic_y (bool): Specifies y-axis periodicity. Defaults to True.
        """
        # Store the new parameter
        self.t_prime = t_prime

        # Initialize the parent class (TJModelMPOBuilder). This will set up
        # nx, ny, t, j, num_sites, initialize all operators, and build the
        # FSM with the nearest-neighbor t and J terms.
        super().__init__(nx, ny, t, j, periodic_y)

    def build_fsm(self):
        """Constructs the FSM, including NNN terms.

        This method first calls the parent's `build_fsm` to handle the
        standard t-J terms, and then adds the new t' hopping terms for
        diagonal next-nearest neighbors.
        """
        # 1. Build the FSM with the standard t-J terms first.
        #    This populates the FSM with all nearest-neighbor interactions.
        super().build_fsm()

        # 2. Now, add the next-nearest-neighbor (t') hopping terms.
        #    We only add terms with a non-zero coefficient.
        if abs(self.t_prime) < 1e-14:
            return

        print("\nAdding next-nearest-neighbor (t') terms...")

        # Iterate over each site to find its NNN pairs.
        # NNNs are typically diagonal neighbors.
        for x in range(self.nx):
            for y in range(self.ny):
                # The starting site index
                i = x * self.ny + y

                # --- Find diagonal NNN pairs ---

                # Pair 1: (x, y) and (x+1, y+1)
                if x < self.nx - 1 and self.ny > 1:
                    is_periodic_bond = self.periodic_y and (y == self.ny - 1)
                    is_open_bond = (not self.periodic_y) and (y < self.ny - 1)
                    if is_periodic_bond or is_open_bond:
                        k = (x + 1) * self.ny + (y + 1) % self.ny
                        self._add_t_prime_terms(i, k)

                # Pair 2: (x, y) and (x+1, y-1)
                if x < self.nx - 1 and self.ny > 1:
                    is_periodic_bond = self.periodic_y and (y == 0)
                    is_open_bond = (not self.periodic_y) and (y > 0)
                    if is_periodic_bond or is_open_bond:
                        # For periodic, y=0 connects to y=ny-1. For open, y>0 connects to y-1.
                        y_neighbor = (y - 1 + self.ny) % self.ny
                        k = (x + 1) * self.ny + y_neighbor
                        self._add_t_prime_terms(i, k)

    def _add_t_prime_terms(self, i: int, k: int):
        """Adds the t' hopping terms between two NNN sites i and k.

        Args:
            i (int): The index of the first site.
            k (int): The index of the second (NNN) site.
        """
        # The logic is identical to the nearest-neighbor hopping, just with
        # a different coefficient (t_prime) and site pair (i, k).
        # The FSM handles the longer distance automatically.
        insert_ops = [self.identity, self.fermi_string, self.identity]
        self.fsm.add_term(-self.t_prime, [self.adag_up, self.a_up], [i, k], insert_ops)
        self.fsm.add_term(-self.t_prime, [self.adag_down, self.a_down], [i, k], insert_ops)
        self.fsm.add_term(-self.t_prime, [self.a_up, self.adag_up], [i, k], insert_ops)
        self.fsm.add_term(-self.t_prime, [self.a_down, self.adag_down], [i, k], insert_ops)


if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Define model parameters for a 3x3 system with NNN hopping.
    model_params = {
        'nx': 3,
        'ny': 3,
        't': 1.0,
        'j': 0.4,
        't_prime': -0.2,  # A non-zero t' for the NNN hopping
        'periodic_y': True
    }

    # 2. Instantiate the new builder.
    print(f"Building MPO for a {model_params['nx']}x{model_params['ny']} "
          f"t-t'-J model...")
    ttprime_j_builder = TTPrimeJModelMPOBuilder(**model_params)
    print("Build complete.")

    # 3. Display the symbolic MPO. It will be much larger now due to the
    #    long-range interactions.
    # ttprime_j_builder.display_symbolic_mpo() # This can be very long to print

    # 4. Display the bond dimensions. This is the most informative output.
    #    We expect the bond dimensions to be larger than the simple t-J model.
    ttprime_j_builder.display_bond_dimensions()

    # 5. Generate the numerical MPO.
    mpo_tensors = ttprime_j_builder.get_mpo()
    if mpo_tensors:
        print("\nSuccessfully generated numerical MPO for the t-t'-J model.")
        print(f"Shape of site 0 MPO tensor: {mpo_tensors[0].shape}")
