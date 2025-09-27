#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A dedicated class for constructing the t-J model MPO using the FSM library.

This module provides the TJModelMPOBuilder class, which encapsulates the logic
for building the Matrix Product Operator (MPO) representation of the
two-dimensional t-J model Hamiltonian on a square lattice. It leverages the
provided `finite_state_machine` module to automatically generate and compress
the MPO.
"""

from auto_mpo import FSM, NamedData, generate_mpo_hardcore_boson_operators

class TJModelMPOBuilder:
    """Builds and manages the MPO for the 2D t-J model.

    This class provides a high-level interface to construct the Hamiltonian
    for the t-J model on an Nx x Ny square lattice. It handles the mapping of
    the 2D lattice to a 1D chain, defines the necessary quantum operators,
    and uses a Finite-State Machine (FSM) to automatically generate the
    corresponding MPO.

    Attributes:
        nx (int): The number of sites in the x-direction.
        ny (int): The number of sites in the y-direction.
        t (float): The hopping parameter coefficient.
        j (float): The Heisenberg exchange parameter coefficient.
        periodic_y (bool): If True, applies periodic boundary conditions in the
            y-direction. Otherwise, open boundary conditions are used.
        num_sites (int): The total number of lattice sites (Nx * Ny).
        fsm (FSM): An instance of the FSM class that holds the graphical
            representation of the Hamiltonian.
    """

    def __init__(self, nx: int, ny: int, t: float, j: float, periodic_y: bool = True):
        """Initializes the TJModelMPOBuilder with model parameters.

        Args:
            nx (int): The number of sites along the x-dimension of the lattice.
            ny (int): The number of sites along the y-dimension of the lattice.
            t (float): The coefficient for the hopping terms.
            j (float): The coefficient for the Heisenberg exchange terms.
            periodic_y (bool): Specifies whether to use periodic boundary
                conditions along the y-axis. Defaults to True.
        """
        self.nx = nx
        self.ny = ny
        self.t = t
        self.j = j
        self.periodic_y = periodic_y
        self.num_sites = nx * ny
        self.fsm = None

        # Define all required operators upon initialization.
        self._initialize_operators()

        # Immediately build the FSM representation of the Hamiltonian.
        self.build_fsm()

    def _initialize_operators(self):
        """Generates and stores all necessary operators for the t-J model.

        This method retrieves the hard-core boson operators and wraps them in
        the `NamedData` class, which is required for the FSM to efficiently
        compare and manage them.
        """
        # Retrieve all operators from the generator function.
        ops = generate_mpo_hardcore_boson_operators()

        # Wrap each operator matrix in a NamedData object for the FSM.
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
        """Constructs the Finite-State Machine for the t-J Hamiltonian.

        This method iterates through each site of the 2D lattice, mapping it
        to a 1D index. It then adds the Hamiltonian terms for interactions
        with horizontal and vertical neighbors to the FSM.
        """
        self.fsm = FSM(self.num_sites)

        # Iterate over each site in the 2D lattice.
        for x in range(self.nx):
            for y in range(self.ny):
                # Map the 2D coordinate (x, y) to a 1D site index `i`
                # using a "snake-like" pattern.
                i = x * self.ny + y

                # --- Add Horizontal Bond Terms ---
                # Connect site `i` with its right neighbor `(x+1, y)`.
                if x < self.nx - 1:
                    j = (x + 1) * self.ny + y
                    self._add_bond_terms(i, j)

                # --- Add Vertical Bond Terms ---
                # Connect site `i` with its lower neighbor `(x, y+1)`.
                if self.ny > 1:
                    # Check if this bond should be added based on boundary conditions.
                    is_periodic_bond = self.periodic_y and (y == self.ny - 1)
                    is_open_bond = (not self.periodic_y) and (y < self.ny - 1)

                    if is_periodic_bond or is_open_bond:
                        j = x * self.ny + (y + 1) % self.ny
                        self._add_bond_terms(i, j)

    def _add_bond_terms(self, i: int, j: int):
        """Adds all Hamiltonian terms for a bond between sites i and j.

        This helper method adds the kinetic hopping terms (parameter t) and
        the Heisenberg exchange terms (parameter J) for a single pair of
        neighboring sites.

        Args:
            i (int): The index of the first site.
            j (int): The index of the second site.
        """
        # --- Hopping Terms (t) ---
        # These terms require a Fermi string between the two operators.
        insert_ops = [self.identity, self.fermi_string, self.identity]
        self.fsm.add_term(-self.t, [self.adag_up, self.a_up], [i, j], insert_ops)
        self.fsm.add_term(-self.t, [self.adag_down, self.a_down], [i, j], insert_ops)
        self.fsm.add_term(-self.t, [self.a_up, self.adag_up], [i, j], insert_ops)
        self.fsm.add_term(-self.t, [self.a_down, self.adag_down], [i, j], insert_ops)

        # --- Heisenberg Exchange Terms (J) ---
        # S_i . S_j = Sz_i*Sz_j + 1/2*(Sp_i*Sm_j + Sm_i*Sp_j)
        self.fsm.add_term(self.j, [self.s_z, self.s_z], [i, j])
        self.fsm.add_term(self.j / 2, [self.s_plus, self.s_minus], [i, j])
        self.fsm.add_term(self.j / 2, [self.s_minus, self.s_plus], [i, j])

        # The t-J model often includes a density-density term from the
        # large-U expansion of the Hubbard model: -J/4 * n_i * n_j
        self.fsm.add_term(-self.j / 4, [self.n_tot, self.n_tot], [i, j])

    def get_mpo(self):
        """Generates and returns the numerical MPO tensors from the FSM.

        Returns:
            list[np.ndarray] | None: A list of NumPy arrays, where each array
            is an MPO tensor for a site. Returns None if the FSM has not been
            built.
        """
        if self.fsm is None:
            print("Error: FSM has not been built. Please run build_fsm() first.")
            return None
        return self.fsm.to_mpo()

    def display_symbolic_mpo(self):
        """Prints a symbolic representation of the MPO to the console.

        This is a useful debugging tool to inspect the structure and operators
        of the generated MPO without dealing with the numerical tensor data.
        """
        if self.fsm is None:
            print("Error: FSM has not been built.")
            return

        print("=" * 50)
        print(f"Symbolic MPO for {self.nx}x{self.ny} t-J Model (t={self.t}, J={self.j})")
        print("=" * 50)
        self.fsm.print_symbolic_mpo()

    def display_bond_dimensions(self):
        """Prints the bond dimension of the MPO at each virtual bond."""
        if self.fsm is None:
            print("Error: FSM has not been built.")
            return

        print("=" * 50)
        print("MPO Bond Dimensions")
        print("=" * 50)
        self.fsm.print_bond_dimensions()


if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Define the model parameters for a 3x2 ladder system.
    model_params = {
        'nx': 3,
        'ny': 2,
        't': 3.0,
        'j': 1.0,
        'periodic_y': False  # Use open boundaries for a ladder geometry
    }

    # 2. Instantiate the builder. The FSM is constructed automatically.
    print(f"Building MPO for a {model_params['nx']}x{model_params['ny']} "
          f"t-J ladder...")
    tj_model_builder = TJModelMPOBuilder(**model_params)
    print("Build complete.")

    # 3. Display the symbolic representation of the generated MPO.
    tj_model_builder.display_symbolic_mpo()

    # 4. Display the resulting bond dimensions of the compressed MPO.
    tj_model_builder.display_bond_dimensions()

    # 5. (Optional) Generate the actual numerical MPO tensors.
    numerical_mpo = tj_model_builder.get_mpo()
    if numerical_mpo:
        print("\nSuccessfully generated numerical MPO tensors.")
        print(f"Number of MPO tensors: {len(numerical_mpo)}")
        print(f"Shape of the MPO tensor for site 0: {numerical_mpo[0].shape}")
