import numpy as np
import torch
import sympy as sp
from typing import List, Callable, Dict, Any, Optional

from IPython.display import display

from . import operator as op
from .hamiltonian import Hamiltonian


class BaseModel(Hamiltonian):
    """
    An abstract base class for physical models.

    It extends the standard Hamiltonian to include a method for generating
    a symbolic representation of the Hamiltonian using SymPy.
    """

    def get_symbolic_hamiltonian(self) -> sp.Expr:
        """
        Generates the symbolic (mathematical) formula for the Hamiltonian.

        This method should be implemented by each specific model subclass.

        Returns:
            sp.Expr: A SymPy expression representing the Hamiltonian.
        """
        raise NotImplementedError(
            "Each model must implement its own symbolic Hamiltonian."
        )

    def print_hamiltonian(self):
        """
        Generates and pretty-prints the symbolic Hamiltonian in a text-based
        console using Unicode.
        """
        sp.init_printing(use_unicode=True)
        symbolic_H = self.get_symbolic_hamiltonian()
        print("Symbolic Hamiltonian:")
        sp.pprint(symbolic_H)

    def display_hamiltonian(self):
        """
        Displays the symbolic Hamiltonian rendered beautifully with LaTeX
        in a Jupyter Notebook environment.
        """
        # Configure SymPy to output LaTeX that MathJax can render
        sp.init_printing(use_latex='mathjax')
        symbolic_H = self.get_symbolic_hamiltonian()
        print("Symbolic Hamiltonian:")
        # The 'display' function from IPython is the key to rendering the object
        display(symbolic_H)


class BaseModel2D(BaseModel):
    """
    An abstract base class for 2D lattice models.

    Handles the logic for 2D geometry, including mapping 2D coordinates
    to 1D indices and finding nearest neighbors.
    """
    def __init__(self, width: int, height: int, **kwargs):
        """
        Initializes the 2D model.

        Args:
            width (int): The number of sites in the x-direction.
            height (int): The number of sites in the y-direction.
        """
        self.width = width
        self.height = height
        total_sites = width * height
        super().__init__(total_sites, **kwargs)

    def _map_coord_to_index(self, x: int, y: int) -> int:
        """Maps a 2D coordinate (x, y) to a 1D index using row-major order."""
        return y * self.width + x

    def _add_all_nearest_neighbor_terms(self, pbc: bool, **couplings: Dict[str, Any]):
        """
        A helper function to iterate over all sites and add nearest-neighbor terms.

        Args:
            pbc (bool): Whether to use periodic boundary conditions.
            **couplings: A dictionary of {term_name: (coefficient, operator_pair)}.
                         Example: {"J_x": (jx_val, [op.S_x, op.S_x])}
        """
        for y in range(self.height):
            for x in range(self.width):
                current_site_idx = self._map_coord_to_index(x, y)

                # --- Interaction with neighbor to the right ---
                if x < self.width - 1 or pbc:
                    right_neighbor_x = (x + 1) % self.width
                    right_neighbor_idx = self._map_coord_to_index(right_neighbor_x, y)
                    for term_name, (coeff, op_pair) in couplings.items():
                        if abs(coeff) > 1e-12:
                            self.add_spin_term(coeff, op_pair, [current_site_idx, right_neighbor_idx])

                # --- Interaction with neighbor below ---
                if y < self.height - 1 or pbc:
                    down_neighbor_y = (y + 1) % self.height
                    down_neighbor_idx = self._map_coord_to_index(x, down_neighbor_y)
                    for term_name, (coeff, op_pair) in couplings.items():
                         if abs(coeff) > 1e-12:
                            self.add_spin_term(coeff, op_pair, [current_site_idx, down_neighbor_idx])


class BaseHoneycombModel(BaseModel):
    """
    An abstract base class for models on a 2D honeycomb lattice.

    Handles the complex geometry of the honeycomb lattice, including the
    two-site unit cell, mapping to 1D indices, and identifying the
    three types of nearest-neighbor bonds (x, y, z).
    """
    def __init__(self, u_cells: int, v_cells: int, **kwargs):
        """
        Initializes the honeycomb model.

        Args:
            u_cells (int): The number of unit cells in the first lattice vector direction.
            v_cells (int): The number of unit cells in the second lattice vector direction.
        """
        self.u_cells = u_cells
        self.v_cells = v_cells
        # Each unit cell has 2 sites (sublattices A and B)
        total_sites = 2 * u_cells * v_cells
        super().__init__(total_sites, **kwargs)

    def _map_coord_to_index(self, u: int, v: int, sublattice: int) -> int:
        """Maps a honeycomb coordinate (u, v, sublattice) to a 1D index."""
        return (v * self.u_cells + u) * 2 + sublattice

    def _build_kitaev_terms(self, jx: float, jy: float, jz: float, pbc: bool):
        """
        Iterates through the lattice to build the bond-dependent Kitaev terms.
        """
        # Iterate over all unit cells
        for v in range(self.v_cells):
            for u in range(self.u_cells):
                # Site A (sublattice 0) in the current unit cell (u, v)
                site_a_idx = self._map_coord_to_index(u, v, 0)

                # --- Bond definitions relative to Site A ---

                # 1. Z-bond: Connects A and B inside the same unit cell
                site_b_z_idx = self._map_coord_to_index(u, v, 1)
                if abs(jz) > 1e-12:
                    self.add_spin_term(jz, [op.S_z, op.S_z], [site_a_idx, site_b_z_idx])

                # 2. X-bond: Connects A to B in the unit cell to the 'left'
                u_left = (u - 1 + self.u_cells) % self.u_cells if pbc else u - 1
                if u_left >= 0:
                    site_b_x_idx = self._map_coord_to_index(u_left, v, 1)
                    if abs(jx) > 1e-12:
                        self.add_spin_term(jx, [op.S_x, op.S_x], [site_a_idx, site_b_x_idx])

                # 3. Y-bond: Connects A to B in the unit cell 'above'
                v_up = (v - 1 + self.v_cells) % self.v_cells if pbc else v - 1
                if v_up >= 0:
                    site_b_y_idx = self._map_coord_to_index(u, v_up, 1)
                    if abs(jy) > 1e-12:
                        self.add_spin_term(jy, [op.S_y, op.S_y], [site_a_idx, site_b_y_idx])


class HeisenbergModel(BaseModel):
    """
    Constructs the Hamiltonian for the 1D Heisenberg model.
    ...
    """

    def __init__(
            self,
            lattice_length: int,
            jx: float,
            jy: float,
            jz: float,
            hx: float,
            hy: float,
            hz: float,
            pbc: bool = True,
    ):
        """
        Initializes the 1D Heisenberg model Hamiltonian.
        ...
        """
        super().__init__(lattice_length, spin=0.5)
        self.jx, self.jy, self.jz = jx, jy, jz
        self.hx, self.hy, self.hz = hx, hy, hz
        self.pbc = pbc

        # (The rest of the __init__ method remains the same as before)
        num_interactions = self.lattice_length if self.pbc else self.lattice_length - 1
        for i in range(num_interactions):
            j = (i + 1) % self.lattice_length
            if abs(self.jx) > 1e-12:
                self.add_spin_term(self.jx, [op.S_x, op.S_x], [i, j])
            if abs(self.jy) > 1e-12:
                self.add_spin_term(self.jy, [op.S_y, op.S_y], [i, j])
            if abs(self.jz) > 1e-12:
                self.add_spin_term(self.jz, [op.S_z, op.S_z], [i, j])
        for i in range(self.lattice_length):
            if abs(self.hx) > 1e-12:
                self.add_spin_term(-self.hx, [op.S_x], [i])
            if abs(self.hy) > 1e-12:
                self.add_spin_term(-self.hy, [op.S_y], [i])
            if abs(self.hz) > 1e-12:
                self.add_spin_term(-self.hz, [op.S_z], [i])

    def get_symbolic_hamiltonian(self) -> sp.Expr:
        """Generates the symbolic formula for the Heisenberg Hamiltonian."""
        J_x, J_y, J_z = sp.symbols("J_x J_y J_z")
        h_x, h_y, h_z = sp.symbols("h_x h_y h_z")

        # Use sp.IndexedBase for operators with subscripts
        S_x = sp.IndexedBase("S^x")
        S_y = sp.IndexedBase("S^y")
        S_z = sp.IndexedBase("S^z")
        i, N = sp.symbols("i N", integer=True, positive=True)

        # In physics, sum up to N implies PBC where S_{N+1} -> S_1
        sum_limit = (i, 1, N) if self.pbc else (i, 1, N - 1)

        interaction_term = J_x * S_x[i] * S_x[i + 1] + \
                           J_y * S_y[i] * S_y[i + 1] + \
                           J_z * S_z[i] * S_z[i + 1]
        H_interaction = sp.Sum(interaction_term, sum_limit)

        field_term = h_x * S_x[i] + h_y * S_y[i] + h_z * S_z[i]
        H_field = sp.Sum(field_term, (i, 1, N))

        return H_interaction - H_field


class IsingModel(BaseModel):
    """
    Constructs the Hamiltonian for the 1D Transverse Field Ising Model (TFIM).
    ...
    """

    def __init__(self, lattice_length: int, j_coupling: float, h_field: float, pbc: bool = True):
        """
        Initializes the 1D TFIM Hamiltonian.
        ...
        """
        super().__init__(lattice_length, spin=0.5)
        self.j_coupling = j_coupling
        self.h_field = h_field
        self.pbc = pbc
        # (The rest of the __init__ method remains the same as before)
        num_interactions = self.lattice_length if self.pbc else self.lattice_length - 1
        for i in range(num_interactions):
            j = (i + 1) % self.lattice_length
            self.add_spin_term(-self.j_coupling, [op.S_x, op.S_x], [i, j])
        for i in range(self.lattice_length):
            self.add_spin_term(-self.h_field, [op.S_z], [i])

    def get_symbolic_hamiltonian(self) -> sp.Expr:
        """Generates the symbolic formula for the TFIM Hamiltonian."""
        J, h = sp.symbols("J h")

        # Use sp.IndexedBase for operators
        S_x = sp.IndexedBase("S^x")
        S_z = sp.IndexedBase("S^z")
        i, N = sp.symbols("i N", integer=True, positive=True)

        sum_limit = (i, 1, N) if self.pbc else (i, 1, N - 1)

        H_interaction = sp.Sum(S_x[i] * S_x[i + 1], sum_limit)
        H_field = sp.Sum(S_z[i], (i, 1, N))

        return -J * H_interaction - h * H_field


class KitaevChain(BaseModel):
    """
    Constructs the Hamiltonian for the 1D Kitaev chain.
    ...
    """

    def __init__(
            self,
            lattice_length: int,
            chemical_potential: float,
            hopping: float,
            pairing_gap: float,
            pbc: bool = False
    ):
        """
        Initializes the Kitaev Chain Hamiltonian.
        ...
        """
        super().__init__(lattice_length, spin=0.5, convention="down=1")
        self.mu = chemical_potential
        self.t = hopping
        self.delta = pairing_gap
        self.pbc = pbc
        # (The rest of the __init__ method remains the same as before)
        num_hoppings = self.lattice_length if self.pbc else self.lattice_length - 1
        for j in range(self.lattice_length):
            self.add_fermion_term(-self.mu, [op.c_dag_j, op.c_j], [j, j])
            self.add_prebuilt_term(
                0.5 * self.mu * torch.eye(self.total_dim, dtype=torch.complex64)
            )
        for j in range(num_hoppings):
            k = (j + 1) % self.lattice_length
            self.add_fermion_term(-self.t, [op.c_dag_j, op.c_j], [j, k])
            self.add_fermion_term(-self.t, [op.c_dag_j, op.c_j], [k, j])
            self.add_fermion_term(self.delta, [op.c_j, op.c_j], [j, k])
            self.add_fermion_term(self.delta, [op.c_dag_j, op.c_dag_j], [k, j])

    def get_symbolic_hamiltonian(self) -> sp.Expr:
        """Generates the symbolic formula for the Kitaev Chain Hamiltonian."""
        mu, t, Delta = sp.symbols("mu t Delta")

        # Define fermion operators as NON-COMMUTATIVE symbols
        c_dag = sp.IndexedBase("c^\\dagger", commutative=False)
        c = sp.IndexedBase("c", commutative=False)
        j, N = sp.symbols("j N", integer=True, positive=True)

        sum_limit = (j, 1, N) if self.pbc else (j, 1, N - 1)

        # Chemical potential term (Correct order: c_dag * c)
        H_mu = sp.Sum(-mu * (c_dag[j] * c[j] - sp.Rational(1, 2)), (j, 1, N))

        # Hopping and pairing terms (Correct order)
        hop_pair_term = -t * (c_dag[j] * c[j + 1] + c_dag[j + 1] * c[j]) + \
                        Delta * (c[j] * c[j + 1] + c_dag[j + 1] * c_dag[j])

        H_hop_pair = sp.Sum(hop_pair_term, sum_limit)

        return H_mu + H_hop_pair


class SSHModel(BaseModel):
    """
    Constructs the Hamiltonian for the 1D Su-Schrieffer-Heeger (SSH) model.

    This model is a cornerstone of topological physics, describing spinless
    fermions on a 1D bipartite lattice with alternating hopping amplitudes.
    The Hamiltonian is:
    H = -v * sum_i (c_{A,i}^dag c_{B,i} + h.c.) - w * sum_i (c_{B,i}^dag c_{A,i+1} + h.c.)
    """

    def __init__(self, num_cells: int, intra_cell_hopping: float,
                 inter_cell_hopping: float, pbc: bool = False):
        """
        Initializes the SSH model.

        Args:
            num_cells (int): The number of unit cells in the chain. The total
                             number of sites will be 2 * num_cells.
            intra_cell_hopping (float): The hopping amplitude within a cell (v).
            inter_cell_hopping (float): The hopping amplitude between cells (w).
            pbc (bool): If True, use periodic boundary conditions. Defaults to
                        False, which is standard for observing edge states.
        """
        self.num_cells = num_cells
        self.v = intra_cell_hopping
        self.w = inter_cell_hopping
        self.pbc = pbc

        total_sites = 2 * self.num_cells
        super().__init__(lattice_length=total_sites, spin=0.5)

        # --- Add Hopping Terms ---

        # Intra-cell hopping (v-bonds)
        if abs(self.v) > 1e-12:
            for i in range(self.num_cells):
                site_a = 2 * i
                site_b = 2 * i + 1
                # Term: -v * c_a^dag * c_b
                self.add_fermion_term(-self.v, [op.c_dag_j, op.c_j], [site_a, site_b])
                # Hermitian Conjugate: -v * c_b^dag * c_a
                self.add_fermion_term(-self.v, [op.c_dag_j, op.c_j], [site_b, site_a])

        # Inter-cell hopping (w-bonds)
        if abs(self.w) > 1e-12:
            num_inter_links = self.num_cells if self.pbc else self.num_cells - 1
            for i in range(num_inter_links):
                source_site = 2 * i + 1  # B-site of the i-th cell
                target_site = (2 * (i + 1)) % total_sites  # A-site of the (i+1)-th cell

                # Term: -w * c_{source}^dag * c_{target}
                self.add_fermion_term(-self.w, [op.c_dag_j, op.c_j], [source_site, target_site])
                # Hermitian Conjugate: -w * c_{target}^dag * c_{source}
                self.add_fermion_term(-self.w, [op.c_dag_j, op.c_j], [target_site, source_site])

    def build_single_particle_hamiltonian(self) -> torch.Tensor:
        """
        Builds the single-particle (tight-binding) Hamiltonian matrix.

        For non-interacting models like SSH, this L x L matrix captures all
        the essential physics (band structure, gap, edge states) and is much
        more efficient to diagonalize than the 2^L x 2^L many-body matrix.

        Returns:
            torch.Tensor: The L x L single-particle Hamiltonian matrix.
        """
        L = 2 * self.num_cells
        H_sp = torch.zeros((L, L), dtype=torch.complex64)

        # Intra-cell hopping (v-bonds)
        if abs(self.v) > 1e-12:
            for i in range(self.num_cells):
                site_a = 2 * i
                site_b = 2 * i + 1
                H_sp[site_a, site_b] = -self.v
                H_sp[site_b, site_a] = -self.v

        # Inter-cell hopping (w-bonds)
        if abs(self.w) > 1e-12:
            num_inter_links = self.num_cells if self.pbc else self.num_cells - 1
            for i in range(num_inter_links):
                source_site = 2 * i + 1
                target_site = (2 * (i + 1)) % L
                H_sp[source_site, target_site] = -self.w
                H_sp[target_site, source_site] = -self.w

        return H_sp

    # --- The get_symbolic_hamiltonian method remains the same ---
    def get_symbolic_hamiltonian(self) -> sp.Expr:
        v, w = sp.symbols("v w")
        i = sp.symbols("i", integer=True, positive=True)

        c_dag_A = sp.IndexedBase("c^\\dagger_A", commutative=False)
        c_B = sp.IndexedBase("c_B", commutative=False)
        c_dag_B = sp.IndexedBase("c^\\dagger_B", commutative=False)
        c_A = sp.IndexedBase("c_A", commutative=False)

        SumI = sp.Function("\\sum_{i}")

        H_intra = SumI(c_dag_A[i] * c_B[i] + c_dag_B[i] * c_A[i])
        H_inter = SumI(c_dag_B[i] * c_A[i + 1] + c_dag_A[i + 1] * c_B[i])

        term_v = -v * H_intra
        term_w = -w * H_inter

        # The standard Hamiltonian form includes negative signs
        return sp.Add(term_v, term_w, evaluate=False)


class HeisenbergModel2D(BaseModel2D):
    """Constructs the Hamiltonian for the 2D Heisenberg model."""

    def __init__(self, width: int, height: int, jx: float, jy: float, jz: float, hx: float, hy: float, hz: float, pbc: bool = True):
        super().__init__(width, height, spin=0.5)
        self.jx, self.jy, self.jz = jx, jy, jz
        self.hx, self.hy, self.hz = hx, hy, hz
        self.pbc = pbc

        # Add nearest-neighbor interaction terms
        couplings = {
            "J_x": (self.jx, [op.S_x, op.S_x]),
            "J_y": (self.jy, [op.S_y, op.S_y]),
            "J_z": (self.jz, [op.S_z, op.S_z])
        }
        self._add_all_nearest_neighbor_terms(self.pbc, **couplings)

        # Add external field terms (on-site)
        for i in range(self.lattice_length):
            if abs(self.hx) > 1e-12: self.add_spin_term(-self.hx, [op.S_x], [i])
            if abs(self.hy) > 1e-12: self.add_spin_term(-self.hy, [op.S_y], [i])
            if abs(self.hz) > 1e-12: self.add_spin_term(-self.hz, [op.S_z], [i])

    def get_symbolic_hamiltonian(self) -> sp.Expr:
        """Generates the symbolic formula for the 2D Heisenberg Hamiltonian."""
        J_x, J_y, J_z = sp.symbols("J_x J_y J_z")
        h_x, h_y, h_z = sp.symbols("h_x h_y h_z")

        # Define summation symbols as undefined functions to control printing
        SumNN = sp.Function("\\sum_{\\langle i,j \\rangle}")
        SumI = sp.Function("\\sum_{i}")

        S_ix, S_iy, S_iz = sp.symbols("S^x_i S^y_i S^z_i")
        S_jx, S_jy, S_jz = sp.symbols("S^x_j S^y_j S^z_j")

        # The function call ensures the summation "acts on" the expression
        interaction_term = J_x * S_ix * S_jx + J_y * S_iy * S_jy + J_z * S_iz * S_jz
        H_interaction = SumNN(interaction_term)

        field_term = h_x * S_ix + h_y * S_iy + h_z * S_iz
        H_field = SumI(field_term)

        return H_interaction - H_field


class IsingModel2D(BaseModel2D):
    """Constructs the Hamiltonian for the 2D Transverse Field Ising Model."""

    def __init__(self, width: int, height: int, j_coupling: float, h_field: float, pbc: bool = True):
        super().__init__(width, height, spin=0.5)
        self.j_coupling = j_coupling
        self.h_field = h_field
        self.pbc = pbc

        # Add nearest-neighbor S^x S^x interaction terms
        couplings = { "J_x": (-self.j_coupling, [op.S_x, op.S_x]) }
        self._add_all_nearest_neighbor_terms(self.pbc, **couplings)

        # Add external field terms (on-site)
        for i in range(self.lattice_length):
             if abs(self.h_field) > 1e-12:
                self.add_spin_term(-self.h_field, [op.S_z], [i])

    def get_symbolic_hamiltonian(self) -> sp.Expr:
        """Generates the symbolic formula for the 2D TFIM Hamiltonian."""
        J, h = sp.symbols("J h")

        # Define summation symbols as undefined functions
        SumNN = sp.Function("\\sum_{\\langle i,j \\rangle}")
        SumI = sp.Function("\\sum_{i}")

        S_ix, S_jx = sp.symbols("S^x_i S^x_j")
        S_iz = sp.symbols("S^z_i")

        # The function call structure prevents reordering
        H_interaction = SumNN(S_ix * S_jx)
        H_field = SumI(S_iz)

        return -J * H_interaction - h * H_field


class BoseHubbardModel2D(BaseModel2D):
    """
    Constructs the Hamiltonian for the 2D soft-core Bose-Hubbard model.

    The model is defined on a 2D lattice with a truncated local Hilbert space,
    where each site can be occupied by a maximum of `n_max` bosons.
    The Hamiltonian is:
    H = -t * sum(<i,j>) (b_i^dag b_j + h.c.)
        + (U/2) * sum(i) n_i(n_i - 1)
        - mu * sum(i) n_i
    """

    def __init__(self, width: int, height: int, hopping: float, interaction: float,
                 chemical_potential: float, n_max: int, pbc: bool = True):
        """
        Initializes the 2D Bose-Hubbard model.

        Args:
            width (int): The number of sites in the x-direction.
            height (int): The number of sites in the y-direction.
            hopping (float): The hopping amplitude (t).
            interaction (float): The on-site interaction strength (U).
            chemical_potential (float): The chemical potential (mu).
            n_max (int): The maximum number of bosons allowed per site.
                         The local Hilbert space dimension will be n_max + 1.
            pbc (bool): Whether to use periodic boundary conditions.
        """
        if not isinstance(n_max, int) or n_max < 1:
            raise ValueError("n_max must be an integer >= 1.")

        self.t = hopping
        self.u = interaction
        self.mu = chemical_potential
        self.n_max = n_max
        self.pbc = pbc

        # The local state dimension is n_max + 1 (for 0, 1, ..., n_max bosons)
        n_state = self.n_max + 1
        # The underlying spin representation is (n_state - 1) / 2
        effective_spin = (n_state - 1) / 2

        super().__init__(width, height, spin=effective_spin)

        # --- Define local operators compatible with our builder ---
        # The builder passes 'spin' as a kwarg, but our operators need 'n_state'.
        # We use lambda functions to create compatible wrappers.
        b_dag_local = lambda spin: op.b_dag(n_state=int(2 * spin + 1))
        b_local = lambda spin: op.b_(n_state=int(2 * spin + 1))

        # --- Add nearest-neighbor hopping terms ---
        couplings = {
            "hopping_fwd": (-self.t, [b_dag_local, b_local]),  # -t * b_i^dag b_j
            "hopping_bwd": (-self.t, [b_local, b_dag_local])  # -t * b_j^dag b_i (h.c.)
        }
        self._add_all_nearest_neighbor_terms(self.pbc, **couplings)

        # --- Add on-site terms (Interaction and Chemical Potential) ---
        # The full on-site term is: (U/2) * n_i(n_i-1) - mu * n_i
        # This can be rewritten as: (U/2) * n_i^2 - (U/2 + mu) * n_i

        # Pre-calculate the local operator matrices for n and n^2
        n_local_op = b_dag_local(effective_spin) @ b_local(effective_spin)
        n_squared_local_op = n_local_op @ n_local_op

        # Define functions that return these pre-built matrices.
        # This is how we pass them to the SpinBosonTerm builder.
        def n_op_func(spin):
            return n_local_op

        def n_sq_op_func(spin):
            return n_squared_local_op

        # Add the two parts of the on-site term for each site
        for i in range(self.lattice_length):
            # Add (U/2) * n_i^2 term
            if abs(self.u) > 1e-12:
                self.add_spin_term(self.u / 2.0, [n_sq_op_func], [i])

            # Add -(U/2 + mu) * n_i term
            coeff_n = - (self.u / 2.0 + self.mu)
            if abs(coeff_n) > 1e-12:
                self.add_spin_term(coeff_n, [n_op_func], [i])

    def get_symbolic_hamiltonian(self) -> sp.Expr:
        """Generates the symbolic formula for the 2D Bose-Hubbard model."""
        t, U, mu = sp.symbols("t U mu")

        SumNN = sp.Function("\\sum_{\\langle i,j \\rangle}")
        SumI = sp.Function("\\sum_{i}")

        b_dag_i, b_dag_j = sp.symbols("b^{\\dagger}_i b^{\\dagger}_j", commutative=False)
        b_i, b_j = sp.symbols("b_i b_j", commutative=False)
        # For the formula, it's clearer to show b_dag*b instead of a generic n_i
        n_i_expr = b_dag_i * b_i

        H_hopping = SumNN(b_dag_i * b_j + b_dag_j * b_i)
        H_interaction = SumI(n_i_expr * (n_i_expr - 1))
        H_potential = SumI(n_i_expr)

        # Define each term separately
        term_hop = -t * H_hopping
        term_int = (1 / 2) * U * H_interaction
        term_pot = -mu * H_potential

        # Use sp.Add with evaluate=False to preserve the exact order of terms
        return sp.Add(term_hop, term_int, term_pot, evaluate=False)


class HubbardModel1D(BaseModel):
    """
    Constructs the Hamiltonian for the 1D spinful Hubbard model.

    The model describes interacting fermions (electrons) on a 1D lattice.
    The Hamiltonian is:
    H = -t * sum_{i,sigma} (c_{i,s}^dag c_{i+1,s} + h.c.)
        + U * sum_{i} n_{i,up} n_{i,down}
        - mu * sum_{i,sigma} n_{i,s}
    """
    def __init__(self, lattice_length: int, hopping: float, interaction: float,
                 chemical_potential: float, pbc: bool = True):
        """
        Initializes the 1D Hubbard model.

        Args:
            lattice_length (int): The number of physical sites in the chain.
            hopping (float): The hopping amplitude (t).
            interaction (float): The on-site interaction strength (U).
            chemical_potential (float): The chemical potential (mu).
            pbc (bool): Whether to use periodic boundary conditions.
        """
        self.physical_L = lattice_length
        self.t = hopping
        self.u = interaction
        self.mu = chemical_potential
        self.pbc = pbc

        # For a spinful model, each physical site has two states (up, down).
        # The underlying Jordan-Wigner chain has length 2 * L.
        # The total Hilbert space dimension is 2**(2*L).
        total_jw_length = 2 * self.physical_L
        super().__init__(lattice_length=total_jw_length, spin=0.5)

        # --- Define wrapper functions for spinful operators ---
        # Our FermionTerm builder passes `lattice_length` from the Hamiltonian
        # instance (which is 2*L). However, our spinful operators expect the
        # *physical* lattice length (L). We create wrappers to pass the correct L.
        def c_dag(i, spin):
            return lambda site, length, conv: op.c_dag_j_spinful(site, spin, self.physical_L, conv)
        def c(i, spin):
            return lambda site, length, conv: op.c_j_spinful(site, spin, self.physical_L, conv)

        # --- Add Hopping Terms ---
        num_hoppings = self.physical_L if self.pbc else self.physical_L - 1
        for i in range(num_hoppings):
            j = (i + 1) % self.physical_L
            for spin in ['up', 'down']:
                # Term -t * c_{i,s}^dag c_{j,s}
                self.add_fermion_term(-self.t, [c_dag(i, spin), c(j, spin)], [i, j])
                # Term -t * c_{j,s}^dag c_{i,s} (h.c.)
                self.add_fermion_term(-self.t, [c_dag(j, spin), c(i, spin)], [j, i])

        # --- Add On-site Interaction and Chemical Potential Terms ---
        for i in range(self.physical_L):
            # Interaction term: U * n_{i,up} * n_{i,down}
            # n_{i,up} = c_{i,up}^dag c_{i,up}
            # n_{i,down} = c_{i,down}^dag c_{i,down}
            if abs(self.u) > 1e-12:
                ops_U = [c_dag(i, 'up'), c(i, 'up'), c_dag(i, 'down'), c(i, 'down')]
                sites_U = [i, i, i, i]
                self.add_fermion_term(self.u, ops_U, sites_U)

            # Chemical potential term: -mu * (n_{i,up} + n_{i,down})
            if abs(self.mu) > 1e-12:
                # Up spin part: -mu * n_{i,up}
                self.add_fermion_term(-self.mu, [c_dag(i, 'up'), c(i, 'up')], [i, i])
                # Down spin part: -mu * n_{i,down}
                self.add_fermion_term(-self.mu, [c_dag(i, 'down'), c(i, 'down')], [i, i])

    def get_symbolic_hamiltonian(self) -> sp.Expr:
        """Generates the symbolic formula for the 1D Hubbard model."""
        t, U, mu = sp.symbols("t U mu")

        # Define summation symbols as undefined functions
        Sum_i_sigma = sp.Function("\\sum_{i, \sigma}")
        Sum_i = sp.Function("\\sum_{i}")

        # Define symbolic operators and indices
        sigma = sp.Symbol("\\sigma")
        i = sp.Symbol("i")
        c_dag = sp.IndexedBase("c^\\dagger", commutative=False)
        c = sp.IndexedBase("c", commutative=False)
        n = sp.IndexedBase("n")
        up, down = sp.symbols("\\uparrow \\downarrow")

        # Hopping term
        hopping_expr = c_dag[i, sigma] * c[i + 1, sigma] + c_dag[i + 1, sigma] * c[i, sigma]
        H_hopping = Sum_i_sigma(hopping_expr)

        # Interaction term
        interaction_expr = n[i, up] * n[i, down]
        H_interaction = Sum_i(interaction_expr)

        # Chemical potential term
        potential_expr = n[i, sigma]
        H_potential = Sum_i_sigma(potential_expr)

        # Define each full term
        term_hop = -t * H_hopping
        term_int = U * H_interaction
        term_pot = -mu * H_potential

        # Use sp.Add with evaluate=False to preserve the exact order
        return sp.Add(term_hop, term_int, term_pot, evaluate=False)


class TJModel2D(BaseModel2D):
    """
    Constructs the Hamiltonian for the 2D t-J model.

    This model describes strongly correlated electrons in the limit of large
    on-site repulsion (U -> infinity), forbidding double occupancy.
    The Hamiltonian is:
    H = -t * sum_{<i,j>,s} (P c_{i,s}^dag c_{j,s} P + h.c.)
        + J * sum_{<i,j>} (S_i . S_j - n_i * n_j / 4)
    where P is the projector onto the subspace with no double occupancy.
    """

    def __init__(self, width: int, height: int, hopping: float, exchange: float, pbc: bool = True):
        self.physical_W = width
        self.physical_H = height
        self.physical_L = width * height
        self.t = hopping
        self.j_exch = exchange
        self.pbc = pbc

        # The underlying JW chain has length 2 * L
        total_jw_length = 2 * self.physical_L
        super().__init__(width, height, spin=0.5)
        # Manually override total_dim for the spinful fermionic space 2**(2L)
        self.total_dim = 2 ** total_jw_length
        self.lattice_length = total_jw_length  # Set this for the builder

        # --- Define wrapper functions for spinful operators ---
        def c_dag(i, spin):
            return lambda site, length, conv: op.c_dag_j_spinful(site, spin, self.physical_L, conv)

        def c(i, spin):
            return lambda site, length, conv: op.c_j_spinful(site, spin, self.physical_L, conv)

        def n(i, spin):
            return lambda site, length, conv: op.n_j_spinful(site, spin, self.physical_L, conv)

        # --- Build terms for each nearest-neighbor pair ---
        for y in range(self.physical_H):
            for x in range(self.physical_W):
                i = self._map_coord_to_index(x, y)

                # Neighbors (right and down to avoid double counting)
                neighbors = []
                if x < self.physical_W - 1 or self.pbc:
                    neighbors.append(self._map_coord_to_index((x + 1) % self.physical_W, y))
                if y < self.physical_H - 1 or self.pbc:
                    neighbors.append(self._map_coord_to_index(x, (y + 1) % self.physical_H))

                for j in neighbors:
                    # --- J-term: J * (S_i.S_j - n_i*n_j/4) ---
                    if abs(self.j_exch) > 1e-12:
                        # S_i^z S_j^z term
                        # S_z = 0.5 * (n_up - n_down)
                        # S_i^z S_j^z = 0.25 * (n_i_up - n_i_dn)(n_j_up - n_j_dn)
                        # Expands to: 0.25 * (n_i_up*n_j_up - n_i_up*n_j_dn - n_i_dn*n_j_up + n_i_dn*n_j_dn)
                        self.add_fermion_term(0.25 * self.j_exch, [n(i, 'up'), n(j, 'up')], [i, j])
                        self.add_fermion_term(-0.25 * self.j_exch, [n(i, 'up'), n(j, 'dn')], [i, j])
                        self.add_fermion_term(-0.25 * self.j_exch, [n(i, 'dn'), n(j, 'up')], [i, j])
                        self.add_fermion_term(0.25 * self.j_exch, [n(i, 'dn'), n(j, 'dn')], [i, j])

                        # S_i^x S_j^x + S_i^y S_j^y = 0.5 * (S_i^+ S_j^- + S_i^- S_j^+)
                        # S_i^+ = c_i_up^dag c_i_dn
                        # S_i^- = c_i_dn^dag c_i_up
                        # 0.5 * (c_i_up^dag c_i_dn c_j_dn^dag c_j_up + c_i_dn^dag c_i_up c_j_up^dag c_j_dn)
                        self.add_fermion_term(0.5 * self.j_exch,
                                              [c_dag(i, 'up'), c(i, 'dn'), c_dag(j, 'dn'), c(j, 'up')], [i, i, j, j])
                        self.add_fermion_term(0.5 * self.j_exch,
                                              [c_dag(i, 'dn'), c(i, 'up'), c_dag(j, 'up'), c(j, 'dn')], [i, i, j, j])

                        # -J/4 * n_i * n_j = -J/4 * (n_i_up + n_i_dn)(n_j_up + n_j_dn)
                        # Expands to: -J/4 * (n_i_up*n_j_up + n_i_up*n_j_dn + n_i_dn*n_j_up + n_i_dn*n_j_dn)
                        self.add_fermion_term(-0.25 * self.j_exch, [n(i, 'up'), n(j, 'up')], [i, j])
                        self.add_fermion_term(-0.25 * self.j_exch, [n(i, 'up'), n(j, 'dn')], [i, j])
                        self.add_fermion_term(-0.25 * self.j_exch, [n(i, 'dn'), n(j, 'up')], [i, j])
                        self.add_fermion_term(-0.25 * self.j_exch, [n(i, 'dn'), n(j, 'dn')], [i, j])

                    # --- t-term: -t * sum_{s} (P c_{i,s}^dag c_{j,s} P + h.c.) ---
                    # P c_{i,s}^dag c_{j,s} P is implemented as (1-n_{i,-s}) c_{i,s}^dag c_{j,s} (1-n_{j,-s})
                    if abs(self.t) > 1e-12:
                        for spin, opp_spin in [('up', 'dn'), ('dn', 'up')]:
                            # Hopping from j to i
                            # -t * (1-n_{i,-s}) c_{i,s}^dag c_{j,s} (1-n_{j,-s})
                            # = -t * (c_{i,s}^dag - n_{i,-s}c_{i,s}^dag) * (c_{j,s} - c_{j,s}n_{j,-s})
                            # This is complex. A simpler, equivalent form is:
                            # -t * c_{i,s}^dag c_{j,s} (1-n_{i,-s}) (1-n_{j,s}) ... No.
                            # The standard implementation is by projecting the initial and final states.
                            # tilde_c_dag_i = (1-n_i,-s)c_dag_i,s
                            # tilde_c_j = c_j,s(1-n_j,-s)
                            # The term is -t * (tilde_c_dag_i * tilde_c_j + h.c.)

                            # Forward hopping: j -> i for spin
                            # -t * c_{i,s}^dag * (1-n_{i,-s}) * c_{j,s}
                            self.add_fermion_term(-self.t, [c_dag(i, spin), c(j, spin)], [i, j])
                            self.add_fermion_term(self.t, [c_dag(i, spin), n(i, opp_spin), c(j, spin)], [i, i, j])

                            # Backward hopping: i -> j for spin (h.c.)
                            # -t * c_{j,s}^dag * (1-n_{j,-s}) * c_{i,s}
                            self.add_fermion_term(-self.t, [c_dag(j, spin), c(i, spin)], [j, i])
                            self.add_fermion_term(self.t, [c_dag(j, spin), n(j, opp_spin), c(i, spin)], [j, j, i])

    def get_symbolic_hamiltonian(self) -> sp.Expr:
        t, J, sigma = sp.symbols("t J sigma")
        i, j = sp.symbols("i j")
        SumNN = sp.Function("\\sum_{\\langle i,j \\rangle}")

        c_dag_s = sp.IndexedBase("\\tilde{c}^\\dagger", commutative=False)
        c_s = sp.IndexedBase("\\tilde{c}", commutative=False)
        S_i = sp.Symbol("\\mathbf{S}_i")
        S_j = sp.Symbol("\\mathbf{S}_j")
        n_i, n_j = sp.symbols("n_i n_j")

        H_hopping = SumNN(c_dag_s[i, sigma] * c_s[j, sigma] + sp.Symbol("h.c."))
        H_exchange = SumNN(S_i * S_j - sp.Rational(1, 4) * n_i * n_j)

        term_hop = -t * H_hopping
        term_exch = J * H_exchange

        return sp.Add(term_hop, term_exch, evaluate=False)


class KitaevHoneycombModel(BaseHoneycombModel):
    """
    Constructs the Hamiltonian for the Kitaev Honeycomb Model.

    The model features bond-dependent Ising-like interactions on a
    honeycomb lattice, and is a famous example of a quantum spin liquid.
    H = -J_x * sum_{x-links} S_i^x S_j^x - J_y * sum_{y-links} S_i^y S_j^y
        - J_z * sum_{z-links} S_i^z S_j^z
    """
    def __init__(self, u_cells: int, v_cells: int, jx: float, jy: float, jz: float, pbc: bool = True):
        super().__init__(u_cells, v_cells, spin=0.5)
        self.jx, self.jy, self.jz = jx, jy, jz
        self.pbc = pbc

        # The base class helper does all the heavy lifting of building terms
        self._build_kitaev_terms(-self.jx, -self.jy, -self.jz, self.pbc)

    def get_symbolic_hamiltonian(self) -> sp.Expr:
        Jx, Jy, Jz = sp.symbols("J_x J_y J_z")
        S_ix, S_iy, S_iz = sp.symbols("S^x_i S^y_i S^z_i")
        S_jx, S_jy, S_jz = sp.symbols("S^x_j S^y_j S^z_j")

        # Define summation functions for each link type
        SumX = sp.Function("\\sum_{\\langle i,j \\rangle \\in x}")
        SumY = sp.Function("\\sum_{\\langle i,j \\rangle \\in y}")
        SumZ = sp.Function("\\sum_{\\langle i,j \\rangle \\in z}")

        H_x_term = SumX(S_ix * S_jx)
        H_y_term = SumY(S_iy * S_jy)
        H_z_term = SumZ(S_iz * S_jz)

        term1 = -Jx * H_x_term
        term2 = -Jy * H_y_term
        term3 = -Jz * H_z_term

        return sp.Add(term1, term2, term3, evaluate=False)





