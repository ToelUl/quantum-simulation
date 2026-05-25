"""Pauli-backed model builders for spin and Jordan-Wigner fermion systems."""

from __future__ import annotations

from typing import Any

import sympy as sp
import torch

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - optional notebook dependency
    display = None

from .pauli_hamiltonian import PauliHamiltonian


SpinCoupling = tuple[complex, list[str]]


class BasePauliModel(PauliHamiltonian):
    """Base class for Pauli-backed spin-1/2 lattice models."""

    def get_symbolic_hamiltonian(self) -> sp.Expr:
        raise NotImplementedError(
            "Each Pauli model must implement get_symbolic_hamiltonian()."
        )

    def print_hamiltonian(self) -> None:
        sp.init_printing(use_unicode=True)
        symbolic_h = self.get_symbolic_hamiltonian()
        print("Symbolic Hamiltonian:")
        sp.pprint(symbolic_h)

    def display_hamiltonian(self) -> None:
        if display is None:
            raise ImportError(
                "display_hamiltonian requires IPython to be installed."
            )
        sp.init_printing(use_latex="mathjax")
        symbolic_h = self.get_symbolic_hamiltonian()
        print("Symbolic Hamiltonian:")
        display(symbolic_h)


class BasePauliModel2D(BasePauliModel):
    """Base class for Pauli-backed models on a rectangular 2D lattice."""

    def __init__(self, width: int, height: int, **kwargs: Any) -> None:
        self.width = int(width)
        self.height = int(height)
        super().__init__(self.width * self.height, **kwargs)

    def _map_coord_to_index(self, x: int, y: int) -> int:
        return y * self.width + x

    def _add_all_nearest_neighbor_terms(
            self,
            pbc: bool,
            **couplings: SpinCoupling,
    ) -> None:
        for y in range(self.height):
            for x in range(self.width):
                current_site_idx = self._map_coord_to_index(x, y)

                if x < self.width - 1 or pbc:
                    right_neighbor_x = (x + 1) % self.width
                    right_neighbor_idx = self._map_coord_to_index(
                        right_neighbor_x, y
                    )
                    for coeff, op_pair in couplings.values():
                        if abs(coeff) > 1e-12:
                            self.add_spin_term(
                                coeff,
                                op_pair,
                                [current_site_idx, right_neighbor_idx],
                            )

                if y < self.height - 1 or pbc:
                    down_neighbor_y = (y + 1) % self.height
                    down_neighbor_idx = self._map_coord_to_index(
                        x, down_neighbor_y
                    )
                    for coeff, op_pair in couplings.values():
                        if abs(coeff) > 1e-12:
                            self.add_spin_term(
                                coeff,
                                op_pair,
                                [current_site_idx, down_neighbor_idx],
                            )


class BasePauliHoneycombModel(BasePauliModel):
    """Base class for Pauli-backed models on a honeycomb lattice."""

    def __init__(self, u_cells: int, v_cells: int, **kwargs: Any) -> None:
        self.u_cells = int(u_cells)
        self.v_cells = int(v_cells)
        super().__init__(2 * self.u_cells * self.v_cells, **kwargs)

    def _map_coord_to_index(self, u: int, v: int, sublattice: int) -> int:
        return (v * self.u_cells + u) * 2 + sublattice

    def _build_kitaev_terms(
            self,
            jx: float,
            jy: float,
            jz: float,
            pbc: bool,
    ) -> None:
        for v in range(self.v_cells):
            for u in range(self.u_cells):
                site_a_idx = self._map_coord_to_index(u, v, 0)

                site_b_z_idx = self._map_coord_to_index(u, v, 1)
                if abs(jz) > 1e-12:
                    self.add_spin_term(
                        jz, ["S_z", "S_z"], [site_a_idx, site_b_z_idx]
                    )

                u_left = (u - 1 + self.u_cells) % self.u_cells if pbc else u - 1
                if u_left >= 0:
                    site_b_x_idx = self._map_coord_to_index(u_left, v, 1)
                    if abs(jx) > 1e-12:
                        self.add_spin_term(
                            jx, ["S_x", "S_x"], [site_a_idx, site_b_x_idx]
                        )

                v_up = (v - 1 + self.v_cells) % self.v_cells if pbc else v - 1
                if v_up >= 0:
                    site_b_y_idx = self._map_coord_to_index(u, v_up, 1)
                    if abs(jy) > 1e-12:
                        self.add_spin_term(
                            jy, ["S_y", "S_y"], [site_a_idx, site_b_y_idx]
                        )


class BasePauliPhysicalModel2D(BasePauliModel):
    """Base class for models defined on a 2D physical lattice."""

    def __init__(
            self,
            width: int,
            height: int,
            *,
            modes_per_site: int = 1,
            **kwargs: Any,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.physical_L = self.width * self.height
        self.modes_per_site = int(modes_per_site)
        super().__init__(self.modes_per_site * self.physical_L, **kwargs)

    def _map_coord_to_index(self, x: int, y: int) -> int:
        return y * self.width + x

    def _nearest_neighbor_pairs(self, pbc: bool) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for y in range(self.height):
            for x in range(self.width):
                i = self._map_coord_to_index(x, y)
                if x < self.width - 1 or pbc:
                    j = self._map_coord_to_index((x + 1) % self.width, y)
                    pairs.append((i, j))
                if y < self.height - 1 or pbc:
                    j = self._map_coord_to_index(x, (y + 1) % self.height)
                    pairs.append((i, j))
        return pairs


class PauliHeisenbergModel(BasePauliModel):
    """Pauli-backed 1D spin-1/2 Heisenberg model."""

    def __init__(
            self,
            lattice_length: int,
            jx: float = 1.0,
            jy: float = 1.0,
            jz: float = 1.0,
            hx: float = 0.0,
            hy: float = 0.0,
            hz: float = 0.0,
            pbc: bool = True,
    ) -> None:
        super().__init__(lattice_length)
        self.jx, self.jy, self.jz = jx, jy, jz
        self.hx, self.hy, self.hz = hx, hy, hz
        self.pbc = pbc

        num_interactions = self.lattice_length if self.pbc else self.lattice_length - 1
        for i in range(num_interactions):
            j = (i + 1) % self.lattice_length
            if abs(self.jx) > 1e-12:
                self.add_spin_term(self.jx, ["S_x", "S_x"], [i, j])
            if abs(self.jy) > 1e-12:
                self.add_spin_term(self.jy, ["S_y", "S_y"], [i, j])
            if abs(self.jz) > 1e-12:
                self.add_spin_term(self.jz, ["S_z", "S_z"], [i, j])

        for i in range(self.lattice_length):
            if abs(self.hx) > 1e-12:
                self.add_spin_term(-self.hx, ["S_x"], [i])
            if abs(self.hy) > 1e-12:
                self.add_spin_term(-self.hy, ["S_y"], [i])
            if abs(self.hz) > 1e-12:
                self.add_spin_term(-self.hz, ["S_z"], [i])

    def get_symbolic_hamiltonian(self) -> sp.Expr:
        J_x, J_y, J_z = sp.symbols("J_x J_y J_z")
        h_x, h_y, h_z = sp.symbols("h_x h_y h_z")
        S_x = sp.IndexedBase("S^x")
        S_y = sp.IndexedBase("S^y")
        S_z = sp.IndexedBase("S^z")
        i, N = sp.symbols("i N", integer=True, positive=True)
        sum_limit = (i, 1, N) if self.pbc else (i, 1, N - 1)

        interaction_term = (
            J_x * S_x[i] * S_x[i + 1]
            + J_y * S_y[i] * S_y[i + 1]
            + J_z * S_z[i] * S_z[i + 1]
        )
        H_interaction = sp.Sum(interaction_term, sum_limit)
        field_term = h_x * S_x[i] + h_y * S_y[i] + h_z * S_z[i]
        H_field = sp.Sum(field_term, (i, 1, N))
        return H_interaction - H_field


class PauliIsingModel(BasePauliModel):
    """Pauli-backed 1D transverse-field Ising model."""

    def __init__(
            self,
            lattice_length: int,
            j_coupling: float,
            h_field: float,
            pbc: bool = True,
    ) -> None:
        super().__init__(lattice_length)
        self.j_coupling = j_coupling
        self.h_field = h_field
        self.pbc = pbc

        num_interactions = self.lattice_length if self.pbc else self.lattice_length - 1
        for i in range(num_interactions):
            j = (i + 1) % self.lattice_length
            if abs(self.j_coupling) > 1e-12:
                self.add_spin_term(-self.j_coupling, ["S_x", "S_x"], [i, j])
        for i in range(self.lattice_length):
            if abs(self.h_field) > 1e-12:
                self.add_spin_term(-self.h_field, ["S_z"], [i])

    def get_symbolic_hamiltonian(self) -> sp.Expr:
        J, h = sp.symbols("J h")
        S_x = sp.IndexedBase("S^x")
        S_z = sp.IndexedBase("S^z")
        i, N = sp.symbols("i N", integer=True, positive=True)
        sum_limit = (i, 1, N) if self.pbc else (i, 1, N - 1)
        H_interaction = sp.Sum(S_x[i] * S_x[i + 1], sum_limit)
        H_field = sp.Sum(S_z[i], (i, 1, N))
        return -J * H_interaction - h * H_field


class PauliHeisenbergModel2D(BasePauliModel2D):
    """Pauli-backed 2D spin-1/2 Heisenberg model."""

    def __init__(
            self,
            width: int,
            height: int,
            jx: float,
            jy: float,
            jz: float,
            hx: float,
            hy: float,
            hz: float,
            pbc: bool = True,
    ) -> None:
        super().__init__(width, height)
        self.jx, self.jy, self.jz = jx, jy, jz
        self.hx, self.hy, self.hz = hx, hy, hz
        self.pbc = pbc

        couplings = {
            "J_x": (self.jx, ["S_x", "S_x"]),
            "J_y": (self.jy, ["S_y", "S_y"]),
            "J_z": (self.jz, ["S_z", "S_z"]),
        }
        self._add_all_nearest_neighbor_terms(self.pbc, **couplings)

        for i in range(self.lattice_length):
            if abs(self.hx) > 1e-12:
                self.add_spin_term(-self.hx, ["S_x"], [i])
            if abs(self.hy) > 1e-12:
                self.add_spin_term(-self.hy, ["S_y"], [i])
            if abs(self.hz) > 1e-12:
                self.add_spin_term(-self.hz, ["S_z"], [i])

    def get_symbolic_hamiltonian(self) -> sp.Expr:
        J_x, J_y, J_z = sp.symbols("J_x J_y J_z")
        h_x, h_y, h_z = sp.symbols("h_x h_y h_z")
        SumNN = sp.Function("\\sum_{\\langle i,j \\rangle}")
        SumI = sp.Function("\\sum_{i}")
        S_ix, S_iy, S_iz = sp.symbols("S^x_i S^y_i S^z_i")
        S_jx, S_jy, S_jz = sp.symbols("S^x_j S^y_j S^z_j")
        interaction_term = (
            J_x * S_ix * S_jx + J_y * S_iy * S_jy + J_z * S_iz * S_jz
        )
        H_interaction = SumNN(interaction_term)
        field_term = h_x * S_ix + h_y * S_iy + h_z * S_iz
        H_field = SumI(field_term)
        return H_interaction - H_field


class PauliIsingModel2D(BasePauliModel2D):
    """Pauli-backed 2D transverse-field Ising model."""

    def __init__(
            self,
            width: int,
            height: int,
            j_coupling: float,
            h_field: float,
            pbc: bool = True,
    ) -> None:
        super().__init__(width, height)
        self.j_coupling = j_coupling
        self.h_field = h_field
        self.pbc = pbc

        couplings = {
            "J_x": (-self.j_coupling, ["S_x", "S_x"]),
        }
        self._add_all_nearest_neighbor_terms(self.pbc, **couplings)

        for i in range(self.lattice_length):
            if abs(self.h_field) > 1e-12:
                self.add_spin_term(-self.h_field, ["S_z"], [i])

    def get_symbolic_hamiltonian(self) -> sp.Expr:
        J, h = sp.symbols("J h")
        SumNN = sp.Function("\\sum_{\\langle i,j \\rangle}")
        SumI = sp.Function("\\sum_{i}")
        S_ix, S_jx = sp.symbols("S^x_i S^x_j")
        S_iz = sp.symbols("S^z_i")
        H_interaction = SumNN(S_ix * S_jx)
        H_field = SumI(S_iz)
        return -J * H_interaction - h * H_field


class PauliKitaevHoneycombModel(BasePauliHoneycombModel):
    """Pauli-backed spin-1/2 Kitaev honeycomb model."""

    def __init__(
            self,
            u_cells: int,
            v_cells: int,
            jx: float,
            jy: float,
            jz: float,
            pbc: bool = True,
    ) -> None:
        super().__init__(u_cells, v_cells)
        self.jx, self.jy, self.jz = jx, jy, jz
        self.pbc = pbc
        self._build_kitaev_terms(-self.jx, -self.jy, -self.jz, self.pbc)

    def get_symbolic_hamiltonian(self) -> sp.Expr:
        Jx, Jy, Jz = sp.symbols("J_x J_y J_z")
        S_ix, S_iy, S_iz = sp.symbols("S^x_i S^y_i S^z_i")
        S_jx, S_jy, S_jz = sp.symbols("S^x_j S^y_j S^z_j")
        SumX = sp.Function("\\sum_{\\langle i,j \\rangle \\in x}")
        SumY = sp.Function("\\sum_{\\langle i,j \\rangle \\in y}")
        SumZ = sp.Function("\\sum_{\\langle i,j \\rangle \\in z}")

        H_x_term = SumX(S_ix * S_jx)
        H_y_term = SumY(S_iy * S_jy)
        H_z_term = SumZ(S_iz * S_jz)
        return sp.Add(
            -Jx * H_x_term,
            -Jy * H_y_term,
            -Jz * H_z_term,
            evaluate=False,
        )


class PauliKitaevChain(BasePauliModel):
    """Pauli-backed spinless Kitaev chain."""

    def __init__(
            self,
            lattice_length: int,
            chemical_potential: float,
            hopping: float,
            pairing_gap: float,
            pbc: bool = False,
    ) -> None:
        super().__init__(lattice_length)
        self.mu = chemical_potential
        self.t = hopping
        self.delta = pairing_gap
        self.pbc = pbc

        identity_label = "I" * self.lattice_length
        num_hoppings = self.lattice_length if self.pbc else self.lattice_length - 1
        for j in range(self.lattice_length):
            self.add_fermion_term(-self.mu, ["c_dag", "c"], [j, j])
            self.add_pauli_string(identity_label, 0.5 * self.mu)

        for j in range(num_hoppings):
            k = (j + 1) % self.lattice_length
            self.add_fermion_term(-self.t, ["c_dag", "c"], [j, k])
            self.add_fermion_term(-self.t, ["c_dag", "c"], [k, j])
            self.add_fermion_term(self.delta, ["c", "c"], [j, k])
            self.add_fermion_term(self.delta, ["c_dag", "c_dag"], [k, j])

    def get_symbolic_hamiltonian(self) -> sp.Expr:
        mu, t, Delta = sp.symbols("mu t Delta")
        c_dag = sp.IndexedBase("c^\\dagger", commutative=False)
        c = sp.IndexedBase("c", commutative=False)
        j, N = sp.symbols("j N", integer=True, positive=True)
        sum_limit = (j, 1, N) if self.pbc else (j, 1, N - 1)

        h_mu = sp.Sum(-mu * (c_dag[j] * c[j] - sp.Rational(1, 2)), (j, 1, N))
        hop_pair_term = (
            -t * (c_dag[j] * c[j + 1] + c_dag[j + 1] * c[j])
            + Delta * (c[j] * c[j + 1] + c_dag[j + 1] * c_dag[j])
        )
        h_hop_pair = sp.Sum(hop_pair_term, sum_limit)
        return h_mu + h_hop_pair


class PauliSSHModel(BasePauliModel):
    """Pauli-backed 1D SSH model."""

    def __init__(
            self,
            num_cells: int,
            intra_cell_hopping: float,
            inter_cell_hopping: float,
            pbc: bool = False,
    ) -> None:
        self.num_cells = int(num_cells)
        self.v = intra_cell_hopping
        self.w = inter_cell_hopping
        self.pbc = pbc

        total_sites = 2 * self.num_cells
        super().__init__(lattice_length=total_sites)

        if abs(self.v) > 1e-12:
            for i in range(self.num_cells):
                site_a = 2 * i
                site_b = 2 * i + 1
                self.add_fermion_term(-self.v, ["c_dag", "c"], [site_a, site_b])
                self.add_fermion_term(-self.v, ["c_dag", "c"], [site_b, site_a])

        if abs(self.w) > 1e-12:
            num_inter_links = self.num_cells if self.pbc else self.num_cells - 1
            for i in range(num_inter_links):
                source_site = 2 * i + 1
                target_site = (2 * (i + 1)) % total_sites
                self.add_fermion_term(
                    -self.w, ["c_dag", "c"], [source_site, target_site]
                )
                self.add_fermion_term(
                    -self.w, ["c_dag", "c"], [target_site, source_site]
                )

    def build_single_particle_hamiltonian(self) -> torch.Tensor:
        """Builds the SSH single-particle tight-binding matrix."""
        lattice_length = 2 * self.num_cells
        h_sp = torch.zeros((lattice_length, lattice_length), dtype=torch.complex64)

        if abs(self.v) > 1e-12:
            for i in range(self.num_cells):
                site_a = 2 * i
                site_b = 2 * i + 1
                h_sp[site_a, site_b] = -self.v
                h_sp[site_b, site_a] = -self.v

        if abs(self.w) > 1e-12:
            num_inter_links = self.num_cells if self.pbc else self.num_cells - 1
            for i in range(num_inter_links):
                source_site = 2 * i + 1
                target_site = (2 * (i + 1)) % lattice_length
                h_sp[source_site, target_site] = -self.w
                h_sp[target_site, source_site] = -self.w

        return h_sp

    def get_symbolic_hamiltonian(self) -> sp.Expr:
        v, w = sp.symbols("v w")
        i = sp.symbols("i", integer=True, positive=True)
        c_dag_a = sp.IndexedBase("c^\\dagger_A", commutative=False)
        c_b = sp.IndexedBase("c_B", commutative=False)
        c_dag_b = sp.IndexedBase("c^\\dagger_B", commutative=False)
        c_a = sp.IndexedBase("c_A", commutative=False)
        sum_i = sp.Function("\\sum_{i}")

        h_intra = sum_i(c_dag_a[i] * c_b[i] + c_dag_b[i] * c_a[i])
        h_inter = sum_i(c_dag_b[i] * c_a[i + 1] + c_dag_a[i + 1] * c_b[i])
        return sp.Add(-v * h_intra, -w * h_inter, evaluate=False)


class PauliHubbardModel1D(BasePauliModel):
    """Pauli-backed 1D spinful Hubbard model."""

    def __init__(
            self,
            lattice_length: int,
            hopping: float,
            interaction: float,
            chemical_potential: float,
            pbc: bool = True,
    ) -> None:
        self.physical_L = int(lattice_length)
        self.t = hopping
        self.u = interaction
        self.mu = chemical_potential
        self.pbc = pbc

        super().__init__(lattice_length=2 * self.physical_L)

        num_hoppings = self.physical_L if self.pbc else self.physical_L - 1
        for i in range(num_hoppings):
            j = (i + 1) % self.physical_L
            for spin in ["up", "down"]:
                self.add_spinful_fermion_term(
                    -self.t,
                    ["c_dag", "c"],
                    [i, j],
                    [spin, spin],
                    physical_lattice_length=self.physical_L,
                )
                self.add_spinful_fermion_term(
                    -self.t,
                    ["c_dag", "c"],
                    [j, i],
                    [spin, spin],
                    physical_lattice_length=self.physical_L,
                )

        for i in range(self.physical_L):
            if abs(self.u) > 1e-12:
                self.add_spinful_fermion_term(
                    self.u,
                    ["c_dag", "c", "c_dag", "c"],
                    [i, i, i, i],
                    ["up", "up", "down", "down"],
                    physical_lattice_length=self.physical_L,
                )
            if abs(self.mu) > 1e-12:
                self.add_spinful_fermion_term(
                    -self.mu,
                    ["c_dag", "c"],
                    [i, i],
                    ["up", "up"],
                    physical_lattice_length=self.physical_L,
                )
                self.add_spinful_fermion_term(
                    -self.mu,
                    ["c_dag", "c"],
                    [i, i],
                    ["down", "down"],
                    physical_lattice_length=self.physical_L,
                )

    def get_symbolic_hamiltonian(self) -> sp.Expr:
        t, U, mu = sp.symbols("t U mu")
        sum_i_sigma = sp.Function("\\sum_{i, \\sigma}")
        sum_i = sp.Function("\\sum_{i}")
        sigma = sp.Symbol("\\sigma")
        i = sp.Symbol("i")
        c_dag = sp.IndexedBase("c^\\dagger", commutative=False)
        c = sp.IndexedBase("c", commutative=False)
        n = sp.IndexedBase("n")
        up, down = sp.symbols("\\uparrow \\downarrow")

        hopping_expr = (
            c_dag[i, sigma] * c[i + 1, sigma]
            + c_dag[i + 1, sigma] * c[i, sigma]
        )
        h_hopping = sum_i_sigma(hopping_expr)
        h_interaction = sum_i(n[i, up] * n[i, down])
        h_potential = sum_i_sigma(n[i, sigma])

        return sp.Add(
            -t * h_hopping,
            U * h_interaction,
            -mu * h_potential,
            evaluate=False,
        )


class PauliTJModel2D(BasePauliPhysicalModel2D):
    """Pauli-backed 2D t-J model."""

    def __init__(
            self,
            width: int,
            height: int,
            hopping: float,
            exchange: float,
            pbc: bool = True,
    ) -> None:
        super().__init__(width, height, modes_per_site=2)
        self.t = hopping
        self.j_exch = exchange
        self.pbc = pbc

        for i, j in self._nearest_neighbor_pairs(self.pbc):
            if abs(self.j_exch) > 1e-12:
                self.add_spinful_fermion_term(
                    0.25 * self.j_exch,
                    ["n", "n"],
                    [i, j],
                    ["up", "up"],
                    physical_lattice_length=self.physical_L,
                )
                self.add_spinful_fermion_term(
                    -0.25 * self.j_exch,
                    ["n", "n"],
                    [i, j],
                    ["up", "dn"],
                    physical_lattice_length=self.physical_L,
                )
                self.add_spinful_fermion_term(
                    -0.25 * self.j_exch,
                    ["n", "n"],
                    [i, j],
                    ["dn", "up"],
                    physical_lattice_length=self.physical_L,
                )
                self.add_spinful_fermion_term(
                    0.25 * self.j_exch,
                    ["n", "n"],
                    [i, j],
                    ["dn", "dn"],
                    physical_lattice_length=self.physical_L,
                )

                self.add_spinful_fermion_term(
                    0.5 * self.j_exch,
                    ["c_dag", "c", "c_dag", "c"],
                    [i, i, j, j],
                    ["up", "dn", "dn", "up"],
                    physical_lattice_length=self.physical_L,
                )
                self.add_spinful_fermion_term(
                    0.5 * self.j_exch,
                    ["c_dag", "c", "c_dag", "c"],
                    [i, i, j, j],
                    ["dn", "up", "up", "dn"],
                    physical_lattice_length=self.physical_L,
                )

                self.add_spinful_fermion_term(
                    -0.25 * self.j_exch,
                    ["n", "n"],
                    [i, j],
                    ["up", "up"],
                    physical_lattice_length=self.physical_L,
                )
                self.add_spinful_fermion_term(
                    -0.25 * self.j_exch,
                    ["n", "n"],
                    [i, j],
                    ["up", "dn"],
                    physical_lattice_length=self.physical_L,
                )
                self.add_spinful_fermion_term(
                    -0.25 * self.j_exch,
                    ["n", "n"],
                    [i, j],
                    ["dn", "up"],
                    physical_lattice_length=self.physical_L,
                )
                self.add_spinful_fermion_term(
                    -0.25 * self.j_exch,
                    ["n", "n"],
                    [i, j],
                    ["dn", "dn"],
                    physical_lattice_length=self.physical_L,
                )

            if abs(self.t) > 1e-12:
                for spin, opp_spin in [("up", "dn"), ("dn", "up")]:
                    self.add_spinful_fermion_term(
                        -self.t,
                        ["c_dag", "c"],
                        [i, j],
                        [spin, spin],
                        physical_lattice_length=self.physical_L,
                    )
                    self.add_spinful_fermion_term(
                        self.t,
                        ["c_dag", "n", "c"],
                        [i, i, j],
                        [spin, opp_spin, spin],
                        physical_lattice_length=self.physical_L,
                    )
                    self.add_spinful_fermion_term(
                        -self.t,
                        ["c_dag", "c"],
                        [j, i],
                        [spin, spin],
                        physical_lattice_length=self.physical_L,
                    )
                    self.add_spinful_fermion_term(
                        self.t,
                        ["c_dag", "n", "c"],
                        [j, j, i],
                        [spin, opp_spin, spin],
                        physical_lattice_length=self.physical_L,
                    )

    def get_symbolic_hamiltonian(self) -> sp.Expr:
        t, J, sigma = sp.symbols("t J sigma")
        i, j = sp.symbols("i j")
        sum_nn = sp.Function("\\sum_{\\langle i,j \\rangle}")

        c_dag_s = sp.IndexedBase("\\tilde{c}^\\dagger", commutative=False)
        c_s = sp.IndexedBase("\\tilde{c}", commutative=False)
        S_i = sp.Symbol("\\mathbf{S}_i")
        S_j = sp.Symbol("\\mathbf{S}_j")
        n_i, n_j = sp.symbols("n_i n_j")

        h_hopping = sum_nn(c_dag_s[i, sigma] * c_s[j, sigma] + sp.Symbol("h.c."))
        h_exchange = sum_nn(S_i * S_j - sp.Rational(1, 4) * n_i * n_j)

        return sp.Add(-t * h_hopping, J * h_exchange, evaluate=False)


class PauliBoseHubbardModel2D(BasePauliPhysicalModel2D):
    """Explicit Phase-3 placeholder: general Bose-Hubbard is not supported."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "PauliBoseHubbardModel2D is not supported in the current qubit "
            "Pauli backend. General BoseHubbardModel requires a non-qubit "
            "local Hilbert space and is intentionally left for Phase 3."
        )
