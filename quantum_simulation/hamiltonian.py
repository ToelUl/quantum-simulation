
import torch
import numpy as np
from typing import List, Callable, Union
from abc import ABC, abstractmethod
from . import operator as op
from .operation import nested_kronecker_product
from .domain import generate_k_space


class Term(ABC):
    """Abstract base class for a term in the Hamiltonian."""

    def __init__(self, coefficient: float):
        self.coefficient = coefficient

    @abstractmethod
    def build_matrix(self, lattice_length: int, **kwargs) -> torch.Tensor:
        """
        Constructs the matrix for this term in the full Hilbert space.

        This method must be implemented by subclasses.
        """
        pass


class SpinBosonTerm(Term):
    """
    A term constructed via Kronecker product of local operators.
    Suitable for spins and hard-core bosons.
    """

    def __init__(self, coefficient: float, operators: List[Callable], sites: List[int]):
        super().__init__(coefficient)
        self.operators = operators
        self.sites = sites

    def build_matrix(self, lattice_length: int, **kwargs) -> torch.Tensor:
        spin = kwargs.get("spin", 0.5)
        identity_op = op.identity(spin)

        op_list = [identity_op for _ in range(lattice_length)]
        for op_func, site_idx in zip(self.operators, self.sites):
            if not (0 <= site_idx < lattice_length):
                raise IndexError(f"Site index {site_idx} out of bounds.")
            op_list[site_idx] = op_func(spin=spin)

        term_matrix = nested_kronecker_product(op_list)
        return self.coefficient * term_matrix


class FermionTerm(Term):
    """
    A term constructed via matrix multiplication of global Jordan-Wigner operators.
    Suitable for fermionic systems.
    """

    def __init__(self, coefficient: float, operators: List[Callable], sites: List[int]):
        super().__init__(coefficient)
        self.operators = operators
        self.sites = sites

    def build_matrix(self, lattice_length: int, **kwargs) -> torch.Tensor:
        convention = kwargs.get("convention", "down=1")

        # Build the first operator matrix
        op_func, site_idx = self.operators[0], self.sites[0]
        term_matrix = op_func(site_idx, lattice_length, convention)

        # Multiply by subsequent operator matrices
        for i in range(1, len(self.operators)):
            op_func, site_idx = self.operators[i], self.sites[i]
            next_op_matrix = op_func(site_idx, lattice_length, convention)
            term_matrix = term_matrix @ next_op_matrix

        return self.coefficient * term_matrix


# --- NEW: A Term class for pre-built matrices ---
class PrebuiltTerm(Term):
    """
    A term that holds an already constructed matrix.
    The coefficient is assumed to be included in the matrix.
    """

    def __init__(self, matrix: torch.Tensor):
        # The coefficient is already part of the matrix, so we use 1.0
        super().__init__(coefficient=1.0)
        self.matrix = matrix

    def build_matrix(self, lattice_length: int, **kwargs) -> torch.Tensor:
        # This is simple: just return the stored matrix.
        return self.matrix


# -------------------------------------------------

class Hamiltonian:
    """
    A unified, flexible builder for constructing Hamiltonian matrices for various
    1D lattice models by composing different types of terms.
    """

    def __init__(self, lattice_length: int, **kwargs):
        """
        Initializes the Hamiltonian constructor.

        Args:
            lattice_length (int): The number of sites in the system.
            **kwargs: Additional system-wide parameters like 'spin' or 'convention'.
        """
        self.lattice_length = lattice_length
        self.system_params = kwargs
        self.terms: List[Term] = []

        # Calculate total dimension based on spin (default to 0.5 for spin-1/2)
        spin = self.system_params.get("spin", 0.5)
        dim = int(2 * spin + 1)
        self.total_dim = dim ** self.lattice_length

    def add_spin_term(self, coefficient: float, operators: List[Callable], sites: List[int]):
        """Adds a spin-like (or bosonic) term."""
        term = SpinBosonTerm(coefficient, operators, sites)
        self.terms.append(term)

    def add_fermion_term(self, coefficient: float, operators: List[Callable], sites: List[int]):
        """Adds a fermionic term."""
        term = FermionTerm(coefficient, operators, sites)
        self.terms.append(term)

    def add_prebuilt_term(self, term_matrix: torch.Tensor):
        """
        Adds a term that is already in its full matrix form.

        Args:
            term_matrix (torch.Tensor): A matrix of shape (total_dim, total_dim)
                                        representing the term. The coefficient should
                                        already be multiplied into this matrix.
        """
        if term_matrix.shape != (self.total_dim, self.total_dim):
            raise ValueError(
                f"Shape of pre-built matrix {term_matrix.shape} does not match "
                f"the required shape ({self.total_dim}, {self.total_dim})."
            )

        term = PrebuiltTerm(term_matrix)
        self.terms.append(term)

    # ----------------------------------------------

    def build(self) -> torch.Tensor:
        """
        Constructs the full Hamiltonian by summing the matrices of all added terms.
        """
        H = torch.zeros((self.total_dim, self.total_dim), dtype=torch.complex64)
        for term in self.terms:
            # The magic of polymorphism: we don't care what kind of term it is,
            # we just tell it to build itself! ✨
            H += term.build_matrix(self.lattice_length, **self.system_params)
        return H


def bogoliubov_pseudospin_spectrum(coeff_x: float = 1.0,
                                   coeff_y: float = 1.0,
                                   coeff_z: float = 1.0) -> tuple[float, float]:
    """Calculates the Bogoliubov excitation spectrum from pseudo-spin coefficients.

    This function models the energy spectrum of a system described by a
    Bogoliubov-de Gennes (BdG) Hamiltonian in a pseudo-spin representation.
    The positive and negative branches of the spectrum correspond to the
    excitation energies of quasiparticles and quasiholes, respectively.

    This is a common formulation in condensed matter physics for systems such as
    superconductors or topological materials.

    Args:
        coeff_x: The coefficient for the x-component of the pseudo-spin.
            This could physically correspond to a Zeeman field component, a
            superconducting pairing term, etc.
        coeff_y: The coefficient for the y-component of the pseudo-spin.
        coeff_z: The coefficient for the z-component of the pseudo-spin.

    Returns:
        A tuple containing the positive and negative branches of the energy
        spectrum (E, -E), representing the quasiparticle excitation energies.
    """
    # In the BdG formalism, the calculated energy E represents the
    # quasiparticle energy.
    quasiparticle_energy = np.sqrt(coeff_x ** 2 + coeff_y ** 2 + coeff_z ** 2)

    # The Bogoliubov spectrum is symmetric, containing both positive
    # (quasiparticle) and negative (quasihole) energy branches.
    return quasiparticle_energy, -quasiparticle_energy


def xy_chain_bogoliubov_spectrum(k: Union[float, np.ndarray],
                                jx: float = 1.0,
                                jy: float = 0.0,
                                h: float = 1.0,
                                phi: float = 0.0) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Calculates the Bogoliubov spectrum for the quantum XY chain.

    This function translates the physical parameters of the anisotropic quantum
    XY chain in a transverse field into a set of effective pseudo-spin
    coefficients. It then computes the quasiparticle energy spectrum based on
    these coefficients.

    Args:
        k: The wave vector (momentum). Can be a single float or a NumPy array
           to compute the dispersion relation.
        jx: The exchange coupling strength along the x-axis.
        jy: The exchange coupling strength along the y-axis.
        h: The strength of the transverse magnetic field.
        phi: The anisotropy phase angle, which can introduce DMI-like terms.

    Returns:
        A tuple containing the positive (quasiparticle) and negative
        (quasihole) branches of the energy spectrum for the given wave vector(s).
    """
    # Avoid division by zero if both couplings are zero.
    if jx == 0 and jy == 0:
        # In this case, the spectrum is simply from the transverse field.
        # The pseudo-spin coefficients would be (0, 0, 2h).
        quasiparticle_energy = 2 * h
        return quasiparticle_energy, -quasiparticle_energy

    # Sum of exchange couplings.
    j_sum = jx + jy

    # Dimensionless anisotropy parameter, chi.
    anisotropy_param = (jx - jy) / j_sum

    # Map the XY chain parameters onto the pseudo-spin coefficients (x_k, y_k, z_k).
    # These coefficients represent the effective field acting on the pseudo-spin.
    pseudospin_coeff_x = -2 * anisotropy_param * j_sum * np.sin(2 * phi) * np.sin(k)
    pseudospin_coeff_y = 2 * anisotropy_param * j_sum * np.cos(2 * phi) * np.sin(k)
    pseudospin_coeff_z = 2 * (h - j_sum * np.cos(k))

    # Calculate the final spectrum using the core Bogoliubov function.
    return bogoliubov_pseudospin_spectrum(
        coeff_x=pseudospin_coeff_x,
        coeff_y=pseudospin_coeff_y,
        coeff_z=pseudospin_coeff_z
    )


def xy_chain_ground_energy(lattice_length: int = 4,
                                           jx: float = 1.0,
                                           jy: float = 0.0,
                                           h: float = 1.0,
                                           phi: float = 0.0,
                                           use_abc: bool = True) -> float:
    """Calculates the ground state energy of the quantum XY chain.

    This function computes the ground state energy by summing the Bogoliubov
    quasiparticle excitation energies .

    The boundary conditions (PBC vs. ABC) determine the set of allowed wave
    vectors (k) and may introduce boundary correction terms to the energy.

    Args:
        lattice_length: The number of sites in the 1D lattice (L).
        jx: The exchange coupling strength along the x-axis.
        jy: The exchange coupling strength along the y-axis.
        h: The strength of the transverse magnetic field.
        phi: The anisotropy phase angle.
        use_abc: If False, uses Periodic Boundary Conditions (PBC). If True,
            uses Anti-periodic Boundary Conditions (ABC) and applies a
            boundary correction term.

    Returns:
        The calculated ground state energy of the model.
    """
    # 1. Generate the appropriate k-space based on boundary conditions.
    k_space = generate_k_space(lattice_length, use_abc, positive_only=True)

    # 2. Calculate the spectrum for all k-points in a single vectorized call.
    positive_energies, _ = xy_chain_bogoliubov_spectrum(
        k=k_space, jx=jx, jy=jy, h=h, phi=phi
    )

    # 3. Sum the quasiparticle energies .
    ground_energy = -np.sum(positive_energies)

    # 4. Apply boundary corrections if necessary.
    if not use_abc:
        j_sum = jx + jy
        ground_energy -= 2 * j_sum

    return ground_energy

