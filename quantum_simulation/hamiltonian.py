import torch
from typing import List, Callable, Union
from abc import ABC, abstractmethod
from . import operator as op
from .operation import nested_kronecker_product


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

