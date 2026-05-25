"""Hamiltonian builders backed by Pauli-string algebra.

This module mirrors the role of ``quantum_simulation.hamiltonian`` for
spin-1/2 / qubit systems, but stores Hamiltonians directly as ``PauliSum``
objects rather than dense matrices.
"""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Sequence, Union

import numpy as np
import torch

from .pauli_class import (
    PauliString,
    PauliSum,
    SparseKet,
    adjoint_generator_in_lie_closure_basis,
    coefficients_in_lie_closure_basis,
    evolve_operator_in_lie_closure_basis,
    global_pauli_op_chain,
    heisenberg_expectation_for_density_matrix_in_lie_closure_basis,
    heisenberg_expectation_in_lie_closure_basis,
    lie_closure_basis_symbolic,
)
from .jordan_wigner import (
    jw_spinless_fermion_product,
    jw_spinful_fermion_product,
)


LocalExpansion = List[tuple[str, complex]]


class PauliHamiltonian:
    """Accumulates a qubit Hamiltonian as a symbolic Pauli sum.

    Site labels follow the same convention as ``PauliString``:
    ``label[j]`` acts on site ``j``, and site 0 corresponds to the least-
    significant qubit in bitmask-based routines.
    """

    _LOCAL_SPIN_OPERATORS: Dict[str, LocalExpansion] = {
        "I": [("I", 1.0 + 0.0j)],
        "X": [("X", 1.0 + 0.0j)],
        "Y": [("Y", 1.0 + 0.0j)],
        "Z": [("Z", 1.0 + 0.0j)],
        "S_x": [("X", 0.5 + 0.0j)],
        "S_y": [("Y", 0.5 + 0.0j)],
        "S_z": [("Z", 0.5 + 0.0j)],
        "S_p": [("X", 0.5 + 0.0j), ("Y", 0.0 + 0.5j)],
        "S_m": [("X", 0.5 + 0.0j), ("Y", 0.0 - 0.5j)],
        "n": [("I", 0.5 + 0.0j), ("Z", -0.5 + 0.0j)],
    }

    def __init__(self, lattice_length: int, *, convention: str = "down=1"):
        if lattice_length < 1:
            raise ValueError("lattice_length must be >= 1.")
        if convention != "down=1":
            raise NotImplementedError(
                "PauliHamiltonian currently supports only convention='down=1'."
            )

        self.lattice_length = int(lattice_length)
        self.convention = convention
        self._terms: DefaultDict[str, complex] = defaultdict(complex)

    def _validate_site(self, site: int) -> None:
        if not (0 <= site < self.lattice_length):
            raise IndexError(
                f"Site index {site} out of bounds for length {self.lattice_length}."
            )

    def _add_label(self, label: str, coefficient: complex) -> None:
        if len(label) != self.lattice_length:
            raise ValueError(
                f"Label length {len(label)} does not match lattice_length "
                f"{self.lattice_length}."
            )
        if abs(coefficient) == 0:
            return
        self._terms[label] += complex(coefficient)

    @classmethod
    def _local_spin_operator(cls, name: str) -> LocalExpansion:
        if name not in cls._LOCAL_SPIN_OPERATORS:
            raise ValueError(
                f"Unsupported local spin operator '{name}'. "
                f"Supported operators: {sorted(cls._LOCAL_SPIN_OPERATORS)}"
            )
        return cls._LOCAL_SPIN_OPERATORS[name]

    def add_pauli_string(self, label: str, coefficient: complex = 1.0) -> None:
        """Adds a full-length Pauli string term."""
        self._add_label(label.upper(), coefficient)

    def add_pauli_op_chain(
            self,
            op_type: str,
            coefficient: complex = 1.0,
            *,
            pbc: bool = True,
            spin: float = 0.5,
    ) -> None:
        """Adds a translated Pauli chain sum to the Hamiltonian.

        This mirrors ``quantum_simulation.operator.global_pauli_op_chain``,
        but accumulates the result symbolically as a ``PauliSum``.
        """
        self.add_term(
            global_pauli_op_chain(
                op_type,
                self.lattice_length,
                coefficient=coefficient,
                spin=spin,
                pbc=pbc,
            )
        )

    def add_pauli_term(
            self,
            coefficient: complex,
            operators: Sequence[str],
            sites: Sequence[int],
    ) -> None:
        """Adds a product of explicit Pauli operators on selected sites."""
        if len(operators) != len(sites):
            raise ValueError(
                "operators and sites must have the same length, got "
                f"{len(operators)} and {len(sites)}."
            )
        if len(set(sites)) != len(sites):
            raise ValueError(
                "Repeated sites are not supported by add_pauli_term; "
                "multiply local operators analytically before adding the term."
            )

        label = ["I"] * self.lattice_length
        for op_name, site in zip(operators, sites):
            self._validate_site(site)
            op = op_name.upper()
            if op not in {"I", "X", "Y", "Z"}:
                raise ValueError(
                    f"Unsupported Pauli operator '{op_name}'. "
                    "Expected one of I, X, Y, Z."
                )
            label[site] = op
        self._add_label("".join(label), coefficient)

    def add_spin_term(
            self,
            coefficient: complex,
            operators: Sequence[str],
            sites: Sequence[int],
    ) -> None:
        """Adds a product of local spin-1/2 operators."""
        if len(operators) != len(sites):
            raise ValueError(
                "operators and sites must have the same length, got "
                f"{len(operators)} and {len(sites)}."
            )
        if len(set(sites)) != len(sites):
            raise ValueError(
                "Repeated sites are not supported by add_spin_term; "
                "multiply local operators analytically before adding the term."
            )

        terms: List[tuple[List[str], complex]] = [
            (["I"] * self.lattice_length, complex(coefficient))
        ]
        for op_name, site in zip(operators, sites):
            self._validate_site(site)
            local_expansion = self._local_spin_operator(op_name)
            next_terms: List[tuple[List[str], complex]] = []
            for current_label, current_coeff in terms:
                for pauli_char, local_coeff in local_expansion:
                    label_copy = current_label.copy()
                    label_copy[site] = pauli_char
                    next_terms.append(
                        (label_copy, current_coeff * local_coeff)
                    )
            terms = next_terms

        for label_chars, total_coeff in terms:
            self._add_label("".join(label_chars), total_coeff)

    def add_fermion_term(
            self,
            coefficient: complex,
            operators: Sequence[str],
            sites: Sequence[int],
            *,
            tol: float = 1e-12,
    ) -> None:
        """Adds an ordered spinless fermion monomial via Jordan-Wigner."""
        fermion_term = jw_spinless_fermion_product(
            list(operators),
            list(sites),
            self.lattice_length,
            convention=self.convention,
            tol=tol,
        )
        self.add_term(complex(coefficient) * fermion_term)

    def add_spinful_fermion_term(
            self,
            coefficient: complex,
            operators: Sequence[str],
            sites: Sequence[int],
            spins: Sequence[str],
            *,
            physical_lattice_length: int | None = None,
            tol: float = 1e-12,
    ) -> None:
        """Adds an ordered spinful fermion monomial via Jordan-Wigner."""
        if physical_lattice_length is None:
            if self.lattice_length % 2 != 0:
                raise ValueError(
                    "physical_lattice_length must be specified when the "
                    "Jordan-Wigner chain length is odd."
                )
            physical_lattice_length = self.lattice_length // 2

        fermion_term = jw_spinful_fermion_product(
            list(operators),
            list(sites),
            list(spins),
            int(physical_lattice_length),
            convention=self.convention,
            tol=tol,
        )
        self.add_term(complex(coefficient) * fermion_term)

    def add_term(self, term: Union[PauliString, PauliSum]) -> None:
        """Adds an already constructed PauliString or PauliSum."""
        if isinstance(term, PauliString):
            self._add_label(term.label, term.coeff)
            return
        if not isinstance(term, PauliSum):
            raise TypeError(
                f"term must be a PauliString or PauliSum, got {type(term)}"
            )
        for subterm in term.terms:
            self._add_label(subterm.label, subterm.coeff)

    def build(self, tol: float = 1e-12) -> PauliSum:
        """Builds the Hamiltonian as a simplified PauliSum."""
        terms = [
            PauliString(label, coeff)
            for label, coeff in self._terms.items()
            if abs(coeff) > tol
        ]
        return PauliSum(terms).simplify(tol=tol)

    def to_matrix(self, tol: float = 1e-12) -> np.ndarray:
        """Builds the Hamiltonian and converts it to a dense matrix."""
        return self.build(tol=tol).to_matrix()

    def num_terms(self, tol: float = 1e-12) -> int:
        """Returns the number of nonzero Pauli terms currently stored."""
        return sum(1 for coeff in self._terms.values() if abs(coeff) > tol)

    @staticmethod
    def _standardize_generators(
            operators: Iterable[Union[PauliString, PauliSum]]
    ) -> List[PauliSum]:
        generators: List[PauliSum] = []
        for operator in operators:
            if isinstance(operator, PauliString):
                generators.append(PauliSum([operator]))
            elif isinstance(operator, PauliSum):
                generators.append(operator)
            else:
                raise TypeError(
                    "Operators must be PauliString or PauliSum, got "
                    f"{type(operator)}"
                )
        return generators

    def lie_closure_basis(
            self,
            observables: Iterable[Union[PauliString, PauliSum]] = (),
            *,
            atol: float = 1e-6,
            rtol: float = 1e-5,
            max_iter: int = 100,
    ) -> List[PauliSum]:
        """Builds the Lie-closure basis of H together with selected observables."""
        generators = [self.build()] + self._standardize_generators(observables)
        return lie_closure_basis_symbolic(
            generators,
            atol=atol,
            rtol=rtol,
            max_iter=max_iter,
        )

    def adjoint_generator(
            self,
            basis: List[PauliSum],
            *,
            time: float = 1.0,
            hbar: float = 1.0,
            device: str = "cpu",
    ) -> torch.Tensor:
        """Returns the adjoint-action matrix of (i t / hbar) H on a basis."""
        scaled_hamiltonian = (1j * time / hbar) * self.build()
        return adjoint_generator_in_lie_closure_basis(
            basis, scaled_hamiltonian, device=device
        )

    def evolve_operator(
            self,
            basis: List[PauliSum],
            operator: Union[PauliString, PauliSum],
            *,
            time: float,
            hbar: float = 1.0,
            device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evolves an observable using the adjoint-matrix route."""
        return evolve_operator_in_lie_closure_basis(
            basis,
            self.build(),
            operator,
            time=time,
            hbar=hbar,
            device=device,
        )

    def operator_coefficients(
            self,
            basis: List[PauliSum],
            operator: Union[PauliString, PauliSum],
            *,
            device: str = "cpu",
    ) -> torch.Tensor:
        """Projects an operator onto a Lie-closure basis."""
        return coefficients_in_lie_closure_basis(
            basis, operator, device=device
        )

    def expectation(
            self,
            basis: List[PauliSum],
            operator: Union[PauliString, PauliSum],
            state: SparseKet,
            *,
            time: float,
            hbar: float = 1.0,
            device: str = "cpu",
            op_tol: float = 1e-12,
    ) -> tuple[complex, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes <psi|O(t)|psi> using only the adjoint-matrix route."""
        return heisenberg_expectation_in_lie_closure_basis(
            basis,
            self.build(),
            operator,
            state,
            time=time,
            hbar=hbar,
            device=device,
            op_tol=op_tol,
        )

    def expectation_density(
            self,
            basis: List[PauliSum],
            operator: Union[PauliString, PauliSum],
            density_matrix: Union[np.ndarray, torch.Tensor],
            *,
            time: float,
            hbar: float = 1.0,
            device: str = "cpu",
            op_tol: float = 1e-12,
    ) -> tuple[complex, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes Tr(O(t) rho) using only the adjoint-matrix route."""
        return heisenberg_expectation_for_density_matrix_in_lie_closure_basis(
            basis,
            self.build(),
            operator,
            density_matrix,
            time=time,
            hbar=hbar,
            device=device,
            op_tol=op_tol,
        )
