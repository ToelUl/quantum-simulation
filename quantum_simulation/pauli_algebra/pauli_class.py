"""Defines classes for Pauli string algebra, optimized for quantum computing.

This module provides two main classes:
- PauliString: Represents a single Pauli string (e.g., 1.5*XZYI) with a
               symplectic (X/Z) representation for efficient algebra.
- PauliSum: Represents a linear combination of PauliString objects (e.g.,
            a Hamiltonian), with support for simplification and commutator
            calculations.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Union, Tuple

import numpy as np
import torch

# Module-level cache for 2x2 Pauli matrices
_PAULI_DICT: Dict[str, np.ndarray] = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


# ============================================================
# PauliString: A single Pauli string object
# ============================================================

class PauliString:
    """An efficient Pauli spin string object using a symplectic representation.

    Supports operator algebra (multiplication, addition, commutation) and
    conversion to matrix form.

    Attributes:
        label (str): The string representation (e.g., 'XYZI').
        coeff (complex): The complex coefficient.
        n (int): The number of qubits (length of the string).
        x (np.ndarray): The binary X-part of the symplectic representation
            (uint8 array of length n).
        z (np.ndarray): The binary Z-part of the symplectic representation
            (uint8 array of length n).
    """

    PAULI_TO_XZ: Dict[str, Tuple[int, int]] = {
        'I': (0, 0), 'X': (1, 0), 'Z': (0, 1), 'Y': (1, 1)
    }
    XZ_TO_PAULI: Dict[Tuple[int, int], str] = {
        (0, 0): 'I', (1, 0): 'X', (0, 1): 'Z', (1, 1): 'Y'
    }
    __slots__ = ("label", "coeff", "n", "x", "z", "_x_mask", "_z_mask",
                 "_x_dot_z_mod4")

    def __init__(self, label: str, coeff: complex = 1.0):
        """Initializes the PauliString.

        Args:
            label: The string of Pauli operators (e.g., 'IXYZ').
            coeff: The complex coefficient multiplying the string.

        Raises:
            ValueError: If an invalid character (not 'I', 'X', 'Y', 'Z')
                is found in the label.
        """
        self.label = label.upper()
        self.coeff = complex(coeff)
        self.n = len(self.label)

        if not all(c in self.PAULI_TO_XZ for c in self.label):
            raise ValueError(f"Invalid Pauli character in label: {self.label}")

        self.x = np.array(
            [self.PAULI_TO_XZ[c][0] for c in self.label], dtype=np.uint8
        )
        self.z = np.array(
            [self.PAULI_TO_XZ[c][1] for c in self.label], dtype=np.uint8
        )

        # --- Caches for bitmask representation ---
        self._x_mask: Optional[int] = None
        self._z_mask: Optional[int] = None
        self._x_dot_z_mod4: Optional[int] = None

    def __repr__(self, eps: float = 1e-6) -> str:
        """Returns a human-readable string representation.

        Args:
            eps: Tolerance for snapping the coefficient to 1, -1, 1j, or -1j
                for cleaner printing.

        Returns:
            A string like "(1.5+0j)*XYZI" or "XYZI" if coeff is 1.
        """
        c = self.coeff
        # Snap to {1, -1, 1j, -1j} if very close
        options = [1 + 0j, -1 + 0j, 0 + 1j, 0 - 1j]
        for o in options:
            if abs(c - o) < eps:
                c = o
                break

        # Hide coefficient if it's 1.0
        coeff_str = "" if abs(c - 1) < eps else f"({c})*"
        return f"{coeff_str}{self.label}"

    def multiply(self, other: "PauliString") -> "PauliString":
        """Multiplies this PauliString by another (self * other).

        Uses the symplectic representation for the output label together with
        the sitewise Pauli multiplication table to accumulate the global phase.
        This keeps the phase convention exactly consistent with the concrete
        matrices I, X, Y, Z used by `to_matrix()`.

        Args:
            other: The PauliString to multiply with.

        Returns:
            A new PauliString representing the product.

        Raises:
            ValueError: If the Pauli strings have different lengths (n).
        """
        if self.n != other.n:
            raise ValueError(
                f"Incompatible PauliString lengths: {self.n} vs {other.n}"
            )

        # New symplectic vectors
        x3 = self.x ^ other.x
        z3 = self.z ^ other.z

        # Accumulate the phase from the single-site Pauli multiplication rules:
        # XY=iZ, YZ=iX, ZX=iY, and reversing the order flips the sign.
        phase = 1 + 0j
        for a, b in zip(self.label, other.label):
            if a == 'I' or b == 'I' or a == b:
                continue
            if (a, b) in (('X', 'Y'), ('Y', 'Z'), ('Z', 'X')):
                phase *= 1j
            else:
                phase *= -1j

        new_label = ''.join(
            self.XZ_TO_PAULI[(int(a), int(b))] for a, b in zip(x3, z3)
        )
        new_coeff = self.coeff * other.coeff * phase
        return PauliString(new_label, new_coeff)

    # --- Operator Overloading ---

    def __mul__(
            self, other: Union["PauliString", complex, float, int]
    ) -> "PauliString":
        """Overloads the * operator for P * P or P * scalar."""
        if isinstance(other, PauliString):
            return self.multiply(other)
        elif isinstance(other, (complex, float, int)):
            return PauliString(self.label, self.coeff * other)
        else:
            return NotImplemented

    def __rmul__(
            self, other: Union[complex, float, int]
    ) -> "PauliString":
        """Overloads the * operator for scalar * P."""
        if isinstance(other, (complex, float, int)):
            return PauliString(self.label, other * self.coeff)
        else:
            return NotImplemented

    def __add__(
            self, other: Union["PauliString", "PauliSum"]
    ) -> Union["PauliString", "PauliSum"]:
        """Overloads the + operator."""
        if isinstance(other, PauliString):
            if self.label == other.label:
                # Add coefficients if labels match
                return PauliString(self.label, self.coeff + other.coeff)
            else:
                # Promote to a PauliSum if labels differ
                return PauliSum([self, other])
        elif isinstance(other, PauliSum):
            # PauliSum handles the addition logic
            return other + self
        else:
            return NotImplemented

    def __sub__(
            self, other: Union["PauliString", "PauliSum"]
    ) -> Union["PauliString", "PauliSum"]:
        """Overloads the - operator."""
        return self + ((-1) * other)

    def __matmul__(self, other: "PauliString") -> "PauliString":
        """Defines A @ B as the commutator [A, B] = AB - BA."""
        return self.commutator(other)

    def commutator(self, other: "PauliString") -> "PauliString":
        """Calculates the commutator [A, B] = AB - BA.

        Args:
            other: The other PauliString (B).

        Returns:
            A new PauliString representing the commutator.
        """
        prod1 = self.multiply(other)  # AB
        prod2 = other.multiply(self)  # BA

        # The product of two Pauli strings is always a single Pauli string
        # (up to a phase). AB and BA must have the same resulting label.
        if prod1.label != prod2.label:
            # This should be mathematically impossible
            raise ValueError(
                "Commutator product labels mismatch. This should not happen."
            )

        coeff = prod1.coeff - prod2.coeff
        return PauliString(prod1.label, coeff)

    def anticommutator(self, other: "PauliString") -> "PauliString":
        """Calculates the anti-commutator {A, B} = AB + BA.

        Args:
            other: The other PauliString (B).

        Returns:
            A new PauliString representing the anti-commutator.
        """
        prod1 = self.multiply(other)  # AB
        prod2 = other.multiply(self)  # BA

        if prod1.label != prod2.label:
            # This should be mathematically impossible
            raise ValueError(
                "Anti-commutator product labels mismatch. This should not happen."
            )

        coeff = prod1.coeff + prod2.coeff
        return PauliString(prod1.label, coeff)

    def to_masks(self) -> Tuple[int, int, int]:
        """Returns the bitmask representation of the Pauli string.

        The masks are cached after the first computation.

        Returns:
            A tuple (x_mask, z_mask, x_dot_z_mod4):
            - x_mask (int): uint64 bitmask for X operators (LSB = site 0).
            - z_mask (int): uint64 bitmask for Z operators (LSB = site 0).
            - x_dot_z_mod4 (int): (Number of Y sites) mod 4.
        """
        if self._x_mask is None:
            x_mask = 0
            z_mask = 0
            # Pack (x,z) -> bit masks
            for j, (xj, zj) in enumerate(zip(self.x, self.z)):
                if xj: x_mask |= (1 << j)
                if zj: z_mask |= (1 << j)
            self._x_mask = int(x_mask)
            self._z_mask = int(z_mask)

            # x·z = number of Y's (where xj=zj=1)
            # Use 64-bit int to avoid overflow
            x64 = self.x.astype(np.int64)
            z64 = self.z.astype(np.int64)
            self._x_dot_z_mod4 = int(np.dot(x64, z64)) % 4

        return self._x_mask, self._z_mask, self._x_dot_z_mod4

    def to_matrix(
            self,
            pauli_dict: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """Converts the PauliString to its full (2^n, 2^n) matrix.

        The bitmask-based routines in this module use the convention that
        `label[j]` acts on site `j`, with site 0 stored in the least-significant
        bit of basis-state indices. To keep the dense matrix consistent with
        that convention, the Kronecker factors are applied in reverse label
        order so that `label[0]` acts on the least-significant qubit.

        Args:
            pauli_dict: An optional dictionary of pre-computed 2x2 Pauli
                matrices. If None, the module-level `_PAULI_DICT` is used.

        Returns:
            A (2^n, 2^n) numpy array representing the operator.
        """
        D = _PAULI_DICT if pauli_dict is None else pauli_dict
        mat = np.array([[1]], dtype=complex)
        for c in reversed(self.label):
            mat = np.kron(mat, D[c])
        return self.coeff * mat


# ============================================================
# PauliSum: A linear combination of PauliStrings
# ============================================================

class PauliSum:
    """Represents a linear combination of PauliString objects.

    Supports addition, scalar multiplication, and commutation. Terms with the
    same Pauli label are automatically combined upon simplification.

    Attributes:
        terms (List[PauliString]): The list of PauliString terms.
    """

    def __init__(self, terms: Optional[List[PauliString]] = None):
        """Initializes the PauliSum.

        Args:
            terms: An optional list of PauliString objects.

        Raises:
            TypeError: If any item in 'terms' is not a PauliString.
            ValueError: If terms have inconsistent numbers of qubits (n).
        """
        self.terms: List[PauliString] = []
        if terms:
            if not terms:
                return
            n0 = terms[0].n
            for t in terms:
                if not isinstance(t, PauliString):
                    raise TypeError(
                        f"PauliSum terms must be PauliString, got {type(t)}"
                    )
                if t.n != n0:
                    raise ValueError(
                        "All PauliString terms must have the same length (n)."
                    )
                self.terms.append(t)

    def __repr__(self) -> str:
        """Returns a human-readable string representation."""
        if not self.terms:
            return "0"
        return " + ".join(repr(t) for t in self.terms)

    def __add__(
            self, other: Union[PauliString, "PauliSum"]
    ) -> "PauliSum":
        """Overloads the + operator (Sum + String or Sum + Sum)."""
        if isinstance(other, PauliString):
            return PauliSum(self.terms + [other]).simplify()
        elif isinstance(other, PauliSum):
            return PauliSum(self.terms + other.terms).simplify()
        else:
            return NotImplemented

    def __sub__(
            self, other: Union[PauliString, "PauliSum"]
    ) -> "PauliSum":
        """Overloads the - operator."""
        return self + ((-1) * other)

    def __mul__(
            self, scalar: Union[complex, float, int]
    ) -> "PauliSum":
        """Overloads the * operator for Sum * scalar."""
        if isinstance(scalar, (complex, float, int)):
            new_terms = [
                PauliString(t.label, t.coeff * scalar) for t in self.terms
            ]
            return PauliSum(new_terms)
        else:
            return NotImplemented

    __rmul__ = __mul__  # Handle scalar * Sum

    def __truediv__(self, scalar: Union[complex, float, int]) -> "PauliSum":
        """Overloads the / operator for Sum / scalar."""
        if isinstance(scalar, (complex, float, int)):
            if scalar == 0:
                raise ZeroDivisionError("Cannot divide PauliSum by zero.")
            return self * (1.0 / scalar)
        return NotImplemented

    def __matmul__(self, other: "PauliSum") -> "PauliSum":
        """Defines H1 @ H2 as the commutator [H1, H2]."""
        if not isinstance(other, PauliSum):
            raise TypeError(
                "PauliSum commutator (@) is only defined for "
                "PauliSum @ PauliSum"
            )

        # [A+B, C+D] = [A,C] + [A,D] + [B,C] + [B,D]
        new_terms = []
        for t1 in self.terms:
            for t2 in other.terms:
                new_terms.append(t1.commutator(t2))
        return PauliSum(new_terms).simplify()

    def __rmatmul__(self, other: PauliString) -> "PauliSum":
        """Enables PauliString @ PauliSum syntax."""
        if isinstance(other, PauliString):
            return PauliSum([other]) @ self
        return NotImplemented

    def simplify(self, tol: float = 1e-12) -> "PauliSum":
        """Merges terms with identical Pauli labels and removes near-zero terms.

        Args:
            tol: Absolute tolerance. Terms with |coefficient| <= tol
                are discarded.

        Returns:
            A new, simplified PauliSum.

        Raises:
            ValueError: If terms have inconsistent numbers of qubits (n).
        """
        if not self.terms:
            return PauliSum([])

        n0 = self.terms[0].n
        merged = defaultdict(complex)
        for t in self.terms:
            if t.n != n0:
                raise ValueError(
                    "All PauliString terms must have the same length (n)."
                )
            merged[t.label] += t.coeff

        simplified_terms = [
            PauliString(lbl, c) for lbl, c in merged.items() if abs(c) > tol
        ]
        return PauliSum(simplified_terms)

    def to_masks_and_coeffs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorizes all terms into masks and complex prefactors.

        This is a helper for fast vectorized operations (e.g., `apply_pauli_sum`).

        Returns:
            A tuple (x_masks, z_masks, pref):
            - x_masks (np.ndarray[uint64]): Shape (D,).
            - z_masks (np.ndarray[uint64]): Shape (D,).
            - pref (np.ndarray[complex128]): Shape (D,).
                where pref[k] = coeff_k * i^(x·z)_k (global Y-phase absorbed).
        """
        if not self.terms:
            return (np.empty((0,), dtype=np.uint64),
                    np.empty((0,), dtype=np.uint64),
                    np.empty((0,), dtype=np.complex128))

        D = len(self.terms)
        x_masks = np.empty((D,), dtype=np.uint64)
        z_masks = np.empty((D,), dtype=np.uint64)
        pref = np.empty((D,), dtype=np.complex128)

        # Lookup table for i^k
        lut = np.array([1 + 0j, 1j, -1 + 0j, -1j], dtype=np.complex128)

        for k, t in enumerate(self.terms):
            xm, zm, xdotz = t.to_masks()
            x_masks[k] = np.uint64(xm)
            z_masks[k] = np.uint64(zm)
            # Absorb the i^(x·z) phase into the coefficient
            pref[k] = np.complex128(t.coeff) * lut[xdotz]

        return x_masks, z_masks, pref

    def structure_constants_symbolic(
            self,
            device: str = "cpu",
            atol: float = 1e-8,
            orthonormalize: bool = True,
            clean_digits: int = 12,
    ) -> torch.Tensor:
        """Calculates the structure constants f[g,a,b] for the basis {T_a}.

        The constants are defined such that [T_a, T_b] = sum_g f[g,a,b] T_g.
        This method operates on the symplectic representation without
        constructing full matrices, allowing for GPU acceleration.
        It still materializes a dense (d, d, d) tensor, so large-scale
        production evolution should prefer `adjoint_generator_in_lie_closure_basis`
        and `evolve_operator_in_lie_closure_basis`, which only store a
        dense adjoint matrix of shape (d, d).

        Args:
            device: The torch device to use ('cpu' or 'cuda').
            atol: Absolute tolerance for the orthogonality check.
            orthonormalize: If True, returns constants for the orthonormal
                basis {E_a = T_a / ||T_a||}, where ||T_a|| is the
                Frobenius/Hilbert-Schmidt norm.
            clean_digits: Number of decimal places to round the final
                result to, removing floating point noise.

        Returns:
            A torch.Tensor f[g,a,b] of shape (d, d, d) containing the
            complex structure constants.

        Raises:
            ValueError: If the PauliSum is empty.
        """
        # Ensure basis is simplified (no duplicate labels)
        basis = self.simplify().terms
        if not basis:
            raise ValueError(
                "PauliSum is empty, cannot compute structure constants."
            )

        d = len(basis)  # Dimension of the algebra
        n = basis[0].n  # Number of qubits

        # === 1) Extract symplectic vectors and coefficients ===
        x_np = np.stack([p.x for p in basis], axis=0)  # (d, n), uint8
        z_np = np.stack([p.z for p in basis], axis=0)
        # Cast to a wider int to avoid overflow during dot product
        x = torch.from_numpy(x_np.astype(np.int32)).to(device)
        z = torch.from_numpy(z_np.astype(np.int32)).to(device)
        coeffs = torch.tensor(
            [p.coeff for p in basis], dtype=torch.complex128, device=device
        )  # (d,)

        # === 2) Pairwise symbolic multiplication (vectorized) ===
        # Symplectic vectors of the product P_a * P_b
        x_prod = (x[:, None, :] ^ x[None, :, :])  # (d, d, n)
        z_prod = (z[:, None, :] ^ z[None, :, :])  # (d, d, n)

        # Phase from the single-site Pauli multiplication table in the code
        # order I=0, X=1, Z=2, Y=3 (i.e. code = x + 2 z).
        code = x + 2 * z  # (d, n)
        phase_table = torch.tensor(
            [
                [1 + 0j, 1 + 0j, 1 + 0j, 1 + 0j],
                [1 + 0j, 1 + 0j, -1j, 1j],
                [1 + 0j, 1j, 1 + 0j, -1j],
                [1 + 0j, -1j, 1j, 1 + 0j],
            ],
            dtype=torch.complex128,
            device=device,
        )
        local_phase = phase_table[code[:, None, :], code[None, :, :]]  # (d,d,n)
        phase = torch.prod(local_phase, dim=-1)  # (d, d)

        # Commutator coefficient: [T_a, T_b] = c_ab * P_g
        # c_ab = coeff(T_a * T_b) - coeff(T_b * T_a)
        coeff_prod = coeffs[:, None] * coeffs[None, :] * phase  # coeff(T_a * T_b)
        coeff_comm = coeff_prod - coeff_prod.transpose(0, 1)  # (d, d)

        # === 3) Find f[g,a,b] by matching product string to basis ===
        # We need to find which g satisfies P_g == P_a * P_b
        xg = x[:, None, None, :].expand(d, d, d, n)  # (g, a, b, n)
        zg = z[:, None, None, :].expand(d, d, d, n)
        xab = x_prod[None, :, :, :].expand(d, d, d, n)  # (g, a, b, n)
        zab = z_prod[None, :, :, :].expand(d, d, d, n)

        # match[g, a, b] is True if T_g has the same label as [T_a, T_b]
        match = torch.all((xg == xab) & (zg == zab), dim=-1)  # (d, d, d)

        # f_raw[g,a,b] is the coefficient of T_g in the expansion of [T_a, T_b].
        # Since T_g = coeff_g * P_g, we must divide the commutator coefficient
        # of the raw Pauli label P_g by coeff_g to recover the basis coefficient.
        f_raw = torch.zeros((d, d, d), dtype=torch.complex128, device=device)
        f_raw += match * (coeff_comm[None, :, :] / coeffs[:, None, None])

        # === 4) Orthogonality check (Frobenius inner product) ===
        # <T_a, T_b> = Tr(T_a^† T_b)
        # For Pauli strings: Tr(P_a^† P_b) = 2^n * δ_{label(a), label(b)}
        # So, Gram[a,b] = 2^n * conj(coeff_a) * coeff_b * δ_{label(a), label(b)}
        same_mask = torch.all(
            (x[:, None, :] == x[None, :, :]) &
            (z[:, None, :] == z[None, :, :]),
            dim=-1
        )  # (d, d)
        gram = (2.0 ** n) * (coeffs.conj()[:, None] * coeffs[None, :]) * same_mask
        eye = torch.eye(d, dtype=torch.bool, device=device)
        offdiag = torch.abs(gram)[~eye]
        if torch.any(offdiag > atol):
            print(
                "Warning: Provided PauliSum terms are not orthogonal "
                "under the Frobenius inner product."
            )

        # === 5) (Optional) Transform to orthonormal basis ===
        # E_a = T_a / ||T_a||, where ||T_a|| = sqrt(Tr(T_a^† T_a))
        # ||T_a|| = sqrt(Gram[a,a]) = 2^{n/2} * |coeff_a|
        if orthonormalize:
            norms = (2.0 ** (n / 2.0)) * torch.abs(coeffs)  # (d,)
            # f^E_{g,a,b} = f_raw_{g,a,b} * ||T_g|| / (||T_a|| * ||T_b||)
            scale = norms.view(d, 1, 1) / (
                    norms.view(1, d, 1) * norms.view(1, 1, d)
            )
            f = f_raw * scale
        else:
            f = f_raw

        # === 6) Final cleanup: remove floating point noise ===
        if clean_digits is not None:
            f = torch.round(f.real, decimals=clean_digits) + 1j * torch.round(
                f.imag, decimals=clean_digits
            )
            # Set very small values to exact zero
            tiny = 10.0 ** (-(clean_digits - 2))
            f.real[torch.abs(f.real) < tiny] = 0
            f.imag[torch.abs(f.imag) < tiny] = 0

        return f

    def to_matrix(self) -> np.ndarray:
        """Converts the entire PauliSum to its full matrix representation.

        Returns:
            A (2^n, 2^n) numpy array representing the summed operator.
            Returns a 1x1 zero matrix if the sum is empty.
        """
        if not self.terms:
            # Cannot determine size, return a 1x1 zero
            return np.zeros((1, 1), dtype=complex)

        # Sum the matrices of all terms
        n = self.terms[0].n
        total_mat = np.zeros((2 ** n, 2 ** n), dtype=complex)

        # Use module-level dict for efficiency
        pauli_dict = _PAULI_DICT
        for t in self.terms:
            total_mat += t.to_matrix(pauli_dict)

        return total_mat

    def _get_term_dict(self) -> Dict[str, complex]:
        """Helper method to get the merged {label: coeff} dictionary.

        This is used for symbolic inner products. Unlike simplify(), this
        does not filter out small values, as it's an internal helper.

        Returns:
            A dictionary mapping Pauli labels to their summed complex
            coefficients.
        """
        merged = defaultdict(complex)
        for t in self.terms:
            merged[t.label] += t.coeff
        return merged

    def frob_inner(self, other: "PauliSum") -> complex:
        """Computes the Frobenius inner product Tr(self^† * other) symbolically.

        The inner product is computed efficiently using the orthogonality
        of Pauli strings: Tr(P_a^† P_b) = 2^n * δ_ab.

        Args:
            other: The other PauliSum object (B in Tr(A^† B)).

        Returns:
            The complex scalar value of the Frobenius inner product.

        Raises:
            ValueError: If the two PauliSum objects have different
                numbers of qubits (n).
        """
        if not self.terms and not other.terms:
            return 0.0 + 0.0j

        # Simplify to ensure no duplicate labels
        A = self.simplify()
        B = other.simplify()

        if not A.terms or not B.terms:
            return 0.0 + 0.0j  # Inner product with zero operator is zero

        n_A = A.terms[0].n
        n_B = B.terms[0].n
        if n_A != n_B:
            raise ValueError(
                "Frobenius inner product requires same number of qubits "
                f"(n): {n_A} vs {n_B}"
            )

        n = n_A
        trace_factor = 2.0 ** n

        # Get simplified coefficient maps
        self_merged = A._get_term_dict()
        other_merged = B._get_term_dict()

        inner_prod = 0.0 + 0.0j

        # We only need to iterate over the keys present in 'self'.
        # Terms in 'other' but not 'self' have an inner product of 0.
        for label, coeff_self in self_merged.items():
            # Get the coefficient of the same label from 'other'
            coeff_other = other_merged.get(label, 0.0)

            # Tr(A^† B) = sum_i (coeff_self_i.conj() * coeff_other_i) * 2^n
            inner_prod += coeff_self.conjugate() * coeff_other

        return inner_prod * trace_factor

    def frobenius_norm(self) -> float:
        """Computes the Frobenius norm sqrt(Tr(self^† * self)) symbolically.

        Returns:
            The real scalar value of the Frobenius norm.
        """
        if not self.terms:
            return 0.0

        n = self.terms[0].n
        trace_factor = 2.0 ** n

        # We only need the coefficients of the (simplified) self
        self_merged = self.simplify()._get_term_dict()

        norm_sq = 0.0
        for coeff_self in self_merged.values():
            # |c_i|^2
            norm_sq += abs(coeff_self) ** 2

        norm_sq *= trace_factor

        # The squared norm is guaranteed to be real.
        return np.sqrt(norm_sq)


PauliOperator = Union[PauliString, PauliSum]


def _as_pauli_sum(operator: PauliOperator) -> PauliSum:
    """Normalizes a PauliString/PauliSum input to a PauliSum."""
    if isinstance(operator, PauliString):
        return PauliSum([operator])
    if isinstance(operator, PauliSum):
        return operator
    raise TypeError(
        "Expected a PauliString or PauliSum, got "
        f"{type(operator)}"
    )


def multiply_pauli_operators(
        left: PauliOperator,
        right: PauliOperator,
        *,
        tol: float = 1e-12,
) -> PauliSum:
    """Returns the ordinary product of two Pauli operators.

    Unlike ``@``, which denotes the commutator, this helper distributes the
    matrix product over all terms in two Pauli sums.
    """
    left_sum = _as_pauli_sum(left).simplify(tol=tol)
    right_sum = _as_pauli_sum(right).simplify(tol=tol)

    if not left_sum.terms or not right_sum.terms:
        return PauliSum([])

    new_terms: List[PauliString] = []
    for left_term in left_sum.terms:
        for right_term in right_sum.terms:
            new_terms.append(left_term.multiply(right_term))
    return PauliSum(new_terms).simplify(tol=tol)


def multiply_pauli_factors(
        factors: Iterable[PauliOperator],
        *,
        tol: float = 1e-12,
) -> PauliSum:
    """Returns the ordered product of a sequence of Pauli operators."""
    factor_list = list(factors)
    if not factor_list:
        raise ValueError("multiply_pauli_factors requires at least one factor.")

    product = _as_pauli_sum(factor_list[0]).simplify(tol=tol)
    for factor in factor_list[1:]:
        product = multiply_pauli_operators(product, factor, tol=tol)
        if not product.terms:
            break
    return product.simplify(tol=tol)


def pauli_string_from_sites(
        operators: Union[str, Iterable[str]],
        sites: Iterable[int],
        lattice_length: int,
        coefficient: complex = 1.0,
) -> PauliString:
    """Builds a full-length PauliString from local operators and site indices.

    Args:
        operators: Either a Pauli word like ``"XZY"`` or an iterable of local
            Pauli labels.
        sites: Site indices where the corresponding operators act.
        lattice_length: Total number of qubits.
        coefficient: Overall coefficient of the resulting PauliString.

    Returns:
        A full-length PauliString with identity on all unspecified sites.
    """
    if lattice_length < 1:
        raise ValueError("lattice_length must be >= 1.")

    if isinstance(operators, str):
        operator_list = list(operators.upper())
    else:
        operator_list = [str(op).upper() for op in operators]

    site_list = [int(site) for site in sites]
    if len(operator_list) != len(site_list):
        raise ValueError(
            "operators and sites must have the same length, got "
            f"{len(operator_list)} and {len(site_list)}."
        )
    if len(set(site_list)) != len(site_list):
        raise ValueError(
            "sites must be distinct when building a PauliString from sites."
        )

    label = ["I"] * lattice_length
    for op, site in zip(operator_list, site_list):
        if op not in {"I", "X", "Y", "Z"}:
            raise ValueError(
                f"Unsupported Pauli operator '{op}'. Expected one of I, X, Y, Z."
            )
        if not (0 <= site < lattice_length):
            raise IndexError(
                f"Site index {site} out of bounds for length {lattice_length}."
            )
        label[site] = op

    return PauliString("".join(label), coefficient)


def global_pauli_op_chain_list(
        op_type: str,
        lattice_length: int,
        coefficient: complex = 1.0,
        spin: float = 0.5,
        pbc: bool = True,
) -> List[PauliString]:
    """Symbolically constructs translated Pauli-chain terms on a 1D lattice.

    This is the Pauli-algebra analogue of ``quantum_simulation.operator
    .global_pauli_op_chain_list``. Instead of dense matrices it returns a list
    of translated ``PauliString`` terms.
    """
    if spin != 0.5:
        raise NotImplementedError(
            "global_pauli_op_chain_list in pauli_algebra currently supports "
            "only spin=0.5."
        )

    word = op_type.upper()
    if not word:
        raise ValueError("op_type must be a non-empty Pauli word.")

    op_len = len(word)
    if lattice_length < op_len:
        raise ValueError(
            "lattice_length must be greater than or equal to the length of "
            "op_type."
        )

    num_terms = lattice_length if pbc else (lattice_length - op_len + 1)
    translated_terms: List[PauliString] = []
    for start in range(num_terms):
        sites = [
            ((start + offset) % lattice_length) if pbc else (start + offset)
            for offset in range(op_len)
        ]
        translated_terms.append(
            pauli_string_from_sites(
                word,
                sites,
                lattice_length,
                coefficient=coefficient,
            )
        )

    return translated_terms


def global_pauli_op_chain(
        op_type: str,
        lattice_length: int,
        coefficient: complex = 1.0,
        spin: float = 0.5,
        pbc: bool = True,
) -> PauliSum:
    """Symbolically constructs a summed translated Pauli chain on a 1D lattice.

    This is the Pauli-algebra analogue of ``quantum_simulation.operator
    .global_pauli_op_chain``. It returns a ``PauliSum`` instead of a dense
    matrix.
    """
    return PauliSum(
        global_pauli_op_chain_list(
            op_type,
            lattice_length,
            coefficient=coefficient,
            spin=spin,
            pbc=pbc,
        )
    ).simplify()


# ============================================================
# Lie Algebra Utility Functions
# ============================================================

def lie_closure_basis_symbolic(
        generators: List[PauliSum],
        atol: float = 1e-6,
        rtol: float = 1e-5,
        max_iter: int = 100
) -> List[PauliSum]:
    """Generates an orthonormal basis for the Lie algebra from generators.

    This function performs a symbolic Gram-Schmidt-like procedure. It starts
    with the given generators and iteratively adds new, orthogonal elements
    by computing commutators of all pairs in the current basis.

    Args:
        generators: A list of PauliSum objects serving as the
            initial generators of the Lie algebra.
        atol: Absolute tolerance for the residual norm. A candidate is
            considered zero if its norm is <= atol.
        rtol: Relative tolerance (scaled by candidate norm). A residual
            is kept if rnorm > atol + rtol * xnorm.
        max_iter: Safety cap on expansion rounds.

    Returns:
        A list of PauliSum objects [Q_0, Q_1, ...] that form an
        orthonormal basis (under the Frobenius inner product) for the
        generated Lie algebra.
    """

    # Q will store the building orthonormal basis
    Q: List[PauliSum] = []

    def _add_from_candidate(X: PauliSum) -> bool:
        """Symbolic Gram-Schmidt step.

        Projects X onto span(Q), keeps residual if significant,
        normalizes it, and appends it to Q.

        Args:
            X: The candidate PauliSum to add.

        Returns:
            True if a new basis vector was added, False otherwise.
        """
        # Skip near-zero candidate early
        xnorm = X.frobenius_norm()
        if xnorm <= atol:
            return False

        # R is the residual
        R = X
        for Qi in Q:
            # Project R onto Qi
            # alpha = <Qi, R> = Tr(Qi^† * R)
            alpha = Qi.frob_inner(R)
            # R = R - alpha * Qi
            # (Note: Assumes Q is already orthonormal, so <Qi, Qi> = 1)
            R = R - (alpha * Qi)

        rnorm = R.frobenius_norm()

        # Check if residual is significant
        if rnorm > atol + rtol * xnorm:
            # Orthonormalize and add to basis
            Q.append((1.0 / rnorm) * R)
            return True
        return False

    # 1) Seed: Orthonormalize the input generators
    for A in generators:
        _add_from_candidate(A.simplify())

    # 2) Expand by commutators among the *current basis*
    added = True
    it = 0
    while added and it < max_iter:
        added = False
        k = len(Q)
        if k < 2:
            break  # Need at least 2 basis elements to commute

        # Create all new unique commutators [Qi, Qj] where i < j
        new_candidates: List[PauliSum] = []
        for i in range(k):
            for j in range(i + 1, k):
                # [Qi, Qj] is implemented as Qi @ Qj
                comm = (Q[i] @ Q[j]).simplify()
                if not comm.terms:  # 0 operator
                    continue
                new_candidates.append(comm)

        # Try to add the new candidates to the basis
        for comm_candidate in new_candidates:
            if _add_from_candidate(comm_candidate.simplify()):
                added = True

        it += 1

    if it == max_iter:
        print(f"Warning: Lie closure did not converge after {max_iter} "
              "iterations. The basis may be incomplete.")

    # 3) Return the final orthonormal basis
    return Q


def coefficients_in_lie_closure_basis(
        basis: List[PauliSum],
        operator: Union[PauliString, PauliSum],
        device: str = "cpu"
) -> torch.Tensor:
    """Calculates the expansion coefficients of an operator in an ONB.

    Assumes `basis` is an orthonormal basis {Q_i}, typically from
    `lie_closure_basis_symbolic`.

    This function projects the given `operator` (A) onto this basis, finding
    the complex coefficients {c_i} such that:
        A = sum_i c_i * Q_i

    The coefficients are computed using the Frobenius inner product:
        c_i = <Q_i, A> = Tr(Q_i^† * A)

    Args:
        basis: A list of PauliSum objects [Q_0, Q_1, ...] that form an
            orthonormal basis.
        operator: The PauliString or PauliSum to project onto the basis.
        device: The torch device (e.g., 'cpu' or 'cuda') on which to create
            the resulting tensor.

    Returns:
        A 1D torch.Tensor of dtype complex128 containing the coefficients
        [c_0, c_1, ...].

    Raises:
        TypeError: If 'operator' is not a PauliString or PauliSum.
    """
    # 1. Standardize the input operator into a PauliSum object.
    if isinstance(operator, PauliString):
        op_sum = PauliSum([operator])
    elif isinstance(operator, PauliSum):
        op_sum = operator
    else:
        raise TypeError(
            f"Operator must be a PauliString or PauliSum, got {type(operator)}"
        )

    # 2. Project the operator onto each basis vector via the inner product.
    #    c_i = <Q_i, operator> = Tr(Q_i^† * operator)
    coeffs = [
        q_i.frob_inner(op_sum) for q_i in basis
    ]

    # 3. Convert the list of complex coefficients to a torch.Tensor.
    return torch.tensor(coeffs, dtype=torch.complex128, device=device)


def adjoint_generator_in_lie_closure_basis(
        basis: List[PauliSum],
        generator: Union[PauliString, PauliSum],
        device: str = "cpu",
) -> torch.Tensor:
    """Builds the adjoint-action matrix M of a generator on a Lie basis.

    The matrix is defined by
        [G, Q_beta] = sum_gamma M[gamma, beta] Q_gamma
    for an orthonormal Lie-closure basis {Q_beta}.

    This is the object that appears in the note as the adjoint generator.
    Compared with first materializing a dense structure-constant tensor
    f[gamma, alpha, beta], this routine is often substantially cheaper when
    the generator G is sparse in the chosen basis.

    Args:
        basis: Orthonormal Lie-closure basis [Q_0, Q_1, ...].
        generator: The operator G whose adjoint action is represented.
        device: Torch device for the returned matrix.

    Returns:
        A complex128 tensor M of shape (d, d), where d = len(basis).

    Raises:
        TypeError: If 'generator' is not a PauliString or PauliSum.
    """
    if isinstance(generator, PauliString):
        G = PauliSum([generator])
    elif isinstance(generator, PauliSum):
        G = generator
    else:
        raise TypeError(
            f"Generator must be a PauliString or PauliSum, got {type(generator)}"
        )

    d = len(basis)
    if d == 0:
        return torch.empty((0, 0), dtype=torch.complex128, device=device)

    columns = []
    for Q_beta in basis:
        comm = (G @ Q_beta).simplify()
        columns.append(
            coefficients_in_lie_closure_basis(basis, comm, device=device)
        )

    return torch.stack(columns, dim=1)


def evolve_operator_in_lie_closure_basis(
        basis: List[PauliSum],
        hamiltonian: Union[PauliString, PauliSum],
        operator: Union[PauliString, PauliSum],
        time: float,
        hbar: float = 1.0,
        device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evolves an operator in the Lie-closure basis via the adjoint action.

    This implements the note's formula
        O(t) = exp((i t / hbar) ad_H) O
    in basis-coordinate form.

    Args:
        basis: Orthonormal Lie-closure basis [Q_0, Q_1, ...].
        hamiltonian: Hamiltonian H as a PauliString or PauliSum.
        operator: Observable O as a PauliString or PauliSum.
        time: Evolution time t.
        hbar: Planck constant factor used in (i t / hbar) H.
        device: Torch device for all returned tensors.

    Returns:
        A tuple (w_t, M, w_0):
        - w_t: Coefficients of O(t) in the basis.
        - M: Adjoint generator matrix for (i t / hbar) H.
        - w_0: Initial coefficients of O.

    Raises:
        TypeError: If 'hamiltonian' or 'operator' has an invalid type.
        ZeroDivisionError: If hbar == 0.
    """
    if hbar == 0:
        raise ZeroDivisionError("hbar must be non-zero.")

    if isinstance(hamiltonian, PauliString):
        H = PauliSum([hamiltonian])
    elif isinstance(hamiltonian, PauliSum):
        H = hamiltonian
    else:
        raise TypeError(
            "Hamiltonian must be a PauliString or PauliSum, got "
            f"{type(hamiltonian)}"
        )

    w_0 = coefficients_in_lie_closure_basis(basis, operator, device=device)
    scaled_H = (1j * time / hbar) * H
    M = adjoint_generator_in_lie_closure_basis(
        basis, scaled_H, device=device
    )
    w_t = torch.matrix_exp(M) @ w_0
    return w_t, M, w_0


def basis_expectations_for_sparse_ket(
        basis: List[PauliSum],
        state: "SparseKet",
        device: str = "cpu",
        op_tol: float = 1e-12,
) -> torch.Tensor:
    """Computes <psi|Q_gamma|psi> for every basis element Q_gamma.

    Args:
        basis: Orthonormal Lie-closure basis [Q_0, Q_1, ...].
        state: Sparse pure state |psi>.
        device: Torch device for the returned tensor.
        op_tol: Tolerance passed to `SparseKet.measure_pauli`.

    Returns:
        A complex128 tensor mu with mu[gamma] = <psi|Q_gamma|psi>.
    """
    values = [state.measure_pauli(Q_gamma, op_tol=op_tol) for Q_gamma in basis]
    return torch.tensor(values, dtype=torch.complex128, device=device)


def _as_density_matrix_tensor(
        density_matrix: Union[np.ndarray, torch.Tensor],
        n: int,
        device: str = "cpu",
) -> torch.Tensor:
    """Standardizes a density matrix input and validates its shape."""
    rho = torch.as_tensor(
        density_matrix, dtype=torch.complex128, device=device
    )
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError(
            "density_matrix must be a square 2D array/tensor, got "
            f"shape {tuple(rho.shape)}"
        )

    expected_dim = 1 << n
    if rho.shape[0] != expected_dim:
        raise ValueError(
            "density_matrix dimension does not match the operator qubit count: "
            f"expected {(expected_dim, expected_dim)}, got {tuple(rho.shape)}"
        )

    return rho


def _as_state_vector_tensor(
        state_vector: Union[np.ndarray, torch.Tensor],
        n: int,
        device: str = "cpu",
) -> torch.Tensor:
    """Standardizes a dense pure-state vector input and validates its shape."""
    psi = torch.as_tensor(
        state_vector, dtype=torch.complex128, device=device
    )

    if psi.ndim == 2:
        if psi.shape[0] == 1:
            psi = psi.reshape(-1)
        elif psi.shape[1] == 1:
            psi = psi[:, 0]
        else:
            raise ValueError(
                "state_vector must be a 1D vector or a column/row vector, got "
                f"shape {tuple(psi.shape)}"
            )
    elif psi.ndim != 1:
        raise ValueError(
            "state_vector must be a 1D vector or a column/row vector, got "
            f"shape {tuple(psi.shape)}"
        )

    expected_dim = 1 << n
    if psi.shape[0] != expected_dim:
        raise ValueError(
            "state_vector dimension does not match the operator qubit count: "
            f"expected {(expected_dim,)}, got {tuple(psi.shape)}"
        )

    return psi


def _parity_signs_for_z_mask(
        indices: torch.Tensor,
        z_mask: int,
) -> torch.Tensor:
    """Returns (-1)^{popcount(indices & z_mask)} as a complex tensor."""
    if z_mask == 0:
        return torch.ones(
            indices.shape[0], dtype=torch.complex128, device=indices.device
        )

    parity = torch.zeros(
        indices.shape[0], dtype=torch.int8, device=indices.device
    )
    mask = int(z_mask)
    bit = 0
    while mask:
        if mask & 1:
            parity ^= ((indices >> bit) & 1).to(torch.int8)
        mask >>= 1
        bit += 1

    return (1 - 2 * parity).to(torch.complex128)


def measure_pauli_sum_state_vector(
        operator: Union[PauliString, PauliSum],
        state_vector: Union[np.ndarray, torch.Tensor],
        device: str = "cpu",
        op_tol: float = 1e-12,
) -> complex:
    """Computes <psi|H|psi> without forming any Pauli operator matrices."""
    if isinstance(operator, PauliString):
        H = PauliSum([operator])
    elif isinstance(operator, PauliSum):
        H = operator
    else:
        raise TypeError(
            "Operator must be a PauliString or PauliSum, got "
            f"{type(operator)}"
        )

    H = H.simplify(tol=op_tol)
    if not H.terms:
        return 0.0 + 0.0j

    n = H.terms[0].n
    psi = _as_state_vector_tensor(state_vector, n=n, device=device)
    dim = psi.shape[0]
    indices = torch.arange(dim, dtype=torch.int64, device=psi.device)
    x_masks, z_masks, pref = H.to_masks_and_coeffs()
    pref_t = torch.as_tensor(pref, dtype=torch.complex128, device=psi.device)

    sign_cache: Dict[int, torch.Tensor] = {}
    total = torch.tensor(0.0 + 0.0j, dtype=torch.complex128, device=psi.device)

    for k in range(len(H.terms)):
        x_mask = int(x_masks[k])
        z_mask = int(z_masks[k])
        cols = indices ^ x_mask

        if z_mask not in sign_cache:
            sign_cache[z_mask] = _parity_signs_for_z_mask(indices, z_mask)

        total += pref_t[k] * torch.sum(
            sign_cache[z_mask] * psi * psi[cols].conj()
        )

    return total.item()


def measure_pauli_sum_density_matrix(
        operator: Union[PauliString, PauliSum],
        density_matrix: Union[np.ndarray, torch.Tensor],
        device: str = "cpu",
        op_tol: float = 1e-12,
) -> complex:
    """Computes Tr(H rho) without forming any Pauli operator matrices.

    For each Pauli term, this uses the bitmask action on computational-basis
    indices to evaluate the trace directly from density-matrix entries.

    Args:
        operator: The PauliString or PauliSum operator H.
        density_matrix: Dense matrix rho with shape (2^n, 2^n).
        device: Torch device for the computation.
        op_tol: Tolerance used when simplifying the Pauli operator.

    Returns:
        The complex scalar value Tr(H rho).

    Raises:
        TypeError: If 'operator' is not a PauliString or PauliSum.
        ValueError: If the density matrix shape is incompatible.
    """
    if isinstance(operator, PauliString):
        H = PauliSum([operator])
    elif isinstance(operator, PauliSum):
        H = operator
    else:
        raise TypeError(
            "Operator must be a PauliString or PauliSum, got "
            f"{type(operator)}"
        )

    H = H.simplify(tol=op_tol)
    if not H.terms:
        return 0.0 + 0.0j

    n = H.terms[0].n
    rho = _as_density_matrix_tensor(density_matrix, n=n, device=device)
    dim = rho.shape[0]
    indices = torch.arange(dim, dtype=torch.int64, device=rho.device)
    x_masks, z_masks, pref = H.to_masks_and_coeffs()
    pref_t = torch.as_tensor(pref, dtype=torch.complex128, device=rho.device)

    sign_cache: Dict[int, torch.Tensor] = {}
    total = torch.tensor(0.0 + 0.0j, dtype=torch.complex128, device=rho.device)

    for k in range(len(H.terms)):
        x_mask = int(x_masks[k])
        z_mask = int(z_masks[k])
        cols = indices ^ x_mask

        if z_mask not in sign_cache:
            sign_cache[z_mask] = _parity_signs_for_z_mask(indices, z_mask)

        total += pref_t[k] * torch.sum(sign_cache[z_mask] * rho[indices, cols])

    return total.item()


def basis_expectations_for_state_vector(
        basis: List[PauliSum],
        state_vector: Union[np.ndarray, torch.Tensor],
        device: str = "cpu",
        op_tol: float = 1e-12,
) -> torch.Tensor:
    """Computes mu_gamma = <psi|Q_gamma|psi> for every basis element Q_gamma."""
    if not basis:
        return torch.empty((0,), dtype=torch.complex128, device=device)

    n = basis[0].terms[0].n
    psi = _as_state_vector_tensor(state_vector, n=n, device=device)
    values = [
        measure_pauli_sum_state_vector(
            Q_gamma, psi, device=device, op_tol=op_tol
        )
        for Q_gamma in basis
    ]
    return torch.tensor(values, dtype=torch.complex128, device=device)


def basis_expectations_for_density_matrix(
        basis: List[PauliSum],
        density_matrix: Union[np.ndarray, torch.Tensor],
        device: str = "cpu",
        op_tol: float = 1e-12,
) -> torch.Tensor:
    """Computes mu_gamma = Tr(Q_gamma rho) for every basis element Q_gamma."""
    if not basis:
        return torch.empty((0,), dtype=torch.complex128, device=device)

    n = basis[0].terms[0].n
    rho = _as_density_matrix_tensor(density_matrix, n=n, device=device)
    values = [
        measure_pauli_sum_density_matrix(
            Q_gamma, rho, device=device, op_tol=op_tol
        )
        for Q_gamma in basis
    ]
    return torch.tensor(values, dtype=torch.complex128, device=device)


def expectation_value_in_lie_closure_basis(
        operator_coeffs: torch.Tensor,
        basis_expectations: torch.Tensor,
) -> complex:
    """Contracts evolved operator coefficients with basis expectation values.

    For O = sum_gamma w_gamma Q_gamma and
        mu_gamma = Tr(Q_gamma rho),
    this returns sum_gamma w_gamma mu_gamma.

    Args:
        operator_coeffs: Coefficients w_gamma of the operator in the basis.
        basis_expectations: Moments mu_gamma = Tr(Q_gamma rho).

    Returns:
        The complex expectation value.

    Raises:
        ValueError: If the input vectors do not have the same shape.
    """
    if operator_coeffs.shape != basis_expectations.shape:
        raise ValueError(
            "operator_coeffs and basis_expectations must have the same shape, "
            f"got {operator_coeffs.shape} and {basis_expectations.shape}"
        )

    return torch.sum(operator_coeffs * basis_expectations).item()


def heisenberg_expectation_in_lie_closure_basis(
        basis: List[PauliSum],
        hamiltonian: Union[PauliString, PauliSum],
        operator: Union[PauliString, PauliSum],
        state: "SparseKet",
        time: float,
        hbar: float = 1.0,
        device: str = "cpu",
        op_tol: float = 1e-12,
) -> Tuple[complex, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluates <psi|O(t)|psi> using Lie-basis evolution.

    Args:
        basis: Orthonormal Lie-closure basis [Q_0, Q_1, ...].
        hamiltonian: Hamiltonian H.
        operator: Observable O.
        state: Sparse pure state |psi>.
        time: Evolution time t.
        hbar: Planck constant factor used in (i t / hbar) H.
        device: Torch device for intermediate tensors.
        op_tol: Tolerance passed to `SparseKet.measure_pauli`.

    Returns:
        A tuple (expectation_value, w_t, M, mu):
        - expectation_value: <psi|O(t)|psi>
        - w_t: Evolved coefficients of O(t)
        - M: Adjoint generator matrix
        - mu: Basis moments mu_gamma = <psi|Q_gamma|psi>
    """
    w_t, M, _ = evolve_operator_in_lie_closure_basis(
        basis,
        hamiltonian,
        operator,
        time=time,
        hbar=hbar,
        device=device,
    )
    mu = basis_expectations_for_sparse_ket(
        basis, state, device=device, op_tol=op_tol
    )
    value = expectation_value_in_lie_closure_basis(w_t, mu)
    return value, w_t, M, mu


def heisenberg_expectation_for_density_matrix_in_lie_closure_basis(
        basis: List[PauliSum],
        hamiltonian: Union[PauliString, PauliSum],
        operator: Union[PauliString, PauliSum],
        density_matrix: Union[np.ndarray, torch.Tensor],
        time: float,
        hbar: float = 1.0,
        device: str = "cpu",
        op_tol: float = 1e-12,
) -> Tuple[complex, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluates Tr(O(t) rho) using Lie-basis evolution and a dense rho input.

    Args:
        basis: Orthonormal Lie-closure basis [Q_0, Q_1, ...].
        hamiltonian: Hamiltonian H.
        operator: Observable O.
        density_matrix: Dense matrix rho with shape (2^n, 2^n).
        time: Evolution time t.
        hbar: Planck constant factor used in (i t / hbar) H.
        device: Torch device for intermediate tensors.
        op_tol: Tolerance used when evaluating basis moments.

    Returns:
        A tuple (expectation_value, w_t, M, mu):
        - expectation_value: Tr(O(t) rho)
        - w_t: Evolved coefficients of O(t)
        - M: Adjoint generator matrix
        - mu: Basis moments mu_gamma = Tr(Q_gamma rho)
    """
    w_t, M, _ = evolve_operator_in_lie_closure_basis(
        basis,
        hamiltonian,
        operator,
        time=time,
        hbar=hbar,
        device=device,
    )
    mu = basis_expectations_for_density_matrix(
        basis,
        density_matrix,
        device=device,
        op_tol=op_tol,
    )
    value = expectation_value_in_lie_closure_basis(w_t, mu)
    return value, w_t, M, mu


# ============================================================
# SparseKet: A sparse state vector
# ============================================================

class SparseKet:
    """A sparse state vector representation.

    |psi> = sum_j amps[j] |indices[j]>

    Attributes:
        n (int): The number of qubits.
        indices (np.ndarray): 1D array of uint64 bitstring indices.
            LSB = site 0.
        amps (torch.Tensor): 1D complex128 tensor of amplitudes.
        device (str): The torch device where `amps` is stored.
    """
    __slots__ = ("n", "indices", "amps", "device")

    def __init__(self, n: int, indices: np.ndarray, amps: torch.Tensor,
                 device: str = "cpu"):
        """Initializes the SparseKet.

        Args:
            n: The number of qubits.
            indices: 1D uint64 numpy array of basis state indices.
            amps: 1D complex128 torch tensor of amplitudes.
            device: The torch device.
        """
        self.n = int(n)
        self.indices = np.asarray(indices, dtype=np.uint64)
        self.amps = torch.as_tensor(amps, dtype=torch.complex128, device=device)
        if self.indices.shape[0] != self.amps.shape[0]:
            raise ValueError(
                f"indices (len {self.indices.shape[0]}) and "
                f"amps (len {self.amps.shape[0]}) length mismatch."
            )
        self.device = device

    @staticmethod
    def basis(n: int, index: int, device: str = "cpu") -> "SparseKet":
        """Creates a computational basis state |index>.

        Args:
            n: Number of qubits.
            index: The basis state index (as an integer).
            device: The torch device.

        Returns:
            A SparseKet representing the basis state.
        """
        return SparseKet(
            n,
            np.array([np.uint64(index)], dtype=np.uint64),
            torch.tensor([1 + 0j], dtype=torch.complex128, device=device),
            device=device
        )

    def clone(self) -> "SparseKet":
        """Returns a deep copy of the SparseKet."""
        return SparseKet(self.n, self.indices.copy(), self.amps.clone(),
                         self.device)

    def coalesce(self, tol: float = 0.0) -> "SparseKet":
        """Merges duplicate indices by summing their amplitudes.

        Also removes any entries with |amplitude| <= tol.

        Args:
            tol: Tolerance for pruning near-zero amplitudes.

        Returns:
            A new, coalesced SparseKet.
        """
        if self.indices.size == 0:
            return self

        # Sort by index to bring duplicates together
        order = np.argsort(self.indices, kind="mergesort")
        idx_sorted = self.indices[order]
        amps_sorted = self.amps[order]

        # Find unique indices and the inverse mapping
        unique, inv = np.unique(idx_sorted, return_inverse=True)
        inv_t = torch.as_tensor(inv, dtype=torch.int64, device=self.device)

        # Perform a scatter-add to sum amplitudes for identical indices
        sums = torch.zeros((unique.shape[0],), dtype=torch.complex128,
                           device=self.device)
        sums.index_add_(0, inv_t, amps_sorted)

        # Filter out near-zero amplitudes
        if tol > 0:
            keep = torch.abs(sums) > tol
            unique = unique[keep.cpu().numpy()]
            sums = sums[keep]

        return SparseKet(self.n, unique, sums, device=self.device)

    def apply_pauli_sum_vectorized(
            self, H: "PauliSum", merge: bool = True, tol: float = 0.0
    ) -> "SparseKet":
        """Vectorized application of a PauliSum to this sparse ket.

        Computes `out = H |psi> = (sum_k P_k) |psi>`.
        Complexity is O(m * D), where m = |psi| sparsity and D = |H| sparsity.
        No (2^n) sized objects are created.

        Vectorized steps:
        1. new_indices[m, D] = indices[m, 1] ^ x_masks[1, D]
        2. signs[m, D] = (-1)^{popcount(indices[m, 1] & z_masks[1, D])}
        3. new_amps[m, D] = amps[m, 1] * pref[1, D] * signs[m, D]
        4. Flatten new_indices and new_amps and coalesce.

        Args:
            H: The PauliSum operator to apply.
            merge: If True, coalesce the result.
            tol: Tolerance for pruning amplitudes during coalescence.

        Returns:
            A new SparseKet representing H |psi>.
        """
        if not H.terms:
            return self.clone()
        if not self.indices.size:
            return SparseKet(self.n, np.array([], dtype=np.uint64),
                             torch.tensor([], dtype=torch.complex128),
                             self.device)

        x_masks, z_masks, pref = H.to_masks_and_coeffs()  # (D,), (D,), (D,)
        m = self.indices.shape[0]
        D = x_masks.shape[0]

        # --- 1. Compute new indices (broadcast) ---
        # new_idx[m, D] = b_i ^ x_k
        new_idx = (self.indices[:, None] ^ x_masks[None, :]).reshape(-1)

        # --- 2. Compute signs (broadcast) ---
        # signs = (-1)^{popcount(b_i & z_k)}
        if np.any(z_masks != 0):
            # Use numpy's fast popcount
            parity = (np.bitwise_count(
                self.indices[:, None] & z_masks[None, :]) & 1
            ).astype(np.int8)  # (m, D)
            signs_np = (1 - 2 * parity).astype(np.int8)  # in {+1, -1}
        else:
            # Optimization: if all z_masks are 0, all signs are +1
            signs_np = np.ones((m, D), dtype=np.int8)

        # --- 3. Compute new amplitudes (torch broadcast) ---
        pref_t = torch.as_tensor(pref, dtype=torch.complex128,
                                 device=self.device).view(1, D)
        signs_t = torch.as_tensor(signs_np, dtype=torch.complex128,
                                  device=self.device)

        # amps2d[m, D] = amp_i * pref_k * sign_ik
        amps2d = self.amps.view(m, 1) * pref_t * signs_t
        new_amps = amps2d.reshape(-1)  # (m*D,)

        # --- 4. Create new SparseKet and coalesce ---
        out = SparseKet(self.n, new_idx, new_amps, device=self.device)
        return out.coalesce(tol=tol) if merge else out

    def norm(self) -> float:
        """Computes the L2 norm ||psi||_2 = sqrt(sum |amps|^2)."""
        if self.amps.numel() == 0:
            return 0.0
        return float(torch.sqrt(torch.sum(torch.abs(self.amps) ** 2)).item())

    def normalize_(self, eps: float = 1e-12) -> "SparseKet":
        """In-place normalization.

        Does nothing if the norm is <= eps.

        Args:
            eps: Tolerance threshold below which normalization is skipped.

        Returns:
            The normalized SparseKet (self).
        """
        nrm = self.norm()
        if nrm > eps:
            self.amps /= nrm
        return self

    def measure_pauli(
            self,
            operator: Union[PauliString, PauliSum],
            op_tol: float = 1e-12
        ) -> complex:
        """Computes the expectation value <psi| H |psi>.

        This is done efficiently by:
        1. Computing |phi> = H |psi> using `apply_pauli_sum_vectorized`.
        2. Computing the inner product <psi|phi> using a sparse dot product.

        Note: The operator H is not assumed to be Hermitian, so the
        result can be a complex number.

        Args:
            operator: The PauliString or PauliSum (H) to measure.
            op_tol: Tolerance for coalescing the operator and intermediate
                state vector.

        Returns:
            The complex expectation value <psi| H |psi>.

        Raises:
            TypeError: If 'operator' is not a PauliString or PauliSum.
        """
        if isinstance(operator, PauliString):
            H = PauliSum([operator])
        elif isinstance(operator, PauliSum):
            H = operator
        else:
            raise TypeError(
                "Operator must be a PauliString or PauliSum, got "
                f"{type(operator)}"
            )

        H = H.simplify(tol=op_tol)
        if not H.terms or self.indices.size == 0:
            return 0.0 + 0.0j

        phi_ket = self.apply_pauli_sum_vectorized(H, merge=True, tol=op_tol)

        if phi_ket.indices.size == 0:
            return 0.0 + 0.0j

        psi_c = self.coalesce(tol=op_tol)
        if psi_c.indices.size == 0:
            return 0.0 + 0.0j

        amps_psi = psi_c.amps
        idx_psi = psi_c.indices
        amps_phi = phi_ket.amps
        idx_phi = phi_ket.indices

        m_psi = idx_psi.shape[0]
        m_phi = idx_phi.shape[0]
        i = 0
        j = 0

        total_exp_val = torch.tensor(
            0.0 + 0.0j, dtype=torch.complex128, device=self.device
        )

        while i < m_psi and j < m_phi:
            if idx_psi[i] < idx_phi[j]:
                i += 1
            elif idx_psi[i] > idx_phi[j]:
                j += 1
            else:
                total_exp_val += amps_psi[i].conj() * amps_phi[j]
                i += 1
                j += 1

        return total_exp_val.item()
