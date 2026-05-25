"""Jordan-Wigner helpers for the Pauli-string backend."""

from __future__ import annotations

from .pauli_class import (
    PauliString,
    PauliSum,
    multiply_pauli_factors,
)


def _identity_label(num_modes: int) -> str:
    return "I" * num_modes


def _validate_mode(mode: int, num_modes: int) -> None:
    if not (0 <= mode < num_modes):
        raise IndexError(
            f"Fermionic mode index {mode} out of bounds for {num_modes} modes."
        )


def _normalize_fermion_operator_name(name: str) -> str:
    normalized = name.strip().lower()
    aliases = {
        "c": "c",
        "annihilation": "c",
        "destroy": "c",
        "cdag": "c_dag",
        "c_dag": "c_dag",
        "create": "c_dag",
        "creation": "c_dag",
        "n": "n",
        "number": "n",
        "i": "i",
        "identity": "i",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported fermion operator '{name}'. "
            "Expected one of c, c_dag/cdag, n, I."
        )
    return aliases[normalized]


def _normalize_spin_label(spin: str) -> str:
    normalized = spin.strip().lower()
    aliases = {
        "up": "up",
        "uparrow": "up",
        "dn": "down",
        "down": "down",
        "downarrow": "down",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported spin label '{spin}'. Expected 'up' or 'down'/'dn'."
        )
    return aliases[normalized]


def jw_spinless_fermion_operator(
        operator: str,
        mode: int,
        num_modes: int,
        *,
        convention: str = "down=1",
) -> PauliSum:
    """Maps a spinless fermion operator to a PauliSum via Jordan-Wigner."""
    if convention != "down=1":
        raise NotImplementedError(
            "Jordan-Wigner Pauli backend currently supports only "
            "convention='down=1'."
        )
    _validate_mode(mode, num_modes)
    op = _normalize_fermion_operator_name(operator)

    if op == "i":
        return PauliSum([PauliString(_identity_label(num_modes), 1.0)])

    if op == "n":
        label_z = ["I"] * num_modes
        label_z[mode] = "Z"
        return PauliSum([
            PauliString(_identity_label(num_modes), 0.5),
            PauliString("".join(label_z), -0.5),
        ])

    label_x = ["I"] * num_modes
    label_y = ["I"] * num_modes
    for j in range(mode):
        label_x[j] = "Z"
        label_y[j] = "Z"
    label_x[mode] = "X"
    label_y[mode] = "Y"

    if op == "c":
        return PauliSum([
            PauliString("".join(label_x), 0.5),
            PauliString("".join(label_y), 0.5j),
        ])

    return PauliSum([
        PauliString("".join(label_x), 0.5),
        PauliString("".join(label_y), -0.5j),
    ])


def jw_spinful_mode_index(
        site: int,
        spin: str,
        physical_lattice_length: int,
) -> int:
    """Returns the Jordan-Wigner mode index for a spinful fermion."""
    if not (0 <= site < physical_lattice_length):
        raise IndexError(
            f"Physical site index {site} out of bounds for "
            f"length {physical_lattice_length}."
        )
    return 2 * site + (0 if _normalize_spin_label(spin) == "up" else 1)


def jw_spinful_fermion_operator(
        operator: str,
        site: int,
        spin: str,
        physical_lattice_length: int,
        *,
        convention: str = "down=1",
) -> PauliSum:
    """Maps a spinful fermion operator to a PauliSum via Jordan-Wigner."""
    mode = jw_spinful_mode_index(site, spin, physical_lattice_length)
    return jw_spinless_fermion_operator(
        operator,
        mode,
        2 * physical_lattice_length,
        convention=convention,
    )


def jw_spinless_fermion_product(
        operators: list[str],
        sites: list[int],
        num_modes: int,
        *,
        convention: str = "down=1",
        tol: float = 1e-12,
) -> PauliSum:
    """Maps an ordered spinless fermion monomial to a PauliSum."""
    if len(operators) != len(sites):
        raise ValueError(
            "operators and sites must have the same length, got "
            f"{len(operators)} and {len(sites)}."
        )
    factors = [
        jw_spinless_fermion_operator(
            operator,
            site,
            num_modes,
            convention=convention,
        )
        for operator, site in zip(operators, sites)
    ]
    return multiply_pauli_factors(factors, tol=tol)


def jw_spinful_fermion_product(
        operators: list[str],
        sites: list[int],
        spins: list[str],
        physical_lattice_length: int,
        *,
        convention: str = "down=1",
        tol: float = 1e-12,
) -> PauliSum:
    """Maps an ordered spinful fermion monomial to a PauliSum."""
    if len(operators) != len(sites) or len(operators) != len(spins):
        raise ValueError(
            "operators, sites, and spins must have the same length, got "
            f"{len(operators)}, {len(sites)}, and {len(spins)}."
        )
    factors = [
        jw_spinful_fermion_operator(
            operator,
            site,
            spin,
            physical_lattice_length,
            convention=convention,
        )
        for operator, site, spin in zip(operators, sites, spins)
    ]
    return multiply_pauli_factors(factors, tol=tol)

