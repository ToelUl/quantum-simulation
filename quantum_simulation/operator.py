from .group import SU2Group
from .operation import (
    dagger,
    nested_kronecker_product
)
from .transform import (
    jordan_wigner_transform_1d_spin_half_local_to_global,
    jordan_wigner_transform_1d_spin_half_global_to_global,
    lattice_fourier_transform_1d,
    lattice_inverse_fourier_transform_1d,
    to_global_operator,
)
import torch


def pauli_x(spin=0.5):
    """Returns the Pauli-X matrix for a given spin."""
    spin_group = SU2Group(spin=spin)
    return spin_group.algebra.sigma_x


def pauli_y(spin=0.5):
    """Returns the Pauli-Y matrix for a given spin."""
    spin_group = SU2Group(spin=spin)
    return spin_group.algebra.sigma_y


def pauli_z(spin=0.5):
    """Returns the Pauli-Z matrix for a given spin."""
    spin_group = SU2Group(spin=spin)
    return spin_group.algebra.sigma_z


def S_x(spin=0.5):
    """Returns the spin-X operator for a given spin."""
    spin_group = SU2Group(spin=spin)
    return spin_group.algebra.S_x


def S_y(spin=0.5):
    """Returns the spin-Y operator for a given spin."""
    spin_group = SU2Group(spin=spin)
    return spin_group.algebra.S_y


def S_z(spin=0.5):
    """Returns the spin-Z operator for a given spin."""
    spin_group = SU2Group(spin=spin)
    return spin_group.algebra.S_z


def S_p(spin=0.5):
    """Returns the spin raising operator for a given spin."""
    spin_group = SU2Group(spin=spin)
    return spin_group.algebra.S_plus


def S_m(spin=0.5):
    """Returns the spin lowering operator for a given spin."""
    spin_group = SU2Group(spin=spin)
    return spin_group.algebra.S_minus


def identity(spin=0.5):
    """Returns the identity operator for a given spin."""
    spin_group = SU2Group(spin=spin)
    return spin_group.identity


def time_reversal_operator_u(spin=0.5):
    """Returns the time-reversal operator U for a given spin."""
    spin_group = SU2Group(spin=spin)
    return spin_group.exponential_map(torch.tensor([0.0, torch.pi, 0.0]))


def b_(n_state=2, convention="down=1"):
    """hard-core boson annihilation operator"""
    if convention == "down=1":
        return S_p((n_state-1)/2)
    if convention == "up=1":
        return S_m((n_state-1)/2)


def b_dag(n_state=2, convention="down=1"):
    """hard-core boson creation operator"""
    if convention == "down=1":
        return S_m((n_state-1)/2)
    if convention == "up=1":
        return S_m((n_state-1)/2)


def c_j(j, lattice_length=2 ,convention="down=1"):
    """Annihilation operator on site j"""
    return jordan_wigner_transform_1d_spin_half_local_to_global(
        j, lattice_length, b_(convention=convention), convention)


def c_dag_j(j, lattice_length=2 ,convention="down=1"):
    """Creation operator on site j"""
    return jordan_wigner_transform_1d_spin_half_local_to_global(
        j, lattice_length, b_dag(convention=convention), convention)


def majorana_alpha_j(j, lattice_length=2, convention="down=1"):
    """Majorana operator alpha_j = c_j + c_j^dagger"""
    return c_j(j, lattice_length, convention) + c_dag_j(j, lattice_length, convention)


def majorana_beta_j(j, lattice_length=2, convention="down=1"):
    """Majorana operator beta_j = -i(c_j - c_j^dagger)"""
    return -1j * (c_j(j, lattice_length, convention) - c_dag_j(j, lattice_length, convention))


def c_k(k, lattice_length=2, phi=0.0, convention="down=1"):
    """Annihilation operator in momentum space"""
    return lattice_fourier_transform_1d(k, lattice_length, c_j, phi=phi, convention=convention)


def c_dag_k(k, lattice_length=2, phi=0.0, convention="down=1"):
    """Creation operator in momentum space"""
    return dagger(c_k(k, lattice_length, phi, convention))


def b_j(j, lattice_length=2, n_state=2 ,convention="down=1"):
    """Annihilation operator of hard-core boson on site j"""
    return to_global_operator(j,lattice_length, b_(n_state=n_state, convention=convention))


def b_dag_j(j, lattice_length=2, n_state=2  ,convention="down=1"):
    """Creation operator of hard-core boson on site j"""
    return to_global_operator(j,lattice_length, b_dag(n_state=n_state, convention=convention))


def n_j(j, lattice_length=2 ,convention="down=1"):
    """number operator on site j"""
    return b_dag_j(j, lattice_length=lattice_length, convention=convention) \
           @ b_j(j, lattice_length=lattice_length, convention=convention)


def total_number_operator(lattice_length=2, convention="down=1"):
    """total number operator"""
    if lattice_length < 1:
        raise ValueError("lattice_length >= 1")
    n = n_j(0, lattice_length, convention)
    for i in range(1, lattice_length, 1):
        n += n_j(i, lattice_length, convention)
    return n


def c_j_spinful(i, spin_up_down, lattice_length, convention="down=1"):
    """
    Annihilation operator for a fermion at physical site <i> with spin <spin>.
    'spin' can be 'up' (0) or 'down' (1).
    """
    spin_offset = 0 if spin_up_down == 'up' else 1
    k = 2 * i + spin_offset
    total_length = 2 * lattice_length

    return jordan_wigner_transform_1d_spin_half_local_to_global(
        k, total_length, b_(convention=convention), convention)


def c_dag_j_spinful(i, spin_up_down, lattice_length, convention="down=1"):
    """
    Creation operator for a fermion at physical site <i> with spin <spin>.
    'spin' can be 'up' (0) or 'down' (1).
    """
    spin_offset = 0 if spin_up_down == 'up' else 1
    k = 2 * i + spin_offset
    total_length = 2 * lattice_length

    return jordan_wigner_transform_1d_spin_half_local_to_global(
        k, total_length, b_dag(convention=convention), convention)


def n_j_spinful(i, spin, lattice_length, convention="down=1"):
    """
    Number operator for a fermion at physical site <i> with spin <spin>.
    """
    c_dag = c_dag_j_spinful(i, spin, lattice_length, convention)
    c = c_j_spinful(i, spin, lattice_length, convention)
    return c_dag @ c


def neg_1_powered_by_n_operator(lattice_length=2):
    """(-1)^(total number operator)"""
    return jordan_wigner_transform_1d_spin_half_local_to_global(
        lattice_length-1, lattice_length, pauli_z(), convention="down=1")


def fermionic_spinor(
    k: float,
    lattice_length: int,
    phi: float = 0.0,
    convention: str = "down=1"
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Constructs the fermionic two-component spinor for a given momentum k.

    The spinor is:
    $$ \hat{\Psi}_k = \begin{pmatrix} \hat{c}_k \\ \hat{c}_{-k}^\dagger \end{pmatrix} $$

    Args:
        k (float): The wave vector (momentum).
        lattice_length (int): The number of sites in the lattice.
        phi (float, optional): Phase factor for the Fourier transform. Defaults to 0.0.
        convention (str, optional): The Jordan-Wigner convention. Defaults to "down=1".

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the operators
        (c_k, c_dag_{-k}).
    """
    op_ck = c_k(k, lattice_length, phi, convention)
    op_c_dag_mk = c_dag_k(-k, lattice_length, phi, convention)
    return op_ck, op_c_dag_mk


def fermionic_spinor_dagger(
    k: float,
    lattice_length: int,
    phi: float = 0.0,
    convention: str = "down=1"
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Constructs the dagger of the fermionic two-component spinor.

    The dagger spinor is:
    $$ \hat{\Psi}_k^\dagger = (\hat{c}_k^\dagger, \hat{c}_{-k}) $$

    Args:
        k (float): The wave vector (momentum).
        lattice_length (int): The number of sites in the lattice.
        phi (float, optional): Phase factor for the Fourier transform. Defaults to 0.0.
        convention (str, optional): The Jordan-Wigner convention. Defaults to "down=1".

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the operators
        (c_dag_k, c_{-k}).
    """
    op_c_dag_k = c_dag_k(k, lattice_length, phi, convention)
    op_c_mk = c_k(-k, lattice_length, phi, convention)
    return op_c_dag_k, op_c_mk