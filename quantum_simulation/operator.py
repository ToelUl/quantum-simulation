from .group import SU2Group
from .operation import (
    dagger,
    nested_kronecker_product,
    nested_matrix_product,
)
from .transform import (
    jordan_wigner_transform_1d_spin_half_local_to_global,
    jordan_wigner_transform_1d_spin_half_global_to_global,
    lattice_fourier_transform_1d,
    lattice_inverse_fourier_transform_1d,
    to_global_operator,
)
from typing import List
import torch


def pauli_x(spin=0.5) -> torch.Tensor:
    """Returns the Pauli-X matrix for a given spin.

    Args:
        spin (float): The spin value (default is 0.5).

    Returns:
        torch.Tensor: The Pauli-X matrix.
    """
    spin_group = SU2Group(spin=spin)
    return spin_group.algebra.sigma_x


def pauli_y(spin=0.5) -> torch.Tensor:
    """Returns the Pauli-Y matrix for a given spin.

    Args:
        spin (float): The spin value (default is 0.5).

    Returns:
        torch.Tensor: The Pauli-Y matrix.
    """
    spin_group = SU2Group(spin=spin)
    return spin_group.algebra.sigma_y


def pauli_z(spin=0.5) -> torch.Tensor:
    """Returns the Pauli-Z matrix for a given spin.

    Args:
        spin (float): The spin value (default is 0.5).

    Returns:
        torch.Tensor: The Pauli-Z matrix.
    """
    spin_group = SU2Group(spin=spin)
    return spin_group.algebra.sigma_z


def pauli_string(pauli_str: str, spin: float = 0.5) -> torch.Tensor:
    """Constructs a Pauli string operator from a list of Pauli matrices.

    Args:
        pauli_str (str): A string representing the Pauli operators (e.g., 'XZYI').
        spin (float): The spin value (default is 0.5).

    Returns:
        torch.Tensor: The resulting Pauli string operator.
    """
    pauli_dict = {
        'I': identity(spin),
        'X': pauli_x(spin),
        'Y': pauli_y(spin),
        'Z': pauli_z(spin)
    }
    operators = [pauli_dict[p] for p in pauli_str]
    return nested_kronecker_product(operators)


def global_pauli_op_chain(op_type: str, lattice_length: int, spin: float = 0.5, pbc: bool = True) -> torch.Tensor:
    """Constructs a global Pauli operator on a 1D lattice.
    Args:
        op_type (str): A string representing the type of Pauli operator
            (e.g., 'X', 'Y', 'Z', 'XX', 'YY', 'ZZ', 'XXX', ...).
        lattice_length (int): The length of the 1D lattice.
        spin (float): The spin value (default is 0.5).
        pbc (bool): Whether to apply periodic boundary conditions (default is True).

    Returns:
        torch.Tensor: The resulting global Pauli operator.
    """
    pauli_dict = {
        'I': identity(spin),
        'X': pauli_x(spin),
        'Y': pauli_y(spin),
        'Z': pauli_z(spin)
    }
    op_len = len(op_type)

    if lattice_length < op_len:
        raise ValueError("lattice_length must be greater than or equal to the length of op_type")

    if lattice_length == op_len and not pbc:
        temp_op_list = [to_global_operator(position, lattice_length, pauli_dict[op])
                        for position, op in enumerate(op_type)]
        if len(temp_op_list) == 1:
            return temp_op_list[0]
        else:
            return nested_matrix_product(temp_op_list)

    else:
        global_op = torch.zeros((int((2 * spin + 1) ** lattice_length), int((2 * spin + 1) ** lattice_length)),
                                dtype=torch.complex64)
        if pbc:
            num_terms = lattice_length
        else:
            num_terms = lattice_length - op_len + 1

        for i in range(num_terms):
            temp_op_list = []
            for position, op in enumerate(op_type):
                if pbc:
                    site_index = (position + i) % lattice_length
                else:
                    site_index = position + i

                temp_op_list.append(to_global_operator(site_index, lattice_length, pauli_dict[op]))

            if len(temp_op_list) == 1:
                temp_op = temp_op_list[0]
            else:
                temp_op = nested_matrix_product(temp_op_list)

            global_op += temp_op

        return global_op


def global_pauli_op_chain_list(op_type: str, lattice_length: int, spin: float = 0.5, pbc: bool = True) -> List[torch.Tensor]:
    """Constructs a list of global Pauli operators on a 1D lattice.
    Args:
        op_type (str): A string representing the type of Pauli operator
            (e.g., 'X', 'Y', 'Z', 'XX', 'YY', 'ZZ', 'XXX', ...).
        lattice_length (int): The length of the 1D lattice.
        spin (float): The spin value (default is 0.5).
        pbc (bool): Whether to apply periodic boundary conditions (default is True).

    Returns:
        List[torch.Tensor]: A list of resulting global Pauli operators.
        e.g., for op_type='XX' and lattice_length=4, it returns [X1X2, X2X3, X3X4].
    """
    pauli_dict = {
        'I': identity(spin),
        'X': pauli_x(spin),
        'Y': pauli_y(spin),
        'Z': pauli_z(spin)
    }
    op_len = len(op_type)

    if lattice_length < op_len:
        raise ValueError("lattice_length must be greater than or equal to the length of op_type")

    if lattice_length == op_len and not pbc:
        temp_op_list = [to_global_operator(position, lattice_length, pauli_dict[op])
                        for position, op in enumerate(op_type)]
        if len(temp_op_list) == 1:
            return temp_op_list
        else:
            return [nested_matrix_product(temp_op_list)]

    else:
        global_op = []
        if pbc:
            num_terms = lattice_length
        else:
            num_terms = lattice_length - op_len + 1

        for i in range(num_terms):
            temp_op_list = []
            for position, op in enumerate(op_type):
                if pbc:
                    site_index = (position + i) % lattice_length
                else:
                    site_index = position + i

                temp_op_list.append(to_global_operator(site_index, lattice_length, pauli_dict[op]))

            if len(temp_op_list) == 1:
                temp_op = temp_op_list[0]
            else:
                temp_op = nested_matrix_product(temp_op_list)

            global_op.append(temp_op)

        return global_op


def S_x(spin=0.5) -> torch.Tensor:
    """Returns the spin-X operator for a given spin.

    Args:
        spin (float): The spin value (default is 0.5).

    Returns:
        torch.Tensor: The spin-X operator.
    """
    spin_group = SU2Group(spin=spin)
    return spin_group.algebra.S_x


def S_y(spin=0.5) -> torch.Tensor:
    """Returns the spin-Y operator for a given spin.

    Args:
        spin (float): The spin value (default is 0.5).

    Returns:
        torch.Tensor: The spin-Y operator.
    """
    spin_group = SU2Group(spin=spin)
    return spin_group.algebra.S_y


def S_z(spin=0.5) -> torch.Tensor:
    """Returns the spin-Z operator for a given spin.

    Args:
        spin (float): The spin value (default is 0.5).

    Returns:
        torch.Tensor: The spin-Z operator.
    """
    spin_group = SU2Group(spin=spin)
    return spin_group.algebra.S_z


def S_p(spin=0.5) -> torch.Tensor:
    """Returns the spin raising operator for a given spin.

    Args:
        spin (float): The spin value (default is 0.5).

    Returns:
        torch.Tensor: The spin raising operator.
    """
    spin_group = SU2Group(spin=spin)
    return spin_group.algebra.S_plus


def S_m(spin=0.5) -> torch.Tensor:
    """Returns the spin lowering operator for a given spin.

    Args:
        spin (float): The spin value (default is 0.5).

    Returns:
        torch.Tensor: The spin lowering operator.
    """
    spin_group = SU2Group(spin=spin)
    return spin_group.algebra.S_minus


def identity(spin=0.5) -> torch.Tensor:
    """Returns the identity operator for a given spin.

    Args:
        spin (float): The spin value (default is 0.5).

    Returns:
        torch.Tensor: The identity operator.
    """
    spin_group = SU2Group(spin=spin)
    return spin_group.identity


def time_reversal_operator_u(spin=0.5) -> torch.Tensor:
    """Returns the time-reversal operator U for a given spin.

    Args:
        spin (float): The spin value (default is 0.5).

    Returns:
        torch.Tensor: The time-reversal operator U.
    """
    spin_group = SU2Group(spin=spin)
    return spin_group.exponential_map(torch.tensor([0.0, torch.pi, 0.0]))


def b_(n_state=2, convention="down=1") -> torch.Tensor:
    """hard-core boson annihilation operator

    Args:
        n_state (int): number of states (default is 2).
        convention (str): Whether "down=1" or "up=1" (default is "down=1").

    Returns:
        torch.Tensor: The hard-core boson annihilation operator.
    """
    if convention == "down=1":
        return S_p((n_state-1)/2)
    if convention == "up=1":
        return S_m((n_state-1)/2)


def b_dag(n_state=2, convention="down=1") -> torch.Tensor:
    """hard-core boson creation operator

    Args:
        n_state (int): number of states (default is 2).
        convention (str): Whether "down=1" or "up=1" (default is "down=1").

    Returns:
        torch.Tensor: The hard-core boson creation operator.
    """
    if convention == "down=1":
        return S_m((n_state-1)/2)
    if convention == "up=1":
        return S_m((n_state-1)/2)


def c_j(j, lattice_length=2 ,convention="down=1") -> torch.Tensor:
    """Annihilation operator on site j

    Args:
        j (int): site index.
        lattice_length (int): number of sites in the lattice (default is 2).
        convention (str): Whether "down=1" or "up=1" (default is "down=1").

    Returns:
        torch.Tensor: The annihilation operator on site j.
    """
    return jordan_wigner_transform_1d_spin_half_local_to_global(
        j, lattice_length, b_(convention=convention), convention)


def c_dag_j(j, lattice_length=2 ,convention="down=1") -> torch.Tensor:
    """Creation operator on site j

    Args:
        j (int): site index.
        lattice_length (int): number of sites in the lattice (default is 2).
        convention (str): Whether "down=1" or "up=1" (default is "down=1").

    Returns:
        torch.Tensor: The creation operator on site j.
    """
    return jordan_wigner_transform_1d_spin_half_local_to_global(
        j, lattice_length, b_dag(convention=convention), convention)


def majorana_alpha_j(j, lattice_length=2, convention="down=1") -> torch.Tensor:
    """Majorana operator alpha_j = c_j + c_j^dagger

    Args:
        j (int): site index.
        lattice_length (int): number of sites in the lattice (default is 2).
        convention (str): Whether "down=1" or "up=1" (default is "down=1").

    Returns:
        torch.Tensor: The Majorana operator alpha_j.
    """
    return c_j(j, lattice_length, convention) + c_dag_j(j, lattice_length, convention)


def majorana_beta_j(j, lattice_length=2, convention="down=1") -> torch.Tensor:
    """Majorana operator beta_j = -i(c_j - c_j^dagger)

    Args:
        j (int): site index.
        lattice_length (int): number of sites in the lattice (default is 2).
        convention (str): Whether "down=1" or "up=1" (default is "down=1").

    Returns:
        torch.Tensor: The Majorana operator beta_j.
    """
    return -1j * (c_j(j, lattice_length, convention) - c_dag_j(j, lattice_length, convention))


def c_k(k, lattice_length=2, phi=0.0, convention="down=1") -> torch.Tensor:
    """Annihilation operator in momentum space

    Args:
        k (float): wave vector.
        lattice_length (int): number of sites in the lattice (default is 2).
        phi (float): phase factor for Fourier transform (default is 0.0).
        convention (str): Whether "down=1" or "up=1" (default is "down=1").

    Returns:
        torch.Tensor: The annihilation operator in momentum space.
    """
    return lattice_fourier_transform_1d(k, lattice_length, c_j, phi=phi, convention=convention)


def c_dag_k(k, lattice_length=2, phi=0.0, convention="down=1") -> torch.Tensor:
    """Creation operator in momentum space

    Args:
        k (float): wave vector.
        lattice_length (int): number of sites in the lattice (default is 2).
        phi (float): phase factor for Fourier transform (default is 0.0).
        convention (str): Whether "down=1" or "up=1" (default is "down=1").

    Returns:
        torch.Tensor: The creation operator in momentum space.
    """
    return dagger(c_k(k, lattice_length, phi, convention))


def b_j(j, lattice_length=2, n_state=2 ,convention="down=1") -> torch.Tensor:
    """Annihilation operator of hard-core boson on site j

    Args:
        j (int): site index.
        lattice_length (int): number of sites in the lattice (default is 2).
        n_state (int): number of states (default is 2).
        convention (str): Whether "down=1" or "up=1" (default is "down=1").

    Returns:
        torch.Tensor: The annihilation operator of hard-core boson on site j.
    """
    return to_global_operator(j,lattice_length, b_(n_state=n_state, convention=convention))


def b_dag_j(j, lattice_length=2, n_state=2  ,convention="down=1") -> torch.Tensor:
    """Creation operator of hard-core boson on site j

    Args:
        j (int): site index.
        lattice_length (int): number of sites in the lattice (default is 2).
        n_state (int): number of states (default is 2).
        convention (str): Whether "down=1" or "up=1" (default is "down=1").

    Returns:
        torch.Tensor: The creation operator of hard-core boson on site j.
    """
    return to_global_operator(j,lattice_length, b_dag(n_state=n_state, convention=convention))


def n_j(j, lattice_length=2 ,convention="down=1") -> torch.Tensor:
    """number operator on site j

    Args:
        j (int): site index.
        lattice_length (int): number of sites in the lattice (default is 2).
        convention (str): Whether "down=1" or "up=1" (default is "down=1").

    Returns:
        torch.Tensor: The number operator on site j.
    """
    return b_dag_j(j, lattice_length=lattice_length, convention=convention) \
           @ b_j(j, lattice_length=lattice_length, convention=convention)


def total_number_operator(lattice_length=2, convention="down=1") -> torch.Tensor:
    """total number operator

    Args:
        lattice_length (int): number of sites in the lattice (default is 2).
        convention (str): Whether "down=1" or "up=1" (default is "down=1").

    Returns:
        torch.Tensor: The total number operator.
    """
    if lattice_length < 1:
        raise ValueError("lattice_length >= 1")
    n = n_j(0, lattice_length, convention)
    for i in range(1, lattice_length, 1):
        n += n_j(i, lattice_length, convention)
    return n


def c_j_spinful(i, spin_up_down, lattice_length, convention="down=1") -> torch.Tensor:
    """
    Annihilation operator for a fermion at physical site <i> with spin <spin>.
    'spin' can be 'up' (0) or 'down' (1).

    Args:
        i (int): site index.
        spin_up_down (str): 'up' or 'down'.
        lattice_length (int): number of sites in the lattice.
        convention (str): Whether "down=1" or "up=1" (default is "down=1").

    Returns:
        torch.Tensor: The annihilation operator for a fermion at site i with spin.
    """
    spin_offset = 0 if spin_up_down == 'up' else 1
    k = 2 * i + spin_offset
    total_length = 2 * lattice_length

    return jordan_wigner_transform_1d_spin_half_local_to_global(
        k, total_length, b_(convention=convention), convention)


def c_dag_j_spinful(i, spin_up_down, lattice_length, convention="down=1") -> torch.Tensor:
    """
    Creation operator for a fermion at physical site <i> with spin <spin>.
    'spin' can be 'up' (0) or 'down' (1).

    Args:
        i (int): site index.
        spin_up_down (str): 'up' or 'down'.
        lattice_length (int): number of sites in the lattice.
        convention (str): Whether "down=1" or "up=1" (default is "down=1").

    Returns:
        torch.Tensor: The creation operator for a fermion at site i with spin.
    """
    spin_offset = 0 if spin_up_down == 'up' else 1
    k = 2 * i + spin_offset
    total_length = 2 * lattice_length

    return jordan_wigner_transform_1d_spin_half_local_to_global(
        k, total_length, b_dag(convention=convention), convention)


def n_j_spinful(i, spin, lattice_length, convention="down=1") -> torch.Tensor:
    """
    Number operator for a fermion at physical site <i> with spin <spin>.

    Args:
        i (int): site index.
        spin (str): 'up' or 'down'.
        lattice_length (int): number of sites in the lattice.
        convention (str): Whether "down=1" or "up=1" (default is "down=1").

    Returns:
        torch.Tensor: The number operator for a fermion at site i with spin.
    """
    c_dag = c_dag_j_spinful(i, spin, lattice_length, convention)
    c = c_j_spinful(i, spin, lattice_length, convention)
    return c_dag @ c


def neg_1_powered_by_n_operator(lattice_length=2) -> torch.Tensor:
    """(-1)^(total number operator)

    Args:
        lattice_length (int): number of sites in the lattice (default is 2).

    Returns:
        torch.Tensor: The operator (-1)^(total number operator).
    """
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