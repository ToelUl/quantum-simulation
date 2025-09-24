from .operation import nested_kronecker_product, dagger
from typing import Union, Literal, Tuple
import torch
import numpy as np

def pauli_z():
    return torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

def jordan_wigner_transform_1d_spin_half_local_to_global(j, lattice_length, local_op, convention = "down=1"):
    if lattice_length == 1:
        return local_op
    else:
        operators_list = []
        if convention == "down=1":
            for k in range(j):
                operators_list.append(pauli_z())
        if convention == "up=1":
            for k in range(j):
                operators_list.append(-pauli_z())
        operators_list.append(local_op)
        for k in range(lattice_length-j-1):
            operators_list.append(torch.eye(2))
        return nested_kronecker_product(operators_list)

def jordan_wigner_transform_1d_spin_half_global_to_global(j, lattice_length, global_op, convention = "down=1"):
    if lattice_length == 1:
        return global_op
    else:
        operators_list = []
        if convention == "down=1":
            for k in range(j):
                operators_list.append(pauli_z())
        if convention == "up=1":
            for k in range(j):
                operators_list.append(-pauli_z())
        operators_list.append(torch.eye(2))
        for k in range(lattice_length-j-1):
            operators_list.append(torch.eye(2))
        return nested_kronecker_product(operators_list) @ global_op

def lattice_fourier_transform_1d(k, lattice_length, operator, phi=0, convention="down=1", err=1e-15):
    c = 0
    for j in range(lattice_length):
        c += np.exp(-1j*(k*(j+1)))*operator(j, lattice_length, convention)
    c_i_to_c_k = c*np.exp(-1j*phi)/np.sqrt(lattice_length)
    for i in range(c_i_to_c_k.shape[0]):
        for j in range(c_i_to_c_k.shape[1]):
            a = c_i_to_c_k[i][j] * c_i_to_c_k[i][j].conj()
            if a.real <= err:
                c_i_to_c_k[i][j] = 0
    return c_i_to_c_k

def lattice_inverse_fourier_transform_1d(j, lattice_length, operator, phi=0, convention="down=1", ABC_mode=False, err=1e-15):
    if ABC_mode:
        n_list = [i for i in range(1, int(lattice_length / 2) + 1, 1)]
        k_list_positive_part = [(2 * n - 1) * np.pi / lattice_length for n in n_list]
        k_list = [-i for i in k_list_positive_part[::-1]] + k_list_positive_part
    else:
        n_list = [(-lattice_length/2)+1+i for i in range(lattice_length)]
        k_list = [2*n*np.pi/lattice_length for n in n_list]

    c = 0
    for k in k_list:
        c += np.exp(1j*(j+1)*k)*operator(k, lattice_length, phi, convention)
    c_k_to_c_i = c*np.exp(1j*phi)/np.sqrt(lattice_length)
    for i in range(c_k_to_c_i.shape[0]):
        for j in range(c_k_to_c_i.shape[1]):
            a = c_k_to_c_i[i][j] * c_k_to_c_i[i][j].conj()
            if a.real <= err:
                c_k_to_c_i[i][j] = 0
    return c_k_to_c_i

def to_global_operator(j, lattice_length, local_operator: torch.Tensor):
    I = torch.eye(local_operator.shape[0])
    if lattice_length == 1:
        return local_operator
    else:
        operators = []
        for _ in range(j):
            operators.append(I)
        operators.append(local_operator)
        for _ in range(lattice_length - j - 1):
            operators.append(I)
        return nested_kronecker_product(operators)


