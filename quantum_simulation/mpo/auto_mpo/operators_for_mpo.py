#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A collection of utility functions to generate quantum mechanical operators.

This module provides functions for creating matrix representations of common
operators used in quantum mechanics, such as spin operators for arbitrary spin-S
systems and operators for hard-core bosons.
"""

import numpy as np

def generate_mpo_spin_operators(spin_dim=2, operator_name=None):
    """Generates spin operator matrices for a given local Hilbert space dimension.

    The total spin `S` is determined from the dimension `d` via S = (d - 1) / 2.
    For example, `spin_dim=2` corresponds to S=1/2, and `spin_dim=3` corresponds
    to S=1.

    Args:
        spin_dim (int): The dimension of the local Hilbert space. Must be >= 2.
        operator_name (str, optional): The name of the specific operator to
            return (e.g., 'Sz', 'Sp', 'Id'). If None, a dictionary containing
            all generated operators is returned.

    Returns:
        np.ndarray | dict[str, np.ndarray]: A NumPy array for the specified
            operator, or a dictionary of all operators if `operator_name` is None.
    """
    if spin_dim < 2:
        raise ValueError("Spin dimension must be 2 or greater.")

    # Total spin S is derived from the Hilbert space dimension.
    spin_S = (spin_dim - 1) / 2.0

    # The matrix elements for S+ and S- are derived from the formula:
    # <S, m'|S+|S, m> = sqrt(S(S+1) - m(m+1)) * delta_{m', m+1}
    # The basis is ordered from m=S down to m=-S.
    m_values = np.arange(spin_S, -spin_S - 1, -1)
    s_plus_diag = [np.sqrt(spin_S * (spin_S + 1) - m * (m + 1)) for m in m_values[1:]]

    # Construct the base operators
    s_plus = np.diag(s_plus_diag, k=1)
    s_minus = np.diag(s_plus_diag, k=-1)
    s_z = np.diag(m_values)
    s_x = 0.5 * (s_plus + s_minus)
    s_y = -0.5j * (s_plus - s_minus)
    identity = np.identity(spin_dim)
    zero_op = np.zeros((spin_dim, spin_dim))

    # Store all operators in a dictionary for easy access.
    operators = {
        'Sp': s_plus,
        'Sm': s_minus,
        'Sz': s_z,
        'Sx': s_x,
        'Sy': s_y,
        'Id': identity,
        'S0': zero_op,
    }

    if operator_name:
        return operators.get(operator_name)
    else:
        return operators


def generate_mpo_hardcore_boson_operators(operator_name=None):
    """Generates hard-core boson and associated spin operator matrices.

    This function assumes a 3-dimensional local Hilbert space, which is common
    for models like the t-J model or Hubbard model with hardcore constraint
    (no double occupancy).

    The basis states are ordered as: `{|↑⟩, |↓⟩, |empty⟩}`.

    Args:
        operator_name (str, optional): The name of the specific operator to
            return (e.g., 'adagup', 'n_tot', 'F'). If None, a dictionary
            of all generated operators is returned.

    Returns:
        np.ndarray | dict[str, np.ndarray]: A NumPy array for the specified
            operator, or a dictionary of all operators if `operator_name` is None.
    """
    # Define creation and annihilation operators based on the {|↑⟩, |↓⟩, |empty⟩} basis.
    # adag_up: maps |empty⟩ -> |↑⟩ (state 2 -> state 0)
    adag_up = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=float)
    # adag_down: maps |empty⟩ -> |↓⟩ (state 2 -> state 1)
    adag_down = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    a_up = adag_up.T
    a_down = adag_down.T

    # Number operators
    n_up = adag_up @ a_up
    n_down = adag_down @ a_down
    n_tot = n_up + n_down

    # Jordan-Wigner string operator F = (-1)^n_tot
    # For basis {|↑⟩, |↓⟩, |empty⟩}, n_tot is {1, 1, 0}, so F is {-1, -1, 1}.
    fermi_string = np.diag([-1.0, -1.0, 1.0])

    # Corresponding spin-1/2 operators in this restricted Hilbert space
    s_z = 0.5 * (n_up - n_down)
    # s_plus: maps |↓⟩ -> |↑⟩ (state 1 -> state 0)
    s_plus = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
    s_minus = s_plus.T
    s_x = 0.5 * (s_plus + s_minus)
    s_y = -0.5j * (s_plus - s_minus)

    identity = np.identity(3)

    # Store all operators in a dictionary.
    operators = {
        'adag_up': adag_up,
        'adag_down': adag_down,
        'a_up': a_up,
        'a_down': a_down,
        'n_up': n_up,
        'n_down': n_down,
        'n_tot': n_tot,
        'F': fermi_string,
        'Id': identity,
        'Sz': s_z,
        'Sx': s_x,
        'Sy': s_y,
        'Sp': s_plus,
        'Sm': s_minus,
    }

    if operator_name:
        return operators.get(operator_name)
    else:
        return operators
