from typing import List
import numpy as np
import torch
from .domain import generate_k_space
from .operation import bra_o_ket


def transverse_xy_chain_instant_quench_bogo_coef(
        dynamic_time: float, momentum: float,
        jx: float, jy: float, h_initial: float, h_final: float,
        quench_time: float = 0, phi: float = 0.0) -> tuple[float, float]:
    """Compute the Bogoliubov coefficients for the transverse XY chain after an instantaneous quench.
    Reffering to the formulas in https://scipost.org/SciPostPhys.1.1.003/pdf (Eqs. 11-19).

    Args:
        dynamic_time (float): The time at which to evaluate the coefficients.
        momentum (float): The momentum value.
        jx (float): The coupling constant in the x-direction.
        jy (float): The coupling constant in the y-direction.
        h_initial (float): The initial transverse field before the quench.
        h_final (float): The final transverse field after the quench.
        quench_time (float): The time at which the quench occurs. Default is 0 (instantaneous quench).
        phi (float): The anisotropy phase angle, which can introduce DMI-like terms. Default is 0.0.

    Returns:
        tuple[float, float]: The Bogoliubov coefficients (u, v) at the specified time.
    """
    jx = jx / 4  # Normalize by 4 for consistency with standard conventions.
    jy = jy / 4  # Normalize by 4 for consistency with standard conventions.
    h_initial = h_initial / 2  # Normalize by 2 for consistency with standard conventions.
    h_final = h_final / 2  # Normalize by 2 for consistency with standard conventions.
    j_sum = jx + jy
    anisotropy_param = (jx - jy) / j_sum
    pseudospin_coeff_x = -2 * anisotropy_param * j_sum * np.sin(2 * phi) * np.sin(momentum)
    pseudospin_coeff_y = 2 * anisotropy_param * j_sum * np.cos(2 * phi) * np.sin(momentum)
    pseudospin_coeff_z = 2 * (h_initial - j_sum * np.cos(momentum))
    epsilon_k = np.sqrt(pseudospin_coeff_x**2 + pseudospin_coeff_y**2 + pseudospin_coeff_z**2)
    a_k = 2 * (h_final - j_sum * np.cos(momentum))
    b_k = 2 * j_sum * anisotropy_param * np.sin(momentum)
    omega_k = np.sqrt(a_k**2 + b_k**2)
    u_k_0 = (pseudospin_coeff_z + epsilon_k) / np.sqrt(2 * epsilon_k * (epsilon_k + pseudospin_coeff_z))
    v_nk_dag_0 = (2 * anisotropy_param * j_sum * np.sin(momentum)) / np.sqrt(2 * epsilon_k * (epsilon_k + pseudospin_coeff_z))
    c3_u = np.exp(-1j * omega_k * quench_time) * ((omega_k - a_k) * u_k_0 - b_k * v_nk_dag_0) / (2 * omega_k)
    c4_u = np.exp(1j * omega_k * quench_time) * ((omega_k + a_k) * u_k_0 + b_k * v_nk_dag_0) / (2 * omega_k)
    c3_v = np.exp(-1j * omega_k * quench_time) * ((omega_k + a_k) * v_nk_dag_0 - b_k * u_k_0) / (2 * omega_k)
    c4_v = np.exp(1j * omega_k * quench_time) * ((omega_k - a_k) * v_nk_dag_0 + b_k * u_k_0) / (2 * omega_k)
    uk = c3_u * np.exp(1j * omega_k * dynamic_time) + c4_u * np.exp(-1j * omega_k * dynamic_time)
    v_nk_dag = c3_v * np.exp(1j * omega_k * dynamic_time) + c4_v * np.exp(-1j * omega_k * dynamic_time)

    return uk, v_nk_dag


def cc_dag_correlator_xy_chain_instant_quench(
        i: int, j: int,
        dynamic_time: float, lattice_length: int,
        jx: float, jy: float, h_initial: float, h_final: float,
        quench_time: float = 0, phi: float = 0.0) -> complex:
    """Compute the <c_i c_j^dagger> correlator for the transverse XY chain after an instantaneous quench.
    Reffering to the formulas in https://scipost.org/SciPostPhys.1.1.003/pdf (Eqs. 26).

    Args:
        i (int): The first site index.
        j (int): The second site index.
        dynamic_time (float): The time at which to evaluate the correlator.
        lattice_length (int): The length of the lattice.
        jx (float): The coupling constant in the x-direction.
        jy (float): The coupling constant in the y-direction.
        h_initial (float): The initial transverse field before the quench.
        h_final (float): The final transverse field after the quench.
        quench_time (float): The time at which the quench occurs. Default is 0 (instantaneous quench).
        phi (float): The anisotropy phase angle, which can introduce DMI-like terms. Default is 0.0.

    Returns:
        complex: The <c_i c_j^dagger> correlator at the specified time.
    """
    k_space = generate_k_space(lattice_length, anti_periodic_bc=True)

    def coefs_of_k(momentum):
        return transverse_xy_chain_instant_quench_bogo_coef(
            dynamic_time, momentum,
            jx, jy, h_initial, h_final,
            quench_time, phi
        )

    dk = k_space[1] - k_space[0]
    integral_list = [dk * np.exp(-1j * k * (i - j)) * coefs_of_k(k)[0] * np.conj(coefs_of_k(k)[0]) for k in k_space]

    return np.sum(integral_list) / (k_space[-1] - k_space[0])


def cc_correlator_xy_chain_instant_quench(
        i: int, j: int,
        dynamic_time: float, lattice_length: int,
        jx: float, jy: float, h_initial: float, h_final: float,
        quench_time: float = 0, phi: float = 0.0) -> complex:
    """Compute the <c_i c_j> correlator for the transverse XY chain after an instantaneous quench.
    Reffering to the formulas in https://scipost.org/SciPostPhys.1.1.003/pdf (Eqs. 27).

    Args:
        i (int): The first site index.
        j (int): The second site index.
        dynamic_time (float): The time at which to evaluate the correlator.
        lattice_length (int): The length of the lattice.
        jx (float): The coupling constant in the x-direction.
        jy (float): The coupling constant in the y-direction.
        h_initial (float): The initial transverse field before the quench.
        h_final (float): The final transverse field after the quench.
        quench_time (float): The time at which the quench occurs. Default is 0 (instantaneous quench).
        phi (float): The anisotropy phase angle, which can introduce DMI-like terms. Default is 0.0.

    Returns:
        complex: The <c_i c_j> correlator at the specified time.
    """
    k_space = generate_k_space(lattice_length, anti_periodic_bc=True)

    def coefs_of_k(momentum):
        return transverse_xy_chain_instant_quench_bogo_coef(
            dynamic_time, momentum,
            jx, jy, h_initial, h_final,
            quench_time, phi
        )

    dk = k_space[1] - k_space[0]
    integral_list = [dk * np.exp(-1j * k * (i - j)) * coefs_of_k(k)[0] * np.conj(coefs_of_k(k)[1]) for k in k_space]

    return 1j * np.sum(integral_list) / (k_space[-1] - k_space[0])


def g_contractor(
        i: int, j: int,
        dynamic_time: float, lattice_length: int,
        jx: float, jy: float, h_initial: float, h_final: float,
        quench_time: float = 0, phi: float = 0.0) -> np.ndarray:
    """Compute the g_{i,j} correlator for the transverse XY chain after an instantaneous quench.
    Reffering to the formulas in https://scipost.org/SciPostPhys.1.1.003/pdf (Eqs. 30).

    Args:
        i (int): The first site index.
        j (int): The second site index.
        dynamic_time (float): The time at which to evaluate the correlator.
        lattice_length (int): The length of the lattice.
        jx (float): The coupling constant in the x-direction.
        jy (float): The coupling constant in the y-direction.
        h_initial (float): The initial transverse field before the quench.
        h_final (float): The final transverse field after the quench.
        quench_time (float): The time at which the quench occurs. Default is 0 (instantaneous quench).
        phi (float): The anisotropy phase angle, which can introduce DMI-like terms. Default is 0.0.

    Returns:
        np.ndarray: The g_{i,j} correlator at the specified time.
    """
    k_space = generate_k_space(lattice_length, anti_periodic_bc=True)
    factor = (k_space[-1] - k_space[0]) / np.pi
    cc = cc_correlator_xy_chain_instant_quench(
        i, j, dynamic_time, lattice_length, jx, jy, h_initial, h_final, quench_time, phi
    )
    cc_dag = cc_dag_correlator_xy_chain_instant_quench(
        i, j, dynamic_time, lattice_length, jx, jy, h_initial, h_final, quench_time, phi
    )

    return factor * (np.real(cc) - cc_dag) + 1.0 * (i == j)


def tf_xy_chain_mz_instant_quench(time_sequence: List[float],
                                  lattice_length: int,
                                  jx: float, jy: float,
                                  h_initial: float, h_final: float,
                                  quench_time: float = 0,
                                  phi: float = 0.0) -> np.ndarray:
    """Compute the time evolution of the magnetization along z for the transverse XY chain after an instantaneous quench.
    Reffering to the formulas in https://scipost.org/SciPostPhys.1.1.003/pdf (Eqs. 23).

    Args:
        time_sequence (List[float]): A list of time points at which to evaluate the magnetization.
        lattice_length (int): The length of the lattice.
        jx (float): The coupling constant in the x-direction.
        jy (float): The coupling constant in the y-direction.
        h_initial (float): The initial transverse field before the quench.
        h_final (float): The final transverse field after the quench.
        quench_time (float): The time at which the quench occurs. Default is 0 (instantaneous quench).
        phi (float): The anisotropy phase angle, which can introduce DMI-like terms. Default is 0.0.

    Returns:
        np.ndarray: An array of magnetization values at the specified time points.
    """
    mz_time_evolution = np.array(
        [-np.real(g_contractor(0, 0, t, lattice_length, jx, jy, h_initial, h_final, quench_time, phi)) for t in time_sequence])

    return mz_time_evolution - (mz_time_evolution[0] - 1.0)


def dynamic_of_observables(time: List[float],
                           observable_op: List[torch.Tensor],
                           initial_state: torch.Tensor,
                           hamiltonian_op: torch.Tensor) -> torch.Tensor:
    """Compute the time evolution of observables under a given Hamiltonian.

    Args:
        time (List[float]): A list of time points at which to evaluate the observables.
        observable_op (List[torch.Tensor]): A list of observable operators.
        initial_state (torch.Tensor): The initial state vector.
        hamiltonian_op (torch.Tensor): The Hamiltonian operator.

    Returns:
        torch.Tensor: A tensor of shape (len(time), len(observable_op)) containing the expectation values
        of the observables at each time point.
    """
    device = initial_state.device
    dtype = hamiltonian_op.dtype
    hamiltonian_eigenvalues, hamiltonian_eigenvectors = torch.linalg.eigh(hamiltonian_op)
    # eigenvalues: (D,)
    # eigenvectors: (D, D)

    # (D, D) @ (D, 1) -> (D, 1)
    initial_state_in_eigenbasis = torch.matmul(
        hamiltonian_eigenvectors.conj().T,
        initial_state
    )
    t_tensor = torch.tensor(time, device=device).to(dtype)  # Shape: (T)

    # (T, 1) * (1, D) -> (T, D)
    time_evolution_phases = torch.exp(
        -1j * t_tensor.unsqueeze(-1) * hamiltonian_eigenvalues.unsqueeze(0)
    )

    # (T, D) * (1, D) (from (D, 1).T) -> (T, D)
    evolved_states_in_eigenbasis = time_evolution_phases * initial_state_in_eigenbasis.T

    # (D, D) @ (D, T) -> (D, T)
    evolved_states_T = torch.matmul(
        hamiltonian_eigenvectors,
        evolved_states_in_eigenbasis.T
    )

    evolved_states = evolved_states_T.T  # Shape: (T, D)

    # N * (D, D) -> (N, D, D)
    ops_stack = torch.stack(observable_op, dim=0)

    # 'ket': |psi(t)>
    # (T, D) -> (T, 1, D, 1)
    # dim T for time, dim 1 for N operators
    ket_batch = evolved_states.unsqueeze(1).unsqueeze(-1)

    # 'bra': <psi(t)|
    # (T, D) -> (T, 1, 1, D)
    # (.conj())
    bra_batch = evolved_states.conj().unsqueeze(1).unsqueeze(-2)

    # (N, D, D) -> (1, N, D, D)
    # dim 1 for broadcast T, dim N for operators
    o_batch = ops_stack.unsqueeze(0)

    # bra_batch: (T, 1, 1, D)
    # o_batch:   (1, N, D, D)
    # ket_batch: (T, 1, D, 1)
    # einsum '...' is (T, N)
    # einsum('...ij,...jk,...kl->...il')
    # i=1, j=D, k=D, l=1
    # output '...il' will be (T, N, 1, 1)
    observables_time_evolution = bra_o_ket(
        bra_batch,
        o_batch,
        ket_batch,
        real=True
    )
    # (T, N, 1, 1) -> (T, N)
    return observables_time_evolution.squeeze(-1).squeeze(-1)


