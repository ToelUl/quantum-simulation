import torch
import torch.nn as nn
import numpy as np
from typing import List, Any, Optional
from .mps import MPS
from .eigensolver import lanczos_ground_state
from .ncon_torch import ncon_torch


def _as_mpo_list(M_or_Ms: Any, Nsites: int, device: Any, dtype: torch.dtype) -> List[torch.Tensor]:
    """Ensures the MPO is a list of tensors on the correct device and dtype."""
    if isinstance(M_or_Ms, (list, tuple)):
        assert len(M_or_Ms) == Nsites, f"MPO list length {len(M_or_Ms)} != Nsites {Nsites}"
        Ms = [m.to(device, dtype) for m in M_or_Ms]
    else:
        Ms = [M_or_Ms.to(device, dtype) for _ in range(Nsites)]
    return Ms


def _check_mpo_shapes(Ms: List[torch.Tensor], ML: torch.Tensor, MR: torch.Tensor, chid: int) -> None:
    """Performs sanity checks on MPO tensor shapes for connectivity."""
    num_steps = len(Ms)
    for p, M in enumerate(Ms):
        assert M.ndim == 4, f"Ms[{p}] must be rank-4 (Wl,Wr,d,d), got {M.shape}"
        assert M.shape[2] == chid and M.shape[3] == chid, \
            f"Ms[{p}] physical dims {M.shape[2:]} != (d,d)=({chid},{chid})"
        if p < num_steps - 1:
            Wr = M.shape[1]
            Wl_next = Ms[p+1].shape[0]
            assert Wr == Wl_next, \
                f"MPO bond mismatch: Ms[{p}].Wr={Wr} != Ms[{p+1}].Wl={Wl_next}"
    assert ML.shape[0] == Ms[0].shape[0], \
        f"ML W={ML.shape[0]} must match Ms[0].Wl={Ms[0].shape[0]}"
    assert MR.shape[0] == Ms[-1].shape[1], \
        f"MR W={MR.shape[0]} must match Ms[-1].Wr={Ms[-1].shape[1]}"


# ==============================================================================
# Contraction backend selection
# ==============================================================================
_CONTRACT_BACKEND = 'einsum'  # resolved, internal use

def set_contract_backend(backend: str = 'auto', device: Any = 'cpu') -> None:
    """Resolves and sets the global contraction backend."""
    global _CONTRACT_BACKEND
    dev = device if isinstance(device, str) else getattr(device, "type", "cpu")
    if backend == 'auto':
        _CONTRACT_BACKEND = 'ncon' if dev == 'cpu' else 'einsum'
    elif backend in ('einsum', 'ncon'):
        _CONTRACT_BACKEND = backend
    else:
        raise ValueError(f"Unknown backend: {backend!r} (valid: 'auto'|'einsum'|'ncon')")
    print(f"[DMRG] Contraction backend resolved to '{_CONTRACT_BACKEND}' on device='{dev}'.")

def _resolve_backend(backend: str, device_type: str) -> str:
    """Resolves the contraction backend based on device type."""
    if backend == 'auto':
        return 'ncon' if device_type == 'cpu' else 'einsum'
    return backend


class _BaseContractModule(nn.Module):
    """A helper base class that resolves the contraction backend per device."""
    def __init__(self, contract_backend: str = 'auto'):
        super().__init__()
        if contract_backend not in ('auto', 'einsum', 'ncon'):
            raise ValueError("contract backend must be 'auto'|'einsum'|'ncon'")
        self.contract_backend = contract_backend

    def _resolved(self, ref: torch.Tensor) -> str:
        """Resolves the backend based on the reference tensor's device."""
        return _resolve_backend(self.contract_backend, ref.device.type)


class ApplyMPO(_BaseContractModule):
    r"""Applies an effective MPO to a vectorized two-site state."""
    def __init__(self, contract_backend: str = 'auto'):
        super().__init__(contract_backend)

    def forward(self,
                psi_flat: torch.Tensor,
                L: torch.Tensor,
                M1: torch.Tensor,
                M2: torch.Tensor,
                R: torch.Tensor,
                out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs the MPO application on the two-site state."""
        be = self._resolved(L)
        if be == 'ncon':
            psi_tensor = psi_flat.view(L.shape[2], M1.shape[3], M2.shape[3], R.shape[2])
            res = ncon_torch(
                [psi_tensor, L, M1, M2, R],
                [[1, 3, 5, 7], [2, -1, 1], [2, 4, -2, 3], [4, 6, -3, 5], [6, -4, 7]]
            )
            if out is not None:
                h, i, j, k = res.shape
                if out.numel() != h * i * j * k or out.device != res.device or out.dtype != res.dtype:
                    raise ValueError(f"[ApplyMPO] out shape/device/dtype mismatch.")
                out.view(h, i, j, k).copy_(res)
                return out.view(-1)
            return res.reshape(-1)

        device, dtype = psi_flat.device, psi_flat.dtype
        b, h, a = L.shape
        _b, d, i, c = M1.shape
        assert _b == b
        _d, f, j, e = M2.shape
        assert _d == d
        _f, k, g = R.shape
        assert _f == f
        psi = psi_flat.view(a, c, e, g)
        t01 = torch.einsum('aceg,bha->bcegh', psi, L)
        t02 = torch.einsum('bcegh,fkg->bcehfk', t01, R)
        t03 = torch.einsum('bcehfk,dfje->bcdhjk', t02, M2)
        res = torch.einsum('bcdhjk,bdic->hijk', t03, M1)
        if out is not None:
            if out.numel() != h * i * j * k or out.device != device or out.dtype != dtype:
                raise ValueError(f"[ApplyMPO] out shape/device/dtype mismatch.")
            out.view(h, i, j, k).copy_(res)
            return out.view(-1)
        return res.contiguous().view(-1)


class LeftEnvUpdate(_BaseContractModule):
    r"""Computes the updated left environment tensor in a DMRG sweep."""
    def forward(self, Lp: torch.Tensor, M: torch.Tensor, Ap: torch.Tensor) -> torch.Tensor:
        """Performs the left environment update contraction."""
        be = self._resolved(Lp)
        if be == 'einsum':
            t01 = torch.einsum('abc,cek->abek', Lp, Ap)
            t02 = torch.einsum('abek,bdj->aekdj', t01, torch.conj(Ap))
            return torch.einsum('aekdj,aide->ijk', t02, M)
        else:
            return ncon_torch(
                [Lp, M, Ap, torch.conj(Ap)],
                [[2, 1, 4], [2, -1, 3, 5], [4, 5, -3], [1, 3, -2]]
            )


class RightEnvUpdate(_BaseContractModule):
    r"""Computes the updated right environment tensor in a DMRG sweep."""
    def forward(self, M: torch.Tensor, Rnext: torch.Tensor, Bp1: torch.Tensor) -> torch.Tensor:
        """Performs the right environment update contraction."""
        be = self._resolved(Rnext)
        if be == 'einsum':
            t01 = torch.einsum('abc,kec->abke', Rnext, Bp1)
            t02 = torch.einsum('iade,jdb->iaejb', M, torch.conj(Bp1))
            return torch.einsum('iaejb,abke->ijk', t02, t01)
        else:
            return ncon_torch(
                [M, Rnext, Bp1, torch.conj(Bp1)],
                [[-1, 2, 3, 5], [2, 1, 4], [-3, 5, 4], [-2, 3, 1]]
            )


class EnvironmentManager:
    """Manages the lifecycle of DMRG environment tensors (L and R)."""
    def __init__(self, mpo: List[torch.Tensor], device: Any, dtype: torch.dtype,
                 contract_backend: str = 'auto'):
        """Initializes the EnvironmentManager."""
        self.mpo = mpo
        self.Nsites = len(mpo)
        self.device = device
        self.dtype = dtype

        self.L_cache = [None] * (self.Nsites + 1)
        self.R_cache = [None] * (self.Nsites + 1)

        Wl_0 = self.mpo[0].shape[0]
        self.L_cache[0] = torch.ones(Wl_0, 1, 1, device=device, dtype=dtype)
        Wr_N1 = self.mpo[self.Nsites - 1].shape[1]
        self.R_cache[self.Nsites] = torch.ones(Wr_N1, 1, 1, device=device, dtype=dtype)

        self._left_upd = LeftEnvUpdate(contract_backend)
        self._right_upd = RightEnvUpdate(contract_backend)

    def get_L(self, site_idx: int) -> torch.Tensor:
        """Returns the environment to the left of `site_idx`."""
        if self.L_cache[site_idx] is None:
            raise RuntimeError(f"L[{site_idx}] requested but not yet computed.")
        return self.L_cache[site_idx]

    def get_R(self, site_idx: int) -> torch.Tensor:
        """Returns the environment to the right of `site_idx - 1`."""
        if self.R_cache[site_idx] is None:
            raise RuntimeError(f"R[{site_idx}] requested but not yet computed.")
        return self.R_cache[site_idx]

    def update_L(self, site_idx: int, A_tensor: torch.Tensor):
        """Computes L[site_idx + 1] using L[site_idx], M[site_idx], and A[site_idx]."""
        Lp = self.get_L(site_idx)
        Mp = self.mpo[site_idx]
        L_new = self._left_upd(Lp, Mp, A_tensor)
        self.L_cache[site_idx + 1] = L_new

    def update_R(self, site_idx: int, B_tensor: torch.Tensor):
        """Computes and caches R[site_idx] using R[site_idx + 1], M[site_idx], and B[site_idx]."""
        R_next = self.get_R(site_idx + 1)
        M = self.mpo[site_idx]
        R_new = self._right_upd(M, R_next, B_tensor)
        self.R_cache[site_idx] = R_new


# ==============================================================================
# Sweeps class for managing DMRG parameters
# ==============================================================================
class Sweeps:
    """Manages DMRG sweep parameters, inspired by ITensor's Sweeps object.

    This class provides a convenient way to define schedules for various
    parameters used during the DMRG sweeps, such as the maximum bond dimension
    (maxdim), noise, and Krylov dimension.

    Attributes:
        numsweeps (int): The total number of sweeps to be performed.
        maxdim (List[int]): A list of maximum bond dimensions for each sweep.
        noise (List[float]): A list of noise terms for each sweep.
        krylov_dim (List[int]): A list of Krylov subspace dimensions for each sweep.
        cutoff (List[float]): A list of SVD truncation cutoffs for each sweep.
    """
    def __init__(self, numsweeps: int):
        """Initializes the Sweeps object.

        Args:
            numsweeps (int): The total number of sweeps.

        Raises:
            ValueError: If numsweeps is not a positive integer.
        """
        if numsweeps <= 0:
            raise ValueError("Number of sweeps must be positive.")
        self.numsweeps = numsweeps
        # Initialize with default values
        self.maxdim = [10] * numsweeps
        self.noise = [0.0] * numsweeps
        self.krylov_dim = [4] * numsweeps
        self.cutoff = [1e-12] * numsweeps
        self.reortho = [False] * numsweeps
        self.maxreortho = [1] * numsweeps

    def set_schedule(self, **kwargs):
        """Sets a schedule for one or more parameters.

        This method allows specifying different values for parameters across
        different sweeps. The length of each parameter list must match the
        total number of sweeps.

        Args:
            **kwargs: Keyword arguments where the key is the parameter name
                (e.g., 'maxdim') and the value is a list of values for each sweep.

        Raises:
            AttributeError: If a specified parameter does not exist.
            ValueError: If the length of a schedule list does not match numsweeps.

        Example:
            sweeps = Sweeps(3)
            sweeps.set_schedule(maxdim=[10, 20, 100], noise=[1e-6, 1e-7, 0.0])
        """
        for key, val in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"Sweeps object does not have parameter '{key}'")
            if len(val) != self.numsweeps:
                raise ValueError(f"Length of schedule for '{key}' ({len(val)}) "
                                 f"does not match the number of sweeps ({self.numsweeps})")
            setattr(self, key, val)

    def __repr__(self) -> str:
        """Returns a string representation of the Sweeps object."""
        output = f"Sweeps({self.numsweeps}):\n"
        output += f"  maxdim: {self.maxdim}\n"
        output += f"  noise: {self.noise}\n"
        output += f"  krylov_dim: {self.krylov_dim}\n"
        output += f"  cutoff: {self.cutoff}\n"
        output += f"  reortho: {self.reortho}\n"
        output += f"  maxreortho: {self.maxreortho}\n"
        return output


# ==============================================================================
#                         DMRG driver as nn.Module
# ==============================================================================

class DMRG(nn.Module):
    """PyTorch DMRG main driver, implemented as a nn.Module."""
    def __init__(self, mps: MPS, mpo: Any, *, device: Any = 'cpu', dtype: torch.dtype = torch.float64,
                 contract_backend: str = 'auto'):
        """Initializes the DMRG solver.
        mps (MPS): The MPS object to be optimized.
        """
        super().__init__()
        set_contract_backend(contract_backend, device)
        self.device = device
        self.dtype = dtype
        self.contract_backend = contract_backend
        self.mps = mps
        if isinstance(mpo[0], np.ndarray):
            self.Nsites = len(mpo)
            self.Ms = _as_mpo_list(mpo, self.Nsites, device, dtype)
        else:
            self.Nsites = len(mpo)
            self.Ms = [m.to(device, dtype) for m in mpo]
        self.ML = torch.ones(self.Ms[0].shape[0], 1, 1, device=device, dtype=dtype)
        self.MR = torch.ones(self.Ms[-1].shape[1], 1, 1, device=device, dtype=dtype)
        self.chid = self.Ms[0].shape[2]
        _check_mpo_shapes(self.Ms, self.ML, self.MR, self.chid)
        # Initialize compute modules
        self.apply_mpo = ApplyMPO(contract_backend)

    def _sweep(self,
               direction: str,
               mps: MPS,
               env_manager: EnvironmentManager,
               sweep_params: dict,
               Ekeep: List[float]) -> List[float]:
        """Performs a single DMRG half-sweep (either right or left).

        This is the core workhorse method of the DMRG algorithm. It iterates
        through the MPS, updating two sites at a time to minimize the energy.

        The procedure for each two-site block (p, p+1) is:
        1.  Construct the two-site wavefunction psi_{p,p+1} by contracting
            the relevant MPS tensors and Schmidt values.
        2.  Find the ground state of the effective Hamiltonian for this block
            using the Lanczos algorithm. The effective Hamiltonian is formed
            by the MPO tensors for the block and the environment tensors (L, R).
        3.  Optionally, add a small amount of random noise to the ground state
            wavefunction to help escape local minima, especially in early sweeps.
        4.  Perform a Singular Value Decomposition (SVD) on the optimized
            two-site wavefunction to split it back into two individual site tensors.
        5.  Truncate the bond dimension between the sites based on the maximum
            allowed dimension (chi) and the SVD truncation cutoff.
        6.  Update the MPS tensors (A and B forms) and the Schmidt values (sWeight)
            with the new, optimized tensors from the SVD.
        7.  Update the environment tensor (L or R) for the next step in the sweep.

        Args:
            direction (str): The direction of the sweep, either 'right' or 'left'.
            mps (MPS): The MPS object being optimized.
            env_manager (EnvironmentManager): The manager for L and R environment tensors.
            sweep_params (dict): A dictionary containing all parameters for the current sweep
                                 (chi, noise, krydim, cutoff, etc.).
            Ekeep (List[float]): A list to store the energy at each optimization step.

        Returns:
            List[float]: The updated list of energies.
        """
        # Unpack sweep parameters
        chi = sweep_params['chi']
        noise = sweep_params['noise']
        krydim = sweep_params['krydim']
        cutoff = sweep_params['cutoff']
        dispon = sweep_params['dispon']
        updateon = sweep_params['updateon']
        maxit = sweep_params['maxit']
        reortho = sweep_params['reortho']
        maxreortho = sweep_params['maxreortho']
        sweep_idx = sweep_params['sweep_idx']
        numsweeps = sweep_params['numsweeps']

        A, B, sWeight = mps.A, mps.B, mps.sWeight

        site_range = range(self.Nsites - 1) if direction == 'right' else range(self.Nsites - 2, -1, -1)

        for p in site_range:
            # --- 1. Construct the two-site wavefunction psi_{p, p+1} ---
            if direction == 'right':
                chil, _, _ = B[p].shape
                _, _, chir = B[p + 1].shape
                sw = sWeight[p]
                if sw.dim() == 2: sw = torch.diagonal(sw, 0)
                sw = sw.to(B[p].dtype)

                # Optimized construction: (sWeight * B[p]) @ B[p+1]
                l, s, m = B[p].shape
                _m, t, r = B[p+1].shape
                ls, tr = l * s, t * r

                psi_scaled = torch.mul(B[p], sw.view(-1, 1, 1))
                psi_mm = torch.matmul(psi_scaled.reshape(ls, m), B[p+1].reshape(m, tr))
            else: # direction == 'left'
                chil, _, _ = A[p].shape
                _, _, chir = A[p+1].shape
                sw = sWeight[p + 2]
                if sw.dim() == 2: sw = torch.diagonal(sw, 0)
                sw = sw.to(A[p + 1].dtype)

                # Optimized construction: A[p] @ (A[p+1] * sWeight)
                l, s, m = A[p].shape
                _m, t, r = A[p+1].shape
                ls, tr = l * s, t * r

                psi_scaled = torch.mul(A[p + 1], sw.view(1, 1, -1))
                psi_mm = torch.matmul(A[p].reshape(ls, m), psi_scaled.reshape(m, tr))

            psiGround_flat = psi_mm.view(-1)

            # --- 2. Lanczos ground state search ---
            if updateon:
                Lp, Rp2 = env_manager.get_L(p), env_manager.get_R(p + 2)
                M1, M2 = self.Ms[p], self.Ms[p + 1]
                operator_args = (Lp, M1, M2, Rp2)
                psiGround_flat, Entemp = lanczos_ground_state(
                    psiGround_flat, self.apply_mpo, operator_args,
                    num_restarts=maxit, krylov_dim=krydim,
                    pro=reortho, pro_max_reorth=maxreortho,
                )
                Ekeep.append(Entemp)

            # --- 3. (Optional) Add noise ---
            if noise > 0:
                noise_vec = torch.rand_like(psiGround_flat)
                noise_vec.div_(torch.linalg.norm(noise_vec))
                psiGround_flat = psiGround_flat + noise_vec * noise
                psiGround_flat.div_(torch.linalg.norm(psiGround_flat))

            # --- 4. SVD, Truncation, and MPS Tensor Update ---
            U, S, Vh = torch.linalg.svd(psiGround_flat.view(chil * self.chid, self.chid * chir), full_matrices=False)

            chitemp_maxdim = min(S.numel(), chi)
            chitemp_cutoff = S.numel()
            if cutoff > 0.0:
                S_squared = S ** 2
                cumulative_sum = torch.cumsum(S_squared / torch.sum(S_squared), dim=0)
                chitemp_cutoff = torch.searchsorted(cumulative_sum, 1.0 - cutoff, right=True) + 1

            chitemp = min(chitemp_maxdim, chitemp_cutoff)
            chitemp = max(1, chitemp)

            # Update MPS tensors
            sv = S[:chitemp]
            denom = sv.norm().add_(torch.finfo(sv.dtype).eps)

            A[p] = nn.Parameter(U[:, :chitemp].view(chil, self.chid, chitemp), requires_grad=False)
            sWeight[p + 1] = nn.Parameter(sv.div(denom), requires_grad=False)
            B[p + 1] = nn.Parameter(Vh[:chitemp, :].view(chitemp, self.chid, chir), requires_grad=False)

            # --- 5. Update Environment Tensor ---
            if direction == 'right':
                env_manager.update_L(p, A[p])
            else: # direction == 'left'
                env_manager.update_R(p + 1, B[p + 1])

            if dispon == 2 and updateon:
                print(f'Sweep: {sweep_idx} of {numsweeps}, Loc: {p}, Energy: {Ekeep[-1]:.6f}')

        return Ekeep


    def forward(self, sweeps: Sweeps, *,
                dispon=2, updateon=True, maxit=2):
        """Executes the DMRG sweep algorithm.

        Args:
            sweeps (Sweeps): A Sweeps object defining the sweep schedule.
            dispon (int): Display option (1 or 2 for different verbosity).
            updateon (bool): If True, perform energy optimization.
            maxit (int): Number of restarts for the Lanczos solver.

        Returns:
            A tuple containing:
            - List[float]: A list of energies at each update step.
            - MPS: The final optimized MPS object.
        """
        Ekeep = []
        env_manager = EnvironmentManager(self.Ms, self.device, self.dtype, self.contract_backend)

        # Pre-compute all R environments from the initial right-canonical MPS
        for p in range(self.Nsites - 1, self.mps.ortho_center, -1):
            env_manager.update_R(p, self.mps.B[p])

        # --- Main Sweep Loop ---
        for k in range(1, sweeps.numsweeps + 1):
            sweep_params = {
                'chi': sweeps.maxdim[k-1], 'noise': sweeps.noise[k-1],
                'krydim': sweeps.krylov_dim[k-1], 'cutoff': sweeps.cutoff[k-1],
                'reortho': sweeps.reortho[k-1], 'maxreortho': sweeps.maxreortho[k-1],
                'dispon': dispon, 'updateon': updateon, 'maxit': maxit,
                'sweep_idx': k, 'numsweeps': sweeps.numsweeps
            }

            # --- Left-to-Right Sweep ---
            Ekeep = self._sweep('right', self.mps, env_manager, sweep_params, Ekeep)
            self.mps.ortho_center = self.Nsites - 1

            # Update boundary A[Nsites-1] and sWeight[Nsites] after the LTR sweep
            sw = self.mps.sWeight[self.Nsites - 1]
            if sw.dim() == 2: sw = torch.diagonal(sw, 0)
            B_last = self.mps.B[self.Nsites - 1]
            Atemp = (sw.to(B_last.dtype).view(-1, 1, 1) * B_last).reshape(
                B_last.shape[0] * self.chid, B_last.shape[2]
            )
            U, S, Vh = torch.linalg.svd(Atemp, full_matrices=False)
            A_right = U.view(B_last.shape[0], self.chid, B_last.shape[2])
            eps = torch.finfo(S.dtype).eps
            sW_right = (S.unsqueeze(1) * Vh).to(self.dtype) / (S.norm().add_(eps))
            self.mps.A[self.Nsites - 1] = nn.Parameter(A_right, requires_grad=False)
            self.mps.sWeight[self.Nsites] = nn.Parameter(sW_right, requires_grad=False)

            # --- Right-to-Left Sweep ---
            Ekeep = self._sweep('left', self.mps, env_manager, sweep_params, Ekeep)
            self.mps.ortho_center = 0

            # Update boundary B[0] and sWeight[0] after the RTL sweep
            sw = self.mps.sWeight[1]
            if sw.dim() == 2: sw = torch.diagonal(sw, 0)
            A0_scaled = self.mps.A[0] * sw.to(self.mps.A[0].dtype).view(1, 1, -1)
            chil, _, chir = self.mps.A[0].shape
            Atemp = A0_scaled.reshape(chil, self.chid * chir)
            U, S, Vh = torch.linalg.svd(Atemp, full_matrices=False)
            self.mps.B[0] = nn.Parameter(Vh.view(chil, self.chid, chir), requires_grad=False)
            sW0 = (U * S.unsqueeze(0)).to(self.dtype) / (S.norm().add_(eps))
            self.mps.sWeight[0] = nn.Parameter(sW0, requires_grad=False)


            if dispon == 1:
                print(f'Sweep: {k} of {sweeps.numsweeps}, Energy: {Ekeep[-1]:.12f}, Bond dim: {self.mps.A[self.Nsites//2].shape[-1]}, Noise: {sweeps.noise[k-1]:.2e}, Cutoff: {sweeps.cutoff[k-1]:.2e}')

        return Ekeep, self.mps


