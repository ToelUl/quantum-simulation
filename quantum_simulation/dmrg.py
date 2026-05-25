"""DMRG utilities and solver components for matrix product state optimization.

This module contains helpers for validating matrix product operators (MPOs),
selecting tensor contraction backends, updating DMRG environments, configuring
sweep schedules, and running a two-site DMRG optimization on an ``MPS``.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Any, Optional
from .mps import MPS
from .eigensolver import lanczos_ground_state
from .ncon_torch import ncon_torch


def _as_mpo_list(M_or_Ms: Any, Nsites: int, device: Any, dtype: torch.dtype) -> List[torch.Tensor]:
    """Converts an MPO specification into a per-site tensor list.

    Args:
        M_or_Ms (Any): Either a single MPO tensor to reuse on every site, or a
            list/tuple containing one MPO tensor per site. Each tensor is
            expected to support ``.to(device, dtype)``.
        Nsites (int): Number of lattice sites in the MPS/MPO chain. When
            ``M_or_Ms`` is a list or tuple, its length must equal this value.
        device (Any): Target PyTorch device, such as ``"cpu"``,
            ``"cuda"``, or a ``torch.device``.
        dtype (torch.dtype): Target tensor dtype used by the DMRG solver.

    Returns:
        List[torch.Tensor]: MPO tensors moved to ``device`` and ``dtype``, one
        tensor per site.

    Raises:
        AssertionError: If a per-site MPO list has a length different from
        ``Nsites``.
    """
    if isinstance(M_or_Ms, (list, tuple)):
        assert len(M_or_Ms) == Nsites, f"MPO list length {len(M_or_Ms)} != Nsites {Nsites}"
        Ms = [m.to(device, dtype) for m in M_or_Ms]
    else:
        Ms = [M_or_Ms.to(device, dtype) for _ in range(Nsites)]
    return Ms


def _check_mpo_shapes(Ms: List[torch.Tensor], ML: torch.Tensor, MR: torch.Tensor, chid: int) -> None:
    """Validates MPO and boundary-environment connectivity.

    Args:
        Ms (List[torch.Tensor]): Per-site MPO tensors. Each tensor must have
            shape ``(W_left, W_right, chid, chid)``, where ``W_left`` and
            ``W_right`` are MPO bond dimensions.
        ML (torch.Tensor): Left boundary environment with leading dimension
            matching ``Ms[0].shape[0]``.
        MR (torch.Tensor): Right boundary environment with leading dimension
            matching ``Ms[-1].shape[1]``.
        chid (int): Physical Hilbert-space dimension per site. Both physical
            legs of every MPO tensor must equal this value.

    Raises:
        AssertionError: If any MPO tensor rank, physical dimension, MPO bond,
        or boundary dimension is inconsistent.
    """
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
    """Resolves and stores the global tensor-contraction backend.

    Args:
        backend (str): Requested backend. Use ``"einsum"`` for
            ``torch.einsum``, ``"ncon"`` for ``ncon_torch``, or ``"auto"`` to
            choose ``"ncon"`` on CPU and ``"einsum"`` otherwise.
        device (Any): Device used for backend resolution. Accepts a string or
            an object with a ``type`` attribute, such as ``torch.device``.

    Raises:
        ValueError: If ``backend`` is not ``"auto"``, ``"einsum"``, or
        ``"ncon"``.
    """
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
    """Chooses the concrete contraction backend for a device type.

    Args:
        backend (str): Requested backend. ``"auto"`` is resolved using
            ``device_type``; ``"einsum"`` and ``"ncon"`` are returned as-is.
        device_type (str): PyTorch device type string, for example ``"cpu"``
            or ``"cuda"``.

    Returns:
        str: The concrete backend name, either ``"einsum"`` or ``"ncon"``
        when ``backend`` is valid.
    """
    if backend == 'auto':
        return 'ncon' if device_type == 'cpu' else 'einsum'
    return backend


class _BaseContractModule(nn.Module):
    """Base module for contractions that can switch backend by device.

    Attributes:
        contract_backend (str): Requested backend policy. ``"auto"`` resolves
            per reference tensor device; ``"einsum"`` and ``"ncon"`` force a
            specific backend.
    """

    def __init__(self, contract_backend: str = 'auto'):
        """Initializes the backend policy shared by contraction modules.

        Args:
            contract_backend (str): Requested contraction backend. Must be
                ``"auto"``, ``"einsum"``, or ``"ncon"``.

        Raises:
            ValueError: If ``contract_backend`` is not one of the supported
            backend names.
        """
        super().__init__()
        if contract_backend not in ('auto', 'einsum', 'ncon'):
            raise ValueError("contract backend must be 'auto'|'einsum'|'ncon'")
        self.contract_backend = contract_backend

    def _resolved(self, ref: torch.Tensor) -> str:
        """Resolves the backend using a reference tensor's device.

        Args:
            ref (torch.Tensor): Tensor whose ``device.type`` determines the
                concrete backend when ``contract_backend`` is ``"auto"``.

        Returns:
            str: Concrete backend name used for the current contraction.
        """
        return _resolve_backend(self.contract_backend, ref.device.type)


class ApplyMPO(_BaseContractModule):
    r"""Applies a two-site effective MPO to a vectorized local state."""

    def __init__(self, contract_backend: str = 'auto'):
        """Initializes the two-site MPO application module.

        Args:
            contract_backend (str): Requested contraction backend. Use
                ``"auto"`` to resolve by device, or force ``"einsum"`` or
                ``"ncon"``.
        """
        super().__init__(contract_backend)

    def forward(self,
                psi_flat: torch.Tensor,
                L: torch.Tensor,
                M1: torch.Tensor,
                M2: torch.Tensor,
                R: torch.Tensor,
                out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Applies the effective Hamiltonian to a flattened two-site state.

        Args:
            psi_flat (torch.Tensor): Flattened two-site state with logical
                shape ``(chi_left_in, d1_in, d2_in, chi_right_in)``. The flat
                length must equal ``L.shape[2] * M1.shape[3] * M2.shape[3] *
                R.shape[2]``.
            L (torch.Tensor): Left environment tensor with shape
                ``(W_left, chi_left_out, chi_left_in)``.
            M1 (torch.Tensor): MPO tensor for the first optimized site with
                shape ``(W_left, W_mid, d1_out, d1_in)``.
            M2 (torch.Tensor): MPO tensor for the second optimized site with
                shape ``(W_mid, W_right, d2_out, d2_in)``.
            R (torch.Tensor): Right environment tensor with shape
                ``(W_right, chi_right_out, chi_right_in)``.
            out (Optional[torch.Tensor]): Optional preallocated flat output
                buffer. If provided, it must have the same device and dtype as
                the contraction result and contain exactly ``chi_left_out *
                d1_out * d2_out * chi_right_out`` elements.

        Returns:
            torch.Tensor: Flattened output state with logical shape
            ``(chi_left_out, d1_out, d2_out, chi_right_out)``.

        Raises:
            ValueError: If ``out`` is provided with an incompatible number of
            elements, device, or dtype.
            AssertionError: If the MPO/environment bond dimensions are
            inconsistent in the ``"einsum"`` backend.
        """
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
    r"""Updates the left environment tensor during a DMRG sweep."""

    def forward(self, Lp: torch.Tensor, M: torch.Tensor, Ap: torch.Tensor) -> torch.Tensor:
        """Contracts one site into the left DMRG environment.

        Args:
            Lp (torch.Tensor): Existing left environment ``L[p]`` with shape
                ``(W_left, chi_left, chi_left)``.
            M (torch.Tensor): MPO tensor at site ``p`` with shape
                ``(W_left, W_right, phys_dim, phys_dim)``.
            Ap (torch.Tensor): Left-canonical MPS tensor ``A[p]`` with shape
                ``(chi_left, phys_dim, chi_right)``.

        Returns:
            torch.Tensor: Updated left environment ``L[p + 1]`` with shape
            ``(W_right, chi_right, chi_right)``.
        """
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
    r"""Updates the right environment tensor during a DMRG sweep."""

    def forward(self, M: torch.Tensor, Rnext: torch.Tensor, Bp1: torch.Tensor) -> torch.Tensor:
        """Contracts one site into the right DMRG environment.

        Args:
            M (torch.Tensor): MPO tensor at the site being absorbed with shape
                ``(W_left, W_right, phys_dim, phys_dim)``.
            Rnext (torch.Tensor): Existing right environment to the right of
                the site, with shape ``(W_right, chi_right, chi_right)``.
            Bp1 (torch.Tensor): Right-canonical MPS tensor for the absorbed
                site, with shape ``(chi_left, phys_dim, chi_right)``.

        Returns:
            torch.Tensor: Updated right environment with shape
            ``(W_left, chi_left, chi_left)``.
        """
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
    """Caches and updates left and right DMRG environment tensors.

    Attributes:
        mpo (List[torch.Tensor]): Per-site MPO tensors used for environment
            contractions.
        Nsites (int): Number of sites in the MPO/MPS chain.
        device (Any): Device on which boundary environments are created.
        dtype (torch.dtype): Dtype used for boundary environments.
        L_cache (List[Optional[torch.Tensor]]): Cached left environments,
            where ``L_cache[p]`` is the environment left of site ``p``.
        R_cache (List[Optional[torch.Tensor]]): Cached right environments,
            where ``R_cache[p]`` is the environment starting at site ``p``.
    """

    def __init__(self, mpo: List[torch.Tensor], device: Any, dtype: torch.dtype,
                 contract_backend: str = 'auto'):
        """Initializes environment caches and boundary tensors.

        Args:
            mpo (List[torch.Tensor]): Per-site MPO tensors with shape
                ``(W_left, W_right, phys_dim, phys_dim)``.
            device (Any): Device used for the initial boundary environments.
            dtype (torch.dtype): Dtype used for the initial boundary
                environments.
            contract_backend (str): Requested contraction backend for
                environment updates. Use ``"auto"``, ``"einsum"``, or
                ``"ncon"``.
        """
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
        """Returns a cached left environment.

        Args:
            site_idx (int): Site boundary index. ``0`` refers to the left
                boundary, and ``L[site_idx]`` is the environment immediately
                left of site ``site_idx``.

        Returns:
            torch.Tensor: Cached left environment tensor.

        Raises:
            RuntimeError: If the requested environment has not been computed.
        """
        if self.L_cache[site_idx] is None:
            raise RuntimeError(f"L[{site_idx}] requested but not yet computed.")
        return self.L_cache[site_idx]

    def get_R(self, site_idx: int) -> torch.Tensor:
        """Returns a cached right environment.

        Args:
            site_idx (int): Site boundary index. ``Nsites`` refers to the
                right boundary, and ``R[site_idx]`` is the environment
                immediately right of site ``site_idx - 1``.

        Returns:
            torch.Tensor: Cached right environment tensor.

        Raises:
            RuntimeError: If the requested environment has not been computed.
        """
        if self.R_cache[site_idx] is None:
            raise RuntimeError(f"R[{site_idx}] requested but not yet computed.")
        return self.R_cache[site_idx]

    def update_L(self, site_idx: int, A_tensor: torch.Tensor):
        """Computes and caches ``L[site_idx + 1]``.

        Args:
            site_idx (int): Site index ``p`` whose MPO tensor ``M[p]`` and
                left-canonical MPS tensor are absorbed into ``L[p]``.
            A_tensor (torch.Tensor): Left-canonical MPS tensor ``A[p]`` with
                shape ``(chi_left, phys_dim, chi_right)``.
        """
        Lp = self.get_L(site_idx)
        Mp = self.mpo[site_idx]
        L_new = self._left_upd(Lp, Mp, A_tensor)
        self.L_cache[site_idx + 1] = L_new

    def update_R(self, site_idx: int, B_tensor: torch.Tensor):
        """Computes and caches ``R[site_idx]``.

        Args:
            site_idx (int): Site index ``p`` whose MPO tensor ``M[p]`` and
                right-canonical MPS tensor are absorbed into ``R[p + 1]``.
            B_tensor (torch.Tensor): Right-canonical MPS tensor ``B[p]`` with
                shape ``(chi_left, phys_dim, chi_right)``.
        """
        R_next = self.get_R(site_idx + 1)
        M = self.mpo[site_idx]
        R_new = self._right_upd(M, R_next, B_tensor)
        self.R_cache[site_idx] = R_new


# ==============================================================================
# Sweeps class for managing DMRG parameters
# ==============================================================================
class Sweeps:
    """Stores per-sweep DMRG optimization schedules.

    Attributes:
        numsweeps (int): Total number of full left-to-right/right-to-left
            sweeps to perform.
        maxdim (List[int]): Maximum kept bond dimension for each sweep.
        noise (List[float]): Random-noise amplitude added to the local
            two-site state for each sweep.
        krylov_dim (List[int]): Krylov subspace dimension used by the Lanczos
            eigensolver for each sweep.
        cutoff (List[float]): SVD truncation cutoff for discarded singular
            weight in each sweep.
        reortho (List[bool]): Whether Lanczos reorthogonalization is enabled
            for each sweep.
        maxreortho (List[int]): Maximum number of reorthogonalization passes
            for each sweep when ``reortho`` is enabled.
    """
    def __init__(self, numsweeps: int):
        """Initializes default schedules for a fixed number of sweeps.

        Args:
            numsweeps (int): Number of full DMRG sweeps. Must be positive.

        Raises:
            ValueError: If ``numsweeps`` is not positive.
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
        """Sets one or more per-sweep parameter schedules.

        Args:
            **kwargs: Mapping from schedule name to a sequence of values, one
                value per sweep. Supported names are existing attributes such
                as ``maxdim``, ``noise``, ``krylov_dim``, ``cutoff``,
                ``reortho``, and ``maxreortho``. Each value sequence must have
                length ``numsweeps``.

        Raises:
            AttributeError: If a schedule name is not an attribute of this
            object.
            ValueError: If any schedule length differs from ``numsweeps``.

        Examples:
            >>> sweeps = Sweeps(3)
            >>> sweeps.set_schedule(maxdim=[10, 20, 100], noise=[1e-6, 1e-7, 0.0])
        """
        for key, val in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"Sweeps object does not have parameter '{key}'")
            if len(val) != self.numsweeps:
                raise ValueError(f"Length of schedule for '{key}' ({len(val)}) "
                                 f"does not match the number of sweeps ({self.numsweeps})")
            setattr(self, key, val)

    def __repr__(self) -> str:
        """Builds a human-readable summary of all sweep schedules.

        Returns:
            str: Multi-line string containing ``numsweeps`` and each
            configured schedule.
        """
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
    """Two-site DMRG solver implemented as a PyTorch module.

    Attributes:
        mps (MPS): Matrix product state optimized in place by ``forward``.
        Ms (List[torch.Tensor]): Per-site MPO tensors used as the Hamiltonian.
        Nsites (int): Number of sites in the MPS/MPO chain.
        chid (int): Physical Hilbert-space dimension per site.
        contract_backend (str): Requested contraction backend policy.
    """

    def __init__(self, mps: MPS, mpo: Any, *, device: Any = 'cpu', dtype: torch.dtype = torch.float64,
                 contract_backend: str = 'auto'):
        """Initializes the DMRG solver.

        Args:
            mps (MPS): Initial MPS to optimize. The object is stored and
                updated in place during sweeps.
            mpo (Any): Hamiltonian MPO as a per-site sequence. Each site tensor
                is expected to have shape ``(W_left, W_right, chid, chid)`` and
                support conversion to ``device`` and ``dtype``.
            device (Any): PyTorch device used for MPOs, environments, and
                contractions. Defaults to ``"cpu"``.
            dtype (torch.dtype): Floating-point or complex dtype used for DMRG
                tensors. Defaults to ``torch.float64``.
            contract_backend (str): Requested contraction backend. Use
                ``"auto"``, ``"einsum"``, or ``"ncon"``.

        Raises:
            ValueError: If ``contract_backend`` is invalid.
            AssertionError: If MPO tensor shapes or boundary dimensions are
            inconsistent.
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
        """Performs one left-to-right or right-to-left DMRG half-sweep.

        For each neighboring block ``(p, p + 1)``, this method builds the
        two-site wavefunction, applies Lanczos to the effective Hamiltonian
        when updates are enabled, optionally adds noise, splits the optimized
        state by SVD, truncates the bond, updates the MPS tensors, and refreshes
        the relevant environment cache.

        Args:
            direction (str): Sweep direction. ``"right"`` performs a
                left-to-right update; ``"left"`` performs a right-to-left
                update.
            mps (MPS): MPS object being optimized. Its ``A``, ``B``, and
                ``sWeight`` tensors are modified in place.
            env_manager (EnvironmentManager): Cache manager that provides the
                current ``L`` and ``R`` environments and stores updated ones.
            sweep_params (dict): Parameters for this sweep. Expected keys are
                ``chi`` for maximum bond dimension, ``noise`` for random-noise
                amplitude, ``krydim`` for Lanczos Krylov dimension, ``cutoff``
                for SVD truncation cutoff, ``dispon`` for print verbosity,
                ``updateon`` for enabling Lanczos optimization, ``maxit`` for
                Lanczos restarts, ``reortho`` and ``maxreortho`` for Lanczos
                reorthogonalization, plus ``sweep_idx`` and ``numsweeps`` for
                progress messages.
            Ekeep (List[float]): Existing list of local ground-state energies.
                New energies are appended when ``updateon`` is true.

        Returns:
            List[float]: The same energy list with any newly computed local
            energies appended.
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
        """Runs the full DMRG optimization schedule.

        Args:
            sweeps (Sweeps): Sweep schedule that supplies ``maxdim``,
                ``noise``, ``krylov_dim``, ``cutoff``, ``reortho``, and
                ``maxreortho`` values for each full sweep.
            dispon (int): Display verbosity. ``1`` prints one summary per full
                sweep, ``2`` prints each local two-site update, and other values
                suppress these messages.
            updateon (bool): If true, run Lanczos local optimization. If false,
                skip the eigensolver and only rebuild/split the current two-site
                state.
            maxit (int): Number of Lanczos restart attempts passed to
                ``lanczos_ground_state`` for each local optimization.

        Returns:
            Tuple[List[float], MPS]: A pair ``(Ekeep, mps)`` where ``Ekeep``
            contains local energies recorded during enabled updates and ``mps``
            is the optimized MPS object.
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


