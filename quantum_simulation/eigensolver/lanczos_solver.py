"""Preallocated-buffer Lanczos solver (workspace + engine).

This module provides a buffer-backed Lanczos implementation that reuses
preallocated tensors across calls. It is designed for heavy inner-loop usage
(e.g., within DMRG sweeps) to minimize memory allocations and improve GPU
throughput.

Key features:
  * Workspace caching keyed by (n, krylov_dim, device, dtype).
  * All frequently used arrays are registered as buffers to avoid autograd
    tracking and to enable persistent reuse.
  * Supports best-vector-restart and (full) partial reorthogonalization (PRO).
  * If the provided `linear_operator` accepts an `out=` argument, the
    matrix-vector product will be written directly into a preallocated buffer
    to avoid temporary allocations.
"""

import torch
from typing import Tuple, Callable, Optional


class _LanczosWorkspace(torch.nn.Module):
    """Reusable buffers for a fixed problem signature.

    A workspace is bound to a unique signature (``n``, ``krylov_dim``, device,
    dtype). It preallocates and holds tensors required by the Lanczos routine
    and exposes them as registered buffers to:

      1) avoid gradient tracking (inference-only),
      2) persist across calls (no frequent allocation/deallocation),
      3) allow DDP / module state management if needed.

    Attributes:
      n: Problem dimension (size of the vector space).
      k: Effective Krylov subspace dimension (``k = max(1, krylov_dim)``).
      device: Torch device where buffers live.
      dtype: Complex/real dtype of vectors in the original space.
      real_dtype: Real counterpart of ``dtype`` used for tridiagonal/eigendecomp.
    """
    def __init__(self, n: int, krylov_dim: int, device, dtype, real_dtype):
        super().__init__()
        k = max(1, int(krylov_dim))
        self.n = int(n)
        self.k = k
        self.device = device
        self.dtype = dtype
        self.real_dtype = real_dtype

        # Krylov basis and scalars
        self.register_buffer("K", torch.empty(n, k, device=device, dtype=dtype))
        self.register_buffer("alphas", torch.empty(k, device=device, dtype=real_dtype))
        self.register_buffer("betas", torch.empty(max(0, k-1), device=device, dtype=real_dtype))

        # Tridiagonal & eig workspace (real dtype)
        self.register_buffer("T", torch.empty(k, k, device=device, dtype=real_dtype))
        self.register_buffer("evals", torch.empty(k, device=device, dtype=real_dtype))
        self.register_buffer("evecs", torch.empty(k, k, device=device, dtype=real_dtype))
        self.register_buffer("y", torch.empty(k, device=device, dtype=real_dtype))
        self.register_buffer("order_idx", torch.empty(k, dtype=torch.long, device=device))

        # Temporary vectors in the original space (vector dtype)
        self.register_buffer("vec_cur", torch.empty(n, device=device, dtype=dtype))
        self.register_buffer("vec_prev", torch.empty(n, device=device, dtype=dtype))
        self.register_buffer("w", torch.empty(n, device=device, dtype=dtype))
        self.register_buffer("tmp_n", torch.empty(n, device=device, dtype=dtype))
        self.register_buffer("tmp_n2", torch.empty(n, device=device, dtype=dtype))

        # PRO overlaps
        self.register_buffer("ov", torch.empty(k, device=device, dtype=dtype))


class _LanczosEngine(torch.nn.Module):
    """Cached, buffer-backed Lanczos ground-state solver.

    This engine manages a cache of :class:`_LanczosWorkspace` objects keyed by
    ``(n, krylov_dim, device, dtype)`` and runs a memory-lean Lanczos routine
    using those workspaces. It supports:

      * Best-vector-restart for robust convergence.
      * Partial reorthogonalization (PRO): full reorthogonalization per step
        (optionally repeated up to ``pro_max_reorth`` times).
      * Early stopping: via small ``beta_j`` or Ritz residual estimate.

    Notes:
      * All computations take place under ``torch.inference_mode()``.
      * If ``linear_operator`` accepts an ``out=`` kwarg, the matvec result is
        written directly into a preallocated buffer to avoid temporaries.
    """
    def __init__(self):
        super().__init__()
        self._workspaces = {}  # key -> _LanczosWorkspace

    @staticmethod
    def _real_dtype_of(dtype: torch.dtype) -> torch.dtype:
        """Return the real dtype corresponding to ``dtype``.

        Args:
          dtype: Input dtype (possibly complex).

        Returns:
          The corresponding real dtype for tridiagonal/eigendecomposition.
        """
        if dtype == torch.complex128: return torch.float64
        if dtype == torch.complex64:  return torch.float32
        return dtype

    def _key(self, n: int, krylov_dim: int, device, dtype, real_dtype):
        """Build a unique cache key for a workspace."""
        return (int(n), int(krylov_dim), device.type, device.index, dtype, real_dtype)

    def _ensure_ws(self, n: int, krylov_dim: int, device, dtype) -> _LanczosWorkspace:
        """Get or create a workspace for the given signature.

        If no workspace exists for the signature (``n``, ``krylov_dim``,
        ``device``, ``dtype``), a new one is created, cached, and registered
        as a submodule.

        Args:
          n: Vector space dimension.
          krylov_dim: Target Krylov dimension.
          device: Torch device for buffers.
          dtype: Vector dtype (may be complex).

        Returns:
          A :class:`_LanczosWorkspace` bound to this signature.
        """
        real_dtype = self._real_dtype_of(dtype)
        key = self._key(n, krylov_dim, device, dtype, real_dtype)
        ws = self._workspaces.get(key)
        if ws is None:
            ws = _LanczosWorkspace(n, krylov_dim, device, dtype, real_dtype)
            self._workspaces[key] = ws
            # Register as child module to tie lifetime to the engine.
            self.add_module(f"ws_{len(self._workspaces)}", ws)
        return ws

    @staticmethod
    def _normalize_to(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Copy ``x`` into ``out`` and normalize ``out`` in-place.

        Args:
          x: Source vector.
          out: Destination vector (modified in-place).

        Returns:
          The normalized ``out`` tensor.
        """
        out.copy_(x)
        nrm = torch.linalg.norm(out)
        if nrm == 0:
            # Degenerate input; re-seed with random then normalize.
            out.uniform_(-1, 1)
            nrm = torch.linalg.norm(out)
        out.div_(nrm)
        return out

    def forward(
        self,
        initial_vector: torch.Tensor,
        linear_operator: Callable,
        operator_args: Tuple = (),
        num_restarts: int = 2,
        krylov_dim: int = 4,
        *,
        pro: bool = False,
        pro_max_reorth: int = 1,
        beta_tol: float = 1e-12,
        ritz_residual_tol: Optional[float] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Compute the approximate ground-state eigenpair via Lanczos.

        This routine builds a Krylov subspace up to ``krylov_dim``, diagonalizes
        the tridiagonal projection, and returns the smallest Ritz pair. It uses
        preallocated buffers from a cached workspace to minimize allocations.

        The linear operator is provided as a Python callable:
        ``linear_operator(v, *operator_args) -> w``. If the operator supports an
        ``out=`` keyword, it should follow the convention:
        ``linear_operator(v, *operator_args, out=w) -> out`` and write the
        result into ``out`` in-place (return value is ignored).

        Args:
          initial_vector: (n,) complex/real tensor; starting vector.
          linear_operator: Callable that applies A to a vector v.
          operator_args: Extra positional arguments passed to the operator.
          num_restarts: Number of restart rounds.
          krylov_dim: Maximum dimension of the Krylov subspace per round.
          pro: If True, perform full reorthogonalization against the current
            Krylov basis at each step (can be repeated with ``pro_max_reorth``).
          pro_max_reorth: Number of PRO passes per step (>=1).
          beta_tol: Early-stop threshold on ``||w||``.
          ritz_residual_tol: If provided, stop when the residual estimate
            ``|beta_tail * y[-1]|`` drops below this value.

        Returns:
          A tuple ``(v_gs, e_gs)``:
            * ``v_gs``: (n,) normalized approximate ground-state eigenvector.
            * ``e_gs``: float, corresponding Rayleigh quotient.
        """
        with torch.inference_mode():
            device = initial_vector.device
            dtype = initial_vector.dtype
            n = initial_vector.numel()

            ws = self._ensure_ws(n, krylov_dim, device, dtype)
            real_dtype = ws.real_dtype

            # Best-so-far vector/energy (initialize with the provided start vector).
            best_vec = self._normalize_to(initial_vector, ws.vec_cur.clone())  # detached copy
            best_e = torch.tensor(float("inf"), device=device, dtype=real_dtype)

            for _ in range(num_restarts):
                # Always restart from the best vector found so far.
                self._normalize_to(best_vec, ws.vec_cur)

                # ---------------------------
                # Build the Krylov subspace
                # ---------------------------
                m_max = max(1, int(krylov_dim))
                ws.alphas.zero_()
                if ws.betas.numel() > 0:
                    ws.betas.zero_()

                beta_prev = None
                beta_tail = torch.tensor(0.0, device=device, dtype=real_dtype)
                c = 0  # actual Krylov dimension attained this round

                for j in range(m_max):
                    # Store current basis vector v_j
                    ws.K[:, j].copy_(ws.vec_cur)
                    c += 1

                    # w = A @ v_j
                    try:
                        _ = linear_operator(ws.vec_cur, *operator_args, out=ws.w)
                    except TypeError:
                        res = linear_operator(ws.vec_cur, *operator_args)
                        ws.w.copy_(res)

                    # alpha_j = <v_j, w>
                    alpha_j = torch.vdot(ws.vec_cur, ws.w).real.to(real_dtype)
                    ws.alphas[j] = alpha_j

                    # w <- w - alpha_j v_j - beta_{j-1} v_{j-1}
                    ws.w.add_(ws.vec_cur, alpha=-alpha_j.to(dtype))
                    if beta_prev is not None:
                        ws.w.add_(ws.vec_prev, alpha=-beta_prev.to(dtype))

                    # PRO: full reorthogonalization against the existing basis
                    if pro and j > 0:
                        Kc = ws.K[:, :j]
                        ovj = ws.ov[:j]
                        for _ in range(pro_max_reorth):
                            torch.addmv(ovj, Kc.conj().transpose(0, 1), ws.w, beta=0.0, alpha=1.0, out=ovj)
                            torch.addmv(ws.tmp_n, Kc, ovj, beta=0.0, alpha=1.0, out=ws.tmp_n)
                            ws.w.add_(ws.tmp_n, alpha=-1.0)

                    # beta_j = ||w||
                    beta_j = torch.linalg.norm(ws.w).to(real_dtype)

                    # Early stop if convergence or max dimension reached
                    last = (j == m_max - 1)
                    if beta_j.item() < beta_tol or last:
                        beta_tail = beta_j
                        break

                    # v_{j+1} = w / beta_j
                    ws.vec_prev.copy_(ws.vec_cur)
                    ws.vec_cur.copy_(ws.w)
                    ws.vec_cur.div_(beta_j.to(dtype))
                    ws.betas[j] = beta_j
                    beta_prev = beta_j

                # ---------------------------
                # Solve the projected problem
                # ---------------------------
                Tloc = ws.T[:c, :c]
                Tloc.zero_()
                Tloc.diagonal().copy_(ws.alphas[:c])
                if c > 1:
                    Tloc.diagonal(1).copy_(ws.betas[:c-1])
                    Tloc.diagonal(-1).copy_(ws.betas[:c-1])

                # Eigen-decomposition of T_c
                evals, evecs = torch.linalg.eigh(Tloc)
                order = torch.argsort(evals.real)
                theta = evals[order[0]].real.to(real_dtype)
                y = evecs[:, order[0]]

                # Update best vector and energy only if the new energy is lower.
                if theta < best_e:
                    Kc = ws.K[:, :c]
                    torch.addmv(ws.tmp_n, Kc, y.to(dtype), beta=0.0, alpha=1.0, out=ws.tmp_n)
                    best_vec = self._normalize_to(ws.tmp_n, ws.tmp_n2)
                    best_e = theta

                # Residual estimate for early exit
                if ritz_residual_tol is not None:
                    res_est = (beta_tail.abs() * y[-1].abs()).item()
                    if res_est < ritz_residual_tol:
                        break

            return best_vec, float(best_e.item())


# Singleton engine and drop-in function
_ENGINE = _LanczosEngine()


def lanczos_ground_state(
    initial_vector: torch.Tensor,
    linear_operator: Callable,
    operator_args: Tuple = (),
    num_restarts: int = 2,
    krylov_dim: int = 4,
    *,
    pro: bool = False,
    pro_max_reorth: int = 1,
    beta_tol: float = 1e-12,
    ritz_residual_tol: Optional[float] = None,
) -> Tuple[torch.Tensor, float]:
    """Lanczos ground-state with cached workspaces (drop-in wrapper).

    This function is a thin wrapper around a process-wide singleton
    :class:`_LanczosEngine`. It delegates the computation to the engine, which
    maintains preallocated workspaces keyed by ``(n, krylov_dim, device, dtype)``
    and reuses them across calls. The goal is to minimize memory allocations and
    improve throughput in tight inner loops (e.g., DMRG sweeps).

    Args:
      initial_vector: Shape ``(n,)``. Starting vector (complex or real).
      linear_operator: Callable that applies the operator ``A`` to a vector ``v``.
      operator_args: Positional arguments forwarded to ``linear_operator``.
      num_restarts: Number of restart rounds.
      krylov_dim: Maximum Krylov subspace dimension per round (``k``).
      pro: If ``True``, perform full reorthogonalization against the current
        Krylov basis at each step (can be repeated via ``pro_max_reorth``).
      pro_max_reorth: Number of reorthogonalization passes per step (>= 1).
      beta_tol: Early-stop threshold on the recurrence norm ``||w||``.
      ritz_residual_tol: If set, stop early when the residual estimate
        ``|beta_tail * y[-1]|`` falls below this value.

    Returns:
      Tuple[torch.Tensor, float]:
        * ``v_gs``: Shape ``(n,)``. Normalized approximate ground-state vector.
        * ``e_gs``: Python ``float``. Corresponding Rayleigh quotient.
    """
    return _ENGINE(
        initial_vector,
        linear_operator,
        operator_args,
        num_restarts,
        krylov_dim,
        pro=pro,
        pro_max_reorth=pro_max_reorth,
        beta_tol=beta_tol,
        ritz_residual_tol=ritz_residual_tol,
    )

# ==============================================================================
# --- Example Usage ---
# ==============================================================================

if __name__ == '__main__':
    # 1. Define the linear operator (our "Hamiltonian")
    H = torch.tensor([
        [4.0, 1.0, 0.0, 0.5],
        [1.0, 3.0, 0.2, 0.0],
        [0.0, 0.2, 2.0, 0.3],
        [0.5, 0.0, 0.3, 5.0]
    ], dtype=torch.float64)

    true_eigenvalues, _ = torch.linalg.eigh(H)
    true_ground_energy = true_eigenvalues[0].item()

    def apply_hamiltonian(vec: torch.Tensor, hamiltonian: torch.Tensor) -> torch.Tensor:
        return hamiltonian @ vec

    # 2. Set up the Lanczos calculation
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H = H.to(device)

    initial_vector = torch.rand(H.shape[0], device=device, dtype=H.dtype)
    func_args = (H,)

    print(f"Running Lanczos on device: {device}")
    print("-" * 30)

    # 3. Run the Lanczos solver
    ground_state_vec, ground_energy = lanczos_ground_state(
        initial_vector,
        apply_hamiltonian,
        func_args,
        num_restarts=8,
        krylov_dim=4
    )

    # 4. Compare the results
    print(f"True Ground Energy: {true_ground_energy:.8f}")
    print(f"Lanczos Ground Energy: {ground_energy:.8f}")
    print(f"Error: {abs(true_ground_energy - ground_energy):.2e}")
    print("\nFinal Ground State Vector (first 4 elements):")
    print(ground_state_vec)

