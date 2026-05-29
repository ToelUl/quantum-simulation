"""DMRG observables and diagnostics for matrix product states.

This module provides a correctness-first measurement layer for DMRG results in
this project. The public functions cover common quantities used in DMRG
workflows, including ground-state energy estimates, energy density, local
expectation values, two-point correlation functions, entanglement entropy,
truncation diagnostics, and energy variance.

The current measurement implementations use dense contractions for local
expectations, two-point functions, and energy variance. This keeps the behavior
straightforward to validate and suitable for small systems and tutorials. The
same public API can later be backed by environment-based MPS/MPO contractions
for large production calculations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional, Union

import torch


NumberLike = Union[float, int, complex, torch.Tensor]


def _scalar_to_float(value: NumberLike) -> float:
    """Convert a scalar-like value to a Python ``float``.

    Complex-valued inputs are converted using their real component. Tensor
    inputs must contain exactly one element.

    Args:
        value: Python scalar or scalar tensor to convert.

    Returns:
        The real component of ``value`` as a Python ``float``.

    Raises:
        ValueError: If ``value`` is a tensor with more than one element.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Expected a scalar tensor.")
        return float(value.detach().real.cpu().item())
    if isinstance(value, complex):
        return float(value.real)
    return float(value)


def ground_state_energy(energies: Union[Sequence[NumberLike], torch.Tensor]) -> float:
    """Return the final DMRG energy as the ground-state estimate.

    The current DMRG driver records one local energy per two-site update. The
    final entry is treated as the best available ground-state energy estimate
    for the completed sweep schedule.

    Args:
        energies: Energy trace returned by ``DMRG.forward`` or a one-dimensional
            tensor containing local update energies.

    Returns:
        The last entry in the energy trace.

    Raises:
        ValueError: If the energy trace is empty.
    """
    if isinstance(energies, torch.Tensor):
        flat = energies.reshape(-1)
        if flat.numel() == 0:
            raise ValueError("Energy trace is empty.")
        return _scalar_to_float(flat[-1])

    if len(energies) == 0:
        raise ValueError("Energy trace is empty.")
    return _scalar_to_float(energies[-1])


def energy_density(energy: NumberLike, num_sites: int) -> float:
    """Compute the ground-state energy density.

    Args:
        energy: Total energy of the system.
        num_sites: Number of lattice sites in the system.

    Returns:
        The energy per site, ``energy / num_sites``.

    Raises:
        ValueError: If ``num_sites`` is not positive.
    """
    if num_sites <= 0:
        raise ValueError("num_sites must be positive.")
    return _scalar_to_float(energy) / int(num_sites)


def excitation_gap(*args: Any, **kwargs: Any) -> float:
    """Future interface for the excitation gap ``Delta = E1 - E0``.

    Excited-state targeting is not implemented in the current DMRG driver. This
    function intentionally reserves the API location for future excited-state
    DMRG or another solver capable of estimating ``E1``.

    Args:
        *args: Reserved for a future excited-state API.
        **kwargs: Reserved for a future excited-state API.

    Returns:
        The excitation gap once an excited-state solver is implemented.

    Raises:
        NotImplementedError: Always raised until excited-state support exists.
    """
    raise NotImplementedError(
        "Excitation gap requires excited-state DMRG or another E1 solver, "
        "which is not implemented yet."
    )


def _validate_site(mps: Any, site: int) -> None:
    """Validate that a site index is inside an MPS chain.

    Args:
        mps: MPS-like object with an ``Nsites`` attribute.
        site: Site index to validate.

    Raises:
        ValueError: If ``site`` is outside ``[0, mps.Nsites)``.
    """
    if not 0 <= site < mps.Nsites:
        raise ValueError(f"site must satisfy 0 <= site < {mps.Nsites}, got {site}.")


def _validate_bond(mps: Any, bond: int) -> None:
    """Validate that a bond index is inside an MPS Schmidt-weight list.

    Args:
        mps: MPS-like object with an ``sWeight`` sequence.
        bond: Bond index to validate.

    Raises:
        ValueError: If ``bond`` is outside the stored Schmidt-weight range.
    """
    max_bond = len(mps.sWeight) - 1
    if not 0 <= bond <= max_bond:
        raise ValueError(f"bond must satisfy 0 <= bond <= {max_bond}, got {bond}.")


def _as_schmidt_vector(weights: torch.Tensor) -> torch.Tensor:
    """Convert stored Schmidt weights to a one-dimensional vector.

    Boundary weights may be stored as small matrices in the current MPS
    implementation. Matrix inputs are interpreted through their diagonal, while
    all other inputs are flattened. The returned values are non-negative.

    Args:
        weights: Stored Schmidt-weight tensor from ``mps.sWeight``.

    Returns:
        One-dimensional tensor containing non-negative Schmidt values.
    """
    tensor = weights.detach()
    if tensor.ndim == 2:
        tensor = torch.diagonal(tensor, 0)
    else:
        tensor = tensor.reshape(-1)
    return tensor.abs()


def entanglement_entropy(mps: Any, bond: Optional[int] = None, eps: float = 1e-15) -> Union[float, list[float]]:
    """Compute von Neumann entanglement entropy from MPS Schmidt weights.

    For Schmidt values ``lambda_alpha``, the entropy is computed as
    ``-sum(lambda_alpha**2 * log(lambda_alpha**2))`` after normalizing the
    Schmidt probabilities. Boundary bonds are allowed and typically return zero
    entropy.

    Args:
        mps: MPS-like object with ``sWeight`` tensors.
        bond: Optional bond index. If omitted, entropy is computed for every
            stored bond, including boundary bonds.
        eps: Probability cutoff used to avoid ``log(0)``.

    Returns:
        A single entropy value when ``bond`` is provided, otherwise a list of
        entropy values for all stored bonds.

    Raises:
        ValueError: If ``bond`` is outside the stored Schmidt-weight range.
    """
    def entropy_for_bond(bond_idx: int) -> float:
        _validate_bond(mps, bond_idx)
        lambdas = _as_schmidt_vector(mps.sWeight[bond_idx])
        probabilities = (lambdas ** 2).real
        total = probabilities.sum()
        if total <= 0:
            return 0.0
        probabilities = probabilities / total
        probabilities = probabilities[probabilities > eps]
        if probabilities.numel() == 0:
            return 0.0
        return float(-(probabilities * torch.log(probabilities)).sum().cpu().item())

    if bond is None:
        return [entropy_for_bond(idx) for idx in range(len(mps.sWeight))]
    return entropy_for_bond(bond)


def truncation_errors(dmrg_or_history: Any) -> list[float]:
    """Return discarded weights recorded during DMRG SVD truncations.

    Args:
        dmrg_or_history: Either a DMRG solver instance with a ``history``
            attribute or a mapping containing a ``"discarded_weights"`` entry.

    Returns:
        Discarded weights recorded after each two-site SVD update.

    Raises:
        TypeError: If ``dmrg_or_history`` is neither a solver-like object nor a
            history mapping.
        KeyError: If the history mapping does not contain
            ``"discarded_weights"``.
        ValueError: If any recorded value is not scalar-like.
    """
    history = dmrg_or_history
    if hasattr(dmrg_or_history, "history"):
        history = dmrg_or_history.history
    if not isinstance(history, Mapping):
        raise TypeError("Expected a DMRG object with .history or a history mapping.")
    if "discarded_weights" not in history:
        raise KeyError("history does not contain 'discarded_weights'.")
    return [_scalar_to_float(value) for value in history["discarded_weights"]]


def _dense_state_from_mps(mps: Any, *, form: str = "B") -> torch.Tensor:
    """Contract all MPS tensors into a dense state vector.

    The current DMRG driver returns the final optimized state with the
    orthogonality center at site 0; in that state, the ``B`` tensors reconstruct
    the optimized wavefunction. Dense reconstruction is intended for small
    systems and validation utilities.

    Args:
        mps: MPS-like object with ``A`` and ``B`` tensor lists.
        form: Tensor form to contract. Must be ``"A"`` or ``"B"``.

    Returns:
        Flattened dense state vector.

    Raises:
        ValueError: If ``form`` is invalid or the number of tensors does not
            match ``mps.Nsites``.
    """
    if form not in {"A", "B"}:
        raise ValueError("form must be 'A' or 'B'.")
    tensors = list(getattr(mps, form))
    if len(tensors) != mps.Nsites:
        raise ValueError("MPS tensor count does not match mps.Nsites.")
    state = tensors[0].detach()
    for tensor in tensors[1:]:
        state = torch.tensordot(state, tensor.detach(), dims=([-1], [0]))
    return state.reshape(-1)


def _promote_dtype(*dtypes: torch.dtype) -> torch.dtype:
    """Return the common promoted dtype for several PyTorch dtypes.

    Args:
        *dtypes: Dtypes to promote using ``torch.promote_types``.

    Returns:
        A dtype that can represent all input dtypes.
    """
    dtype = dtypes[0]
    for other in dtypes[1:]:
        dtype = torch.promote_types(dtype, other)
    return dtype


def _as_operator_tensor(operator: Any, *, device: torch.device, dtype: torch.dtype, physical_dim: int) -> torch.Tensor:
    """Convert and validate a local operator tensor.

    Args:
        operator: Array-like local operator.
        device: Target device for the operator tensor.
        dtype: Preferred dtype used together with operator dtype promotion.
        physical_dim: Expected local Hilbert-space dimension.

    Returns:
        Operator tensor on ``device`` with promoted dtype.

    Raises:
        ValueError: If the operator is not a square matrix with shape
            ``(physical_dim, physical_dim)``.
    """
    tensor = torch.as_tensor(operator, device=device)
    target_dtype = _promote_dtype(dtype, tensor.dtype)
    tensor = tensor.to(device=device, dtype=target_dtype)
    if tensor.ndim != 2 or tensor.shape[0] != physical_dim or tensor.shape[1] != physical_dim:
        raise ValueError(
            f"operator must have shape ({physical_dim}, {physical_dim}), got {tuple(tensor.shape)}."
        )
    return tensor


def _apply_local_operator(
    state: torch.Tensor,
    operator: torch.Tensor,
    site: int,
    *,
    num_sites: int,
    physical_dim: int,
) -> torch.Tensor:
    """Apply a single-site operator to a dense many-body state.

    Args:
        state: Flattened dense state vector with size ``physical_dim**num_sites``.
        operator: Local operator with shape ``(physical_dim, physical_dim)``.
        site: Site where the operator acts.
        num_sites: Number of lattice sites in the dense state.
        physical_dim: Local Hilbert-space dimension.

    Returns:
        Flattened dense state vector after applying ``operator`` at ``site``.
    """
    state_tensor = state.reshape([physical_dim] * num_sites)
    applied = torch.tensordot(operator, state_tensor, dims=([1], [site]))
    permutation = list(range(1, site + 1)) + [0] + list(range(site + 1, num_sites))
    return applied.permute(permutation).contiguous().reshape(-1)


def _expectation(state: torch.Tensor, applied_state: torch.Tensor) -> torch.Tensor:
    """Compute ``<state|applied_state> / <state|state>``.

    Args:
        state: Reference state vector.
        applied_state: State vector after applying an operator to ``state``.

    Returns:
        Scalar expectation value.

    Raises:
        ValueError: If ``state`` has zero norm.
    """
    norm = torch.vdot(state, state)
    if norm.abs() == 0:
        raise ValueError("Cannot compute expectation value for a zero-norm state.")
    return torch.vdot(state, applied_state) / norm


def local_expectation(mps: Any, operator: Any, site: int, *, form: str = "B") -> torch.Tensor:
    """Compute a local expectation value ``<O_site>``.

    The MPS is first reconstructed as a dense state vector, then the local
    operator is applied to the requested site. This implementation is intended
    for small-system diagnostics and tutorial examples.

    Args:
        mps: MPS-like object to measure.
        operator: Local operator with shape ``(mps.chid, mps.chid)``.
        site: Site index where the operator acts.
        form: MPS tensor form used for dense reconstruction. The default
            ``"B"`` matches the state returned by the current DMRG driver.

    Returns:
        Scalar expectation value as a tensor on the MPS device.

    Raises:
        ValueError: If ``site`` or ``operator`` is invalid, or if the dense MPS
            state has zero norm.
    """
    _validate_site(mps, site)
    state = _dense_state_from_mps(mps, form=form)
    operator_tensor = _as_operator_tensor(
        operator,
        device=state.device,
        dtype=state.dtype,
        physical_dim=mps.chid,
    )
    state = state.to(operator_tensor.dtype)
    applied = _apply_local_operator(
        state,
        operator_tensor,
        site,
        num_sites=mps.Nsites,
        physical_dim=mps.chid,
    )
    return _expectation(state, applied)


def two_point_correlation(
    mps: Any,
    op_i: Any,
    i: int,
    op_j: Any,
    j: int,
    *,
    form: str = "B",
) -> torch.Tensor:
    """Compute a two-point correlation function ``<O_i O_j>``.

    For distinct sites, ``op_j`` is applied first and ``op_i`` second, which is
    equivalent to ``O_i O_j``. For ``i == j``, the local product ``op_i @ op_j``
    is measured at the shared site.

    Args:
        mps: MPS-like object to measure.
        op_i: Local operator acting at site ``i``.
        i: Site index for ``op_i``.
        op_j: Local operator acting at site ``j``.
        j: Site index for ``op_j``.
        form: MPS tensor form used for dense reconstruction.

    Returns:
        Scalar two-point correlation value as a tensor.

    Raises:
        ValueError: If either site index or operator shape is invalid, or if
            the dense MPS state has zero norm.
    """
    _validate_site(mps, i)
    _validate_site(mps, j)
    state = _dense_state_from_mps(mps, form=form)
    op_i_tensor = _as_operator_tensor(op_i, device=state.device, dtype=state.dtype, physical_dim=mps.chid)
    op_j_tensor = _as_operator_tensor(op_j, device=state.device, dtype=state.dtype, physical_dim=mps.chid)
    common_dtype = _promote_dtype(state.dtype, op_i_tensor.dtype, op_j_tensor.dtype)
    state = state.to(common_dtype)
    op_i_tensor = op_i_tensor.to(common_dtype)
    op_j_tensor = op_j_tensor.to(common_dtype)

    if i == j:
        return local_expectation(mps, op_i_tensor @ op_j_tensor, i, form=form)

    applied = _apply_local_operator(
        state,
        op_j_tensor,
        j,
        num_sites=mps.Nsites,
        physical_dim=mps.chid,
    )
    applied = _apply_local_operator(
        applied,
        op_i_tensor,
        i,
        num_sites=mps.Nsites,
        physical_dim=mps.chid,
    )
    return _expectation(state, applied)


def _mpo_tensor_list(mpo: Any) -> list[torch.Tensor]:
    """Normalize supported MPO containers to a tensor list.

    Supported inputs include a DMRG solver with ``Ms``, an MPO builder with
    ``get_mpo()``, a callable returning an MPO list, or an explicit iterable of
    MPO tensors.

    Args:
        mpo: MPO source to normalize.

    Returns:
        List of MPO tensors.

    Raises:
        TypeError: If ``mpo`` is a single tensor rather than a per-site list.
        ValueError: If the resulting MPO tensor list is empty.
    """
    if hasattr(mpo, "Ms"):
        mpo = mpo.Ms
    elif hasattr(mpo, "get_mpo"):
        mpo = mpo.get_mpo()
    elif callable(mpo) and not isinstance(mpo, torch.Tensor):
        mpo = mpo()

    if isinstance(mpo, torch.Tensor):
        raise TypeError("energy_variance requires an MPO tensor list, not a single tensor.")

    tensors = [torch.as_tensor(tensor.detach() if isinstance(tensor, torch.Tensor) else tensor) for tensor in mpo]
    if not tensors:
        raise ValueError("MPO tensor list is empty.")
    return tensors


def _dense_operator_from_mpo(mpo: Any) -> torch.Tensor:
    """Contract an MPO tensor list into a dense matrix.

    MPO tensors are expected to follow the convention
    ``(W_left, W_right, physical_out, physical_in)`` and to have compatible
    virtual bonds. This dense contraction is intended for small-system
    diagnostics, especially energy variance checks.

    Args:
        mpo: MPO source accepted by ``_mpo_tensor_list``.

    Returns:
        Dense Hamiltonian/operator matrix.

    Raises:
        TypeError: If ``mpo`` is a single tensor rather than a per-site list.
        ValueError: If the MPO is empty, has invalid tensor ranks, incompatible
            physical dimensions, incompatible virtual bonds, or a non-scalar
            right boundary bond.
    """
    tensors = _mpo_tensor_list(mpo)
    device = tensors[0].device
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = _promote_dtype(dtype, tensor.dtype)
    tensors = [tensor.to(device=device, dtype=dtype) for tensor in tensors]

    for site, tensor in enumerate(tensors):
        if tensor.ndim != 4:
            raise ValueError(f"MPO tensor {site} must be rank-4, got shape {tuple(tensor.shape)}.")
        if tensor.shape[2] != tensor.shape[3]:
            raise ValueError(f"MPO tensor {site} physical dimensions must match, got {tuple(tensor.shape[2:])}.")
        if site < len(tensors) - 1 and tensor.shape[1] != tensors[site + 1].shape[0]:
            raise ValueError(
                f"MPO bond mismatch between sites {site} and {site + 1}: "
                f"{tensor.shape[1]} != {tensors[site + 1].shape[0]}."
            )

    acc = tensors[0][0]
    for tensor in tensors[1:]:
        acc = torch.tensordot(acc, tensor, dims=([0], [0]))
        acc = acc.movedim(acc.ndim - 3, 0)

    if acc.shape[0] != 1:
        raise ValueError(f"Right boundary MPO bond dimension must be 1, got {acc.shape[0]}.")

    op_tensor = acc[0]
    num_sites = len(tensors)
    axes = list(range(0, 2 * num_sites, 2)) + list(range(1, 2 * num_sites, 2))
    op_tensor = op_tensor.permute(axes).contiguous()
    dense_dim = int(torch.prod(torch.tensor([tensor.shape[2] for tensor in tensors])).item())
    return op_tensor.reshape(dense_dim, dense_dim)


def energy_variance(mps: Any, mpo: Any, *, form: str = "B") -> torch.Tensor:
    """Compute the energy variance ``Var(H) = <H^2> - <H>^2``.

    The MPS and MPO are converted to dense representations. This implementation
    is a validation baseline for small systems and tutorials; large production
    calculations should use optimized environment contractions.

    Args:
        mps: MPS-like object representing the state to diagnose.
        mpo: MPO source representing the Hamiltonian.
        form: MPS tensor form used for dense reconstruction.

    Returns:
        Real scalar tensor containing the energy variance.

    Raises:
        TypeError: If ``mpo`` cannot be normalized to a per-site tensor list.
        ValueError: If the MPS state has zero norm, the MPO is invalid, or the
            dense MPO dimension does not match the dense state dimension.
    """
    state = _dense_state_from_mps(mps, form=form)
    hamiltonian = _dense_operator_from_mpo(mpo)
    common_dtype = _promote_dtype(state.dtype, hamiltonian.dtype)
    state = state.to(device=hamiltonian.device, dtype=common_dtype)
    hamiltonian = hamiltonian.to(dtype=common_dtype)

    if hamiltonian.shape[0] != state.numel():
        raise ValueError(
            f"Dense MPO dimension {hamiltonian.shape[0]} does not match state dimension {state.numel()}."
        )

    norm = torch.vdot(state, state)
    if norm.abs() == 0:
        raise ValueError("Cannot compute energy variance for a zero-norm state.")
    h_state = hamiltonian @ state
    energy = torch.vdot(state, h_state) / norm
    h2 = torch.vdot(h_state, h_state) / norm
    variance = h2.real - energy.real.pow(2)

    eps = torch.finfo(variance.dtype).eps if variance.dtype.is_floating_point else torch.finfo(torch.float64).eps
    tolerance = 100 * eps * max(1.0, abs(float(h2.real.detach().cpu())), abs(float(energy.real.detach().cpu())) ** 2)
    if variance.item() < 0 and abs(variance.item()) < tolerance:
        variance = torch.zeros((), device=variance.device, dtype=variance.dtype)
    return variance
