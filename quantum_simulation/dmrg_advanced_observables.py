"""Advanced DMRG observables for finite matrix product states.

This module builds on :mod:`quantum_simulation.dmrg_observables` and provides
higher-level measurements that are common in DMRG studies: structure factors,
finite-chain correlation-length estimates, entanglement spectra, string order
parameters, ground-state fidelity, and many-body polarization.

The implementations intentionally use dense state reconstruction as a
correctness baseline for small systems and tutorials. This keeps physical
conventions explicit and easy to test. The public APIs are designed so the
internal dense contractions can later be replaced by MPS/MPO environment
contractions for larger systems.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import torch

from .dmrg_observables import (
    _apply_local_operator,
    _as_operator_tensor,
    _as_schmidt_vector,
    _dense_state_from_mps,
    _expectation,
    _promote_dtype,
    _validate_bond,
    _validate_site,
)


__all__ = [
    "structure_factor",
    "correlation_length",
    "entanglement_spectrum",
    "string_order_parameter",
    "ground_state_fidelity",
    "many_body_polarization",
]


def _real_dtype_for(dtype: torch.dtype) -> torch.dtype:
    """Return the real counterpart of a PyTorch dtype.

    Args:
        dtype: Real or complex PyTorch dtype.

    Returns:
        ``torch.float64`` for ``torch.complex128``, ``torch.float32`` for
        ``torch.complex64``, and ``dtype`` for real floating-point inputs.
    """
    if dtype == torch.complex128:
        return torch.float64
    if dtype == torch.complex64:
        return torch.float32
    return dtype


def _dense_state_and_operator(mps: Any, operator: Any, *, form: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a dense MPS state and validated local operator.

    Args:
        mps: MPS-like object with ``Nsites`` and ``chid`` attributes.
        operator: Local operator with shape ``(mps.chid, mps.chid)``.
        form: MPS tensor form used for dense reconstruction.

    Returns:
        Pair ``(state, operator_tensor)`` with compatible dtype and device.

    Raises:
        ValueError: If dense reconstruction fails or the operator shape is
            incompatible with ``mps.chid``.
    """
    state = _dense_state_from_mps(mps, form=form)
    operator_tensor = _as_operator_tensor(
        operator,
        device=state.device,
        dtype=state.dtype,
        physical_dim=mps.chid,
    )
    state = state.to(operator_tensor.dtype)
    return state, operator_tensor


def _local_expectation_from_state(
    state: torch.Tensor,
    operator: torch.Tensor,
    site: int,
    *,
    num_sites: int,
    physical_dim: int,
) -> torch.Tensor:
    """Measure a local operator using an already reconstructed dense state.

    Args:
        state: Flattened dense many-body state.
        operator: Local operator with shape ``(physical_dim, physical_dim)``.
        site: Site where the operator acts.
        num_sites: Number of lattice sites.
        physical_dim: Local Hilbert-space dimension.

    Returns:
        Scalar local expectation value.
    """
    applied = _apply_local_operator(
        state,
        operator,
        site,
        num_sites=num_sites,
        physical_dim=physical_dim,
    )
    return _expectation(state, applied)


def _two_point_from_state(
    state: torch.Tensor,
    op_i: torch.Tensor,
    i: int,
    op_j: torch.Tensor,
    j: int,
    *,
    num_sites: int,
    physical_dim: int,
) -> torch.Tensor:
    """Measure ``<O_i O_j>`` using an already reconstructed dense state.

    Args:
        state: Flattened dense many-body state.
        op_i: Local operator acting at site ``i``.
        i: Site index for ``op_i``.
        op_j: Local operator acting at site ``j``.
        j: Site index for ``op_j``.
        num_sites: Number of lattice sites.
        physical_dim: Local Hilbert-space dimension.

    Returns:
        Scalar two-point expectation value.
    """
    if i == j:
        return _local_expectation_from_state(
            state,
            op_i @ op_j,
            i,
            num_sites=num_sites,
            physical_dim=physical_dim,
        )

    applied = _apply_local_operator(
        state,
        op_j,
        j,
        num_sites=num_sites,
        physical_dim=physical_dim,
    )
    applied = _apply_local_operator(
        applied,
        op_i,
        i,
        num_sites=num_sites,
        physical_dim=physical_dim,
    )
    return _expectation(state, applied)


def _as_q_values(q_values: Optional[Union[Sequence[float], torch.Tensor]], *, num_sites: int, device: torch.device) -> torch.Tensor:
    """Normalize momentum values for structure-factor calculations.

    Args:
        q_values: Optional sequence of momentum values. If omitted, the default
            grid is ``2*pi*n/num_sites`` for ``n = 0, ..., num_sites - 1``.
        num_sites: Number of lattice sites.
        device: Device for the returned tensor.

    Returns:
        One-dimensional tensor of momentum values with dtype ``torch.float64``.
    """
    if q_values is None:
        n = torch.arange(num_sites, device=device, dtype=torch.float64)
        return 2.0 * torch.pi * n / num_sites
    return torch.as_tensor(q_values, device=device, dtype=torch.float64).reshape(-1)


def structure_factor(
    mps: Any,
    operator: Any,
    q_values: Optional[Union[Sequence[float], torch.Tensor]] = None,
    *,
    connected: bool = False,
    form: str = "B",
) -> dict[str, torch.Tensor]:
    """Compute the finite-chain structure factor ``S(q)``.

    The implemented convention is
    ``S(q) = (1 / L) * sum_ij exp(i*q*(i-j)) * C_ij``, where ``C_ij`` is either
    ``<O_i O_j>`` or the connected correlator
    ``<O_i O_j> - <O_i><O_j>``.

    Args:
        mps: MPS-like object to measure.
        operator: Local operator ``O`` with shape ``(mps.chid, mps.chid)``.
        q_values: Optional momentum values. If omitted, uses the discrete grid
            ``2*pi*n/L``.
        connected: If true, subtract ``<O_i><O_j>`` from the two-point matrix.
        form: MPS tensor form used for dense reconstruction.

    Returns:
        Dictionary with keys:
            ``"q_values"``: Momentum values used in the calculation.
            ``"values"``: Complex structure-factor values at each momentum.
            ``"correlation_matrix"``: Matrix ``C_ij`` used in the Fourier sum.

    Raises:
        ValueError: If the operator shape is invalid or the MPS state has zero
            norm.
    """
    state, operator_tensor = _dense_state_and_operator(mps, operator, form=form)
    num_sites = mps.Nsites
    physical_dim = mps.chid

    local_values = torch.empty(num_sites, device=state.device, dtype=state.dtype)
    corr = torch.empty(num_sites, num_sites, device=state.device, dtype=state.dtype)

    for i in range(num_sites):
        local_values[i] = _local_expectation_from_state(
            state,
            operator_tensor,
            i,
            num_sites=num_sites,
            physical_dim=physical_dim,
        )

    for i in range(num_sites):
        for j in range(num_sites):
            corr[i, j] = _two_point_from_state(
                state,
                operator_tensor,
                i,
                operator_tensor,
                j,
                num_sites=num_sites,
                physical_dim=physical_dim,
            )

    if connected:
        corr = corr - local_values[:, None] * local_values[None, :]

    q_tensor = _as_q_values(q_values, num_sites=num_sites, device=state.device)
    positions = torch.arange(num_sites, device=state.device, dtype=torch.float64)
    separations = positions[:, None] - positions[None, :]
    values = []
    for q in q_tensor:
        phase = torch.exp(1j * q * separations).to(corr.dtype)
        values.append((phase * corr).sum() / num_sites)
    values_tensor = torch.stack(values) if values else torch.empty(0, device=state.device, dtype=corr.dtype)

    return {
        "q_values": q_tensor,
        "values": values_tensor,
        "correlation_matrix": corr,
    }


def correlation_length(
    mps: Any,
    operator: Any,
    *,
    reference_site: Optional[int] = None,
    fit_range: Optional[tuple[int, int]] = None,
    connected: bool = True,
    form: str = "B",
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Estimate a finite-chain correlation length from exponential decay.

    The estimator fits ``log(abs(C(r)))`` against distance ``r`` using a local
    correlator measured relative to ``reference_site``. It assumes the selected
    data are compatible with an exponential decay ``C(r) ~ A * exp(-r / xi)``.

    Args:
        mps: MPS-like object to measure.
        operator: Local operator used to build ``C(reference_site, j)``.
        reference_site: Site used as the correlation origin. Defaults to
            ``mps.Nsites // 2``.
        fit_range: Optional inclusive distance window ``(r_min, r_max)``. If
            omitted, all nonzero distances are considered.
        connected: If true, fit the connected correlator
            ``<O_i O_j> - <O_i><O_j>``.
        form: MPS tensor form used for dense reconstruction.
        eps: Minimum absolute correlation magnitude retained for the log fit.

    Returns:
        Dictionary with keys ``"xi"``, ``"distances"``, ``"correlations"``,
        ``"fit_range"``, ``"slope"``, and ``"intercept"``.

    Raises:
        ValueError: If there are fewer than two valid fit points or if the fitted
            slope is non-negative.
    """
    state, operator_tensor = _dense_state_and_operator(mps, operator, form=form)
    num_sites = mps.Nsites
    physical_dim = mps.chid

    if reference_site is None:
        reference_site = num_sites // 2
    _validate_site(mps, reference_site)

    ref_expectation = _local_expectation_from_state(
        state,
        operator_tensor,
        reference_site,
        num_sites=num_sites,
        physical_dim=physical_dim,
    )

    distances = []
    correlations = []
    for site in range(num_sites):
        if site == reference_site:
            continue
        corr = _two_point_from_state(
            state,
            operator_tensor,
            reference_site,
            operator_tensor,
            site,
            num_sites=num_sites,
            physical_dim=physical_dim,
        )
        if connected:
            local_site = _local_expectation_from_state(
                state,
                operator_tensor,
                site,
                num_sites=num_sites,
                physical_dim=physical_dim,
            )
            corr = corr - ref_expectation * local_site
        distances.append(abs(site - reference_site))
        correlations.append(corr)

    real_dtype = _real_dtype_for(state.dtype)
    distance_tensor = torch.tensor(distances, device=state.device, dtype=real_dtype)
    correlation_tensor = torch.stack(correlations) if correlations else torch.empty(0, device=state.device, dtype=state.dtype)

    if fit_range is None:
        r_min = 1
        r_max = int(distance_tensor.max().item()) if distance_tensor.numel() else 0
    else:
        r_min, r_max = fit_range
        if r_min < 1 or r_max < r_min:
            raise ValueError("fit_range must be an inclusive window with 1 <= r_min <= r_max.")

    magnitudes = correlation_tensor.abs().to(real_dtype)
    mask = (distance_tensor >= r_min) & (distance_tensor <= r_max) & (magnitudes > eps)
    fit_distances = distance_tensor[mask]
    fit_magnitudes = magnitudes[mask]

    if fit_distances.numel() < 2:
        raise ValueError("At least two nonzero correlation values are required to fit a correlation length.")

    y = torch.log(fit_magnitudes)
    x = torch.stack([fit_distances, torch.ones_like(fit_distances)], dim=1)
    solution = torch.linalg.lstsq(x, y.unsqueeze(1)).solution.squeeze(1)
    slope = float(solution[0].detach().cpu().item())
    intercept = float(solution[1].detach().cpu().item())
    if slope >= 0:
        raise ValueError("Fitted correlation decay slope is non-negative; cannot estimate a finite xi.")

    return {
        "xi": -1.0 / slope,
        "distances": distance_tensor,
        "correlations": correlation_tensor,
        "fit_range": (int(r_min), int(r_max)),
        "slope": slope,
        "intercept": intercept,
    }


def entanglement_spectrum(mps: Any, bond: Optional[int] = None, eps: float = 1e-15) -> Union[torch.Tensor, list[torch.Tensor]]:
    """Compute the entanglement spectrum ``-log(lambda_alpha**2)``.

    Schmidt probabilities are normalized before taking the logarithm. Entries
    with probability below ``eps`` are discarded.

    Args:
        mps: MPS-like object with ``sWeight`` tensors.
        bond: Optional bond index. If omitted, spectra for all stored bonds are
            returned.
        eps: Probability cutoff used before taking ``log``.

    Returns:
        A single sorted spectrum tensor when ``bond`` is provided, otherwise a
        list of spectrum tensors for every stored bond.

    Raises:
        ValueError: If ``bond`` is outside the stored Schmidt-weight range.
    """
    def spectrum_for_bond(bond_idx: int) -> torch.Tensor:
        _validate_bond(mps, bond_idx)
        lambdas = _as_schmidt_vector(mps.sWeight[bond_idx])
        probabilities = (lambdas ** 2).real
        total = probabilities.sum()
        if total <= 0:
            return torch.empty(0, device=probabilities.device, dtype=probabilities.dtype)
        probabilities = probabilities / total
        probabilities = probabilities[probabilities > eps]
        if probabilities.numel() == 0:
            return torch.empty(0, device=probabilities.device, dtype=probabilities.dtype)
        return torch.sort(-torch.log(probabilities)).values

    if bond is None:
        return [spectrum_for_bond(idx) for idx in range(len(mps.sWeight))]
    return spectrum_for_bond(bond)


def string_order_parameter(
    mps: Any,
    left_operator: Any,
    string_operator: Any,
    right_operator: Any,
    i: int,
    j: int,
    *,
    form: str = "B",
) -> torch.Tensor:
    """Compute a finite string order parameter.

    The measured operator is
    ``O_left(i) * prod_{k=i+1}^{j-1} O_string(k) * O_right(j)``. If
    ``j == i + 1``, the string product is empty and treated as identity.

    Args:
        mps: MPS-like object to measure.
        left_operator: Endpoint operator acting at site ``i``.
        string_operator: Operator repeated on sites between ``i`` and ``j``.
        right_operator: Endpoint operator acting at site ``j``.
        i: Left endpoint site.
        j: Right endpoint site. Must satisfy ``i < j``.
        form: MPS tensor form used for dense reconstruction.

    Returns:
        Scalar string order parameter as a tensor.

    Raises:
        ValueError: If endpoints or operator shapes are invalid, or if the MPS
            state has zero norm.
    """
    _validate_site(mps, i)
    _validate_site(mps, j)
    if i >= j:
        raise ValueError("string_order_parameter requires i < j.")

    state = _dense_state_from_mps(mps, form=form)
    left = _as_operator_tensor(left_operator, device=state.device, dtype=state.dtype, physical_dim=mps.chid)
    string = _as_operator_tensor(string_operator, device=state.device, dtype=state.dtype, physical_dim=mps.chid)
    right = _as_operator_tensor(right_operator, device=state.device, dtype=state.dtype, physical_dim=mps.chid)
    common_dtype = _promote_dtype(state.dtype, left.dtype, string.dtype, right.dtype)
    state = state.to(common_dtype)
    left = left.to(common_dtype)
    string = string.to(common_dtype)
    right = right.to(common_dtype)

    applied = _apply_local_operator(state, right, j, num_sites=mps.Nsites, physical_dim=mps.chid)
    for site in range(j - 1, i, -1):
        applied = _apply_local_operator(applied, string, site, num_sites=mps.Nsites, physical_dim=mps.chid)
    applied = _apply_local_operator(applied, left, i, num_sites=mps.Nsites, physical_dim=mps.chid)
    return _expectation(state, applied)


def ground_state_fidelity(mps_a: Any, mps_b: Any, *, form_a: str = "B", form_b: str = "B") -> torch.Tensor:
    """Compute the normalized ground-state fidelity between two MPS objects.

    The returned value is ``abs(<psi_a|psi_b>)`` after normalizing both dense
    states. The two MPS objects must describe the same Hilbert space.

    Args:
        mps_a: First MPS-like object.
        mps_b: Second MPS-like object.
        form_a: Tensor form used to reconstruct ``mps_a``.
        form_b: Tensor form used to reconstruct ``mps_b``.

    Returns:
        Real scalar tensor containing the fidelity.

    Raises:
        ValueError: If the two MPS objects have incompatible Hilbert spaces or
            if either state has zero norm.
    """
    if mps_a.Nsites != mps_b.Nsites or mps_a.chid != mps_b.chid:
        raise ValueError("MPS objects must have matching Nsites and chid values.")

    state_a = _dense_state_from_mps(mps_a, form=form_a)
    state_b = _dense_state_from_mps(mps_b, form=form_b)
    common_dtype = _promote_dtype(state_a.dtype, state_b.dtype)
    state_a = state_a.to(common_dtype)
    state_b = state_b.to(device=state_a.device, dtype=common_dtype)

    norm_a = torch.linalg.norm(state_a)
    norm_b = torch.linalg.norm(state_b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cannot compute fidelity for a zero-norm state.")
    return torch.abs(torch.vdot(state_a / norm_a, state_b / norm_b))


def many_body_polarization(
    mps: Any,
    number_operator: Any,
    *,
    origin: Union[int, float] = 0,
    form: str = "B",
) -> dict[str, torch.Tensor]:
    """Compute the Resta many-body polarization for a finite chain.

    The measured unitary is
    ``exp((2*pi*i/L) * sum_j (j + origin) * n_j)``. The returned polarization
    is ``angle(<U>) / (2*pi)``. Changing ``origin`` changes the phase convention;
    this matters in sectors with nonzero total number.

    Args:
        mps: MPS-like object to measure.
        number_operator: Local number operator ``n_j``.
        origin: Coordinate offset in the phase factor. Use ``0`` for sites
            ``0, ..., L-1`` and ``1`` for sites ``1, ..., L``.
        form: MPS tensor form used for dense reconstruction.

    Returns:
        Dictionary with keys:
            ``"polarization"``: Wrapped polarization in units of the lattice
                period.
            ``"expectation"``: Complex expectation value ``<U>``.
            ``"phase"``: Principal phase angle of ``<U>``.

    Raises:
        ValueError: If ``number_operator`` has an invalid shape or if the MPS
            state has zero norm.
    """
    state = _dense_state_from_mps(mps, form=form)
    number = _as_operator_tensor(number_operator, device=state.device, dtype=state.dtype, physical_dim=mps.chid)
    promoted_dtype = _promote_dtype(state.dtype, number.dtype)
    complex_dtype = torch.complex128 if promoted_dtype in (torch.float64, torch.complex128) else torch.complex64
    state = state.to(complex_dtype)
    number = number.to(complex_dtype)

    applied = state
    for site in range(mps.Nsites - 1, -1, -1):
        coordinate = float(site + origin)
        generator = (2j * torch.pi * coordinate / mps.Nsites) * number
        local_unitary = torch.matrix_exp(generator)
        applied = _apply_local_operator(
            applied,
            local_unitary,
            site,
            num_sites=mps.Nsites,
            physical_dim=mps.chid,
        )

    expectation = _expectation(state, applied)
    phase = torch.angle(expectation)
    polarization = phase / (2.0 * torch.pi)
    return {
        "polarization": polarization,
        "expectation": expectation,
        "phase": phase,
    }
