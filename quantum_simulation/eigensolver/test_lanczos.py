import pytest
import torch
import math
import numpy as np
from .lanczos_solver import lanczos_ground_state


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]

@pytest.fixture
def simple_hamiltonian() -> torch.Tensor:
    """Provides a simple, small, and well-behaved Hermitian matrix."""
    return torch.tensor([
        [4.0, 1.0, 0.0, 0.5],
        [1.0, 3.0, 0.2, 0.0],
        [0.0, 0.2, 2.0, 0.3],
        [0.5, 0.0, 0.3, 5.0]
    ], dtype=torch.float64)  # Use float64 for high precision comparison


def apply_hamiltonian(vector: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """A simple linear operator that performs matrix-vector multiplication."""
    return H @ vector

def atol_for(dtype: torch.dtype) -> float:
    """Provides a reasonable absolute tolerance for energy comparisons by dtype.

    Args:
        dtype: The torch.dtype of the tensors being compared.

    Returns:
        The absolute tolerance value.
    """
    # Set a reasonable energy tolerance for different dtypes.
    if dtype in (torch.float64, torch.complex128):
        return 1e-10
    if dtype in (torch.float32, torch.complex64):
        return 5e-5
    return 1e-8

def make_random_hermitian(n: int, dtype: torch.dtype, device: str, seed: int = 0) -> torch.Tensor:
    """Creates a random N x N Hermitian matrix.

    Args:
        n: The dimension of the square matrix.
        dtype: The data type of the matrix (can be real or complex).
        device: The device to create the matrix on ('cpu' or 'cuda').
        seed: A random seed for reproducibility.

    Returns:
        A random Hermitian torch.Tensor.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    if dtype.is_complex:
        A = torch.randn((n, n), dtype=dtype, device=device, generator=g)
        # Make it Hermitian: H = (A + A^H) / 2
        H = 0.5 * (A + A.conj().T)
    else:
        A = torch.randn((n, n), dtype=dtype, device=device, generator=g)
        H = 0.5 * (A + A.T)
    # Slightly increase diagonal dominance to avoid instability with single precision.
    H = H + 0.1 * torch.eye(n, dtype=dtype, device=device)
    return H

def rayleigh(v: torch.Tensor, H: torch.Tensor) -> float:
    """Calculates the Rayleigh quotient <v|H|v> / <v|v>.

    Args:
        v: The vector.
        H: The Hermitian operator (matrix).

    Returns:
        The real-valued Rayleigh quotient.
    """
    v = v / torch.linalg.norm(v)
    num = (torch.vdot(v, H @ v) if v.dtype.is_complex else torch.dot(v, (H @ v)))
    return float(num.real.item() if v.dtype.is_complex else num.item())

def test_correctness_basic(simple_hamiltonian):
    """
    Tests if the Lanczos solver finds the correct ground energy with a fixed seed.
    This is a basic sanity check.
    """
    # Set a seed for reproducibility
    torch.manual_seed(0)

    H = simple_hamiltonian
    dim = H.shape[0]

    true_eigenvalues, _ = torch.linalg.eigh(H)
    true_ground_energy = true_eigenvalues[0].item()

    initial_vector = torch.rand(dim, dtype=H.dtype)
    _, found_energy = lanczos_ground_state(
        initial_vector,
        apply_hamiltonian,
        operator_args=(H,),
        num_restarts=8,
        krylov_dim=4
    )

    assert np.isclose(found_energy, true_ground_energy, atol=1e-8)


def test_zero_initial_vector(simple_hamiltonian):
    """Tests if the function handles a zero initial vector gracefully."""
    H = simple_hamiltonian
    dim = H.shape[0]

    initial_vector = torch.zeros(dim, dtype=H.dtype)

    found_vector, found_energy = lanczos_ground_state(
        initial_vector,
        apply_hamiltonian,
        operator_args=(H,),
    )

    assert found_vector is not None
    assert isinstance(found_energy, float)
    assert np.isclose(torch.linalg.norm(found_vector).item(), 1.0)


# --- Combined and Robust Test for Devices and Data Types ---

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_solver_on_various_configs(simple_hamiltonian, device, dtype):
    """
    Tests the function's stability across different devices and data types
    with a fixed random seed to ensure reproducibility.
    """
    # Set a seed for reproducible random initial vector
    torch.manual_seed(42)

    H = simple_hamiltonian.to(device=device, dtype=dtype)
    dim = H.shape[0]

    true_eigenvalues, _ = torch.linalg.eigh(H)
    true_ground_energy = true_eigenvalues[0].real.item()

    # Create initial vector on the correct device and with the correct dtype
    initial_vector = torch.rand(dim, device=device, dtype=dtype)

    # Increase restarts to ensure convergence, especially for lower precision
    _, found_energy = lanczos_ground_state(
        initial_vector,
        apply_hamiltonian,
        operator_args=(H,),
        num_restarts=8,
        krylov_dim=4
    )

    # Use a looser tolerance for float32 and complex64
    tolerance = 1e-8 if dtype == torch.float64 else 1e-5
    assert np.isclose(found_energy, true_ground_energy, atol=tolerance)


def test_restart_with_pro(simple_hamiltonian):
    """Tests that advanced features (restarts, PRO, residual tol) run and converge."""
    torch.manual_seed(10) # Add seed for reproducibility
    H = simple_hamiltonian.to(dtype=torch.float32)
    v0 = torch.rand(H.size(0), dtype=H.dtype)
    true_eigs, _ = torch.linalg.eigh(H)
    E0 = true_eigs[0].item()

    gs, E = lanczos_ground_state(
        v0, lambda x, A: A @ x, (H,),
        num_restarts=6, krylov_dim=8,
        pro=True,
        ritz_residual_tol=1e-8,
    )
    assert abs(E - E0) < 1e-5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64])
def test_pro_correctness_small(device, dtype):
    """Tests if PRO gives the correct energy for small systems."""
    torch.manual_seed(0)
    n = 16
    H = make_random_hermitian(n, dtype, device, seed=123)
    true_eigs, _ = torch.linalg.eigh(H)
    E0 = float(true_eigs[0].real.item())
    v0 = torch.randn(n, dtype=dtype, device=device)

    gs, E = lanczos_ground_state(
        v0, lambda x, A: A @ x, (H,),
        num_restarts=6, krylov_dim=12,
        pro=True,
        ritz_residual_tol=1e-9 if dtype in (torch.float64, torch.complex128) else 1e-6,
    )
    assert np.isclose(E, E0, atol=atol_for(dtype))


def test_restart_monotonic_energy_cpu64():
    """Tests that energy is monotonically non-increasing with more restarts."""
    device, dtype = "cpu", torch.float64
    torch.manual_seed(1)
    n = 32
    H = make_random_hermitian(n, dtype, device, seed=7)
    true_eigs, _ = torch.linalg.eigh(H)
    E0 = float(true_eigs[0].item())
    v0 = torch.randn(n, dtype=dtype, device=device)

    Es = []
    for R in (1, 3, 6):
        _, E = lanczos_ground_state(
            v0, lambda x, A: A @ x, (H,),
            num_restarts=R, krylov_dim=12,
            pro=True,
            ritz_residual_tol=1e-12,
        )
        Es.append(E)
    assert Es[0] >= Es[1] - 1e-12 and Es[1] >= Es[2] - 1e-12
    assert abs(Es[-1] - E0) < atol_for(dtype)


@pytest.mark.parametrize("device", DEVICES)
def test_pro_not_worse_than_no_pro(device):
    """Tests that enabling PRO is not worse than disabling it."""
    dtype = torch.float32  # Single precision best shows the stability benefits of PRO.
    torch.manual_seed(2)
    n = 24
    H = make_random_hermitian(n, dtype, device, seed=9)
    v0 = torch.randn(n, dtype=dtype, device=device)

    _, E_no = lanczos_ground_state(
        v0, lambda x, A: A @ x, (H,),
        num_restarts=5, krylov_dim=12,
        pro=False,
        ritz_residual_tol=1e-6,
    )
    _, E_pro = lanczos_ground_state(
        v0, lambda x, A: A @ x, (H,),
        num_restarts=5, krylov_dim=12,
        pro=True,
        ritz_residual_tol=1e-6,
    )
    # Allow for minor fluctuations, but PRO should not be significantly worse.
    assert E_pro <= E_no + 2e-5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_init_and_seed_reproducibility(device, dtype):
    """Tests for reproducibility from a zero vector with a fixed seed."""
    torch.manual_seed(42)
    n = 20
    H = make_random_hermitian(n, dtype, device, seed=21)
    v0 = torch.zeros(n, dtype=dtype, device=device)

    gs1, E1 = lanczos_ground_state(
        v0, lambda x, A: A @ x, (H,),
        num_restarts=6, krylov_dim=12,
        pro=True,
        ritz_residual_tol=1e-8 if dtype == torch.float64 else 1e-6,
    )
    torch.manual_seed(42)
    gs2, E2 = lanczos_ground_state(
        v0, lambda x, A: A @ x, (H,),
        num_restarts=6, krylov_dim=12,
        pro=True,
        ritz_residual_tol=1e-8 if dtype == torch.float64 else 1e-6,
    )
    # The vector's phase/sign might differ, but the energy should be identical.
    assert np.isclose(E1, E2, atol=atol_for(dtype))


def test_beta_tol_early_stop_sanity():
    """Tests that a large beta_tol forces early, stable termination."""
    device, dtype = "cpu", torch.float64
    torch.manual_seed(5)
    n = 32
    H = make_random_hermitian(n, dtype, device, seed=33)
    v0 = torch.randn(n, dtype=dtype, device=device)
    rq0 = rayleigh(v0, H)

    _, E = lanczos_ground_state(
        v0, lambda x, A: A @ x, (H,),
        num_restarts=2, krylov_dim=6,
        pro=True,
        beta_tol=1e-2,                      # Intentionally large to force early stop.
        ritz_residual_tol=None,
    )
    assert math.isfinite(E)
    # The result should not be worse than the initial Rayleigh quotient (Ritz principle).
    assert E <= rq0 + 1e-12


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_linear_operator_interface(device, dtype):
    """Tests that the operator API works with a generic matvec function."""
    torch.manual_seed(6)
    n = 28
    H = make_random_hermitian(n, dtype, device, seed=77)
    true_eigs, _ = torch.linalg.eigh(H)
    E0 = float(true_eigs[0].real.item())
    v0 = torch.randn(n, dtype=dtype, device=device)

    def mv(x, A):
        # Simulate a general linear operator that only performs matvec.
        return A @ x

    _, E = lanczos_ground_state(
        v0, mv, (H,),
        num_restarts=5, krylov_dim=12,
        pro=True,
        ritz_residual_tol=1e-6,
    )
    assert np.isclose(E, E0, atol=atol_for(dtype))


def test_nearly_degenerate_groundstate_cpu64():
    """Tests that the solver finds the correct ground state with nearly degenerate eigenvalues."""
    device, dtype = "cpu", torch.float64
    torch.manual_seed(7)
    n = 20
    # Construct a nearly degenerate spectrum via similarity transform.
    diag = torch.linspace(0.0, 5.0, n, dtype=dtype, device=device)
    diag[0] = 0.0
    diag[1] = 1e-4  # Almost degenerate with the ground state.
    D = torch.diag(diag)
    A = torch.randn((n, n), dtype=dtype, device=device)
    Q, _ = torch.linalg.qr(A)  # Random orthogonal matrix.
    H = Q @ D @ Q.T

    true_eigs, _ = torch.linalg.eigh(H)
    E0 = float(true_eigs[0].item())
    v0 = torch.randn(n, dtype=dtype, device=device)

    _, E = lanczos_ground_state(
        v0, lambda x, A: A @ x, (H,),
        num_restarts=8, krylov_dim=18,
        pro=True,
        ritz_residual_tol=1e-12,
    )
    assert abs(E - E0) < 2e-10

