import math

import torch

from .base import LieAlgebraBase, LieGroupBase


class URLieAlgebra(LieAlgebraBase):
    """Implementation of the Lie algebra u(r) in the defining matrix representation."""

    def __init__(self, rank: int) -> None:
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError("rank must be a positive integer.")
        self.rank = rank
        super().__init__(rep_dim=rank, lie_alg_dim=rank ** 2)

        generators = self._build_generators()
        self.register_buffer("generators_tensor", generators)

    def _build_generators(self) -> torch.Tensor:
        gens: list[torch.Tensor] = []
        dtype = torch.complex64
        sqrt2 = math.sqrt(2.0)

        # Diagonal generators D_a = i E_aa.
        for a in range(self.rank):
            g = torch.zeros((self.rank, self.rank), dtype=dtype)
            g[a, a] = 1j
            gens.append(g)

        # Off-diagonal generators in fixed lexicographic order.
        for a in range(self.rank):
            for b in range(a + 1, self.rank):
                skew_real = torch.zeros((self.rank, self.rank), dtype=dtype)
                skew_real[a, b] = 1.0 / sqrt2
                skew_real[b, a] = -1.0 / sqrt2
                gens.append(skew_real)

        for a in range(self.rank):
            for b in range(a + 1, self.rank):
                sym_imag = torch.zeros((self.rank, self.rank), dtype=dtype)
                sym_imag[a, b] = 1j / sqrt2
                sym_imag[b, a] = 1j / sqrt2
                gens.append(sym_imag)

        return torch.stack(gens)

    def generators(self) -> torch.Tensor:
        return self.generators_tensor

    def matrix_from_coefficients(self, coeffs: torch.Tensor) -> torch.Tensor:
        if coeffs.shape[-1] != self.lie_alg_dim:
            raise ValueError(f"Coefficient tensor must have last dimension {self.lie_alg_dim}.")
        gens = self.generators().to(coeffs.device)
        coeffs = coeffs.to(gens.dtype)
        return torch.einsum("...g,gij->...ij", coeffs, gens)

    def coefficients_from_matrix(self, alge_elem: torch.Tensor) -> torch.Tensor:
        if alge_elem.shape[-2:] != (self.rank, self.rank):
            raise ValueError(f"Lie algebra element must have shape (..., {self.rank}, {self.rank}).")
        gens = self.generators().to(alge_elem.device)
        coeffs = torch.einsum("gij,...ij->...g", gens.conj(), alge_elem)
        return coeffs.real

    def structure_constants(self) -> torch.Tensor:
        gens = self.generators()
        brackets = LieAlgebraBase.lie_bracket(gens[:, None, :, :], gens[None, :, :, :])
        coeffs = self.coefficients_from_matrix(brackets)
        return coeffs.to(torch.float32)

    def inner_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape[-2:] != (self.rank, self.rank):
            raise ValueError(f"Lie algebra element 'a' must have shape (..., {self.rank}, {self.rank}).")
        if b.shape[-2:] != (self.rank, self.rank):
            raise ValueError(f"Lie algebra element 'b' must have shape (..., {self.rank}, {self.rank}).")
        prod = torch.einsum("...ij,...ij->...", a.conj(), b)
        return prod.real

    def is_orthogonal(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        inner_prod = self.inner_product(a, b)
        return torch.allclose(inner_prod, torch.zeros_like(inner_prod), atol=1e-4)

    def is_anti_hermitian(self, a: torch.Tensor) -> bool:
        if a.shape[-2:] != (self.rank, self.rank):
            raise ValueError(f"Lie algebra element must have shape (..., {self.rank}, {self.rank}).")
        return torch.allclose(a.conj().transpose(-2, -1), -a, atol=1e-4)

    def is_traceless(self, a: torch.Tensor) -> bool:
        if a.shape[-2:] != (self.rank, self.rank):
            raise ValueError(f"Lie algebra element must have shape (..., {self.rank}, {self.rank}).")
        trace = torch.einsum("...ii->...", a)
        return torch.allclose(trace, torch.zeros_like(trace), atol=1e-4)


class URankGroup(LieGroupBase):
    """Canonical implementation of the Lie group U(r) in the defining matrix representation."""

    def __init__(self, rank: int) -> None:
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError("rank must be a positive integer.")
        identity = torch.eye(rank, dtype=torch.complex64)
        super().__init__(rep_dim=rank, identity=identity)
        self.rank = rank
        self.algebra = URLieAlgebra(rank=rank)

    def elements(self) -> torch.Tensor:
        raise NotImplementedError("Continuous group; cannot list all elements.")

    def product(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        if g1.shape != g2.shape:
            raise ValueError("Group elements g1 and g2 must have the same shape.")
        return torch.matmul(g1, g2)

    def inverse(self, g: torch.Tensor) -> torch.Tensor:
        if g.shape[-2:] != (self.rank, self.rank):
            raise ValueError(f"Group element must have shape (..., {self.rank}, {self.rank}).")
        return g.conj().transpose(-2, -1)

    def exponential_map(self, a: torch.Tensor) -> torch.Tensor:
        if a.shape[-1] != self.algebra.lie_alg_dim:
            raise ValueError(f"Input tensor a must have last dimension {self.algebra.lie_alg_dim}.")
        alge_elem = self.algebra.matrix_from_coefficients(a)
        return torch.matrix_exp(alge_elem)

    def left_action(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if g.shape[-2:] != (self.rank, self.rank):
            raise ValueError(f"Group element must have shape (..., {self.rank}, {self.rank}).")
        if x.shape[-1] != self.rank:
            raise ValueError(f"Vector must have shape (..., {self.rank}) or (..., {self.rank}, {self.rank}).")
        if x.shape[-2:] == (self.rank, self.rank):
            return torch.einsum("...ij,...jk->...ik", g, x)
        return torch.einsum("...ij,...j->...i", g, x)

    def right_action(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if g.shape[-2:] != (self.rank, self.rank):
            raise ValueError(f"Group element must have shape (..., {self.rank}, {self.rank}).")
        if x.shape[-1] != self.rank:
            raise ValueError(f"Dual vector must have shape (..., {self.rank}) or (..., {self.rank}, {self.rank}).")
        g_inv = self.inverse(g)
        if x.shape[-2:] == (self.rank, self.rank):
            return torch.einsum("...ij,...jk->...ik", x, g_inv)
        return torch.einsum("...i,...ij->...j", x, g_inv)

    def adjoint_action(self, g: torch.Tensor, alge_elem: torch.Tensor) -> torch.Tensor:
        if g.shape[-2:] != (self.rank, self.rank):
            raise ValueError(f"Group element must have shape (..., {self.rank}, {self.rank}).")
        if alge_elem.shape[-2:] != (self.rank, self.rank):
            raise ValueError(f"Lie algebra element must have shape (..., {self.rank}, {self.rank}).")
        g_inv = self.inverse(g)
        return torch.einsum("...ij,...jk,...kl->...il", g, alge_elem, g_inv)

    def matrix_representation(self, g: torch.Tensor) -> torch.Tensor:
        return g

    def left_action_on_Rn(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.left_action(g, x)

    def right_action_on_Rn(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.right_action(g, x)

    def normalize_group_elements(self, g: torch.Tensor) -> torch.Tensor:
        if g.shape[-2:] != (self.rank, self.rank):
            raise ValueError(f"Group element must have shape (..., {self.rank}, {self.rank}).")
        g = g.to(self.identity.dtype)
        u, _, vh = torch.linalg.svd(g, full_matrices=False)
        return torch.matmul(u, vh)

    def random_element(
        self,
        sample_size: int = 1,
        generator: torch.Generator = None,
        apply_map: bool = True,
    ) -> torch.Tensor:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive.")
        device = self.identity.device
        coeffs = torch.randn(
            sample_size,
            self.algebra.lie_alg_dim,
            generator=generator,
            device=device,
            dtype=self.identity.real.dtype,
        )
        if apply_map:
            return self.exponential_map(coeffs)
        return coeffs

    def is_unitary(self, g: torch.Tensor) -> bool:
        if g.shape[-2:] != (self.rank, self.rank):
            raise ValueError(f"Group element must have shape (..., {self.rank}, {self.rank}).")
        identity = self.identity.expand_as(g)
        return torch.allclose(torch.matmul(self.inverse(g), g), identity, atol=1e-4)

    def is_determinant_one(self, g: torch.Tensor) -> bool:
        det = self.determinant(g)
        det_abs = det.abs() if det.is_complex() else det
        return torch.allclose(det_abs, torch.ones_like(det_abs), atol=1e-4)
