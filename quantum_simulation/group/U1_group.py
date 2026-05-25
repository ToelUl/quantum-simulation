import numpy as np
import torch

from .base import LieAlgebraBase, LieGroupBase


class U1LieAlgebra(LieAlgebraBase):
    """Implementation of the Lie algebra for the U(1) group."""

    def __init__(self, rep_dim: int = 1) -> None:
        """
        Initialize the U(1) Lie algebra.

        Args:
            rep_dim (int, optional): Representation dimension. Defaults to 1.
        """
        lie_alg_dim = 1  # For u(1), the Lie algebra dimension is 1
        super().__init__(rep_dim=rep_dim, lie_alg_dim=lie_alg_dim)
        self.generator = torch.tensor(1j, dtype=torch.complex64, requires_grad=False)

    def generators(self) -> list:
        """
        Return the generator of the U(1) Lie algebra.

        Returns:
            list[torch.Tensor]: List containing the generator tensor.
        """
        return [self.generator]

    def structure_constants(self) -> torch.Tensor:
        """
        Return the structure constants of the u(1) Lie algebra.

        Returns:
            torch.Tensor: A tensor of shape (1, 1, 1) with all zeros.
        """
        return torch.zeros((1, 1, 1), dtype=torch.float32)

    @staticmethod
    def _to_scalar(a: torch.Tensor) -> torch.Tensor:
        """
        Reduce a (1, 1) matrix-form Lie algebra element to its scalar representation.

        Accepts both the scalar form ``(...,)`` and the matrix form ``(..., 1, 1)``.
        """
        if a.dim() >= 2 and a.shape[-2:] == (1, 1):
            return a.squeeze(-1).squeeze(-1)
        return a

    def is_traceless(self, a: torch.Tensor) -> bool:
        """
        Check whether a u(1) Lie algebra element is traceless.

        Notes:
            For u(1), elements are 1x1 anti-Hermitian matrices of the form ``iθ``.
            Their trace equals the element itself, so this check returns True
            *only* for the zero element. The method is provided for API
            consistency with :meth:`SU2LieAlgebra.is_traceless`; it is **not**
            a meaningful structural property of u(1) (in particular, the
            generator ``i`` is not traceless).

        Args:
            a (torch.Tensor): Lie algebra element. Either a scalar tensor of
                shape ``(...,)`` (purely imaginary) or a matrix-form tensor of
                shape ``(..., 1, 1)``.

        Returns:
            bool: True if every entry of ``a`` is approximately zero.
        """
        a_scalar = self._to_scalar(a)
        return torch.allclose(a_scalar, torch.zeros_like(a_scalar), atol=1e-4)

    def inner_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hilbert–Schmidt inner product ``Tr(a† b) = conj(a) * b``
        for two u(1) elements.

        Args:
            a (torch.Tensor): First Lie algebra element of shape ``(...,)`` or ``(..., 1, 1)``.
            b (torch.Tensor): Second Lie algebra element of shape ``(...,)`` or ``(..., 1, 1)``.

        Returns:
            torch.Tensor: ``conj(a) * b`` (a complex scalar per batch element).
        """
        a_s = self._to_scalar(a)
        b_s = self._to_scalar(b)
        if not torch.is_complex(a_s):
            a_s = a_s.to(torch.complex64)
        if not torch.is_complex(b_s):
            b_s = b_s.to(torch.complex64)
        return a_s.conj() * b_s

    def is_orthogonal(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        """
        Check whether two u(1) elements are orthogonal under the
        Hilbert–Schmidt inner product.

        Notes:
            u(1) is one-dimensional, so any two non-zero elements are
            proportional and therefore *not* orthogonal. This method is
            provided for API consistency with :meth:`SU2LieAlgebra.is_orthogonal`.

        Args:
            a (torch.Tensor): First Lie algebra element.
            b (torch.Tensor): Second Lie algebra element.

        Returns:
            bool: True if ``⟨a, b⟩ ≈ 0``.
        """
        prod = self.inner_product(a, b)
        return torch.allclose(prod, torch.zeros_like(prod), atol=1e-4)

    def is_anti_hermitian(self, a: torch.Tensor) -> bool:
        """
        Check whether a u(1) element is anti-Hermitian.

        For 1x1 complex matrices, anti-Hermitian is equivalent to
        ``a + conj(a) = 0``, i.e. the real part vanishes. This is exactly the
        defining property of u(1) (every element has the form ``iθ`` with
        ``θ ∈ ℝ``).

        Args:
            a (torch.Tensor): Lie algebra element of shape ``(...,)`` or ``(..., 1, 1)``.

        Returns:
            bool: True if ``a`` is anti-Hermitian.
        """
        a_scalar = self._to_scalar(a)
        if not torch.is_complex(a_scalar):
            # A real (non-complex) tensor is anti-Hermitian only if it is zero.
            return torch.allclose(a_scalar, torch.zeros_like(a_scalar), atol=1e-4)
        return torch.allclose(a_scalar.real, torch.zeros_like(a_scalar.real), atol=1e-4)


class U1Group(LieGroupBase):
    """Implementation of the U(1) Lie group."""

    def __init__(self) -> None:
        """
        Initialize the U(1) group.
        """
        rep_dim = 1
        identity = torch.ones((rep_dim, rep_dim), dtype=torch.complex64, requires_grad=False)
        super().__init__(rep_dim=rep_dim, identity=identity)
        self.algebra = U1LieAlgebra()

    def elements(self) -> torch.Tensor:
        """
        Return a tensor containing group elements.

        Note:
            U(1) is continuous; listing all elements is not feasible.

        Raises:
            NotImplementedError: Always raised for continuous groups.
        """
        raise NotImplementedError("Continuous group; cannot list all elements.")

    def product(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        """
        Compute the product of two U(1) group elements (complex multiplication).

        Args:
            g1 (torch.Tensor): Group element 1.
            g2 (torch.Tensor): Group element 2.

        Returns:
            torch.Tensor: The product g1 * g2.

        Raises:
            ValueError: If g1 and g2 have incompatible shapes.
        """
        if g1.shape != g2.shape:
            raise ValueError("Group elements g1 and g2 must have the same shape.")
        return g1 * g2

    def inverse(self, g: torch.Tensor) -> torch.Tensor:
        """
        Compute the inverse of a U(1) group element (complex conjugate).

        Args:
            g (torch.Tensor): Group element.

        Returns:
            torch.Tensor: The inverse of g.
        """
        return torch.conj(g)

    def exponential_map(self, a: torch.Tensor) -> torch.Tensor:
        """
        Map an element from the Lie algebra to the U(1) group via the exponential map.

        Args:
            a (torch.Tensor): A real scalar or batch of scalars representing angles.

        Returns:
            torch.Tensor: The corresponding group element.
        """
        generator = self.algebra.generators()[0]
        return torch.exp(a * generator)

    def left_action(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the left group action on vectors in ℂⁿ.

        Args:
            g (torch.Tensor): Group element (scalar) with shape (...,).
            x (torch.Tensor): Vectors in ℂⁿ with shape (..., n).

        Returns:
            torch.Tensor: The transformed vectors.
        """
        return g * x

    def right_action(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the right group action on dual vectors in ℂⁿ.

        Args:
            g (torch.Tensor): Group element (scalar) with shape (...,).
            x (torch.Tensor): Dual vectors in ℂⁿ with shape (..., n).

        Returns:
            torch.Tensor: The transformed dual vectors.
        """
        return x * self.inverse(g)

    def random_element(
        self,
        sample_size: int = 1,
        generator: torch.Generator = None,
        apply_map: bool = True,
    ) -> torch.Tensor:
        """
        Generate a batch of random U(1) group elements.

        Args:
            sample_size (int, optional): Number of random elements to sample. Defaults to 1.
            generator (torch.Generator, optional): Random generator for reproducibility.
            apply_map (bool, optional): If True, return the element via exponential map;
                if False, return the raw angle. Defaults to True.

        Returns:
            torch.Tensor:
                - If apply_map is True: shape (sample_size, 1, 1).
                - Otherwise: shape (sample_size,).
        """
        device = self.identity.device
        theta = torch.rand(sample_size, generator=generator, device=device) * (2 * np.pi)
        if apply_map:
            g = self.exponential_map(theta)
            return g.view(sample_size, 1, 1)
        else:
            return theta

    @staticmethod
    def _to_scalar(g: torch.Tensor) -> torch.Tensor:
        """
        Reduce a (1, 1) matrix-form U(1) element to its scalar representation.

        Accepts both the scalar form ``(...,)`` and the matrix form ``(..., 1, 1)``.
        """
        if g.dim() >= 2 and g.shape[-2:] == (1, 1):
            return g.squeeze(-1).squeeze(-1)
        return g

    def is_unitary(self, g: torch.Tensor) -> bool:
        """
        Check whether a U(1) element is unitary, i.e. lies on the unit circle.

        For U(1), every group element has the form ``g = e^(iθ)`` (a complex
        scalar or, equivalently, a 1x1 complex matrix). The unitarity condition
        ``g̅ g = 1`` reduces to ``|g| = 1``.

        Args:
            g (torch.Tensor): Group element of shape ``(...,)`` or ``(..., 1, 1)``.

        Returns:
            bool: True if every element satisfies ``|g| ≈ 1``.
        """
        g_scalar = self._to_scalar(g)
        abs_g = g_scalar.abs()
        return torch.allclose(abs_g, torch.ones_like(abs_g), atol=1e-4)

    def is_determinant_one(self, g: torch.Tensor) -> bool:
        """
        Check whether ``|det(g)| = 1`` for a U(1) element.

        Notes:
            For 1x1 representations, ``det(g) = g`` itself, so this method is
            mathematically equivalent to :meth:`is_unitary`. As in
            :meth:`SU2Group.is_determinant_one`, the check is performed on
            ``|det(g)|`` rather than ``det(g)`` so that complex determinants
            on the unit circle are accepted.

        Args:
            g (torch.Tensor): Group element of shape ``(...,)`` or ``(..., 1, 1)``.

        Returns:
            bool: True if ``|det(g)| ≈ 1``.
        """
        g_scalar = self._to_scalar(g)
        abs_g = g_scalar.abs()
        return torch.allclose(abs_g, torch.ones_like(abs_g), atol=1e-4)
