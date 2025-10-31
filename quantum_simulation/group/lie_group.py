import numpy as np
import torch

from .base import LieAlgebraBase, LieGroupBase


class SU2LieAlgebra(LieAlgebraBase):
    """Implementation of the su(2) Lie algebra with arbitrary spin."""

    def __init__(self, spin: float = 0.5) -> None:
        """
        Initialize the su(2) Lie algebra for a given spin.

        Args:
            spin (float, optional): Spin value (e.g., 0.5, 1.0, 1.5, ...). Must be a positive multiple of 0.5.

        Raises:
            ValueError: If spin is not a positive multiple of 0.5.
        """
        if spin <= 0 or (2 * spin) % 1 != 0:
            raise ValueError("Spin must be a positive multiple of 0.5.")
        self.spin = spin
        rep_dim = int(2 * spin + 1)
        lie_alg_dim = 3  # For su(2), the Lie algebra dimension is 3
        super().__init__(rep_dim=rep_dim, lie_alg_dim=lie_alg_dim)

        # Generate spin matrices
        self.S_z = self._generate_sz()
        self.S_plus, self.S_minus = self._generate_s_plus_minus()
        self.S_x = 0.5 * (self.S_plus + self.S_minus)
        self.S_y = (self.S_plus - self.S_minus) / 2j

        # Pauli matrices
        self.sigma_x = self.S_x / self.spin
        self.sigma_y = self.S_y / self.spin
        self.sigma_z = self.S_z / self.spin

        # Generators: e_j = -i * S_j
        self.e_1 = -1j * self.S_x
        self.e_2 = -1j * self.S_y
        self.e_3 = -1j * self.S_z
        self.generators_list = [self.e_1, self.e_2, self.e_3]

    def _generate_sz(self) -> torch.Tensor:
        """
        Generate the S_z matrix.

        Returns:
            torch.Tensor: The S_z matrix of shape (dim, dim) with complex entries.
        """
        dim = int(2 * self.spin + 1)
        m_values = torch.tensor([self.spin - i for i in range(dim)], dtype=torch.complex64, requires_grad=False)
        return torch.diag(m_values)

    def _generate_s_plus_minus(self) -> tuple:
        """
        Generate the S_+ and S_- matrices.

        Returns:
            tuple: A tuple (S_plus, S_minus) of tensors.
        """
        dim = int(2 * self.spin + 1)
        sp = torch.zeros((dim, dim), dtype=torch.complex64, requires_grad=False)
        for i in range(dim - 1):
            j = self.spin - i  # Corresponds to m = s - i
            b_j = np.sqrt((self.spin + j) * (self.spin + 1 - j))
            sp[i, i + 1] = b_j
        sm = sp.transpose(-2, -1)
        return sp, sm

    def generators(self) -> torch.Tensor:
        """
        Return the generators of the Lie algebra.

        Returns:
            torch.Tensor: A tensor of shape (3, rep_dim, rep_dim) containing the generators.
        """
        return torch.stack(self.generators_list)

    def get_spin_operators(self) -> tuple:
        """
        Return the spin operators.

        Returns:
            tuple: (S_x, S_y, S_z, S_plus, S_minus)
        """
        return self.S_x, self.S_y, self.S_z, self.S_plus, self.S_minus

    def structure_constants(self) -> torch.Tensor:
        """
        Return the structure constants f_{i,j,k} of the su(2) Lie algebra.

        Returns:
            torch.Tensor: A tensor of shape (3, 3, 3) with complex entries.
        """
        f = torch.zeros((3, 3, 3), dtype=torch.complex64)
        f[0, 1, 2] = 1.0
        f[1, 2, 0] = 1.0
        f[2, 0, 1] = 1.0
        f[1, 0, 2] = -1.0
        f[2, 1, 0] = -1.0
        f[0, 2, 1] = -1.0
        return f

    def is_traceless(self, a: torch.Tensor) -> bool:
        """
        Check whether a Lie algebra element is traceless.

        Args:
            a (torch.Tensor): Lie algebra element.

        Returns:
            bool: True if "a" is traceless, False otherwise.

        Raises:
            ValueError: If "a" does not have the correct shape.
        """
        if a.shape[-2:] != (self.rep_dim, self.rep_dim):
            raise ValueError(f"Lie algebra element must have shape (..., {self.rep_dim}, {self.rep_dim}).")
        trace = torch.einsum("...ii->...", a)
        return torch.allclose(trace, torch.zeros_like(trace), atol=1e-4)

    def inner_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute the inner product of two Lie algebra elements.

        Args:
            a (torch.Tensor): First Lie algebra element.
            b (torch.Tensor): Second Lie algebra element.

        Returns:
            torch.Tensor: The inner product ⟨a, b⟩.

        Raises:
            ValueError: If a and b do not have the correct shapes.
        """
        if a.shape[-2:] != (self.rep_dim, self.rep_dim):
            raise ValueError(f"Lie algebra element 'a' must have shape (..., {self.rep_dim}, {self.rep_dim}).")
        if b.shape[-2:] != (self.rep_dim, self.rep_dim):
            raise ValueError(f"Lie algebra element 'b' must have shape (..., {self.rep_dim}, {self.rep_dim}).")
        prod = torch.einsum("...ij,...ji->...", a.conj().transpose(-2,-1), b)
        return prod / np.sum(np.arange(-self.spin, self.spin + 1) ** 2)

    def is_orthogonal(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        """
        Check whether two Lie algebra elements are orthogonal.

        Args:
            a (torch.Tensor): First Lie algebra element.
            b (torch.Tensor): Second Lie algebra element.

        Returns:
            bool: True if ⟨a, b⟩ = 0, False otherwise.

        Raises:
            ValueError: If a and b do not have the correct shapes.
        """
        inner_prod = self.inner_product(a, b)
        return torch.allclose(inner_prod, torch.zeros_like(inner_prod), atol=1e-4)

    def is_anti_hermitian(self, a: torch.Tensor) -> bool:
        """
        Check whether a Lie algebra element is anti-Hermitian.

        Args:
            a (torch.Tensor): Lie algebra element.

        Returns:
            bool: True if a is anti-Hermitian, False otherwise.

        Raises:
            ValueError: If a does not have the correct shape.
        """
        if a.shape[-2:] != (self.rep_dim, self.rep_dim):
            raise ValueError(f"Lie algebra element must have shape (..., {self.rep_dim}, {self.rep_dim}).")
        return torch.allclose(a.conj().transpose(-2, -1), -a, atol=1e-4)


class SU2Group(LieGroupBase):
    """Implementation of the SU(2) Lie group with arbitrary spin representation."""

    def __init__(self, spin: float = 0.5) -> None:
        """
        Initialize the SU(2) group for a given spin.

        Args:
            spin (float, optional): Spin value (e.g., 0.5, 1.0, 1.5, ...).

        Raises:
            ValueError: If spin is not a positive multiple of 0.5.
        """
        if spin <= 0 or (2 * spin) % 1 != 0:
            raise ValueError("Spin must be a positive multiple of 0.5.")
        algebra = SU2LieAlgebra(spin=spin)
        rep_dim = algebra.rep_dim
        identity = torch.eye(rep_dim, dtype=torch.complex64, requires_grad=False)
        super().__init__(rep_dim=rep_dim, identity=identity)
        self.spin = spin
        self.algebra = algebra

    def product(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        """
        Compute the product of two SU(2) group elements.

        Args:
            g1 (torch.Tensor): Group element 1.
            g2 (torch.Tensor): Group element 2.

        Returns:
            torch.Tensor: The product g1 ⋅ g2.

        Raises:
            ValueError: If g1 and g2 have incompatible shapes.
        """
        if g1.shape != g2.shape:
            raise ValueError("Group elements g1 and g2 must have the same shape.")
        return torch.matmul(g1, g2) if g1.dim() == 2 else torch.einsum("...ij,...jk->...ik", g1, g2)

    def inverse(self, g: torch.Tensor) -> torch.Tensor:
        """
        Compute the inverse of an SU(2) group element.

        Args:
            g (torch.Tensor): Group element.

        Returns:
            torch.Tensor: The inverse g⁻¹.
        """
        return g.conj().transpose(-2, -1)

    def exponential_map(self, a: torch.Tensor) -> torch.Tensor:
        """
        Map a Lie algebra element to a group element via the exponential map.

        Args:
            a (torch.Tensor): Real coefficients with shape (..., 3). The Lie algebra element is given by
                a[..., 0] * e_1 + a[..., 1] * e_2 + a[..., 2] * e_3, where e_j are the generators.

        Returns:
            torch.Tensor: The corresponding group element with shape (..., rep_dim, rep_dim).

        Raises:
            ValueError: If the last dimension of 'a' is not 3.
        """
        if a.shape[-1] != 3:
            raise ValueError("Input tensor a must have last dimension 3.")
        e1, e2, e3 = self.algebra.generators()
        e1 = e1.to(self.identity.device)
        e2 = e2.to(self.identity.device)
        e3 = e3.to(self.identity.device)
        alge_elem = a[..., 0, None, None] * e1 + a[..., 1, None, None] * e2 + a[..., 2, None, None] * e3
        return torch.matrix_exp(alge_elem)

    def left_action(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the left group action on a vector x in ℂⁿ (n = 2s + 1) using torch.matmul.

        Args:
            g (torch.Tensor): Group element with shape (..., rep_dim, rep_dim).
            x (torch.Tensor): Vector in ℂⁿ with shape (..., rep_dim).

        Returns:
            torch.Tensor: The transformed vector with shape (..., rep_dim).

        Raises:
            ValueError: If g or x do not have the correct shapes.
        """
        if g.shape[-2:] != (self.rep_dim, self.rep_dim):
            raise ValueError(f"Group element must have shape (..., {self.rep_dim}, {self.rep_dim}).")
        if x.shape[-1] != self.rep_dim:
            raise ValueError(f"Vector must have shape (..., {self.rep_dim}) or (..., {self.rep_dim}, {self.rep_dim}).")
        if x.shape[-2:] == (self.rep_dim, self.rep_dim):
            return torch.einsum("...ij,...jk->...ik", g, x)
        else:
            return torch.einsum("...ij,...j->...i", g, x)

    def right_action(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the right group action on a dual vector x in ℂⁿ (n = 2s + 1) using torch.matmul.

        Args:
            g (torch.Tensor): Group element with shape (..., rep_dim, rep_dim).
            x (torch.Tensor): Dual vector in ℂⁿ with shape (..., rep_dim).

        Returns:
            torch.Tensor: The transformed dual vector with shape (..., rep_dim).

        Raises:
            ValueError: If g or x do not have the correct shapes.
        """
        if g.shape[-2:] != (self.rep_dim, self.rep_dim):
            raise ValueError(f"Group element must have shape (..., {self.rep_dim}, {self.rep_dim}).")
        if x.shape[-1] != self.rep_dim:
            raise ValueError(f"Dual vector must have shape (..., {self.rep_dim}).")
        g_inv = self.inverse(g)
        if x.shape[-2:] == (self.rep_dim, self.rep_dim):
            return torch.einsum("...ij,...jk->...ik", x, g_inv)
        else:
            return torch.einsum("...i,...ij->...j", x, g_inv)

    def adjoint_action(self, g: torch.Tensor, alge_elem: torch.Tensor) -> torch.Tensor:
        """
        Apply the adjoint action of the group on a Lie algebra element.

        Args:
            g (torch.Tensor): Group element.
            alge_elem (torch.Tensor): Lie algebra element.

        Returns:
            torch.Tensor: The transformed Lie algebra element.

        Raises:
            ValueError: If g or alge_elem do not have the correct shapes.
        """
        if g.shape[-2:] != (self.rep_dim, self.rep_dim):
            raise ValueError(f"Group element must have shape (..., {self.rep_dim}, {self.rep_dim}).")
        g_inv = self.inverse(g)
        return torch.einsum("...ij,...jk,...kl->...il", g, alge_elem, g_inv)

    def random_element(
        self,
        sample_size: int = 1,
        generator: torch.Generator = None,
        apply_map: bool = True,
    ) -> torch.Tensor:
        """
        Generate a batch of random SU(2) group elements.

        Args:
            sample_size (int, optional): Number of random elements to sample. Defaults to 1.
            generator (torch.Generator, optional): Random generator for reproducibility.
            apply_map (bool, optional): If True, return the group element via exponential map;
                if False, return the raw parameter vector (shape: (sample_size, 3)). Defaults to True.

        Returns:
            torch.Tensor:
                - If apply_map is True: shape (sample_size, rep_dim, rep_dim).
                - Otherwise: shape (sample_size, 3).
        """
        device = self.identity.device
        theta = torch.rand(sample_size, generator=generator, device=device) * (2 * torch.pi)
        phi = torch.rand(sample_size, generator=generator, device=device) * (2 * torch.pi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        n_x = sin_theta * torch.cos(phi)
        n_y = sin_theta * torch.sin(phi)
        n_z = cos_theta
        a = torch.stack([n_x, n_y, n_z], dim=1,)  # shape (sample_size, 3)
        if apply_map:
            return self.exponential_map(a)
        else:
            return a

    def is_unitary(self, g: torch.Tensor) -> bool:
        """
        Check whether a group element is unitary.

        Args:
            g (torch.Tensor): Group element.

        Returns:
            bool: True if g is unitary, False otherwise.

        Raises:
            ValueError: If g does not have the correct shape.
        """
        if g.shape[-len(self.element_shape):] != self.element_shape:
            raise ValueError(f"Group element must have shape (..., {self.element_shape}).")
        if self.element_shape == ():
            return torch.allclose(torch.abs(g), torch.tensor(1.0, dtype=g.dtype, device=g.device), atol=1e-4)
        else:
            identity = self.identity.expand_as(g)
            return torch.allclose(torch.einsum("...ij,...jk->...ik", self.inverse(g), g), identity, atol=1e-4)

    def is_determinant_one(self, g: torch.Tensor) -> bool:
        """
        Check whether the determinant of a group element is 1.

        Args:
            g (torch.Tensor): Group element.

        Returns:
            bool: True if det(g) == 1, False otherwise.
        """
        det = self.determinant(g)
        det_abs = det.abs() if det.is_complex() else det
        return torch.allclose(det_abs, torch.tensor(1.0, dtype=det_abs.dtype, device=det.device), atol=1e-4)