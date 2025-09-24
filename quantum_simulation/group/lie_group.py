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
        self.S_z = self._generate_S_z()
        self.S_plus, self.S_minus = self._generate_S_plus_minus()
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

    def _generate_S_z(self) -> torch.Tensor:
        """
        Generate the S_z matrix.

        Returns:
            torch.Tensor: The S_z matrix of shape (dim, dim) with complex entries.
        """
        dim = int(2 * self.spin + 1)
        m_values = torch.tensor([self.spin - i for i in range(dim)], dtype=torch.complex64, requires_grad=False)
        return torch.diag(m_values)

    def _generate_S_plus_minus(self) -> tuple:
        """
        Generate the S_+ and S_- matrices.

        Returns:
            tuple: A tuple (S_plus, S_minus) of tensors.
        """
        dim = int(2 * self.spin + 1)
        S_plus = torch.zeros((dim, dim), dtype=torch.complex64, requires_grad=False)
        for i in range(dim - 1):
            j = self.spin - i  # Corresponds to m = s - i
            b_j = np.sqrt((self.spin + j) * (self.spin + 1 - j))
            S_plus[i, i + 1] = b_j
        S_minus = S_plus.transpose(-2, -1)
        return S_plus, S_minus

    def generators(self) -> list:
        """
        Return the generators e_j = -i * S_j.

        Returns:
            list[torch.Tensor]: List of generator tensors.
        """
        return self.generators_list

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

    def elements(self) -> torch.Tensor:
        """
        Return a tensor containing group elements.

        Note:
            SU(2) is continuous; listing all elements is not feasible.

        Raises:
            NotImplementedError: Always raised for continuous groups.
        """
        raise NotImplementedError("Continuous group; cannot list all elements.")

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
        return torch.matmul(g1, g2) if g1.dim() == 2 else torch.bmm(g1, g2)

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
            a (torch.Tensor): Real coefficients with shape (..., 3).

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

    def left_action_on_Cn(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the left group action on a vector x in ℂⁿ (n = 2s + 1) using torch.matmul.

        Args:
            g (torch.Tensor): Group element with shape (..., rep_dim, rep_dim).
            x (torch.Tensor): Vector in ℂⁿ with shape (..., rep_dim).

        Returns:
            torch.Tensor: The transformed vector with shape (..., rep_dim).
        """
        if g.shape[-2:] != (self.rep_dim, self.rep_dim):
            raise ValueError(f"Group element must have shape (..., {self.rep_dim}, {self.rep_dim}).")
        if x.shape[-1] != self.rep_dim:
            raise ValueError(f"Vector must have shape (..., {self.rep_dim}) or (..., {self.rep_dim}, {self.rep_dim}).")
        if x.shape[-2:] == (self.rep_dim, self.rep_dim):
            return torch.einsum("...ij,...jk->...ik", g, x)
        else:
            return torch.einsum("...ij,...j->...i", g, x)

    def right_action_on_Cn(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the right group action on a dual vector x in ℂⁿ (n = 2s + 1) using torch.matmul.

        Args:
            g (torch.Tensor): Group element with shape (..., rep_dim, rep_dim).
            x (torch.Tensor): Dual vector in ℂⁿ with shape (..., rep_dim).

        Returns:
            torch.Tensor: The transformed dual vector with shape (..., rep_dim).
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
