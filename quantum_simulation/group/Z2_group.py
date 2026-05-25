import torch

from .base import GroupBase


class Z2Group(GroupBase):
    r"""The discrete group :math:`\mathbb{Z}_2 = \{+1, -1\}`.

    :math:`\mathbb{Z}_2` is represented here in the simplest possible way:
    raw elements are real signs ``{+1, -1}``, and the matrix representation is
    a real ``(1, 1)`` scalar matrix. This keeps the implementation minimal while
    still matching the scalar-matrix interface expected by the lattice code.

    Conventions:
        * ``rep_dim = 1`` (scalar matrix representation).
        * Group elements are stored in their *raw* form as real signs
          ``s ∈ {+1, -1}`` (``float32``). The matrix representation wraps each
          sign into a ``(1, 1)`` real tensor via
          :meth:`matrix_representation`, which is what ``random_element`` with
          ``apply_map=True`` returns.
        * The group operation is multiplication.
        * Every element is its own inverse, i.e. ``s * s = +1``.

    Example:
        >>> import torch
        >>> g = Z2Group()
        >>> g.rep_dim
        1
        >>> g.elements()
        tensor([ 1., -1.])
        >>> mats = g.random_element(sample_size=4)   # shape (4, 1, 1) real
        >>> mats.dtype
        torch.float32
    """

    def __init__(self) -> None:
        """Initialise the :math:`\\mathbb{Z}_2` group."""
        rep_dim = 1
        identity = torch.ones((rep_dim, rep_dim), dtype=torch.float32)
        super().__init__(rep_dim=rep_dim, identity=identity)
        self.order = 2
        self.register_buffer(
            "group_elements",
            torch.tensor([1.0, -1.0], dtype=torch.float32),
        )

    # ------------------------------------------------------------
    # Discrete group API
    # ------------------------------------------------------------
    def elements(self) -> torch.Tensor:
        """Return all raw elements :math:`\\{+1, -1\\}` as a ``(2,)`` float tensor."""
        return self.group_elements

    def product(self, h: torch.Tensor, h_prime: torch.Tensor) -> torch.Tensor:
        """Group product: ordinary multiplication of ``±1`` signs.

        Args:
            h (torch.Tensor): First element(s) with values in ``{+1, -1}``.
            h_prime (torch.Tensor): Second element(s) with the same shape as ``h``.

        Returns:
            torch.Tensor: ``h * h_prime``.

        Raises:
            ValueError: If ``h`` and ``h_prime`` have mismatched shapes.
        """
        if h.shape != h_prime.shape:
            raise ValueError("Group elements h and h_prime must have the same shape.")
        return h * h_prime

    def inverse(self, h: torch.Tensor) -> torch.Tensor:
        """Inverse of a :math:`\\mathbb{Z}_2` element.

        Every element of :math:`\\mathbb{Z}_2` is its own inverse.
        """
        return h.clone()

    def left_action_on_Rn(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Left action of ``h`` on a real vector ``x``: scalar multiplication.

        Args:
            h (torch.Tensor): Raw sign (``±1``), scalar or batch, or a
                ``(..., 1, 1)`` matrix-form element.
            x (torch.Tensor): Real vector(s). Any shape; broadcasting follows
                standard PyTorch rules.

        Returns:
            torch.Tensor: ``h * x`` broadcast to ``x``'s shape.
        """
        sign = h.squeeze(-1).squeeze(-1) if h.dim() >= 2 else h
        sign = sign.to(x.dtype)
        # Broadcast scalar sign across the trailing dims of ``x``.
        while sign.dim() < x.dim():
            sign = sign.unsqueeze(-1)
        return sign * x

    def right_action_on_Rn(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Right action of ``h^{-1}`` on ``x``.

        Because every element of :math:`\\mathbb{Z}_2` is self-inverse, the
        right action coincides with the left action.
        """
        return self.left_action_on_Rn(h, x)

    def matrix_representation(self, h: torch.Tensor) -> torch.Tensor:
        """Wrap a raw sign ``±1`` as a ``(..., 1, 1)`` real matrix.

        If ``h`` is already in matrix form ``(..., 1, 1)``, it is returned
        unchanged up to dtype normalization.

        Args:
            h (torch.Tensor): A raw sign or a batch of signs.

        Returns:
            torch.Tensor: Real tensor of shape ``(..., 1, 1)``.
        """
        if h.dim() >= 2 and h.shape[-2:] == (1, 1):
            return h.to(self.identity.dtype)
        return h.to(self.identity.dtype).unsqueeze(-1).unsqueeze(-1)

    def determinant(self, h: torch.Tensor) -> torch.Tensor:
        """Determinant of the ``(1, 1)`` representation — equal to the element itself.

        Accepts both the raw-sign form and the matrix form.
        """
        if h.dim() >= 2 and h.shape[-2:] == (1, 1):
            return h.squeeze(-1).squeeze(-1)
        return h.to(torch.float32)

    def normalize_group_elements(self, h: torch.Tensor) -> torch.Tensor:
        """Normalise raw signs ``{+1, -1}`` into the canonical ``[-1, 1]`` range.

        :math:`\\mathbb{Z}_2` signs already live in ``[-1, 1]``, so this is the
        identity map (kept for API consistency with the rest of the project).
        """
        return h.to(torch.float32)

    def random_element(
        self,
        sample_size: int = 1,
        generator: torch.Generator = None,
        apply_map: bool = True,
    ) -> torch.Tensor:
        """Sample uniformly from :math:`\\mathbb{Z}_2`.

        Args:
            sample_size (int, optional): Number of elements to sample. Defaults to 1.
            generator (torch.Generator, optional): Random generator for reproducibility.
            apply_map (bool, optional): If True, return the matrix-form
                ``(sample_size, 1, 1)`` real representation (compatible with
                ``generate_wilson_loops`` / ``Plaquette``). If False, return
                the raw real signs of shape ``(sample_size,)``. Defaults to True.

        Returns:
            torch.Tensor:
                * If ``apply_map`` is True: real tensor of shape
                  ``(sample_size, 1, 1)``.
                * Otherwise: real tensor of shape ``(sample_size,)`` with
                  values in ``{+1, -1}``.
        """
        device = self.identity.device
        indices = torch.randint(
            low=0, high=self.order, size=(sample_size,),
            generator=generator, device=device,
        )
        signs = self.group_elements[indices]
        if apply_map:
            return self.matrix_representation(signs)
        return signs

    def is_unitary(self, h: torch.Tensor) -> bool:
        r"""Check whether every element satisfies :math:`|h| \approx 1`.

        Accepts both raw-sign and matrix-form tensors.
        """
        scalar = h.squeeze(-1).squeeze(-1) if h.dim() >= 2 else h
        abs_h = scalar.abs().to(torch.float32)
        return torch.allclose(abs_h, torch.ones_like(abs_h), atol=1e-4)

    def is_determinant_one(self, h: torch.Tensor) -> bool:
        r"""Check whether :math:`|\det(h)| \approx 1`.

        For the ``(1, 1)`` representation, ``det(h) = h`` itself, so this is
        equivalent to :meth:`is_unitary`.
        """
        return self.is_unitary(h)

    def is_in_group(self, h: torch.Tensor) -> bool:
        """Check whether every entry of ``h`` equals :math:`+1` or :math:`-1`.

        Accepts raw-sign tensors and matrix-form tensors.
        """
        scalar = h.squeeze(-1).squeeze(-1) if h.dim() >= 2 else h
        scalar = scalar.to(torch.float32)
        dist = torch.minimum((scalar - 1.0).abs(), (scalar + 1.0).abs())
        return torch.allclose(dist, torch.zeros_like(dist), atol=1e-5)
