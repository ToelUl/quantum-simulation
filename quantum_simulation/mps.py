import torch
import torch.nn as nn
from typing import List, Optional, Any

# ==============================================================================
#                      Matrix Product State (MPS) Class
# ==============================================================================

class MPS(nn.Module):
    """Encapsulates a Matrix Product State (MPS) and its operations.

    This class manages the MPS tensors (A-form, B-form) and Schmidt values (sWeight),
    ensuring the state remains canonical. It is initialized as a right-canonical
    random MPS, with the orthogonality center at site 0, for numerical stability.

    Attributes:
        Nsites (int): Number of sites in the MPS.
        chid (int): Physical dimension of each site.
        ortho_center (int): The current site of the orthogonality center.
    """
    def __init__(self,
                 Nsites: int,
                 chid: int,
                 initial_bond_dim: int,
                 device: Any,
                 dtype: torch.dtype,
                 seed: Optional[int] = None):
        """Initializes the MPS as a right-canonical random state.

        Args:
            Nsites (int): Number of sites in the chain.
            chid (int): Physical dimension of each site.
            initial_bond_dim (int): The initial maximum bond dimension for the random MPS.
            device: The torch device.
            dtype: The torch dtype.
            seed (Optional[int]): A random seed for reproducibility.
        """
        super().__init__()
        # Store attributes
        self.Nsites = Nsites
        self.chid = chid
        self.device = device
        self.dtype = dtype
        self.ortho_center = 0  # Initially right-canonical at site 0

        # --- Create and canonicalize a random MPS ---
        print("Initializing MPS as a right-canonical random state...")
        # A will hold the MPS tensors. After the loop, A[p] for p > 0 will be
        # in B-form (right-canonical). A[0] will be the orthogonality center.
        A_tensors = random_mps(Nsites, chid, initial_bond_dim, dtype=dtype, device=device, seed=seed)

        for p in range(Nsites - 1, 0, -1):
            chil, d, chir = A_tensors[p].shape
            mat = A_tensors[p].reshape(chil, d * chir).T
            Q, R = torch.linalg.qr(mat)
            new_chil = Q.shape[1]
            A_tensors[p] = Q.T.reshape(new_chil, d, chir)
            A_tensors[p - 1] = torch.einsum('lsc,ck->lsk', A_tensors[p - 1], R.T)

        A_tensors[0] /= torch.linalg.norm(A_tensors[0])
        B_tensors = [t.clone() for t in A_tensors]
        sWeight_tensors = [torch.ones(1, device=device, dtype=dtype) for _ in range(Nsites + 1)]

        # Use ParameterList to ensure tensors are properly registered as part of the
        # module's state, allowing them to be moved with .to() etc.
        # requires_grad=False as they are not optimized via gradients.
        self._A = nn.ParameterList([nn.Parameter(t, requires_grad=False) for t in A_tensors])
        self._B = nn.ParameterList([nn.Parameter(t, requires_grad=False) for t in B_tensors])
        self._sWeight = nn.ParameterList([nn.Parameter(t, requires_grad=False) for t in sWeight_tensors])
        print(f"Initialization complete. Ortho center at site {self.ortho_center}.")

    @property
    def A(self) -> nn.ParameterList:
        """Returns the A-form tensors of the MPS."""
        return self._A

    @property
    def B(self) -> nn.ParameterList:
        """Returns the B-form (right-canonical) tensors of the MPS."""
        return self._B

    @property
    def sWeight(self) -> nn.ParameterList:
        """Returns the Schmidt values on the bonds of the MPS."""
        return self._sWeight

    def position(self, site_idx: int) -> None:
        """Moves the orthogonality center to the specified site.

        This is done by performing a series of QR or SVD decompositions without
        any energy optimization steps. This method is the primary way to
        manipulate the canonical form of the MPS.

        Args:
            site_idx (int): The target site for the orthogonality center.
        """
        if not (0 <= site_idx < self.Nsites):
            raise ValueError(f"site_idx {site_idx} out of bounds for Nsites={self.Nsites}")

        print(f"\nMoving orthogonality center from {self.ortho_center} to {site_idx}...")

        # --- Sweep right to move ortho center to the right ---
        while self.ortho_center < site_idx:
            p = self.ortho_center
            chil, d, chir = self._A[p].shape
            U, S, Vh = torch.linalg.svd(self._A[p].reshape(chil * d, chir), full_matrices=False)

            new_chir = U.shape[1]
            self._A[p] = nn.Parameter(U.reshape(chil, d, new_chir), requires_grad=False)
            self._B[p] = nn.Parameter(self._A[p].clone(), requires_grad=False)
            sv = S / torch.linalg.norm(S)
            self._sWeight[p + 1] = nn.Parameter(sv, requires_grad=False)

            # Absorb S and Vh into the next tensor
            next_tensor = torch.einsum('k,kl,lds->kds', sv, Vh, self._B[p + 1])
            self._A[p + 1] = nn.Parameter(next_tensor, requires_grad=False)

            self.ortho_center += 1
            # After update, B[p+1] should also be updated to match A[p+1]
            self._B[p+1] = nn.Parameter(self._A[p+1].clone(), requires_grad=False)

        # --- Sweep left to move ortho center to the left ---
        while self.ortho_center > site_idx:
            p = self.ortho_center
            chil, d, chir = self._A[p].shape
            mat = self._A[p].reshape(chil, d * chir).T
            Q, R = torch.linalg.qr(mat)

            new_chil = Q.shape[1]
            self._A[p] = nn.Parameter(Q.T.reshape(new_chil, d, chir), requires_grad=False)
            self._B[p] = nn.Parameter(self._A[p].clone(), requires_grad=False)

            # Absorb R into the tensor to the left
            prev_tensor = torch.einsum('lsc,ck->lsk', self._A[p - 1], R.T)
            self._A[p - 1] = nn.Parameter(prev_tensor, requires_grad=False)
            self.ortho_center -= 1
            # B[p-1] becomes the new center, update it to match A[p-1]
            self._B[p-1] = nn.Parameter(self._A[p-1].clone(), requires_grad=False)

        # The final sWeight should be normalized
        norm = torch.linalg.norm(self._A[self.ortho_center])
        self._A[self.ortho_center].data /= norm
        self._B[self.ortho_center].data /= norm

        print(f"Orthogonality center is now at site {self.ortho_center}.")


    def __repr__(self) -> str:
        bond_dims = [t.shape[0] for t in self._A[1:]] + [self._A[-1].shape[2]]
        return (f"MPS(Nsites={self.Nsites}, chid={self.chid}, "
                f"ortho_center={self.ortho_center}, bond_dims={bond_dims})")


def random_mps(n_sites: int, d: int, chi: int, *, dtype, device, seed: Optional[int] = None) -> List[torch.Tensor]:
    """Generates a random MPS with a specified bond dimension."""
    if seed is not None:
        g = torch.Generator(device=device).manual_seed(seed)
        rand = lambda *sh: torch.rand(*sh, generator=g, device=device, dtype=dtype)
    else:
        rand = lambda *sh: torch.rand(*sh, device=device, dtype=dtype)

    A = [torch.zeros(1, device=device, dtype=dtype) for _ in range(n_sites)]
    A[0] = rand(1, d, min(chi, d))
    for k in range(1, n_sites):
        left = A[k - 1].shape[2]
        right_max = min(min(chi, left * d), d ** (n_sites - k - 1))
        A[k] = rand(left, d, right_max)
    return A

