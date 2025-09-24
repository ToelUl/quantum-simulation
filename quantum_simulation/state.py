import math
import numpy as np
import torch
from .operation import nested_kronecker_product
from typing import Optional, Union, List

def spin_state(spin: float, m: float):
    if abs(m) > spin:
        raise ValueError("The absolute value of m should be less than or equal to spin")
    dim = int(2 * spin + 1)
    state = np.zeros((dim, 1), dtype=complex)
    index = int(spin - m)
    state[index][0] = 1.0 + 0j
    return torch.from_numpy(state).to(torch.complex64)

def up_state(spin=0.5):
    return spin_state(spin, spin)

def down_state(spin=0.5):
    return spin_state(spin, -spin)

def spin_one_half_state(state_label: str):
    state_dict = {"u": up_state(), "d": down_state(), ">": (up_state() + down_state())/np.sqrt(2),
                  "<": (up_state() - down_state())/np.sqrt(2)}
    return state_dict[state_label]

def global_spin_one_half_state(state: Union[str]):
    state_list = []
    for i in state:
        state_list.append(spin_one_half_state(i))
    if len(state_list) == 0:
        raise ValueError("The input state string is empty")
    if len(state_list) == 1:
        return state_list[0]
    else:
        return nested_kronecker_product(state_list)

def neel_state(lattice_length: int, convention="even_up"):
    state_list = []
    if convention not in ["even_up", "odd_up", "even_down", "odd_down"]:
        raise ValueError("The input convention is not valid")
    if convention == "even_up" or convention == "odd_down":
        for i in range(lattice_length):
            if i % 2 == 0:
                state_list.append(up_state())
            else:
                state_list.append(down_state())
    elif convention == "even_down" or convention == "odd_up":
        for i in range(lattice_length):
            if i % 2 == 0:
                state_list.append(down_state())
            else:
                state_list.append(up_state())
    else:
        raise ValueError("The input convention is not valid")
    if len(state_list) == 0:
        raise ValueError("The input lattice length is zero")
    if len(state_list) == 1:
        return state_list[0]
    else:
        return nested_kronecker_product(state_list)

def product_state(state_list: List[Union[str, torch.Tensor]]):
    temp_list = []
    for state in state_list:
        if isinstance(state, str):
            temp_list.append(spin_one_half_state(state))
        elif isinstance(state, torch.Tensor):
            temp_list.append(state)
        else:
            raise ValueError("The input state should be a string or a torch tensor")
    if len(temp_list) == 0:
        raise ValueError("The input state list is empty")
    if len(temp_list) == 1:
        return temp_list[0]
    else:
        return nested_kronecker_product(temp_list)

def random_state(dim: int, seed: Optional[int] = None):
    if seed is not None:
        torch.manual_seed(seed)
    real_part = torch.randn((dim, 1))
    imag_part = torch.randn((dim, 1))
    state = real_part + 1j * imag_part
    state = state / torch.norm(state)
    return state

def fock_state(state_string: str, convention: str = "down=1") -> torch.Tensor:
    """
    Constructs a product state for a spinless system from a string
    representing particle occupation.

    Args:
        state_string (str): A string of '0's and '1's describing the state of each
                            lattice site. '0' represents an empty site (vacuum),
                            and '1' represents an occupied site (particle).
        convention (str, optional): Specifies the basis vector corresponding to '1'
                                    (occupied). Defaults to "down=1".
                                    - "up=1": Maps '1' to up_state ([1, 0]^T) and '0'
                                      to down_state ([0, 1]^T). This corresponds to
                                      the mapping |1⟩ ≡ |↑⟩.
                                    - "down=1": Maps '1' to down_state ([0, 1]^T) and '0'
                                      to up_state ([1, 0]^T). This corresponds to
                                      the mapping |1⟩ ≡ |↓⟩.

    Returns:
        torch.Tensor: The tensor representing the final many-body quantum state.

    Raises:
        ValueError: If the convention is invalid, if state_string is empty, or
                    if it contains characters other than '0' and '1'.
    """
    # --- 1. Input Validation ---
    if convention not in ["up=1", "down=1"]:
        raise ValueError("Convention must be either 'up=1' or 'down=1'.")

    if not state_string:
        raise ValueError("The input state_string cannot be empty.")

    if any(char not in '01' for char in state_string):
        raise ValueError("The state_string must only contain '0's and '1's.")

    # --- 2. Set up mapping based on the convention ---
    if convention == "up=1":
        # '1' (occupied) -> |↑⟩
        # '0' (vacuum)   -> |↓⟩
        state_map = {'1': up_state(), '0': down_state()}
    else:  # convention == "down=1"
        # '1' (occupied) -> |↓⟩
        # '0' (vacuum)   -> |↑⟩
        state_map = {'1': down_state(), '0': up_state()}

    # --- 3. Build the list of single-site states ---
    state_list = [state_map[char] for char in state_string]

    # --- 4. Combine the states using the Kronecker product ---
    if len(state_list) == 1:
        return state_list[0]
    else:
        return nested_kronecker_product(state_list)


def view_fock_state(state: torch.Tensor, convention: str = "down=1", threshold: float = 1e-9):
    """
    Prints a human-readable representation of a quantum state in the Fock basis.

    This function decomposes the state vector and displays the basis states
    (e.g., |101⟩) that have a significant amplitude. It is the inverse
    operation of the `fock_state` constructor.

    Args:
        state (torch.Tensor): The state vector to be displayed.
                              Its dimension must be a power of 2.
        convention (str, optional): The convention used to map the computational
                                    basis back to the occupation ('0'/'1') basis.
                                    This MUST match the convention used to create the
                                    state. Defaults to "down=1".
        threshold (float, optional): Amplitudes with a magnitude smaller than this
                                     value will not be displayed. This is useful for
                                     ignoring numerical noise. Defaults to 1e-9.
    """
    # --- 1. Validate Inputs and Get System Size ---
    dim = state.shape[0]
    if dim == 0:
        print("Empty state.")
        return

    # The dimension must be 2^N, so log2(dim) must be an integer.
    num_sites = math.log2(dim)
    if num_sites != int(num_sites):
        raise ValueError(f"The state dimension ({dim}) is not a power of 2.")
    num_sites = int(num_sites)

    if convention not in ["up=1", "down=1"]:
        raise ValueError("Convention must be either 'up=1' or 'down=1'.")

    # --- 2. Find Significant Components and Map to Fock Strings ---
    components = []
    for i in range(dim):
        amplitude = state[i].item()
        # Check if the magnitude of the amplitude is above the threshold
        if abs(amplitude) > threshold:
            # Convert the index to a binary string, padded with leading zeros
            # This is the computational basis |↑↓↑...⟩ representation where ↑=0, ↓=1
            computational_basis_str = format(i, f'0{num_sites}b')

            # Map the computational basis to the fock (occupation) basis
            if convention == "down=1":
                # '1' (occupied) is |↓⟩ (which is computational '1')
                # '0' (vacuum) is |↑⟩ (which is computational '0')
                # The mapping is direct.
                fock_str = computational_basis_str
            else:  # convention == "up=1"
                # '1' (occupied) is |↑⟩ (computational '0')
                # '0' (vacuum) is |↓⟩ (computational '1')
                # The mapping is inverted (flip all bits).
                fock_str = "".join(['1' if bit == '0' else '0' for bit in computational_basis_str])

            components.append((amplitude, fock_str))

    # --- 3. Format and Print the Output ---
    if not components:
        print("State is effectively the zero vector (all amplitudes are below threshold).")
        return

    output_str = []
    for i, (amp, fock) in enumerate(components):
        # Format the amplitude. Show it as complex if imag part is significant.
        if abs(amp.imag) > threshold:
            amp_str = f"({amp.real:.4f}{amp.imag:+.4f}j)"
        else:
            amp_str = f"({amp.real:.4f})"

        term = f"{amp_str}|{fock}⟩"

        # Add a " + " sign for superpositions, but not for the first term
        if i > 0:
            # Handle negative signs gracefully
            if amp.real < 0 and abs(amp.imag) <= threshold:
                output_str.append(f" - {-amp.real:.4f}|{fock}⟩")
            else:
                output_str.append(f" + {term}")
        else:
            output_str.append(term)

    print("".join(output_str))