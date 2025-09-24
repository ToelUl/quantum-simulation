import sympy as sym
import numpy as np
from torch import Tensor
from typing import Union
from IPython.display import Math, display


def show_matrix(matrix: Tensor, precision: int=2):
    matrix = matrix.numpy()
    if len(matrix.shape) > 2:
        raise ValueError('The input matrix should be 2D or 1D')
    if np.iscomplexobj(matrix):
        temp = np.zeros(matrix.shape, dtype=complex)
        if len(matrix.shape) == 1:
            for i in range(matrix.shape[0]):
                temp[i] = round(np.real(matrix[i]), precision) + round(np.imag(matrix[i]), precision) * 1j
        else:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    temp[i][j] = round(np.real(matrix[i][j]), precision) + round(np.imag(matrix[i][j]), precision) * 1j
        return sym.Matrix(temp)
    else:
        temp = np.zeros(matrix.shape)
        if len(matrix.shape) == 1:
            for i in range(matrix.shape[0]):
                temp[i] = round(matrix[i], precision)
        else:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    temp[i][j] = round(matrix[i][j], precision)
        return sym.Matrix(temp)


def jordan_wigner_transform_symbolize(key: Union[str], convention="down=1"):
    """Retrieves the symbolic representation of a Jordan-Wigner formula.

    This function acts as a repository for various symbolic formulas related to
    the Jordan-Wigner transformation. It can return specific formulas as SymPy
    symbols or provide a list of all available formula keys.

    Args:
        key: The identifier for the desired formula. Passing '--list' or
            '--help' will cause a formatted list of all available keys
            and their descriptions to be printed to the console.
        convention: The convention for the K_j string operator. Defaults to
            "down=1". This affects the definition of K_j and b_j.

    Returns:
        A SymPy Symbol object representing the requested formula if the key is
        valid. If the key is '--list' or '--help', the function prints a
        help message and returns None. Returns an error message string if the
        key is invalid.
    """
    # A dictionary mapping keys to their corresponding SymPy symbolic representations.
    symbol_dict = {
        # Pauli Operators and the String Operator (K_j)
        "K_j_prod": Math(r"K_j=\prod^{j-1}_{j'=1}(1-2\hat{n}_{j'})") if convention == "down=1"
        else Math(r"K_j=\prod^{j-1}_{j'=1}(2\hat{n}_{j'}-1)"),
        "K_j_exp": Math(r"\hat{K}_{j} = \prod_{j'=1}^{j-1} e^{i\pi\hat{n}_{j'}} = e^{i\pi\sum_{j'=1}^{j-1}\hat{n}_{j'}}"),
        "sigma_x_j": Math(r"\hat{\sigma}^{x}_{j}=\hat{K}_{j}(\hat{c}^{\dagger}_{j}+\hat{c}_{j})"),
        "sigma_y_j": Math(r"\hat{\sigma}^{y}_{j}=\hat{K}_{j}i(\hat{c}^{\dagger}_{j}-\hat{c}_{j})"),
        "sigma_z_j": Math(r"\hat{\sigma}^{z}_{j}=1-2\hat{n}_{j}"
                                 r"=(\hat{c}^{\dagger}_{j}+\hat{c}_{j})(\hat{c}^{\dagger}_{j}-\hat{c}_{j})"),
        "sigma_x_j_sigma_x_j+1": Math(
            r"\hat{\sigma}^{x}_{j}\hat{\sigma}^{x}_{j+1}=\hat{c}^{\dagger}_{j}\hat{c}^{\dagger}_{j+1}"
            r"+\hat{c}^{\dagger}_{j}\hat{c}_{j+1}+H.c."),
        "sigma_y_j_sigma_y_j+1": Math(
            r"\hat{\sigma}^{y}_{j}\hat{\sigma}^{y}_{j+1}=-(\hat{c}^{\dagger}_{j}\hat{c}^{\dagger}_{j+1}"
            r"-\hat{c}^{\dagger}_{j}\hat{c}_{j+1}+H.c.)"),

        # Hard-core Boson Operators (b_j)
        "b_j_def": Math(r"\hat{b}_{j} = S_{+} = \hat{K}_{j}\hat{c}_{j} = \hat{c}_{j}\hat{K}_{j}")
        if convention == "down=1" else Math(
            r"\hat{b}_{j} = S_{-} = \hat{K}_{j}\hat{c}_{j} = \hat{c}_{j}\hat{K}_{j}"),
        "b_dag_j_b_j": Math(r"\hat{b}^{\dagger}_{j}\hat{b}_{j}=\hat{c}^{\dagger}_{j}\hat{c}_{j}"),
        "b_dag_j_b_dag_j+1": Math(
            r"\hat{b}^{\dagger}_{j}\hat{b}^{\dagger}_{j+1}=\hat{c}^{\dagger}_{j}\hat{c}^{\dagger}_{j+1}"),
        "b_dag_j_b_j+1": Math(r"\hat{b}^{\dagger}_{j}\hat{b}_{j+1}=\hat{c}^{\dagger}_{j}\hat{c}_{j+1}"),
        "b_j_b_j+1": Math(r"\hat{b}_{j}\hat{b}_{j+1}=-\hat{c}_{j}\hat{c}_{j+1}"),
        "b_j_b_dag_j+1": Math(r"\hat{b}_{j}\hat{b}^{\dagger}_{j+1}=-\hat{c}_{j}\hat{c}^{\dagger}_{j+1}"),

        # Fermionic Operators (c_j) and Anti-commutation Relations
        "c_j_def": Math(r"\hat{c}_{j} = \hat{K}_{j}\hat{b}_{j}"),
        "anti_comm_c_c": Math(
            r"\{\hat{c}_{j}, \hat{c}_{k}\} = \hat{c}_{j}\hat{c}_{k} + \hat{c}_{k}\hat{c}_{j} = 0"),
        "anti_comm_cdag_cdag": Math(
            r"\{\hat{c}^{\dagger}_{j}, \hat{c}^{\dagger}_{k}\} = \hat{c}^{\dagger}_{j}\hat{c}^{\dagger}_{k} + \hat{c}^{\dagger}_{k}\hat{c}^{\dagger}_{j} = 0"),
        "anti_comm_c_cdag": Math(
            r"\{\hat{c}_{j}, \hat{c}^{\dagger}_{k}\} = \hat{c}_{j}\hat{c}^{\dagger}_{k} + \hat{c}^{\dagger}_{k}\hat{c}_{j} = \delta_{jk}"),
    }

    # Handle requests for listing all available keys by printing directly.
    if key in ('--list', '--help'):
        # A dictionary mapping keys to their plain-text descriptions for the help message.
        descriptions = {
            "K_j_prod": "String operator K_j (product form)",
            "K_j_exp": "String operator K_j (exponential form)",
            "sigma_x_j": "Pauli-X operator at site j",
            "sigma_y_j": "Pauli-Y operator at site j",
            "sigma_z_j": "Pauli-Z operator at site j",
            "sigma_x_j_sigma_x_j+1": "Product of adjacent Pauli-X operators",
            "sigma_y_j_sigma_y_j+1": "Product of adjacent Pauli-Y operators",
            "b_j_def": "Definition of the hard-core boson operator b_j",
            "b_dag_j_b_j": "Number operator",
            "b_dag_j_b_dag_j+1": "Creation-creation term at adjacent sites",
            "b_dag_j_b_j+1": "Hopping term",
            "b_j_b_j+1": "Annihilation-annihilation term at adjacent sites",
            "b_j_b_dag_j+1": "Swap term",
            "c_j_def": "Definition of the fermion operator c_j",
            "anti_comm_c_c": "Anti-commutation relation {c_j, c_k}",
            "anti_comm_cdag_cdag": "Anti-commutation relation {c_dag_j, c_dag_k}",
            "anti_comm_c_cdag": "Anti-commutation relation {c_j, c_dag_k}",
        }

        print("Available keys and their meanings:")
        print("=" * 80)

        # Determine the maximum key length for pristine alignment.
        max_key_length = max(len(k) for k in descriptions)

        for k, v in descriptions.items():
            if k in symbol_dict:
                # Use the calculated max length for dynamic padding.
                print(f"- {k:<{max_key_length + 2}}: {v}")

        print()  # Add a final newline for spacing.
        return

    # Retrieve the symbol from the dictionary.
    symbol = symbol_dict.get(key)

    # Handle invalid keys by returning an informative error message.
    if symbol is None:
        return f"Error: Invalid key '{key}'. Use '--list' to see all available keys."

    return symbol
