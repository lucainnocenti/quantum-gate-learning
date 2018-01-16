import itertools

import numpy as np
import scipy
import scipy.linalg
import sympy
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.paulialgebra import Pauli
import qutip

from .QubitNetwork import _self_and_pairwise_interactions
from .QubitNetwork import pauli_product


def J(*args):
    return sympy.Symbol('J' + ''.join(str(arg) for arg in args))


def is_diagonal_interaction(int_tuple):
    """True if the tuple represents a diagonal interaction.

    Examples
    --------
    >>> is_diagonal_interaction((2, 2))
    True
    >>> is_diagonal_interaction((2, 0))
    True
    >>> is_diagonal_interaction((1, 1, 2))
    False
    """
    nonzero_indices = [idx for idx in int_tuple if idx != 0]
    return len(set(nonzero_indices)) == 1


def pairwise_interactions_indices(num_qubits):
    """List of 1- and 2- qubit interaction terms."""
    return _self_and_pairwise_interactions(num_qubits)


def pairwise_diagonal_interactions_indices(num_qubits):
    """List of 1- and 2- qubit diagonal interaction terms."""
    all_ints = pairwise_interactions_indices(num_qubits)
    return [interaction for interaction in all_ints if is_diagonal_interaction(interaction)]


def indices_to_hamiltonian(interactions_indices):
    out_ham = None
    for interaction in interactions_indices:
        if out_ham is None:
            out_ham = J(*interaction) * pauli_product(*interaction)
            continue
        out_ham += J(*interaction) * pauli_product(*interaction)
    return out_ham


def commutator(m1, m2):
    return m1 * m2 - m2 * m1


def impose_commutativity(mat, other_mat):
    sols = sympy.solve(commutator(mat, other_mat))
    # before sympy v1.1 we had to use `sols[0]`, but now it seems a set
    # is returned instead
    return mat.subs(sols)


def commuting_generator(gate, interactions='all'):
    if isinstance(gate, qutip.Qobj):
        gate = gate.data.toarray()
    gate = np.asarray(gate)
    num_qubits = int(np.log2(gate.shape[0]))
    # decide what kinds of interactions we want in the general
    # paramatrized Hamiltonian (before imposing commutativity)
    which_interactions = None
    if type(interactions) is str:
        if interactions == 'all':
            which_interactions = pairwise_interactions_indices(num_qubits)
        elif interactions == 'diagonal':
            which_interactions = pairwise_diagonal_interactions_indices(num_qubits)
    if which_interactions is None:
        raise ValueError('which_interactions has not been set yet.')
    # make actual parametrized Hamiltonian
    general_ham = indices_to_hamiltonian(which_interactions)
    # compute principal hamiltonian
    principal_ham = (-1j * scipy.linalg.logm(gate)).real
    # impose commutativity
    return impose_commutativity(general_ham, principal_ham)


def get_pauli_coefficient(matrix, coefficient):
    """Extract given Pauli coefficient from matrix.

    The coefficient must be specified in the form of a tuple whose i-th
    element tells the Pauli operator acting on the i-th qubit.
    For example, `coefficient = (2, 1)` asks for the Y1 X2 coefficient.
    Generally speaking, it should be a valid input to `pauli_product`.

    The function works with sympy objects.
    """
    num_qubits = len(coefficient)
    return sympy.trace(matrix * pauli_product(*coefficient)) / 2**num_qubits


def symbolic_pauli_product(*args, as_tensor_product=False):
    """
    Return symbolic sympy object represing product of Pauli matrices.
    """
    if as_tensor_product:
        tensor_product_elems = []
        for arg in args:
            if arg == 0:
                tensor_product_elems.append(1)
            else:
                tensor_product_elems.append(Pauli(arg))
        return TensorProduct(*tensor_product_elems)

    out_expr = sympy.Integer(1)
    for pos, arg in enumerate(args):
        if arg != 0:
            out_expr *= sympy.Symbol(['X', 'Y', 'Z'][arg - 1]
                                     + '_' + str(pos), commutative=False)
    return out_expr


def pauli_basis(matrix, which_coefficients='all'):
    """Take sympy matrix and decompose in terms of Pauli matrices."""
    num_qubits = sympy.log(matrix.shape[0], 2)
    if which_coefficients == 'all':
        coefficients = pairwise_interactions_indices(num_qubits)
    out_expr = sympy.Integer(0)
    for coefficient in coefficients:
        out_expr += (get_pauli_coefficient(matrix, coefficient)
                     * symbolic_pauli_product(*coefficient))
    return out_expr
