import itertools

import numpy as np
import scipy
import scipy.linalg
import sympy
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.paulialgebra import Pauli
import qutip
from .utils import chop


def pauli_product(*args, return_sympy_obj=True):
    """
    Returns
    -------
    Either a `sympy.Matrix` or a `numpy.ndarray`, representing the requested
    product of Pauli matrices.

    Examples
    --------
    >>> pauli_product(1, 1)
    Matrix([[0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0]])
    """
    for arg in args:
        try:
            if not 0 <= arg <= 3:
                raise ValueError('Each argument must be between 0 and 3.')
        except TypeError:
            raise ValueError('The inputs must be integers.')
    n_qubits = len(args)
    sigmas = [qutip.qeye(2), qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]
    output_matrix = [None] * n_qubits
    for idx, arg in enumerate(args):
        output_matrix[idx] = sigmas[arg]
    output_matrix = qutip.tensor(*output_matrix).data.toarray()
    if return_sympy_obj:
        return sympy.Matrix(output_matrix)
    else:
        return output_matrix


def _self_interactions(num_qubits):
    """Return the indices corresponding to the self-interactions."""
    interactions = []
    for qubit in range(num_qubits):
        for pindex in range(1, 4):
            term = [0] * num_qubits
            term[qubit] = pindex
            interactions.append(tuple(term))
    return interactions


def _nwise_interactions(num_qubits, n):
    """Return interaction indices corresponding to n-qubit interactions.
    """
    if n > num_qubits:
        raise ValueError('`n` must be lower than the number of qubits.')
    interactions = []
    tuples = itertools.combinations(range(num_qubits), n)
    for qtuple in tuples:
        for pindices in itertools.product(*[range(1, 4)] * n):
            term = [0] * num_qubits
            for q, p in zip(qtuple, pindices):
                term[q] = p
            interactions.append(tuple(term))
    return interactions


def _nwise_diagonal_interactions(num_qubits, n):
    """All diagonal n-qubit interactions."""
    all_ints = _nwise_interactions(num_qubits, n)
    diagonal_ints = [tup for tup in all_ints if is_diagonal_interaction(tup)]
    return diagonal_ints


def _at_most_nwise_interactions(num_qubits, n=None, include_identity=False):
    """Return interactions between at most n qubits.

    For example, for `n=2` we get all 1- and 2-qubit interactinos.
    """
    if n is None:
        n = num_qubits
    tuples = sum((_nwise_interactions(num_qubits, n) for n in range(2, n + 1)),
                 _nwise_interactions(num_qubits, 1))
    if include_identity:
        tuples = [(0, ) * num_qubits] + tuples
    return tuples


def _at_most_nwise_diagonal_interactions(num_qubits,
                                         n=None,
                                         include_identity=False):
    """Return diagonal interactions between at most n qubits."""
    if n is None:
        n = num_qubits
    tuples = sum((_nwise_diagonal_interactions(num_qubits, n) for n in range(2, n + 1)),
                 _nwise_diagonal_interactions(num_qubits, 1))
    if include_identity:
        tuples = [(0, ) * num_qubits] + tuples
    return tuples


def J(*args):
    """Simple helper to generate standardised sympy.Symbol objects.
    """
    return sympy.Symbol('J' + ''.join(str(arg) for arg in args))


def is_diagonal_interaction(int_tuple):
    """True if the tuple represents a diagonal interaction.

    A one-qubit interaction is automatically "diagonal".

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
    return _at_most_nwise_interactions(num_qubits, 2)


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
    """Produce Hamiltonian with restricted interactions producing gate.

    Computes a general parametrised Hamiltonian, containing only the
    requested set of interactions, that commutes with `gate`, which
    generally represents the canonical generator of a target gate.
    The idea is that if an Hamiltonian does not commute with the
    canonical generator, then it cannot generate the gate.

    The inputs `gate` should be a np.array or qutip object, while the
    output is given as a sympy symbol.
    """
    if isinstance(gate, qutip.Qobj):
        gate = gate.data.toarray()
    gate = np.asarray(gate)
    num_qubits = int(np.log2(gate.shape[0]))
    # decide what kinds of interactions we want in the general
    # paramatrized Hamiltonian (before imposing commutativity)
    which_interactions = None
    if type(interactions) is str:
        if interactions == 'all':
            which_interactions = _at_most_nwise_interactions(num_qubits, 2)
        elif interactions == 'diagonal':
            which_interactions = _at_most_nwise_diagonal_interactions(num_qubits, 2)
    if which_interactions is None:
        raise ValueError('which_interactions has not been set yet.')
    # make actual parametrized Hamiltonian
    general_ham = indices_to_hamiltonian(which_interactions)
    # compute principal hamiltonian
    principal_ham = chop(-1j * scipy.linalg.logm(gate))
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
                                     + '_' + str(pos + 1), commutative=False)
    return out_expr


def pauli_basis(matrix, which_coefficients='all'):
    """Take sympy matrix and decompose in terms of Pauli matrices."""
    num_qubits = sympy.log(matrix.shape[0], 2)
    if which_coefficients == 'all':
        coefficients = _at_most_nwise_interactions(num_qubits)
    out_expr = sympy.Integer(0)
    for coefficient in coefficients:
        out_expr += (get_pauli_coefficient(matrix, coefficient)
                     * symbolic_pauli_product(*coefficient))
    return out_expr
