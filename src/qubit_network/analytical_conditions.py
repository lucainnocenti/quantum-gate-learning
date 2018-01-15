import itertools

import numpy as np
import scipy
import scipy.linalg
import sympy
import qutip

from .QubitNetwork import _self_and_pairwise_interactions
from .QubitNetwork import pauli_product


def J(*args):
    return sympy.Symbol('J' + ''.join(str(arg) for arg in args))


def is_diagonal_interaction(int_tuple):
    nonzero_indices = [idx for idx in int_tuple if idx != 0]
    return len(set(nonzero_indices)) == 1


def pairwise_interactions_indices(num_qubits):
    return _self_and_pairwise_interactions(num_qubits)


def pairwise_diagonal_interactions_indices(num_qubits):
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
