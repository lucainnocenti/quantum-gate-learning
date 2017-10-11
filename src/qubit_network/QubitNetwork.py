"""
Compute the base object representing the qubit network.
"""
import itertools
import numbers
import numpy as np
import sympy

import qutip
import theano
import theano.tensor as T

from .utils import complex2bigreal


def pauli_product(*args):
    """
    Return sympy.Matrix object represing product of Pauli matrices.

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
    return sympy.Matrix(output_matrix)


def _self_interactions(num_qubits):
    """Return the indices corresponding to the self-interactions."""
    interactions = []
    for qubit in range(num_qubits):
        for pindex in range(1, 4):
            term = [0] * num_qubits
            term[qubit] = pindex
            interactions.append(tuple(term))
    return interactions


def _pairwise_interactions(num_qubits):
    """
    Return the indices corresponding the the pairwise interactions.
    """
    interactions = []
    pairs = itertools.combinations(range(num_qubits), 2)
    for qubit1, qubit2 in pairs:
        for pindex1, pindex2 in itertools.product(*[range(1, 4)] * 2):
            term = [0] * num_qubits
            term[qubit1] = pindex1
            term[qubit2] = pindex2
            interactions.append(tuple(term))
    return interactions


def _self_and_pairwise_interactions(num_qubits):
    """Return list of all possible one- and two-qubit interactions."""
    return _self_interactions(num_qubits) + _pairwise_interactions(num_qubits)


class QubitNetwork:
    """Compute the Hamiltonian for the qubit network.

    The Hamiltonian can be generated in several different ways, depending
    on the arguments given. Note that `QubitNetworkHamiltonian` is not
    supposed to know anything about ancillae, system qubits and so on.
    This class is only to parse input arguments (interactions, topology
    or sympy expression) in order to extract free symbols and matrix
    coefficients of a whole qubit network. The distinction between
    system and ancillary qubits comes next with `QubitNetwork`.

    Parameters
    ----------
    num_qubits : int,
        Number of qubits in the network.
    parameters : string, tuple or list, optional
        If given, it is used to use the parameters in some predefined
        way. Possible values are:
        - 'all': use all 1- and 2-qubit interactions, each one with a
            different parameter assigned.
        - ('all', (...)): use the specified types of intereactions for
            all qubits.
        - list of interactions: use all and only the given interactions.
    """

    def __init__(self,
                 num_qubits=None,
                 sympy_expr=None,
                 free_parameters_order=None,
                 interactions=None,
                 net_topology=None):
        # initialize class attributes
        self.num_qubits = None  # number of qubits in network
        self.matrices = None  # matrix coefficients for free parameters
        self.free_parameters = None  # symbolic parameters of the model
        self.interactions = None  # list of active interactions, if meaningful
        self.net_topology = None

        # Extract lists of parameters and matrices to which each is to
        # be multiplied
        if sympy_expr is not None:
            self._parse_sympy_expr(sympy_expr, free_parameters_order)
        elif interactions is not None:
            self._parse_from_interactions(num_qubits, interactions)
        elif net_topology is not None:
            self._parse_from_topology(num_qubits, net_topology)
        else:
            raise ValueError('One of `sympy_expr`, `interactions` or '
                             '`net_topology` must be given.')

    def _parse_sympy_expr(self, expr, free_parameters_order=None):
        """
        Extract free parameters and matrix coefficients from sympy expr.
        """
        try:
            if free_parameters_order is not None:
                self.free_parameters = free_parameters_order
            else:
                self.free_parameters = list(expr.free_symbols)
            _len = expr.shape[0]
        except TypeError:
            raise TypeError('`expr` must be a sympy MatrixSymbol object.')
        # initialize the list of matrices to which each parameter is multiplied
        self.matrices = []
        # extract the matrix to which each element is multiplied
        for parameter in self.free_parameters:
            self.matrices.append(expr.diff(parameter))
        # extract and store number of qubits of Hamiltonian
        self.num_qubits = int(np.log2(_len))

    def _parse_from_interactions(self, num_qubits, interactions):
        """
        Use value of `interactions` to compute parametrized Hamiltonian.

        When the Hamiltonian is derived from the `interactions`
        parameter, also the `self.interactions` attribute is filled,
        storing the indices corresponding to the interactions that are
        being used (as opposite to what happens when the Hamiltonian is
        computed from a sympy expression).
        """
        def make_symbols_and_matrices(interactions):
            self.free_parameters = []
            self.matrices = []
            for interaction in interactions:
                # create free parameter sympy symbol for interaction
                new_symb = 'J' + ''.join(str(idx) for idx in interaction)
                self.free_parameters.append(sympy.Symbol(new_symb))
                # create matrix coefficient for symbol just created
                self.matrices.append(pauli_product(*interaction))
        # store number of qubits in class
        if num_qubits is None:
            raise ValueError('The number of qubits must be given.')
        else:
            self.num_qubits = num_qubits

        if interactions == 'all':
            self.interactions = _self_and_pairwise_interactions(num_qubits)
        # a tuple of the kind `('all', ((1, 1), (2, 2)))` means that all
        # XX and YY interactions, and no others, should be used.
        elif isinstance(interactions, tuple) and interactions[0] == 'all':
            _interactions = _self_and_pairwise_interactions(num_qubits)
            self.interactions = []
            # filter list of interactions using given filter
            mask = [sorted(tup) for tup in interactions[1]]
            for interaction in _interactions:
                no_zeros = sorted([idx for idx in interaction if idx != 0])
                if no_zeros in mask:
                    self.interactions.append(interaction)
        elif isinstance(interactions, list):
            self.interactions = interactions
        # store values of symbols and matrices for chosen interactions
        if len(self.interactions) == 0:
            raise ValueError('No interaction value has been specified.')
        make_symbols_and_matrices(self.interactions)

    def _parse_from_topology(self, num_qubits, topology):
        """
        Use value of `topology` to compute parametrized Hamiltonian.

        The expected value of `topology` is a dictionary like:
            {((1, 2), 'xx'): 'a',
            ((0, 2), 'xx'): 'a',
            ((0, 1), 'zz'): 'b',
            ((1, 2), 'xy'): 'c'}
        or a dictionary like:
            {(0, 1, 1): a,
            (1, 0, 1): a,
            (3, 3, 0): b,
            (0, 1, 2): c}
        where `a`, `b` and `c` are `sympy.Symbol` instances.
        """
        self.num_qubits = num_qubits
        self.net_topology = topology
        # ensure that all values are sympy symbols
        all_symbols = [sympy.Symbol(str(symb)) for symb in topology.values()]
        # take list of not equal symbols
        symbols = list(set(all_symbols))
        # we try to sort the symbols, but if they are sympy symbols this
        # will fail with a TypeError, in which case we just give up and
        # leave them in whatever order they come out of `set`
        try:
            symbols = sorted(symbols)
        except TypeError:
            symbols = list(symbols)
        self.free_parameters = symbols
        # parse target tuples so that (2, 2) represents the YY interaction
        target_tuples = []
        for tuple_ in topology.keys():
            if isinstance(tuple_[1], str):
                str_spec = list(tuple_[1])
                new_tuple = [0] * num_qubits
                for idx, char in zip(tuple_[0], str_spec):
                    if char == 'x':
                        new_tuple[idx] = 1
                    elif char == 'y':
                        new_tuple[idx] = 2
                    elif char == 'z':
                        new_tuple[idx] = 3
                    else:
                        raise ValueError('Only x, y or z are valid.')
                target_tuples.append(tuple(new_tuple))
            else:
                target_tuples.append(tuple_)
        # Extract matrix coefficients for storing
        # The i-th element of `J` will correspond to the
        # interactions terms associated to the i-th symbol listed
        # in `symbols` (after sorting).
        self.matrices = []
        for idx, symb in enumerate(symbols):
            factor = sympy.Matrix(np.zeros((2 ** num_qubits,) * 2))
            for tuple_, label in zip(target_tuples, all_symbols):
                if label == symb:
                    factor += pauli_product(*tuple_)
            self.matrices.append(factor)

    def get_matrix(self):
        """Return the Hamiltonian matrix as a sympy matrix object."""
        # final_matrix = sympy.MatrixSymbol('H', *self.matrices[0].shape)
        final_matrix = sympy.Matrix(np.zeros(self.matrices[0].shape))
        for matrix, parameter in zip(self.matrices, self.free_parameters):
            final_matrix += parameter * matrix
        return final_matrix
