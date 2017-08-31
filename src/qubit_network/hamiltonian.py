"""
Compute the Hamiltonian of the system.
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


class QubitNetworkHamiltonian:
    """Compute the Hamiltonian for the qubit network.

    The Hamiltonian can be generated in several different ways, depending
    on the arguments given.
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
                 expr=None,
                 interactions=None,
                 net_topology=None):
        # initialize class attributes
        self.num_qubits = None  # number of qubits in network
        self.matrices = None  # matrix coefficients for free parameters
        self.free_parameters = None  # parameters to be trained
        self.interactions = None  # list of active interactions, if meaningful
        self.net_topology = None
        self.initial_values = None  # values from which training starts

        # Extract lists of parameters and matrices to which each is to
        # be multiplied
        if expr is not None:
            self._parse_sympy_expr(expr)
        elif interactions is not None:
            self._parse_from_interactions(num_qubits, interactions)
        elif net_topology is not None:
            self._parse_from_topology(num_qubits, net_topology)
        else:
            raise ValueError('One of `expr`, `interactions` or '
                             '`net_topology` must be given.')

    def _parse_sympy_expr(self, expr):
        """
        Extract free parameters and matrix coefficients from sympy expr.
        """
        try:
            self.free_parameters = expr.free_symbols
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
        symbols = set(all_symbols)
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
        # import ipdb; ipdb.set_trace()
        for idx, symb in enumerate(symbols):
            factor = sympy.Matrix(np.zeros((2 ** num_qubits,) * 2))
            for tuple_, label in zip(target_tuples, all_symbols):
                if label == symb:
                    factor += pauli_product(*tuple_)
            self.matrices.append(factor)

    def _get_bigreal_matrices(self, multiply_by_j=True):
        """
        Multiply each element of `self.matrices` with `-1j`, and return
        them converted to big real form. Or optionally do not multiply
        with the imaginary unit and just return the matrix coefficients
        converted in big real form.
        """
        if multiply_by_j:
            return [complex2bigreal(-1j * matrix).astype(np.float)
                    for matrix in self.matrices]
        else:
            return [complex2bigreal(matrix).astype(np.float)
                    for matrix in self.matrices]

    def build_theano_graph(self):
        """Build theano object corresponding to the Hamiltonian model.

        The free parameters in the output graphs are taken from the sympy
        free symbols in the Hamiltonian, stored in `self.free_parameters`.

        Returns
        -------
        tuple with the shared theano variable representing the parameters
        and the corresponding theano.tensor object for the Hamiltonian
        model.
        """
        # define the theano variables
        parameters = theano.shared(
            value=np.zeros(len(self.free_parameters), dtype=np.float),
            name='J',
            borrow=True  # still not sure what this does
        )
        # multiply variables with matrix coefficients
        bigreal_matrices = self._get_bigreal_matrices()
        theano_graph = T.tensordot(parameters, bigreal_matrices, axes=1)
        # from IPython.core.debugger import set_trace; set_trace()
        return [parameters, theano_graph]

    def get_matrix(self):
        """Return the Hamiltonian matrix as a sympy matrix object."""
        # final_matrix = sympy.MatrixSymbol('H', *self.matrices[0].shape)
        final_matrix = sympy.Matrix(np.zeros(self.matrices[0].shape))
        for matrix, parameter in zip(self.matrices, self.free_parameters):
            final_matrix += parameter * matrix
        return final_matrix

    def set_initial_values(self, values=None):
        """Set initial values for the parameters in the Hamiltonian.

        If no explicit values are given, the parameters are initialized
        with zeros.
        """
        if values is None:
            self.initial_values = np.random.randn(len(self.free_parameters))
        elif isinstance(values, numbers.Number):
            self.initial_values = np.ones(len(self.free_parameters)) * values
        # A dictionary can be used to directly set the values of some of
        # the parameters. Each key of the dictionary can be either a
        # 1) sympy symbol correponding to an interaction, 2) a string
        # with the same name of a symbol of an interaction or 3) a tuple
        # of integers corresponding to a given interactions. This last
        # option is not valid if the Hamiltonian was created using a
        # sympy expression.
        # All the symbols not specified in the dictionary are initialized
        # to zero.
        elif isinstance(values, dict):
            init_values = np.zeros(len(self.free_parameters))
            symbols_dict = dict(zip(
                self.free_parameters, range(len(self.free_parameters))))
            for symb, value in values.items():
                # if `symb` is a single number, make a 1-element tuple
                if isinstance(symb, numbers.Number):
                    symb = (symb,)
                # convert strings to corresponding sympy symbols
                if isinstance(symb, str):
                    symb = sympy.Symbol(symb)
                # `symb` can be a tuple when a key is of the form
                # `(1, 3)` to indicate an X1Z2 interaction.
                elif isinstance(symb, tuple):
                    symb = 'J' + ''.join(str(char) for char in symb)
                try:
                    init_values[symbols_dict[symb]] = value
                except KeyError:
                    raise ValueError('The symbol {} doesn\'t match'
                                     ' any of the names of parameters of '
                                     'the model.'.format(str(symb)))
            self.initial_values(init_values)
        else:
            raise NotImplementedError('Not implemented.')
