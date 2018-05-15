"""
Compute the base object representing the qubit network.
"""
import itertools
import time
import logging
import numpy as np
import sympy
import qutip

from .analytical_conditions import (pauli_product, pauli_basis,
                                    _self_interactions,
                                    _at_most_nwise_interactions)


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
    num_qubits : int
        Number of qubits in the network.
    sympy_expr : sympy object, optional
        A sympy symbolic matrix object. If given, it directly specifies
        the Hamiltonian model to be used during the training. Do not use
        together with `interactions` or `net_topology`.
    free_parameters_order : list of sympy symbols, optional
        This list is used to fix an ordering in the list of free parameters
        of the model. It's required because without it one could e.g. get
        inconsistent results by replacing the "first" symbols (the notion
        of "i-th symbol" would not be assured).
        If not given, it is automatically generated via `sympy`'s `free_symbols`
        method.
    interactions : string, tuple or list, optional
        If given, it is used to use the parameters in some predefined
        way. Possible values are:
        - 'all': use all 1- and 2-qubit interactions, each one with a
            different parameter assigned.
        - ('all', (...)): use the specified types of intereactions for
            all qubits.
        - list of interactions: use all and only the given interactions.

    Attributes
    ----------
    num_qubits : int
    matrices : list of sympy matrices
    free_parameters : list of sympy objects
    interactions
    net_topology
    """

    def __init__(self,
                 num_qubits=None,
                 sympy_expr=None,
                 free_parameters_order=None,
                 interactions=None,
                 net_topology=None):
        logging.debug("I'm inside QubitNetwork")
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

        Attributes
        ----------
        expr : sympy object or tuple
            If a `sympy.Matrix` object, it completely characterises the
            Hamiltonian model to be used.
            If a `tuple`, it should contain two elements. The first one being
            a list of `sympy.Symbol` objects, and the second one being a list
            of matrices to be associated to each `sympy.Symbol`.
        """
        logging.info("Parsing from sympy expression")

        if hasattr(expr, 'free_symbols'):
            logging.info('  Detected sympy.Matrix object.')
            if free_parameters_order is not None:
                logging.info("  A `free_parameters_order` was given.")
                self.free_parameters = free_parameters_order
            else:
                logging.info("  No `free_parameters_order` was given, generati"
                             "ng one automatically.")
                self.free_parameters = list(expr.free_symbols)
            # compute number of qubits from sympy matrix object
            self.num_qubits = int(np.log2(expr.shape[0]))
            logging.info('  Deriving matrices from sympy expression..')
            self.matrices = []
            for parameter in self.free_parameters:
                self.matrices.append(expr.diff(parameter))
        elif isinstance(expr, tuple):
            logging.info('  Detected split sympy model.')
            self.free_parameters = expr[0]
            self.matrices = expr[1]
            self.num_qubits = int(np.log2(self.matrices[0].shape[0]))

    def _parse_from_interactions(self, num_qubits, interactions):
        """
        Use value of `interactions` to compute parametrized Hamiltonian.

        When the Hamiltonian is derived from the `interactions`
        parameter, also the `self.interactions` attribute is filled,
        storing the indices corresponding to the interactions that are
        being used (as opposite to what happens when the Hamiltonian is
        computed from a sympy expression).
        """
        logging.info("Parsing from `interactions`")
        def make_symbols_and_matrices(interactions):
            free_parameters = []
            matrices = []
            for interaction in interactions:
                # create free parameter sympy symbol for interaction
                new_symb = 'J' + ''.join(str(idx) for idx in interaction)
                free_parameters.append(sympy.Symbol(new_symb))
                # create matrix coefficient for symbol just created
                matrices.append(pauli_product(*interaction,
                                              return_sympy_obj=False))
            return free_parameters, matrices

        # store number of qubits in class
        if num_qubits is None:
            raise ValueError('The number of qubits must be given.')
        else:
            self.num_qubits = num_qubits
        logging.info('  Total no. of qubit: {}'.format(self.num_qubits))
        if interactions == 'all':
            logging.info('  Using interactions=\'all\' mode')
            self.interactions = _at_most_nwise_interactions(num_qubits, 2)
        # a tuple of the kind `('all', ((1, 1), (2, 2)))` means that all
        # XX and YY interactions, and no others, should be used.
        elif isinstance(interactions, tuple) and interactions[0] == 'all':
            logging.info('  Explicit interactions specified')
            _interactions = _at_most_nwise_interactions(num_qubits, 2)
            self.interactions = []
            # filter list of interactions using given filter
            mask = [sorted(tup) for tup in interactions[1]]
            for interaction in _interactions:
                no_zeros = sorted([idx for idx in interaction if idx != 0])
                if no_zeros in mask:
                    self.interactions.append(interaction)
        # Otherwise we assume that the input is a list of tuples, with each 
        # representing an n-qubit interaction, like: `[(1, 1), (1, 2), (3, 0)]`
        elif isinstance(interactions, (list, tuple)):
            logging.info('  Using explicit interactions specification')
            self.interactions = list(interactions)
        else:
            raise ValueError('Value of parameter `interaction` not valid.')
        # store values of symbols and matrices for chosen interactions
        if len(self.interactions) == 0:
            raise ValueError('No interaction value has been specified.')

        self.free_parameters, self.matrices = make_symbols_and_matrices(
            self.interactions)

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
        logging.info("Parsing from `net_topology`")
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


    def get_matrix(self, symbolic_paulis=False, sure=False):
        """Return the Hamiltonian matrix as a sympy matrix object."""
        if self.num_qubits > 5 and not sure:
            raise ValueError('This can take quite a while for more than 5 '
                             'qubits. Call the function with the `sure` pa'
                             'rameter if you really want to do this.')
        if not symbolic_paulis:
            final_matrix = sympy.Matrix(np.zeros(self.matrices[0].shape))
            for matrix, parameter in zip(self.matrices, self.free_parameters):
                final_matrix += parameter * matrix
            return final_matrix
        expr = pauli_basis(self.get_matrix())
        return sympy.simplify(expr)
