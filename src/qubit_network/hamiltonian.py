"""
Compute the Hamiltonian of the system.
"""
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


class QubitNetworkHamiltonian:
    """Compute the Hamiltonian for the qubit network.

    The Hamiltonian can be generated in several different ways, depending
    on the arguments given.
    Parameters
    ----------
    n_qubits : int,
        Number of qubits in the network.
    parameters : ???, optional
        If given, it is used to use the parameters in some predefined
        way. Possible values are:
        - 'all': use all 1- and 2-qubit interactions, each one with a
                 different parameter assigned.
    """

    def __init__(self, n_qubits=None, expr=None, parameters=None, topology=None):
        # initialize class attributes
        self.matrices = None
        self.free_parameters = None
        self.initial_values = None
        self.J = None
        # extract lists of parameters and matrices to which each is to
        # be multiplied
        if expr is not None:
            self._parse_sympy_expr(expr)
        else:
            raise NotImplementedError('To be implemented')

    def _parse_sympy_expr(self, expr):
        """
        Extract free parameters and matrix coefficients from sympy expr.
        """
        try:
            self.free_parameters = expr.free_symbols
            self.len = expr.shape[0]
        except TypeError:
            raise TypeError('`expr` must be a sympy MatrixSymbol object.')
        # initialize the list of matrices to which each parameter is multiplied
        self.matrices = []
        # extract the matrix to which each element is multiplied
        for parameter in self.free_parameters:
            self.matrices.append(expr.diff(parameter))

    def _get_bigreal_matrices(self):
        """
        Return the elements of `self.matrices` as big real matrices.
        """
        return [complex2bigreal(matrix).astype(np.float)
                for matrix in self.matrices]

    def build_theano_graph(self):
        """Return a theano object corresponding to the Hamiltonian.

        The free parameters in the output graphs are taken from the sympy
        free symbols in the Hamiltonian, stored in `self.free_parameters`.

        """
        if self.initial_values is None:
            self.set_initial_values()
        # define the theano variables
        self.J = theano.shared(
            value=self.initial_values,
            name='J',
            borrow=True  # still not sure what this does
        )
        # multiply variables with matrix coefficients
        bigreal_matrices = self._get_bigreal_matrices()
        theano_graph = T.tensordot(self.J, bigreal_matrices, axes=1)
        return theano_graph

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
            self.initial_values = np.zeros(len(self.free_parameters))
        else:
            raise NotImplementedError('Not implemented.')
