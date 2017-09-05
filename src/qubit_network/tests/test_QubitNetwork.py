# pylint: skip-file
import unittest
import sys
import os
import inspect

import numpy as np
from numpy.testing import assert_array_equal
import sympy

import qutip
import theano
import theano.tensor as T


class TestQubitNetworkHamiltonian(unittest.TestCase):
    def test_parse_from_sympy_expr(self):
        a, b = sympy.symbols('a, b')
        x1 = pauli_product(1, 0)
        xx = pauli_product(1, 1)
        expr = a * x1 + b * xx
        net = QubitNetwork(sympy_expr=expr)
        hamiltonian_matrix = net.get_matrix()
        self.assertListEqual(
            sympy.flatten(hamiltonian_matrix),
            sympy.flatten(sympy.Matrix([[0, 0, 1.0*a, 1.0*b],
                                        [0, 0, 1.0*b, 1.0*a],
                                        [1.0*a, 1.0*b, 0, 0],
                                        [1.0*b, 1.0*a, 0, 0]])))
        self.assertEqual(net.num_qubits, 2)


if __name__ == '__main__':
    # change path to properly import qubit_network package when called
    # from terminal as script and import modules to test
    CURRENTDIR = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe())))
    PARENTDIR = os.path.dirname(os.path.dirname(CURRENTDIR))
    sys.path.insert(1, PARENTDIR)
    from qubit_network.QubitNetwork import QubitNetwork
    from qubit_network.hamiltonian import pauli_product

    unittest.main()
