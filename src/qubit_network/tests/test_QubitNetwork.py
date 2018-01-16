# pylint: skip-file
import unittest
import sys
import os
import inspect

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal, assert_allclose
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


class TestPauliProduct(unittest.TestCase):
    def test_pauli_products(self):
        def to_numpy(sympy_obj):
            return np.asarray(sympy_obj.tolist(), dtype=np.complex)
        x1 = to_numpy(pauli_product(1, 0))
        xx = to_numpy(pauli_product(1, 1))
        y2 = to_numpy(pauli_product(0, 2))
        yz = to_numpy(pauli_product(2, 3))
        assert_array_equal(x1, [[0, 0, 1.0, 0], [0, 0, 0, 1.0], [1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        assert_array_equal(xx, [[0, 0, 0, 1.0], [0, 0, 1.0, 0], [0, 1.0, 0, 0], [1.0, 0, 0, 0]])
        assert_array_equal(y2, [[0, -1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])
        assert_array_equal(yz, [[0, 0, -1j, 0], [0, 0, 0, 1j], [1j, 0, 0, 0], [0, -1j, 0, 0]])


if __name__ == '__main__':
    # change path to properly import qubit_network package when called
    # from terminal as script and import modules to test
    CURRENTDIR = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe())))
    PARENTDIR = os.path.dirname(os.path.dirname(CURRENTDIR))
    sys.path.insert(1, PARENTDIR)
    from qubit_network.QubitNetwork import QubitNetwork, pauli_product

    unittest.main()
