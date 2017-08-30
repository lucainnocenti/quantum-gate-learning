# pylint: skip-file
import unittest
import sys
import os
import inspect

import numpy as np
from numpy.testing import assert_array_equal
import sympy

import qutip


class TestPauliProduct(unittest.TestCase):
    def test_pauli_product(self):
        self.assertIsInstance(pauli_product(1), sympy.Matrix)
        self.assertIsInstance(pauli_product(1, 2), sympy.Matrix)
        self.assertListEqual(sympy.flatten(pauli_product(1)), [0, 1, 1, 0])
        self.assertListEqual(sympy.flatten(pauli_product(2)), [0, -1j, 1j, 0])
        self.assertListEqual(
            sympy.flatten(pauli_product(1, 1)),
            [0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0]
        )


class TestQubitNetworkHamiltonian(unittest.TestCase):
    def test_parse_from_sympy_expr(self):
        a, b = sympy.symbols('a, b')
        x1 = pauli_product(1, 0)
        xx = pauli_product(1, 1)
        expr = a * x1 + b * xx
        hamiltonian_matrix = QubitNetworkHamiltonian(expr=expr).get_matrix()
        self.assertListEqual(
            sympy.flatten(hamiltonian_matrix),
            sympy.flatten(sympy.Matrix([[0, 0, 1.0*a, 1.0*b],
                                        [0, 0, 1.0*b, 1.0*a],
                                        [1.0*a, 1.0*b, 0, 0],
                                        [1.0*b, 1.0*a, 0, 0]]))
        )


if __name__ == '__main__':
    # change path to properly import qubit_network package when called
    # from terminal as script and import modules to test
    CURRENTDIR = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe())))
    PARENTDIR = os.path.dirname(os.path.dirname(CURRENTDIR))
    sys.path.insert(1, PARENTDIR)
    from qubit_network.hamiltonian import (QubitNetworkHamiltonian,
                                           pauli_product)

    unittest.main()
