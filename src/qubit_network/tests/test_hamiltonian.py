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
        hamiltonian = QubitNetworkHamiltonian(expr=expr)
        hamiltonian_matrix = hamiltonian.get_matrix()
        self.assertListEqual(
            sympy.flatten(hamiltonian_matrix),
            sympy.flatten(sympy.Matrix([[0, 0, 1.0*a, 1.0*b],
                                        [0, 0, 1.0*b, 1.0*a],
                                        [1.0*a, 1.0*b, 0, 0],
                                        [1.0*b, 1.0*a, 0, 0]])))
        self.assertEqual(hamiltonian.num_qubits, 2)

    def test_parse_from_parameters_all(self):
        hamiltonian = QubitNetworkHamiltonian(interactions='all',
                                              num_qubits=2)
        self.assertListEqual(
            hamiltonian.interactions,
            [(1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (0, 3),(1, 1),(1, 2),
             (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
        )
        self.assertEqual(
            hamiltonian.matrices[0],
            sympy.Matrix([[  0,   0, 1.0,   0],
                          [  0,   0,   0, 1.0],
                          [1.0,   0,   0,   0],
                          [  0, 1.0,   0,   0]])
        )
        self.assertEqual(hamiltonian.free_parameters[0],
                         sympy.Symbol('J10'))
        self.assertEqual(hamiltonian.num_qubits, 2)

    def test_parse_from_topology_with_strings(self):
        from sympy import I
        topology = {
            ((0, 1), 'xx'): 'a',
            ((0, 1), 'xy'): 'a',
            ((0,), 'x'): 'b',
            ((1,), 'z'): 'c',
        }
        hamiltonian = QubitNetworkHamiltonian(
            num_qubits=2, net_topology=topology)
        hamiltonian_matrix = hamiltonian.get_matrix()
        a, b, c = sympy.symbols('a, b, c')
        self.assertEqual(hamiltonian.num_qubits, 2)
        self.assertEqual(
            hamiltonian_matrix,
            sympy.Matrix([
                [1.0 * c,  0., 1.0 * b,  a * (1 - I)],
                [0., -1.0 * c, a * (1 + 1.0 * I),  1.0 * b],
                [1.0 * b,  a * (1 - I), 1.0 * c,  0.],
                [a * (1 + 1.0 * I),  1.0 * b, 0., -1.0 * c]]))

    def test_parse_from_topology_with_symbols(self):
        from sympy import I
        a, b, c = sympy.symbols('a, b, c')
        topology = {
            ((0, 1), 'xx'): a,
            ((0, 1), 'xy'): a,
            ((0,), 'x'): b,
            ((1,), 'z'): c,
        }
        hamiltonian = QubitNetworkHamiltonian(
            num_qubits=2, net_topology=topology)
        hamiltonian_matrix = hamiltonian.get_matrix()
        self.assertEqual(hamiltonian.num_qubits, 2)
        self.assertEqual(
            hamiltonian_matrix,
            sympy.Matrix([
                [1.0 * c,  0., 1.0 * b,  a * (1 - I)],
                [0., -1.0 * c, a * (1 + 1.0 * I),  1.0 * b],
                [1.0 * b,  a * (1 - I), 1.0 * c,  0.],
                [a * (1 + 1.0 * I),  1.0 * b, 0., -1.0 * c]]))

    def test_parse_from_topology_with_tuples_notation(self):
        a, b, c = sympy.symbols('a, b, c')
        topology = {
            (1, 1): a,
            (0, 1): a,
            (1, 0): b,
            (0, 3): c,
        }
        # import ipdb; ipdb.set_trace()
        hamiltonian = QubitNetworkHamiltonian(
            num_qubits=2, net_topology=topology)
        hamiltonian_matrix = hamiltonian.get_matrix()
        self.assertEqual(hamiltonian.num_qubits, 2)
        self.assertEqual(
            hamiltonian_matrix,
            sympy.Matrix([
                [1.0 * c,  1.0 * a, 1.0 * b,  1.0 * a],
                [1.0 * a, -1.0 * c, 1.0 * a,  1.0 * b],
                [1.0 * b,  1.0 * a, 1.0 * c,  1.0 * a],
                [1.0 * a,  1.0 * b, 1.0 * a, -1.0 * c]]))


    def test_theano_graph(self):
        a, b = sympy.symbols('a, b')
        x1 = pauli_product(1, 0)
        xx = pauli_product(1, 1)
        expr = a * x1 + b * xx
        hamiltonian = QubitNetworkHamiltonian(expr=expr)
        J, theano_hamiltonian = hamiltonian.build_theano_graph()
        compute_hamiltonian = theano.function([], theano_hamiltonian)
        J.set_value([1, 1])
        assert_array_equal(
            compute_hamiltonian(),
            np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                      [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                      [ 0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.],
                      [ 0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.],
                      [ 0.,  0., -1., -1.,  0.,  0.,  0.,  0.],
                      [ 0.,  0., -1., -1.,  0.,  0.,  0.,  0.],
                      [-1., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
                      [-1., -1.,  0.,  0.,  0.,  0.,  0.,  0.]])
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
