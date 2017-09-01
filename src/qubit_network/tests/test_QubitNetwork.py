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
    
    def test_parse_from_sympy_and_compile(self):
        sympy.init_printing(pretty_print=True)
        J_pars = np.asarray(sympy.symbols('J0:4:4')).reshape((4, 4))
        hamiltonian_model = (pauli_product(0, 0) * J_pars[0, 0] +
                             pauli_product(1, 1) * J_pars[1, 1])
        net = QubitNetwork(num_qubits=2, sympy_expr=hamiltonian_model,
                           initial_values=0)
        J, hamiltonian_graph = net.build_theano_graph()
        compute_hamiltonian = theano.function([], hamiltonian_graph)
        assert_array_equal(compute_hamiltonian(), np.zeros((8, 8)))
        J.set_value([1, 1])
        assert_array_equal(
            compute_hamiltonian()[4:, :4],
            np.array([[-1.,  0.,  0., -1.],
                      [ 0., -1., -1.,  0.],
                      [ 0., -1., -1.,  0.],
                      [-1.,  0.,  0., -1.]])
        )


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
