import unittest
import sys
import os
import inspect

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import sympy

import qutip
import theano
import theano.tensor as T


class TestFidelityGraph(unittest.TestCase):
    def test_evolution_matrix(self):
        a, b = sympy.symbols('a, b')
        x1 = pauli_product(1, 0)
        xx = pauli_product(1, 1)
        expr = a * x1 + b * xx
        net = QubitNetwork(sympy_expr=expr)
        fidelity = FidelityGraph(2, 2, *net.build_theano_graph())
        compute_evolution = theano.function(
            [], fidelity.compute_evolution_matrix())
        # check that if all parameters are zero evolution is the identity
        assert_array_equal(compute_evolution(), np.identity(8))
        # update value of parameters and check updating of evolution
        fidelity.parameters.set_value([1, 0])
        assert_almost_equal(
            compute_evolution(),
            np.array([[0.54030231, 0., 0., 0., 0., 0., 0., 0.84147098],
                      [0., 0.54030231, 0., 0., 0., 0., 0.84147098, 0.],
                      [0., 0., 0.54030231, 0., 0., 0.84147098, 0., 0.],
                      [0., 0., 0., 0.54030231, 0.84147098, 0., 0., 0.],
                      [0., 0., 0., -0.84147098, 0.54030231, 0., 0., 0.],
                      [0., 0., -0.84147098, 0., 0., 0.54030231, 0., 0.],
                      [0., -0.84147098, 0., 0., 0., 0., 0.54030231, 0.],
                      [-0.84147098, 0., 0., 0., 0., 0., 0., 0.54030231]]))

if __name__ == '__main__':
    # change path to properly import qubit_network package when called
    # from terminal as script and import modules to test
    CURRENTDIR = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe())))
    PARENTDIR = os.path.dirname(os.path.dirname(CURRENTDIR))
    sys.path.insert(1, PARENTDIR)
    from qubit_network.QubitNetwork import QubitNetwork
    from qubit_network.hamiltonian import pauli_product
    from qubit_network.model import FidelityGraph

    unittest.main()
