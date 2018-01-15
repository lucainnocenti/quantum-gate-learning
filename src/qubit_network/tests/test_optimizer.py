# pylint: skip-file
import unittest
import sys
import os
import inspect

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import scipy
import sympy

import qutip
import theano
import theano.tensor as T

class TestOptimizer(unittest.TestCase):
    def test_optimizer_cost(self):
        J00, J11 = sympy.symbols('J00 J11')
        hamiltonian_model = pauli_product(0, 0) * J00 + pauli_product(1, 1) * J11
        target_gate = qutip.Qobj(pauli_product(1, 1).tolist(), dims=[[2] * 2] * 2)
        net = QubitNetworkGateModel(sympy_expr=hamiltonian_model, target_gate = target_gate)
        optimizer = Optimizer(net)
        # set parameters to have evolution implement XX gate
        new_parameters = [0, 0]
        new_parameters[net.free_parameters.index(J11)] = np.pi / 2
        optimizer.net.parameters.set_value(new_parameters)
        # check via optimizer.cost that |00> goes to |11>
        fidelity = theano.function([], optimizer.cost, givens={
            net.inputs: complex2bigreal([1, 0, 0, 0]).reshape((1, 8)),
            net.outputs: complex2bigreal([0, 0, 0, 1]).reshape((1, 8))
        })()
        assert_almost_equal(fidelity, np.array(1))


if __name__ == '__main__':
    # change path to properly import qubit_network package when called
    # from terminal as script and import modules to test
    CURRENTDIR = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe())))
    PARENTDIR = os.path.dirname(os.path.dirname(CURRENTDIR))
    sys.path.insert(1, PARENTDIR)
    from qubit_network.QubitNetwork import QubitNetwork, pauli_product
    from qubit_network.model import QubitNetworkGateModel
    from qubit_network.utils import (bigreal2complex, complex2bigreal,
                                     bigreal2qobj, theano_matrix_grad)
    # from qubit_network.theano_qutils import _fidelity_no_ptrace
    from qubit_network.Optimizer import Optimizer

    unittest.main(failfast=True)
