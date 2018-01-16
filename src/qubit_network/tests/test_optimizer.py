# pylint: skip-file
import unittest
import sys
import os
import inspect

import numpy as np
from numpy.testing import (assert_array_equal, assert_almost_equal,
                           assert_array_less)
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
        model = QubitNetworkGateModel(sympy_expr=hamiltonian_model, target_gate = target_gate)
        optimizer = Optimizer(model)
        # set parameters to have evolution implement XX gate
        new_parameters = [0, 0]
        new_parameters[model.free_parameters.index(J11)] = np.pi / 2
        optimizer.net.parameters.set_value(new_parameters)
        # check via optimizer.cost that |00> goes to |11>
        fidelity = theano.function([], optimizer.cost, givens={
            model.inputs: complex2bigreal([1, 0, 0, 0]).reshape((1, 8)),
            model.outputs: complex2bigreal([0, 0, 0, 1]).reshape((1, 8))
        })()
        assert_almost_equal(fidelity, np.array(1))

    def test_grad(self):
        J00, J11 = sympy.symbols('J00 J11')
        hamiltonian_model = pauli_product(0, 0) * J00 + pauli_product(1, 1) * J11
        target_gate = qutip.Qobj(pauli_product(1, 1).tolist(), dims=[[2] * 2] * 2)
        model = QubitNetworkGateModel(sympy_expr=hamiltonian_model,
                                      initial_values=1)
        optimizer = Optimizer(model, target_gate=target_gate,
                              learning_rate=1, decay_rate=0.01)
        optimizer.refill_training_data(sample_size=2)
        train_model = theano.function([], optimizer.cost,
            updates=optimizer.updates,
            givens={
                model.inputs: optimizer.vars['train_inputs'],
                model.outputs: optimizer.vars['train_outputs']
            }
        )
        first_fid = train_model()
        second_fid = train_model()
        assert_array_less(first_fid, second_fid)

    # def test_single_train_step(self):
    #     J00, J11 = sympy.symbols('J00 J11')
    #     hamiltonian_model = pauli_product(0, 0) * J00 + pauli_product(1, 1) * J11
    #     target_gate = qutip.Qobj(pauli_product(1, 1).tolist(), dims=[[2] * 2] * 2)
    #     model = QubitNetworkGateModel(sympy_expr=hamiltonian_model)
    #     optimizer = Optimizer(model, target_gate=target_gate)
    #     optimizer.refill_training_data()
    #     train_model = theano.function([], optimizer.cost,
    #         updates=optimizer.updates,
    #         givens={
    #             model.inputs: optimizer.vars['train_inputs'],
    #             model.outputs: optimizer.vars['train_outputs']
    #         }
    #     )
    #     print(train_model())
    #     print(train_model())


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
