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


class TestFidelityGraph(unittest.TestCase):
    def test_evolution_matrix_x1_xx(self):
        # build simple hamiltonian
        a, b = sympy.symbols('a, b')
        x1 = pauli_product(1, 0)
        xx = pauli_product(1, 1)
        expr = a * x1 + b * xx
        # make model with initially 0 values for interactions
        net = QubitNetworkModel(sympy_expr=expr, initial_values=0)
        compute_evolution = theano.function([], net.compute_evolution_matrix())
        # check that if all parameters are zero evolution is the identity
        assert_array_equal(compute_evolution(), np.identity(8))
        # set a=1, b=0
        newJ = [0, 0]
        newJ[net.free_parameters.index(a)] = 1
        net.parameters.set_value(newJ)
        new_evolution = complex2bigreal(
            scipy.linalg.expm(-1j * np.asarray(x1).astype(np.complex)))
        assert_almost_equal(compute_evolution(), new_evolution)
        # try with a=1.3, b=-3.
        newJ = [0, 0]
        newJ[net.free_parameters.index(a)] = 1.3
        newJ[net.free_parameters.index(b)] = -3
        net.parameters.set_value(newJ)
        new_evolution = complex2bigreal(scipy.linalg.expm(
            -1j * np.asarray(1.3 * x1 - 3 * xx).astype(np.complex)))
        assert_almost_equal(compute_evolution(), new_evolution)

    def test_evolution_matrix_y1_zz(self):
        # make expr
        J20, J33 = sympy.symbols('J20 J33')
        y1 = pauli_product(2, 0)
        y1_as_gate = qutip.Qobj(y1.tolist(), dims=[[2] * 2] * 2)
        zz = pauli_product(3, 3)
        zz_as_gate = qutip.Qobj(zz.tolist(), dims=[[2] * 2] * 2)
        expr = J20 * y1 + J33 * zz
        some_hamiltonian = zz_as_gate + 1.3 * y1_as_gate
        some_target_gate = qutip.Qobj(
            scipy.linalg.expm(-1j * some_hamiltonian.data.toarray()),
            dims=[[2] * 2] * 2)
        # make net with random initial values
        net = QubitNetworkModel(sympy_expr=expr)
        net.target_gate = some_target_gate
        # make training data (this uses the target gate not the current one)
        inputs, outputs = net.generate_training_states(4)
        # set the parameters in the hamiltonian to the correct values
        newJ = [0, 0]
        newJ[net.free_parameters.index(J33)] = 1.
        newJ[net.free_parameters.index(J20)] = 1.3
        net.parameters.set_value(newJ)
        # compute evolution matrix with current (randomly generated) parameters
        evolution_matrix = theano.function([],
                                           net.compute_evolution_matrix())()
        # check evolution matrix
        assert_almost_equal(
            bigreal2complex(evolution_matrix),
            some_target_gate.data.toarray() 
        )


    def test_generation_training_states(self):
        J20, J33 = sympy.symbols('J20 J33')
        y1 = pauli_product(2, 0)
        zz = pauli_product(3, 3)
        expr = J20 * y1 + J33 * zz
        net = QubitNetworkModel(sympy_expr=expr)
        zz_as_gate = qutip.Qobj(zz.tolist(), dims=[[2] * 2] * 2)
        y1_as_gate = qutip.Qobj(y1.tolist(), dims=[[2] * 2] * 2)
        target_gate = zz_as_gate + 1.3 * y1_as_gate
        net.target_gate = target_gate
        inputs, outputs = net.generate_training_states(6)
        self.assertEqual(inputs.shape[0], 6)
        self.assertEqual(outputs.shape[0], 6)
        self.assertEqual(inputs.shape[1], 2 * 2**2)
        qutip_dims = [[2] * 2, [1] * 2]
        for input_, output in zip(inputs, outputs):
            input_ = qutip.Qobj(bigreal2complex(input_), dims=qutip_dims)
            output = qutip.Qobj(bigreal2complex(output), dims=qutip_dims)
            assert_almost_equal(
                (target_gate * input_).data.toarray(),
                output.data.toarray()
            )

    def test_single_fidelity_no_ptrace(self):
        state1 = qutip.rand_ket(4).data.toarray()
        state2 = qutip.rand_ket(4).data.toarray()
        # fidelity via numpy
        fidelity_np = np.abs(np.vdot(state1, state2))**2
        # fidelity via theano
        state1 = complex2bigreal(state1).reshape((1, 8))
        state2 = complex2bigreal(state2).reshape((1, 8))
        state1 = state1.astype(theano.config.floatX)
        state2 = state2.astype(theano.config.floatX)
        fidelity_theano = theano.function(
            [], _fidelity_no_ptrace(0, state1, state2))()
        # check they are compatible
        assert_almost_equal(fidelity_np, fidelity_theano)

    def test_fidelities_no_ptrace_identity(self):
        net = QubitNetworkModel(
            num_qubits=2, interactions='all', initial_values=0)
        # target_gate set to qutip.qeye, so each state is its own target
        net.target_gate = qutip.qeye([2, 2])
        inputs, outputs = net.generate_training_states(10)
        fidelities = net._fidelities_no_ptrace(net.inputs, net.outputs)
        compute_fidelities = theano.function([], fidelities, givens={
            net.inputs: inputs, net.outputs: outputs})
        assert_almost_equal(compute_fidelities(), np.ones(len(inputs)))

    def test_fidelity_no_ptrace_identity(self):
        net = QubitNetworkModel(
            num_qubits=2, interactions='all', initial_values=0)
        # target_gate set to qutip.qeye, so each state is its own target
        net.target_gate = qutip.qeye([2, 2])
        inputs, outputs = net.generate_training_states(4)
        fidelity = theano.function([], net.fidelity(), givens={
            net.inputs: inputs, net.outputs: outputs})()
        assert_almost_equal(fidelity, np.array(1))
    
    def test_fidelity_no_ptrace_y1_zz(self):
        J20, J33 = sympy.symbols('J20 J33')
        y1 = pauli_product(2, 0)
        zz = pauli_product(3, 3)
        expr = J20 * y1 + J33 * zz
        # make gate
        zz_as_gate = qutip.Qobj(zz.tolist(), dims=[[2] * 2] * 2)
        y1_as_gate = qutip.Qobj(y1.tolist(), dims=[[2] * 2] * 2)
        target_gate = zz_as_gate + 1.3 * y1_as_gate
        # create model, parameters initialied at random values
        net = QubitNetworkModel(sympy_expr=expr)
        net.target_gate = target_gate
        # make random inputs and corresponding target outputs
        inputs, outputs = net.generate_training_states(10)
        # compute fidelity from theano functions to test
        fidelities = theano.function([], net.fidelity(return_mean=False),
            givens={net.inputs: inputs, net.outputs: outputs})()
        # extract evolution matrix for given parameters
        evolution_matrix = theano.function(
            [], net.compute_evolution_matrix())()
        evolution_matrix = bigreal2qobj(evolution_matrix)
        # make inputs and outputs into qutip objects, easier to handle
        inputs = [bigreal2qobj(input_) for input_ in inputs]
        outputs = [bigreal2qobj(output) for output in outputs]
        # compute actual outputs
        actual_outputs = [evolution_matrix * in_ for in_ in inputs]
        # recompute fidelities with qutip
        def fid(ket1, ket2):
            ket1 = ket1.data.toarray()
            ket2 = ket2.data.toarray()
            return np.abs(np.vdot(ket1, ket2)) ** 2
        fidelities_check = [
            fid(out, actual_out)
            for out, actual_out in zip(outputs, actual_outputs)
        ]
        # check results are compatible
        assert_almost_equal(fidelities, fidelities_check)

    def test_grad_evolution(self):
        J00, J11 = sympy.symbols('J00 J11')
        hamiltonian = J00 * pauli_product(0, 0) + J11 * pauli_product(1, 1)
        net = QubitNetworkModel(sympy_expr=hamiltonian)
        unitary_evolution = net.compute_evolution_matrix()
        grads_matrix = theano_matrix_grad(unitary_evolution, net.parameters)
        compute_grads = theano.function([], grads_matrix)
        args = [1, 1.1]
        net.parameters.set_value(args)
        grads0, grads1 = compute_grads()
        # manual gradient
        _compute_evol = theano.function([], unitary_evolution)
        def compute_evol(*args):
            net.parameters.set_value([*args])
            return _compute_evol()
        eps = 0.00000001
        manual_grad0 = (compute_evol(args[0] + eps, args[1]) - compute_evol(
            args[0], args[1])) / eps
        # compare results
        assert_almost_equal(grads0, manual_grad0)


class TestOptimizer(unittest.TestCase):
    def test_optimizer_cost(self):
        J00, J11 = sympy.symbols('J00 J11')
        hamiltonian_model = pauli_product(0, 0) * J00 + pauli_product(1, 1) * J11
        target_gate = qutip.Qobj(pauli_product(1, 1).tolist(), dims=[[2] * 2] * 2)
        net = QubitNetworkModel(sympy_expr=hamiltonian_model)
        net.target_gate = target_gate
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
    from qubit_network.QubitNetwork import QubitNetwork
    from qubit_network.hamiltonian import pauli_product
    from qubit_network.model import (QubitNetworkModel, Optimizer,
                                     _fidelity_no_ptrace)
    from qubit_network.utils import (bigreal2complex, complex2bigreal,
                                     bigreal2qobj, theano_matrix_grad)

    unittest.main()
