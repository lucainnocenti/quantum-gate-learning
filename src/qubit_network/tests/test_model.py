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
        a, b = sympy.symbols('a, b')
        x1 = pauli_product(1, 0)
        xx = pauli_product(1, 1)
        expr = a * x1 + b * xx
        net = QubitNetwork(sympy_expr=expr, initial_values=0)
        model = FidelityGraph(2, 2, *net.build_theano_graph(), None)
        compute_evolution = theano.function(
            [], model.compute_evolution_matrix())
        # check that if all parameters are zero evolution is the identity
        assert_array_equal(compute_evolution(), np.identity(8))
        # try with a=1, b=0
        newJ = [0, 0]
        newJ[net.free_parameters.index(a)] = 1
        model.parameters.set_value(newJ)
        new_evolution = complex2bigreal(
            scipy.linalg.expm(-1j * np.asarray(x1).astype(np.complex)))
        assert_almost_equal(compute_evolution(), new_evolution)
        # try with a=1.3, b=-3.
        newJ = [0, 0]
        newJ[net.free_parameters.index(a)] = 1.3
        newJ[net.free_parameters.index(b)] = -3
        model.parameters.set_value(newJ)
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
        net = QubitNetwork(sympy_expr=expr)
        model = FidelityGraph(2, 2, *net.build_theano_graph(), some_target_gate)
        # make training data (this uses the target gate not the current one)
        inputs, outputs = model.generate_training_states(4)
        # set the parameters in the hamiltonian to the correct values
        newJ = [0, 0]
        newJ[net.free_parameters.index(J33)] = 1.
        newJ[net.free_parameters.index(J20)] = 1.3
        model.parameters.set_value(newJ)
        # compute evolution matrix with current (randomly generated) parameters
        evolution_matrix = theano.function([],
                                           model.compute_evolution_matrix())()
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
        net = QubitNetwork(sympy_expr=expr)
        zz_as_gate = qutip.Qobj(zz.tolist(), dims=[[2] * 2] * 2)
        y1_as_gate = qutip.Qobj(y1.tolist(), dims=[[2] * 2] * 2)
        some_gate = zz_as_gate + 1.3 * y1_as_gate
        model = FidelityGraph(2, 2, *net.build_theano_graph(), some_gate)
        inputs, outputs = model.generate_training_states(6)
        self.assertEqual(inputs.shape[0], 6)
        self.assertEqual(outputs.shape[0], 6)
        self.assertEqual(inputs.shape[1], 2 * 2**2)
        qutip_dims = [[2] * 2, [1] * 2]
        for input_, output in zip(inputs, outputs):
            input_ = qutip.Qobj(bigreal2complex(input_), dims=qutip_dims)
            output = qutip.Qobj(bigreal2complex(output), dims=qutip_dims)
            assert_almost_equal(
                (some_gate * input_).data.toarray(),
                output.data.toarray()
            )

    def test_fidelities_no_ptrace_identity(self):
        net = QubitNetwork(num_qubits=2, interactions='all', initial_values=0)
        # target_gate set to qutip.qeye, so each state is its own target
        model = FidelityGraph(
            2, 2, *net.build_theano_graph(), qutip.qeye([2, 2]))
        inputs, outputs = model.generate_training_states(10)
        fidelities = FidelityGraph._fidelities_no_ptrace(model.inputs,
                                                         model.outputs)
        compute_fidelities = theano.function([], fidelities, givens={
            model.inputs: inputs, model.outputs: outputs})
        assert_almost_equal(compute_fidelities(), np.ones(len(inputs)))

    def test_fidelity_no_ptrace_identity(self):
        net = QubitNetwork(num_qubits=2, interactions='all', initial_values=0)
        # target_gate set to qutip.qeye, so each state is its own target
        model = FidelityGraph(
            2, 2, *net.build_theano_graph(), qutip.qeye([2, 2]))
        inputs, outputs = model.generate_training_states(4)
        fidelity = theano.function([], model.fidelity(), givens={
            model.inputs: inputs, model.outputs: outputs})()
        assert_almost_equal(fidelity, np.array(1))
    
    def test_fidelity_no_ptrace_y1_zz(self):
        J20, J33 = sympy.symbols('J20 J33')
        y1 = pauli_product(2, 0)
        zz = pauli_product(3, 3)
        expr = J20 * y1 + J33 * zz
        # parameters initialied at random values
        net = QubitNetwork(sympy_expr=expr)
        # make gate
        zz_as_gate = qutip.Qobj(zz.tolist(), dims=[[2] * 2] * 2)
        y1_as_gate = qutip.Qobj(y1.tolist(), dims=[[2] * 2] * 2)
        some_gate = zz_as_gate + 1.3 * y1_as_gate
        # create model
        model = FidelityGraph(2, 2, *net.build_theano_graph(), some_gate)
        # make random inputs and corresponding target outputs
        inputs, outputs = model.generate_training_states(10)
        # compute fidelity from theano functions to test
        fidelities = theano.function([], model.fidelity(return_mean=False),
            givens={model.inputs: inputs, model.outputs: outputs})()
        # extract evolution matrix for given parameters
        evolution_matrix = theano.function(
            [], model.compute_evolution_matrix())()
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
    from qubit_network.utils import (bigreal2complex, complex2bigreal,
                                     bigreal2qobj)

    unittest.main()
