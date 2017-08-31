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
    def test_evolution_matrix(self):
        a, b = sympy.symbols('a, b')
        x1 = pauli_product(1, 0)
        xx = pauli_product(1, 1)
        expr = a * x1 + b * xx
        net = QubitNetwork(sympy_expr=expr)
        fidelity = FidelityGraph(2, 2, *net.build_theano_graph(), None)
        compute_evolution = theano.function(
            [], fidelity.compute_evolution_matrix())
        # check that if all parameters are zero evolution is the identity
        assert_array_equal(compute_evolution(), np.identity(8))
        # try with a=1, b=0
        newJ = [0, 0]
        newJ[net.free_parameters.index(a)] = 1
        fidelity.parameters.set_value(newJ)
        new_evolution = complex2bigreal(scipy.linalg.expm(
            -1j * np.asarray(x1).astype(np.complex)))
        assert_almost_equal(compute_evolution(), new_evolution)
        # try with a=1.3, b=-3.
        newJ = [0, 0]
        newJ[net.free_parameters.index(a)] = 1.3
        newJ[net.free_parameters.index(b)] = -3
        fidelity.parameters.set_value(newJ)
        new_evolution = complex2bigreal(scipy.linalg.expm(
            -1j * np.asarray(1.3 * x1 - 3 * xx).astype(np.complex)))
        assert_almost_equal(compute_evolution(), new_evolution)

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
    from qubit_network.utils import bigreal2complex, complex2bigreal

    unittest.main()
