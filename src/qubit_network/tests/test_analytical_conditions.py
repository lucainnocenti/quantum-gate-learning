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


class TestAnalyticalConditions(unittest.TestCase):
    def test_pairwise_interactions_indices(self):
        indices = ac.pairwise_interactions_indices(2)
        self.assertListEqual(
            indices,
            [(1,0),(2,0),(3,0),(0,1),(0,2),(0,3),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)]
        )

    def test_pairwise_diagonal_interactions_indices(self):
        indices = ac.pairwise_diagonal_interactions_indices(2)
        self.assertListEqual(
            indices,
            [(1,0),(2,0),(3,0),(0,1),(0,2),(0,3),(1,1),(2,2),(3,3)]
        )

    def test_is_diagonal_interaction(self):
        self.assertTrue(ac.is_diagonal_interaction((1, )))
        self.assertTrue(ac.is_diagonal_interaction((1, 0)))
        self.assertTrue(ac.is_diagonal_interaction((1, 1)))
        self.assertTrue(ac.is_diagonal_interaction((1, 1, 1)))
        self.assertTrue(ac.is_diagonal_interaction((1, 0, 0)))
        self.assertFalse(ac.is_diagonal_interaction((1, 2)))
        self.assertFalse(ac.is_diagonal_interaction((1, 0, 2)))



if __name__ == '__main__':
    # change path to properly import qubit_network package when called
    # from terminal as script and import modules to test
    CURRENTDIR = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe())))
    PARENTDIR = os.path.dirname(os.path.dirname(CURRENTDIR))
    sys.path.insert(1, PARENTDIR)
    import qubit_network.analytical_conditions as ac
    unittest.main(failfast=True)
