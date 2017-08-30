# pylint: skip-file
import unittest
import sys
import os
import inspect

import numpy as np
from numpy.testing import assert_array_equal
import qutip


class TestComplex2BigReal(unittest.TestCase):
    def test_bigreal_conversion(self):
        test_vec1 = np.array([1, 2])
        test_vec2 = np.array([1j, 1 - 2j])
        test_matrix1 = np.array([[0.2, 0.1], [1., -3.]])
        test_matrix2 = qutip.sigmax()
        target_bigreal_vec = np.array([[1], [2], [0], [0]])

        assert_array_equal(
            utils.complex2bigreal(test_vec1), target_bigreal_vec)

        assert_array_equal(
            utils.complex2bigreal(qutip.sigmax()),
            np.array([[0, 1, 0, 0],
                      [1, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]])
        )

        assert_array_equal(
            utils.complex2bigreal(qutip.sigmay()),
            np.array([[0, 0, 0, 1],
                      [0, 0, -1, 0],
                      [0, -1, 0, 0],
                      [1, 0, 0, 0]])
        )
        # test back and forth between complex and bigreal form
        assert_array_equal(
            utils.bigreal2complex(utils.complex2bigreal(test_vec1)),
            test_vec1.reshape((len(test_vec1), 1))
        )
        assert_array_equal(
            utils.bigreal2complex(utils.complex2bigreal(test_vec2)),
            test_vec2.reshape((len(test_vec2), 1))
        )
        assert_array_equal(
            utils.bigreal2complex(utils.complex2bigreal(test_matrix1)),
            test_matrix1
        )
        assert_array_equal(
            utils.bigreal2complex(utils.complex2bigreal(test_matrix2)),
            test_matrix2.data.toarray()
        )



if __name__ == '__main__':
    # change path to properly import qubit_network package when called
    # from terminal as script
    CURRENTDIR = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe())))
    PARENTDIR = os.path.dirname(os.path.dirname(CURRENTDIR))
    sys.path.insert(1, PARENTDIR)
    import qubit_network.utils as utils

    unittest.main()
