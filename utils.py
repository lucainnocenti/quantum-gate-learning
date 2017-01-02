import numpy as np
import qutip


# __all__ = ['complexrandn', 'complex2bigreal', 'get_sigmas_index',
#            'generate_ss_terms', 'chars2pair']


def complexrandn(dim1, dim2):
    """Generates an array of pseudorandom, normally chosen, complex numbers."""
    big_matrix = np.random.randn(dim1, dim2, 2)
    return big_matrix[:, :, 0] + 1.j * big_matrix[:, :, 1]


def complex2bigreal(matrix):
    """Takes an nxn complex matrix and returns a 2nx2n real matrix.

    To avoid the problem of theano and similar libraries not properly
    supporting the gradient of complex objects, we map every complex
    nxn matrix U to a bigger 2nx2n real matrix defined as
    [[Ur, -Ui], [Ui, Ur]], where Ur and Ui are the real and imaginary
    parts of U.
    """
    row1 = np.concatenate((np.real(matrix), -np.imag(matrix)), axis=1)
    row2 = np.concatenate((np.imag(matrix), np.real(matrix)), axis=1)
    return np.concatenate((row1, row2), axis=0)


def get_sigmas_index(indices):
    """Takes a tuple and gives back a length-16 array with a single 1.

    Parameters
    ----------
    indices: a tuple of two integers, each one between 0 and 3.

    Examples
    --------
    >>> get_sigmas_index((1, 0))
    array([ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.])
    >>> get_sigmas_index((0, 3))
    array([ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.])

    """
    all_zeros = np.zeros(4 * 4)
    all_zeros[indices[0] * 4 + indices[1]] = 1.
    return all_zeros


def generate_ss_terms():
    """Returns the tensor products of every combination of two sigmas.

    Generates a list in which each element is the tensor product of two
    Pauli matrices, multiplied by the imaginary unit 1j and converted
    into big real form using complex2bigreal.
    The matrices are sorted in natural order, so that for example the
    3th element is the tensor product of sigma_0 and sigma_3 and the
    4th element is the tensor product of sigma_1 and sigma_0.
    """
    sigmas = [qutip.qeye(2), qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]
    sigma_pairs = []
    for idx1 in range(4):
        for idx2 in range(4):
            term = qutip.tensor(sigmas[idx1], sigmas[idx2])
            term = 1j * term.data.toarray()
            sigma_pairs.append(complex2bigreal(term))
    return np.asarray(sigma_pairs)


def chars2pair(chars):
    out_pair = []
    for idx in range(len(chars)):
        if chars[idx] == 'x':
            out_pair.append(1)
        elif chars[idx] == 'y':
            out_pair.append(2)
        elif chars[idx] == 'z':
            out_pair.append(3)
        else:
            raise ValueError('chars must contain 2 characters, each of'
                             'which equal to either x, y, or z')
    return tuple(out_pair)
