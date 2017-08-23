"""
A collection of utility functions not yet categorized.
"""
from collections import OrderedDict
import json
import numpy as np
import qutip


def complexrandn(dim1, dim2):
    """Generates an array of pseudorandom, normally chosen, complex numbers."""
    big_matrix = np.random.randn(dim1, dim2, 2)
    return big_matrix[:, :, 0] + 1.j * big_matrix[:, :, 1]


def complex2bigreal(matrix):
    """Takes converts from complex to "big real" representation.

    To avoid the problem of theano and similar libraries not properly
    supporting the gradient of complex objects, we map every complex
    nxn matrix U to a bigger 2nx2n real matrix defined as
    [[Ur, -Ui], [Ui, Ur]], where Ur and Ui are the real and imaginary
    parts of U.

    The input argument can be either a qutip object representing a ket,
    or a qutip object representing an operator (a density matrix).
    """

    # if `matrix` is actually a qutip ket...
    if isinstance(matrix, qutip.Qobj) and matrix.shape[1] == 1:
        matrix = matrix.data.toarray()
        matrix = np.concatenate((np.real(matrix), np.imag(matrix)), axis=0)
        return matrix.reshape(matrix.shape[0])

    # else, we assume the input to represent a density matrix. It can be
    # both a 2d numpy array, or a 2d qutip object.
    else:
        if isinstance(matrix, qutip.Qobj):
            matrix = matrix.data.toarray()
        else:
            matrix = np.asarray(matrix)
        row1 = np.concatenate((np.real(matrix), -np.imag(matrix)), axis=1)
        row2 = np.concatenate((np.imag(matrix), np.real(matrix)), axis=1)
        return np.concatenate((row1, row2), axis=0)


def bigreal2complex(matrix):
    matrix = np.asarray(matrix)
    if len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]:
        real_part = matrix[:matrix.shape[0] // 2, :matrix.shape[1] // 2]
        imag_part = matrix[matrix.shape[0] // 2:, :matrix.shape[1] // 2]
        return real_part + 1j * imag_part
    elif len(matrix.shape) == 1:
        real_part = matrix[:matrix.shape[0] // 2]
        imag_part = matrix[matrix.shape[0] // 2:]
        return real_part + 1j * imag_part


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


def pauli_matrix(n_modes, position, which_pauli):
    sigmas = [qutip.qeye(2), qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]
    indices = [0] * n_modes
    indices[position] = which_pauli
    return qutip.tensor(*tuple(sigmas[index] for index in indices))


def pauli_product(*pauli_indices):
    n_modes = len(pauli_indices)
    partial_product = qutip.tensor(*([qutip.qeye(2)] * n_modes))
    for pos, pauli_index in enumerate(pauli_indices):
        partial_product *= pauli_matrix(n_modes, pos, pauli_index)
    return partial_product


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


def dm2ket(dm):
    """Converts density matrix to ket form, assuming it to be pure."""
    outket = dm[:, 0] / dm[0, 0] * np.sqrt(np.abs(dm[0, 0]))
    try:
        return qutip.Qobj(outket, dims=[dm.dims[0], [1] * len(dm.dims[0])])
    except AttributeError:
        # `dm` could be a simple matrix, not a qutip.Qobj object. In
        # this case just return the numpy array
        return outket


def ket_normalize(ket):
    return ket * np.exp(-1j * np.angle(ket[0, 0]))


def detensorize(bigm):
    """Assumes second matrix is 2x2."""
    out = np.zeros((bigm.shape[0] * bigm.shape[1], 2, 2), dtype=np.complex)
    idx = 0
    for row in range(bigm.shape[0] // 2):
        for col in range(bigm.shape[1] // 2):
            trow = 2 * row
            tcol = 2 * col
            foo = np.zeros([2, 2], dtype=np.complex)
            foo = np.zeros([2, 2], dtype=np.complex)
            foo[0, 0] = 1
            foo[0, 1] = bigm[trow, tcol + 1] / bigm[trow, tcol]
            foo[1, 0] = bigm[trow + 1, tcol] / bigm[trow, tcol]
            foo[1, 1] = bigm[trow + 1, tcol + 1] / bigm[trow, tcol]
            out[idx] = foo
            idx += 1
    return out



def chop(arr, eps=1e-5):
    if isinstance(arr, qutip.Qobj):
        _arr = arr.data.toarray()
        _arr.real[np.abs(_arr.real) < eps] = 0.0
        _arr.imag[np.abs(_arr.imag) < eps] = 0.0
        _arr = qutip.Qobj(_arr, dims=arr.dims)
        return _arr
    else:
        arr = np.asarray(arr)
        arr.real[np.abs(arr.real) < eps] = 0.0
        arr.imag[np.abs(arr.imag) < eps] = 0.0
        return arr


def transpose(list_of_lists):
    return list(map(list, zip(*list_of_lists)))


def print_OrderedDict(od):
    outdict = OrderedDict()
    for k, v in od.items():
        outdict[str(k)] = v
    print(json.dumps(outdict, indent=4))


def custom_dataframe_sort(key=None, reverse=False, cmp=None):
    """Make a custom sorter for pandas dataframes."""
    def sorter(df):
        columns = list(df)
        return [
            columns.index(col)
            for col in sorted(columns, key=key, reverse=reverse, cmp=cmp)
        ]
    return sorter
