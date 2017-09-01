"""
A collection of utility functions not yet categorized.
"""
from collections import OrderedDict
import json
import numpy as np
import scipy
import sympy

import qutip

import theano
import theano.tensor as T


def complexrandn(dim1, dim2):
    """Generates an array of pseudorandom, normally chosen, complex numbers."""
    big_matrix = np.random.randn(dim1, dim2, 2)
    return big_matrix[:, :, 0] + 1.j * big_matrix[:, :, 1]


def isvector(arr):
    """Check if a numpy array is a vector-like object."""
    # we are not using `arr.ndims` in case the input is a qutip object
    ndims = len(arr.shape)
    return (ndims == 1
            or (ndims == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1)))


def _complex2bigreal_vector(vector):
    """Convert a complex vector to big real notation."""
    vector = vector.reshape((vector.shape[0], 1))
    return np.concatenate((np.real(vector), np.imag(vector)), axis=0)


def _complex2bigreal_matrix(matrix):
    """Convert complex matrix to big real notation."""
    first_row = np.concatenate((np.real(matrix), -np.imag(matrix)), axis=1)
    second_row = np.concatenate((np.imag(matrix), np.real(matrix)), axis=1)
    return np.concatenate((first_row, second_row), axis=0)


def complex2bigreal(arr):
    """Convert from complex to big real representation.

    To avoid the problem of theano and similar libraries not properly
    supporting the gradient of complex objects, we map every complex
    nxn matrix U to a bigger 2nx2n real matrix defined as
    [[Ur, -Ui], [Ui, Ur]], where Ur and Ui are the real and imaginary
    parts of U.

    The input argument can be either a qutip object representing a ket,
    or a qutip object representing an operator (a density matrix).
    """
    # if qutip object, extract numpy arrays from it
    if isinstance(arr, qutip.Qobj):
        arr = arr.data.toarray()
    arr = np.asarray(arr).astype(np.complex)
    # if `arr` is a vector (possibly of shape Nx1 or 1xN)
    if isvector(arr):
        outarr = _complex2bigreal_vector(arr)
    else:
        outarr = _complex2bigreal_matrix(arr)
    return outarr


def bigreal2complex(arr):
    """Convert numpy array back into regular complex form.

    NOTE: The output will always be a numpy.ndarray of complex dtype
    """
    arr = np.asarray(arr)
    if isvector(arr):
        # `arr` may be a Nx1 or 1xN dimensional vector, or a flat vector
        try:
            arr_len = arr.shape[0] * arr.shape[1]
        except IndexError:
            arr_len = len(arr)
        # make `arr` an Nx1 vector
        arr = arr.reshape((arr_len, 1))
        real_part = arr[:arr.shape[0] // 2]
        imag_part = arr[arr.shape[0] // 2:]
        return real_part + 1j * imag_part
    else:
        real_part = arr[:arr.shape[0] // 2, :arr.shape[1] // 2]
        imag_part = arr[arr.shape[0] // 2:, :arr.shape[1] // 2]
        return real_part + 1j * imag_part


def bigreal2qobj(arr):
    """Convert big real vector into corresponding qutip object."""
    if arr.ndim == 1 or arr.shape[0] != arr.shape[1]:
        arr = bigreal2complex(arr)
        num_qubits = scipy.log2(arr.shape[0]).astype(int)
        return qutip.Qobj(arr, dims=[[2] * num_qubits, [1] * num_qubits])
    elif arr.shape[0] == arr.shape[1]:
        arr = bigreal2complex(arr)
        num_qubits = scipy.log2(arr.shape[0]).astype(int)
        return qutip.Qobj(arr, dims=[[2] * num_qubits] * num_qubits)
    else:
        raise ValueError('Not sure what to do with this here.')


def theano_matrix_grad(matrix, parameters):
    shape = matrix.shape
    num_elements = shape[0] * shape[1]
    flattened_matrix = T.flatten(matrix)
    def grad_element(i, arr):
        return T.grad(arr[i], parameters)
    flattened_grads, _ = theano.scan(fn=grad_element,
                                     sequences=T.arange(num_elements),
                                     non_sequences=flattened_matrix)
    num_gradients = parameters.shape[0]
    return T.reshape(flattened_grads.T, (num_gradients, shape[0], shape[1]))

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
            for col in sorted(columns, key=key, reverse=reverse)
        ]

    return sorter
