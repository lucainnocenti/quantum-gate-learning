import qutip
import theano
import theano.tensor as T


def _split_bigreal_ket(ket):
    """Splits in half a real vector of length 2N

    Given an input ket vector in big real form, returns a pair of real
    vectors, the first containing the first N elements, and the second
    containing the last N elements.
    """
    ket_real = ket[:ket.shape[0] // 2]
    ket_imag = ket[ket.shape[0] // 2:]
    return ket_real, ket_imag


def _ket_to_dm(ket):
    """Builds theano function to convert kets in dms in big real form.

    The input is expected to be a 1d real array, storing the state
    vector as `(psi_real, psi_imag)`, where `psi` is the complex vector.
    The outputs are real and imaginary part of the corresponding density
    matrix.
    """
    ket_real, ket_imag = _split_bigreal_ket(ket)
    ket_real = ket_real[:, None]
    ket_imag = ket_imag[:, None]

    dm_real = ket_real * ket_real.T + ket_imag * ket_imag.T
    dm_imag = ket_imag * ket_real.T - ket_real * ket_imag.T
    return dm_real, dm_imag


def _compute_fidelities_col_fn(col_idx, row_idx, matrix, num_ancillae):
    """
    `_compute_fidelities_col_fn` and `(...)row_fn` are the functions that
    handle the computation of the partial traces. The latter is run on
    each block of rows of the matrix to partial trace, with each block
     containing `2 ** num_ancillae` rows.
    For each block of rows, the former scans through the corresponding
    blocks of columns, taking the trace of each resulting submatrix.
    """
    subm_dim = 2**num_ancillae
    return T.nlinalg.trace(matrix[row_idx * subm_dim:(row_idx + 1) * subm_dim,
                                  col_idx * subm_dim:(col_idx + 1) * subm_dim])


def _compute_fidelities_row_fn(row_idx, matrix, num_ancillae):
    """See `_compute_fidelities_col_fn`."""
    results, _ = theano.scan(
        fn=_compute_fidelities_col_fn,
        sequences=T.arange(matrix.shape[1] // 2**num_ancillae),
        non_sequences=[row_idx, matrix, num_ancillae])
    return results



def _fidelity_with_ptrace(i, matrix, target_states, num_ancillae):
    """
    Compute fidelity between target and obtained states.

    This function is intended to be called in `theano.scan` from
    `QubitNetwork.fidelity`, and it operates with `theano` "symbolic"
    tensor objects. It *does not* operate with numbers.

    Parameters
    ----------
    i : int
        Denotes the element to take from `matrix`. This is necessary
        because of how `theano.scan` works (it is not possible to just
        pass the corresponding matrix[i] element to the function).
    matrix : theano 2d array
        An array of training states. `matrix[i]` is the `i`-th training
        state after evolution through `exp(-1j * H)`, *in big real ket
        form*.
        In other words, `matrix` is the set of states obtained after
        the evolution through the net, to be compared with the
        corresponding set of training target states.
        `matrix[i]` has length `2 * (2 ** num_qubits)`, with
    target_states : theano 2d array
        The set of target states. `target_states[i]` is the state that
        we would like `matrix[i]` to be equal to.
    num_ancillae : int
        The number of ancillae in the network.
    """
    # - `dm_real` and `dm_imag` will be square matrices of length
    #   `2 ** num_qubits`.
    dm_real, dm_imag = _ket_to_dm(matrix[i])
    # `dm_real_traced` and `dm_imag_traced` are square matrices
    # of length `2 ** num_system_qubits`.
    dm_real_traced, _ = theano.scan(
        fn=_compute_fidelities_row_fn,
        sequences=T.arange(dm_real.shape[0] // 2**num_ancillae),
        non_sequences=[dm_real, num_ancillae])
    dm_imag_traced, _ = theano.scan(
        fn=_compute_fidelities_row_fn,
        sequences=T.arange(dm_imag.shape[0] // 2**num_ancillae),
        non_sequences=[dm_imag, num_ancillae])

    #  ---- Old method to compute trace of product of dms: ----
    # target_dm_real, target_dm_imag = _ket_to_dm(target_states[i])

    # prod_real = (T.dot(dm_real_traced, target_dm_real) -
    #              T.dot(dm_imag_traced, target_dm_imag))
    # tr_real = T.nlinalg.trace(prod_real)

    # # we need only take the trace of the real part of the product,
    # # as if \rho and \rho' are two Hermitian matrices, then
    # # Tr(\rho_R \rho'_I) = Tr(\rho_I \rho'_R) = 0.
    # return tr_real

    # ---- New method: ----
    target_real, target_imag = _split_bigreal_ket(target_states[i])

    # `psi` and `psi_tilde` have length 2 * (2 ** numSystemQubits)
    psi = target_states[i][:, None]
    psi_tilde = T.concatenate((-target_imag, target_real))[:, None]
    # `big_dm` is a square matrix with same length
    big_dm = T.concatenate((
        T.concatenate((dm_imag_traced, dm_real_traced), axis=1),
        T.concatenate((-dm_real_traced, dm_imag_traced), axis=1)
    ), axis=0)
    out_fidelity = psi.T.dot(big_dm).dot(psi_tilde)
    return out_fidelity


def _fidelity_no_ptrace(i, states, target_states):
    """
    Compute symbolic fidelity between `states[i]` and `target_states[i]`.

    Inputs and outputs are supposed to be theano variables.
    Both `states[i]` and `target_states[i]` are real vectors of same
    length.
    """
    state = states[i]
    target_state = target_states[i]

    state_real, state_imag = _split_bigreal_ket(state)
    target_state_real, target_state_imag = _split_bigreal_ket(target_state)

    fidelity_real = (T.dot(state_real, target_state_real) +
                     T.dot(state_imag, target_state_imag))
    fidelity_imag = (T.dot(state_real, target_state_imag) -
                     T.dot(state_imag, target_state_real))
    fidelity = fidelity_real ** 2 + fidelity_imag ** 2
    return fidelity


class TheanoQstates:
    """
    Handle states manipulations when states are represented as theano objects.

    Parameters
    ----------
    states : theano.dmatrix
        A theano matrix, each row of which represents a ket state in big
        real notation.
    """
    def __init__(self, states):
        self.states = states
        self.states_form = 'kets'

    def evolve_all_kets(self, evolution_matrix):
        """Evolve all the states.

        Parameters
        ----------
        evolution_matrix : theano.dmatrix
            Theano object representing a (usually unitary) evolution
            matrix, in big real representation.
        Returns
        -------
        Nothing, changes `self.states` inplace.
        """
        if self.states_form == 'kets':
            self.states = T.tensordot(
                evolution_matrix, self.states, axes=([1], [1])).T
        else:
            raise NotImplementedError('To be implemented.')

    def _fidelities_with_ptrace(self, other_states, num_ancillae):
        """Compute fidelities in the case of ancillary qubits.

        This function handles the case of the fidelity when the output
        states are *larger* than the target states. In this case the
        fidelity is computed taking the partial trace with respect to
        the ancillary degrees of freedom of the output, and taking the
        fidelity of the resulting density matrix with the target
        (pure) state.
        """
        num_states = self.states.shape[0]
        fidelities, _ = theano.scan(
            fn=_fidelity_with_ptrace,
            sequences=T.arange(num_states),
            non_sequences=[self.states, other_states, num_ancillae]
        )
        return fidelities

    def _fidelities_no_ptrace(self, other_states):
        """Compute fidelities when there are no ancillary qubits.
        """
        num_states = self.states.shape[0]
        fidelities, _ = theano.scan(
            fn=_fidelity_no_ptrace,
            sequences=T.arange(num_states),
            non_sequences=[self.states, other_states]
        )
        return fidelities

    def fidelities(self, other_states, num_ancillae=0):
        """
        Compute fidelities between states and other states.

        Parameters
        ----------
        num_ancillae : int
            Number of qubit states to be traced out.
        """
        if num_ancillae > 0:
            return self._fidelities_with_ptrace(other_states, num_ancillae)
        else:
            return self._fidelities_no_ptrace(other_states)
