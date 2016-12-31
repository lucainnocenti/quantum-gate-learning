import itertools
from collections import OrderedDict
import qutip
import numpy as np
import theano
import theano.tensor as T
import theano.tensor.slinalg
import theano.tensor.nlinalg


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


def generate_training_data(net, target_unitary, size):
    """Generates a set of training data for the QubitNetwork net."""
    if type(net) != QubitNetwork:
        raise ValueError('net must be an instance of the class QubitNetwork.')
    vectors_size = 2 * 2 ** net.num_system_qubits
    training_states = np.random.randn(size, vectors_size)
    norms = np.sqrt(np.einsum('ij,ij->i', training_states, training_states))
    training_states /= norms[:, np.newaxis]

    # make sure that target_unitary is a numpy array
    target_unitary = np.asarray(target_unitary)
    # make sure that target_unitary has the correct size
    if not target_unitary.shape[0] == vectors_size:
        raise ValueError('Wrong dimensions for target_unitary.')

    return training_states, np.dot(training_states, target_unitary)


def fidelity(net, J, state, target_state):
    """This is the cost function of the model.

    Parameters
    ----------
    net : A QubitNetwork object, containing the necessary data about
          the qubit network to trin (dimensions, used interactions and so on).
    J : A theano.shared variable for the the parameters of the network to
        train using MSGD.
    state : This function computes the fidelity between the state obtained
            evolving *state* through the network with parameters J, traced
            over the ancilla degrees of freedom, and target_state.
    target_state : The label corresponding to the training datum *state*

    Returns
    -------
    A theano function, to be used for a MSGD algorithm
    """

    # net.hs_factors and net.Js_factors are already in big real matrix form,
    # and already multiplied by 1j
    H_factors = np.concatenate((net.hs_factors, net.Js_factors), axis=0)
    # this builds the Hamiltonian of the system (in big real matrix form),
    # already multiplied with the 1j factor and ready for exponentiation
    H = T.tensordot(J, H_factors, axes=1)
    # expH is the unitary evolution of the system
    expH = T.slinalg.expm(H)

    # net.initial_state is the initial state stored as a vector (ket),
    # multiplying it on the left by expH amounts to evolve it
    expH_times_state = T.dot(expH, state)
    # build the density matrix out of the evolved state
    dm = expH_times_state * expH_times_state.T
    dm_real = dm[0:dm.shape[0] // 2, 0:dm.shape[1] // 2]
    dm_imag = dm[0:dm.shape[0] // 2, dm.shape[1] // 2:]
    # partial trace of the density matrix

    def col_fn(col_idx, row_idx, matrix):
        subm_dim = 2 ** net.num_ancillas
        return T.nlinalg.trace(
            matrix[row_idx * subm_dim:(row_idx + 1) * subm_dim,
                   col_idx * subm_dim:(col_idx + 1) * subm_dim])

    def row_fn(row_idx, matrix):
        results, _ = theano.scan(
            fn=col_fn,
            sequences=T.arange(matrix.shape[1] // 2 ** net.num_ancillas),
            non_sequences=[row_idx, matrix]
        )
        return results
    dm_real_traced, _ = theano.scan(
        fn=row_fn,
        sequences=T.arange(dm_real.shape[0] // 2 ** net.num_ancillas),
        non_sequences=[dm_real]
    )
    dm_imag_traced, _ = theano.scan(
        fn=row_fn,
        sequences=T.arange(dm_imag.shape[0] // 2 ** net.num_ancillas),
        non_sequences=[dm_imag]
    )
    dm_traced_r1 = T.concatenate((dm_real_traced, -dm_imag_traced), axis=1)
    dm_traced_r2 = T.concatenate((dm_imag_traced, dm_real_traced), axis=1)
    dm_traced = T.concatenate((dm_traced_r1, dm_traced_r2), axis=0)

    # target_u = complex2bigreal(qutip.qip.fredkin().data.toarray())
    # target_state = T.dot(target_unitary, state)
    # fidelity = target_evolved_state.T.dot(target_u.dot(target_evolved_state))
    return T.dot(target_state.T, T.dot(dm_traced, target_state))


class QubitNetwork:
    def __init__(self, num_qubits,
                 interactions='all', self_interactions='all',
                 system_qubits=None):
        # *self.num_qubits* is the TOTAL number of qubits in the network,
        # regardless of them being system or ancilla qubits
        self.num_qubits = num_qubits
        # Define which qubits belong to the system. The others are all
        # assumed to be ancilla qubits. If *system_qubits* was not explicitly
        # given it is assumed that half of the qubits are the system and half
        # are ancillas
        if system_qubits is None:
            self.system_qubits = tuple(range(num_qubits // 2))
        elif np.all(np.asarray(system_qubits) < num_qubits):
            self.system_qubits = tuple(system_qubits)
        else:
            raise ValueError('Invalid value for system_qubits.')
        # it will still be useful in the following to have direct access
        # to the number of ancilla and system qubits
        self.num_ancillas = self.num_qubits - len(self.system_qubits)
        self.num_system_qubits = len(self.system_qubits)

        # we store all the possible pairs for convenience
        self.pairs = list(itertools.combinations(range(self.num_qubits), 2))
        # decode_interactions_dict fills the self.active_Js variable
        self.active_Js = self.decode_interactions(interactions)
        self.active_hs = self.decode_self_interactions(self_interactions)

        self.num_interactions = self.count_interactions()
        self.num_self_interactions = self.count_self_interactions()

        # Js_factors and hs_factors store the matrices, in big real form,
        # that will have to multiplied by the *Js* and *hs* factors
        self.Js_factors, self.hs_factors = self.build_H_components()

        # the initial state is here build in big real form, with all
        # the qubits initialized in the up position
        # (self.initial_state,
        #  self.initial_system_state) = self.build_initial_state_vector()
        self.ancillas_state = self.build_ancilla_state()

        # self.J is the set of parameters that we want to train
        self.J = theano.shared(
            value=np.zeros(
                self.num_interactions + self.num_self_interactions),
            name='J',
            borrow=True
        )

    def decode_interactions(self, interactions):
        """Returns an OrderedDict with the requested interactions.

        The output of decode_interactions is passed to self.active_Js,
        and is meant to represent all the interactions that are switched
        on in the created qubit network.
        """
        if interactions == 'all':
            allsigmas = [item[0] + item[1]
                         for item in
                         itertools.product(['x', 'y', 'z'], repeat=2)]
            return OrderedDict([(pair, allsigmas) for pair in self.pairs])
        elif isinstance(interactions, tuple):
            if interactions[0] == 'all':
                d = {pair: interactions[1] for pair in self.pairs}
                return OrderedDict(d)
        elif (isinstance(interactions, dict) and
              all(isinstance(k, tuple) for k in interactions.keys())):
            return OrderedDict(interactions)
        else:
            raise ValueError('Invalid value given for interactions.')

    def decode_self_interactions(self, self_interactions):
        if self_interactions == 'all':
            return OrderedDict(
                {idx: ['x', 'y', 'z'] for idx in range(self.num_qubits)})
        elif isinstance(self_interactions, tuple):
            if self_interactions[0] == 'all':
                d = {idx: self_interactions[1]
                     for idx in range(self.num_qubits)}
                return OrderedDict(d)
            else:
                raise ValueError('Invalid value for self_interactions.')
        else:
            raise ValueError('Invalid value of self_interactions.')

    def count_interactions(self):
        count = 0
        for k, v in self.active_Js.items():
            count += len(v)
        return count

    def count_self_interactions(self):
        count = 0
        for k, v in self.active_hs.items():
            count += len(v)
        return count

    def build_H_components(self):
        """Builds the list of factors to be multiplied by the parameters.

        Every element in the output numpy array is the factor to which a
        corresponding network parameters will have to be multiplied.
        More specifically, a 2-element tuple is returned, the first
        element of which containing the pairwise interactions term and the
        second element of which containing the self-interaction terms.

        All the terms are already multiplied by the imaginary unit 1j and
        converted into big real form with complex2bigreal, so that all
        that remains after is to multiply by the parameters and the matrix
        exponential, with something like:

        >>> J = T.dvector('J')
        >>> H = T.tensordot(J, terms, axes=1)
        >>> expH = T.slinalg.expm(H)

        where terms is the output of this function (mind the tuple though).
        """
        terms_template = [qutip.qeye(2) for _ in range(self.num_qubits)]
        Js_factors = []
        hs_factors = []

        sigmas = [qutip.qeye(2),
                  qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]
        # start by building pairwise interactions terms, filling Js_factors
        for pair, directions in self.active_Js.items():
            # - *pair* is a pair of qubit indices, e.g. (0, 2)
            # - *directions* is a list of elements like ss below
            # - *ss* is a two-character string specifying an interaction
            # direction, e.g. 'xx' or 'xy' or 'zy'
            for ss in directions:
                term = terms_template
                term[pair[0]] = sigmas[chars2pair(ss)[0]]
                term[pair[1]] = sigmas[chars2pair(ss)[1]]
                term = complex2bigreal(1j * qutip.tensor(term).data.toarray())
                Js_factors.append(term)

        # proceed building self-interaction terms, filling hs_factors
        for qubit_idx, direction in self.active_hs.items():
            # - now direction is a list of characters among 'x', 'y' and 'z',
            # - s is either 'x', 'y', or 'z'
            for s in direction:
                term = terms_template
                term[qubit_idx] = sigmas[chars2pair(s)[0]]
                term = complex2bigreal(1j * qutip.tensor(term).data.toarray())
                hs_factors.append(term)

        return np.asarray(Js_factors), np.asarray(hs_factors)

    def build_initial_state_vector(self):
        """Probably DEPRECATED."""
        state = qutip.tensor([qutip.basis(2, 0)
                              for _ in range(self.num_qubits)])
        state = state.data.toarray()
        state = np.concatenate((np.real(state), np.imag(state)), axis=0)
        system_state = qutip.tensor([qutip.basis(2, 0)
                                     for _ in range(len(self.system_qubits))])
        system_state = system_state.data.toarray()
        system_state = np.concatenate(
            (np.real(system_state), np.imag(system_state)), axis=0)
        return state, system_state

    def build_ancilla_state(self):
        """Returns an initial ancilla state.

        The generated state has every ancillary qubit in the up position.
        """
        state = qutip.tensor([qutip.basis(2, 0)
                              for _ in range(self.num_ancillas)])
        state = state.data.toarray()
        state = np.concatenate((np.real(state), np.imag(state)), axis=0)
        return state
