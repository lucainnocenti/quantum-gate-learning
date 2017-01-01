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
        return state.reshape(state.shape[0])

    def generate_training_data(self, target_unitary, size):
        """Generates a set of training data for the QubitNetwork net.

        Returns
        -------
        A tuple with two elements: training vectors and labels.
        NOTE: The training and target vectors have different lengths!
              The latter span the whole space while the former only the
              system one.

        training_states: an array of vectors. Each vector represents a
                         state in the full system+ancilla space, in big
                         real form.
        target_states: an array of vectors. Each vector represents a
                       state in only the system space, in big real form.
                       Every such state is generated by evolving a
                       corresponding training_state through the matrix
                       target_unitary.
        """
        vectors_size = 2 * 2 ** self.num_system_qubits
        # generate a number *size* of normalized vectors, each one of
        # length *vectors_size*.
        training_states = np.random.randn(size, vectors_size)
        norms = np.sqrt(
            np.einsum('ij,ij->i', training_states, training_states))
        training_states /= norms[:, np.newaxis]

        # make sure that *target_unitary* is a numpy array
        target_unitary = np.asarray(target_unitary)
        # make sure that *target_unitary* has the correct size
        if not target_unitary.shape[0] == vectors_size:
            raise ValueError('Wrong dimensions for target_unitary.')
        target_states = np.dot(training_states, target_unitary)

        # now compute the tensor product between every element of
        # *target_states* and the ancilla state. Note that this is not
        # done in the usual way due to the use of the big real form.

        # ---------------------- WARNING ---------------------------

        # THE FOLLOWING COMPUTATION ASSUMES A REAL VECTOR FOR THE ANCILLA
        # STATE, AND A SINGLE ANCILLA QUBIT

        # ---------------------- WARNING ---------------------------
        training_states = training_states.reshape(
            training_states.shape[0],
            2,
            training_states.shape[1] // 2
        )
        training_states = np.einsum('kij,l->kilj',
                                    training_states,
                                    self.ancillas_state[:2])
        training_states = training_states.reshape(
            training_states.shape[0], 2 * vectors_size
        )

        return training_states, target_states

    def fidelity(self, states, target_states):
        """This is the cost function of the model.

        Parameters
        ----------
        states : This function computes the fidelity between the states
                obtained evolving the elements of *states* through the
                network with parameters J, traced over the ancilla
                degrees of freedom, and the elements of *target_states*.
                Note that here *states* is a vector whose elements
                represent a state over the whole system+ancilla
                space, in big real form.
        target_states: The labels corresponding to the training data
                      *states*. Note that *target_states* is a an array
                      of vectors representing a system-only state in
                      real big form.

        Returns
        -------
        A theano function, to be used for the MSGD algorithm
        """

        # self.hs_factors and self.Js_factors are already in big real
        # matrix form, and already multiplied by 1j
        H_factors = np.concatenate((self.hs_factors, self.Js_factors), axis=0)
        # this builds the Hamiltonian of the system (in big real matrix form),
        # already multiplied with the 1j factor and ready for exponentiation
        H = T.tensordot(self.J, H_factors, axes=1)
        # expH is the unitary evolution of the system
        expH = T.slinalg.expm(H)

        # expH_times_state is the full output state given by the qubit
        # network.
        # *state* in general is a matrix (array of state vectors) so
        # that *expH_times_state* is also a matrix with a number of
        # rows equal to the number of training vectors.
        expH_times_state = T.tensordot(states, expH, axes=([1], [0]))

        # *col_fn* and *row_fn* are used inside *build_density_matrices*
        # to compute the partial traces
        def col_fn(col_idx, row_idx, matrix):
            subm_dim = 2 ** self.num_ancillas
            return T.nlinalg.trace(
                matrix[row_idx * subm_dim:(row_idx + 1) * subm_dim,
                       col_idx * subm_dim:(col_idx + 1) * subm_dim])

        def row_fn(row_idx, matrix):
            results, _ = theano.scan(
                fn=col_fn,
                sequences=T.arange(matrix.shape[1] // 2 ** self.num_ancillas),
                non_sequences=[row_idx, matrix]
            )
            return results

        # *build_density_matrices* is to be called by the immediately
        # following theano.scan, and its output is given to *dm*.
        # Every call to *build_density_matrices* returns the density
        # matrix obtained after tracing out the ancillary degrees of
        # freedom. Overall *dm* will therefore be an array of such
        # density matrices.
        def compute_fidelities(i, matrix):
            dm = T.dot(
                matrix[i].reshape((matrix[i].shape[0], 1)),
                matrix[i].reshape((1, matrix[i].shape[0]))
            )
            dm_real = dm[0:dm.shape[0] // 2, 0:dm.shape[1] // 2]
            dm_imag = dm[0:dm.shape[0] // 2, dm.shape[1] // 2:]
            dm_real_traced, _ = theano.scan(
                fn=row_fn,
                sequences=T.arange(dm_real.shape[0] // 2 ** self.num_ancillas),
                non_sequences=[dm_real]
            )
            dm_imag_traced, _ = theano.scan(
                fn=row_fn,
                sequences=T.arange(dm_imag.shape[0] // 2 ** self.num_ancillas),
                non_sequences=[dm_imag]
            )
            dm_traced_r1 = T.concatenate(
                (dm_real_traced, -dm_imag_traced),
                axis=1
            )
            dm_traced_r2 = T.concatenate(
                (dm_imag_traced, dm_real_traced),
                axis=1
            )
            dm_traced = T.concatenate((dm_traced_r1, dm_traced_r2), axis=0)
            return T.dot(target_states.T, T.dot(dm_traced, target_states))

        fidelities, _ = theano.scan(
            fn=compute_fidelities,
            sequences=T.arange(expH_times_state.shape[0]),
            non_sequences=expH_times_state
        )
        return T.mean(fidelities)

        # target_u = complex2bigreal(qutip.qip.fredkin().data.toarray())
        # target_state = T.dot(target_unitary, state)
        # fidelity = target_evolved_state.T.dot(target_u.dot(
        # target_evolved_state))

        # we want to implement with theano ops the equivalent of the
        # following numpy.einsum call:
        # np.einsum('ij,ijk,ik->i', target_states, dm, target_states)


def sgd_optimization(learning_rate=0.13, n_epochs=100,
                     batch_size=100):

    print('Building the model...')

    net = QubitNetwork(4, interactions=('all', ['zz']),
                                    self_interactions=('all', ['x', 'y']),
                                    system_qubits=[0, 1, 2])

    # Generate training dataset. In this case the target unitary is
    # fixed to be a Fredkin gate.

    fredkin_gate = qutip.qip.fredkin().data.toarray()
    fredkin_gate = complex2bigreal(fredkin_gate)
    dataset = net.generate_training_data(fredkin_gate, 1000)
    states = theano.shared(
        np.asarray(dataset[0], dtype=theano.config.floatX)
    )
    target_states = theano.shared(
        np.asarray(dataset[1], dtype=theano.config.floatX)
    )

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a minibatch

    # generate symbolic variables for input data and labels
    x = T.dmatrix('x')  # input state (data). Every row is a state vector
    y = T.dmatrix('y')  # output target state (label). As above
    # define the cost function, that is, the fidelity. This is the
    # number we ought to maximize through the training.
    cost = net.fidelity(x, y)

    # compute the gradient of the cost
    g_J = T.grad(cost=cost, wrt=net.J)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = [(net.J, net.J - learning_rate * g_J)]

    # compile the training function `train_model`, that while computing
    # the cost at every iteration (batch), also updates the weights of
    # the network based on the rules defined in `updates`.
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: states[index * batch_size: (index + 1) * batch_size],
            y: target_states[index * batch_size: (index + 1) * batch_size]
        }
    )
