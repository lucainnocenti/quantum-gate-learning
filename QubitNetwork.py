import itertools
from collections import OrderedDict
import os
import qutip
import matplotlib.pyplot as plt
import scipy.linalg
import numpy as np
import theano
import theano.tensor as T
import theano.tensor.slinalg
import theano.tensor.nlinalg
from utils import chars2pair, complex2bigreal, bigreal2complex


class QubitNetwork:
    def __init__(self, num_qubits, system_qubits=None,
                 interactions='all', self_interactions='all',
                 ancillae_state=None,
                 net_topology=None,
                 J=None):
        # *self.num_qubits* is the TOTAL number of qubits in the
        # network, regardless of them being system or ancilla qubits
        self.num_qubits = num_qubits
        # Define which qubits belong to the system. The others are all
        # assumed to be ancilla qubits. If *system_qubits* was not
        # explicitly given it is assumed that half of the qubits are the
        # system and half are ancillae
        if system_qubits is None:
            self.system_qubits = tuple(range(num_qubits // 2))
        elif (isinstance(system_qubits, list) and
                all(qb < num_qubits for qb in system_qubits)):
            self.system_qubits = tuple(system_qubits)
        elif isinstance(system_qubits, int) and system_qubits <= num_qubits:
            self.system_qubits = tuple(range(system_qubits))
        else:
            raise ValueError('Invalid value for system_qubits.')

        # it will still be useful in the following to have direct access
        # to the number of ancilla and system qubits
        self.num_ancillae = self.num_qubits - len(self.system_qubits)
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

        # Build the initial state of the ancillae, if there are any
        if self.num_ancillae > 0:
            if ancillae_state is None:
                self.ancillae_state = self.build_ancilla_state()
            else:
                self.ancillae_state = ancillae_state

        # self.J is the set of parameters that we are going to train
        if J is None:
            if net_topology is None:
                self.J = theano.shared(
                    value=np.random.randn(
                        self.num_interactions + self.num_self_interactions),
                    name='J',
                    borrow=True
                )
            else:
                num_symbols = len(set(net_topology.values()))
                self.J = theano.shared(
                    value=np.random.randn(num_symbols),
                    name='J',
                    borrow=True
                )
        else:
            # if a `net_topology` has been given, check consistency
            if net_topology is not None:
                num_symbols = len(set(s for s in net_topology.values()))
                if np.asarray(J).shape[0] != num_symbols:
                    raise ValueError('The number of specified parameters does '
                                     'is not consistent with the value of `net'
                                     '_topology`.')
            self.J = theano.shared(
                value=np.asarray(J),
                name='J',
                borrow=True
            )

        self.net_topology = net_topology

    def decode_interactions(self, interactions):
        """Returns an OrderedDict with the requested interactions.

        The output of `decode_interactions` is passed to `self.active_Js`, and
        is meant to represent all the interactions that are switched on in the
        created qubit network.

        Parameters
        ----------
        interactions: used to specify the active interactions.
            - 'all': all possible pairwise interactions are active.
            - ('all', directions): for each pair of qubits the active
                interactions are all and only those specified in `directions`.
            - {pair1: dir1, pair2: dir2, ...}: explicitly specify all the
                active interactions.

        Returns
        -------
        An OrderedDict with the specified interactions formatted in a standard
        form.
        """
        if interactions == 'all':
            allsigmas = [item[0] + item[1]
                         for item in
                         itertools.product(['x', 'y', 'z'], repeat=2)]
            return OrderedDict([(pair, allsigmas) for pair in self.pairs])
        elif isinstance(interactions, tuple):
            if interactions[0] == 'all':
                d = [(pair, interactions[1]) for pair in self.pairs]
                return OrderedDict(d)
        # if `interactions` is dictionary-like, then return it back,
        # changing values that are strings (thus representing single
        # interactions corresponding to some qubit pair) converted to
        # lists with the single element equal to that string.
        elif isinstance(interactions, dict):
            parsed_interactions = OrderedDict()
            for k, v in interactions.items():
                if not isinstance(k, tuple):
                    raise ValueError('All keys must be tuples.')
                if isinstance(v, str):
                    parsed_interactions[k] = [v]
                else:
                    parsed_interactions[k] = v
            return parsed_interactions
        else:
            raise ValueError('Invalid value given for interactions.')

    def decode_self_interactions(self, self_interactions):
        if self_interactions == 'all':
            return OrderedDict(
                [(idx, ['x', 'y', 'z']) for idx in range(self.num_qubits)])
        elif isinstance(self_interactions, tuple):
            if self_interactions[0] == 'all':
                d = [(idx, self_interactions[1])
                     for idx in range(self.num_qubits)]
                return OrderedDict(d)
            else:
                raise ValueError('Invalid value for self_interactions.')
        elif (isinstance(self_interactions, dict) and
              all(isinstance(k, int) for k in self_interactions.keys())):
            return OrderedDict(self_interactions)
        else:
            raise ValueError('Invalid value of self_interactions.')

    def count_interactions(self):
        count = 0
        for k, v in self.active_Js.items():
            if isinstance(v, str):
                count += 1
            else:
                count += len(v)
        return count

    def count_self_interactions(self):
        count = 0
        for k, v in self.active_hs.items():
            if isinstance(v, str):
                count += 1
            else:
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

        # start by building pairwise interactions terms, filling
        # `Js_factors`. The order of the elements in the `Js_factors`
        # array is defined by the content of `self.active_Js`.
        for pair, directions in self.active_Js.items():
            # - *pair* is a pair of qubit indices, e.g. (0, 2)
            # - *directions* is a list of elements like ss below
            # - *ss* is a two-character string specifying an interaction
            # direction, e.g. 'xx' or 'xy' or 'zy'
            for ss in directions:
                term = terms_template[:]
                term[pair[0]] = sigmas[chars2pair(ss)[0]]
                term[pair[1]] = sigmas[chars2pair(ss)[1]]
                term = complex2bigreal(-1j * qutip.tensor(term).data.toarray())
                Js_factors.append(term)

        # print(terms_template)

        # proceed building self-interaction terms, filling hs_factors
        for qubit_idx, direction in self.active_hs.items():
            # - now direction is a list of characters among 'x', 'y' and 'z',
            # - s is either 'x', 'y', or 'z'
            if not isinstance(direction, list):
                raise TypeError('`direction` must be a list.')
            for s in direction:
                term = terms_template[:]
                term[qubit_idx] = sigmas[chars2pair(s)[0]]
                # print('qubit {}, dir {}, matrix:\n{}'.format(
                #     qubit_idx, s, qutip.tensor(term)))
                term = complex2bigreal(-1j * qutip.tensor(term).data.toarray())
                hs_factors.append(term)

        return np.asarray(Js_factors), np.asarray(hs_factors)

    def build_ancilla_state(self):
        """Returns an initial ancilla state, as a qutip.Qobj object.

        The generated state has every ancillary qubit in the up position.
        """
        state = qutip.tensor([qutip.basis(2, 0)
                              for _ in range(self.num_ancillae)])
        return state

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

        system_size = 2 ** self.num_system_qubits

        # generate a number `size` of normalized vectors, each one of
        # length `self.num_system_qubits`.
        training_states = [qutip.rand_ket(system_size) for _ in range(size)]
        qutip_dims = [[2 for _ in range(self.num_system_qubits)],
                      [1 for _ in range(self.num_system_qubits)]]
        for idx in range(len(training_states)):
            training_states[idx].dims = qutip_dims

        # evolve all training states
        if not isinstance(target_unitary, qutip.Qobj):
            raise TypeError('`target_unitary` should be a qutip object.')

        target_states = [target_unitary * psi for psi in training_states]

        # now compute the tensor product between every element of
        # `target_states` and the ancillae state, IF there are ancillae
        # over which we have to trace over after
        if self.num_ancillae > 0:
            training_states = [qutip.tensor(psi, self.ancillae_state)
                               for psi in training_states]

        # convert all the computed states in big real form
        training_states = [complex2bigreal(psi) for psi in training_states]
        target_states = [complex2bigreal(psi) for psi in target_states]

        return np.asarray(training_states), np.asarray(target_states)

    def save_to_file(self, outfile):
        import pickle
        data = {
            'num_qubits': self.num_qubits,
            'num_system_qubits': self.num_system_qubits,
            'active_hs': self.active_hs,
            'active_Js': self.active_Js,
            'J': self.J.get_value()
        }
        if not os.path.isabs(outfile):
            outfile = os.path.join(os.path.dirname(__file__), outfile)
        with open(outfile, 'wb') as file:
            pickle.dump(data, file)

    def tuple_to_xs_index(self, pair):
        if not isinstance(pair, tuple):
            raise TypeError('`pair` must be a tuple.')

        # if `pair` represents a self-interaction:
        if isinstance(pair[0], int):
            idx = 0
            found = False
            for qubit, dirs in self.active_hs.items():
                if qubit == pair[0]:
                    idx += dirs.index(pair[1])
                    return idx
                else:
                    idx += len(dirs)
            if not found:
                raise ValueError('The value of `pair` is invalid.')

        # otherwise it should represent a pairwise interaction:
        elif isinstance(pair[0], tuple) and len(pair[0]) == 2:
            idx = 0
            found = False
            for qubits, dirs in self.active_Js.items():
                if qubits == pair[0]:
                    idx += dirs.index(pair[1])
                    return idx
                else:
                    idx += len(dirs)
            if not found:
                raise ValueError('The value of `pair` is invalid.')
        # otherwise fuck it
        else:
            raise ValueError('The first element of `pair` should be an integer'
                             ' number representing a self-interaction, or a tu'
                             'ple of two integer numbers, representing a pairw'
                             'ise interaction')

    def tuple_to_xs_factor(self, pair):
        if not isinstance(pair, tuple):
            raise TypeError('`pair` must be a tuple.')

        # if `pair` represents a self-interaction:
        if isinstance(pair[0], int):
            idx = self.tuple_to_xs_index(pair)
            return self.hs_factors[idx]
        # otherwise it should represent a pairwise interaction:
        elif isinstance(pair[0], tuple) and len(pair[0]) == 2:
            idx = self.tuple_to_xs_index(pair)
            return self.Js_factors[idx]
        # otherwise fuck it
        else:
            raise ValueError('The first element of `pair` should be an integer'
                             ' number representing a self-interaction, or a tu'
                             'ple of two integer numbers, representing a pairw'
                             'ise interaction')

    def build_custom_H_factors(self):
        if self.net_topology is None:
            H_factors = np.concatenate(
                (self.hs_factors, self.Js_factors), axis=0)
            return T.tensordot(self.J, H_factors, axes=1)
        else:
            # the expected form of `self.net_topology` is a dictionary like
            # the following:
            # {
            #   ((1, 2), 'xx'): 'a',
            #   ((1, 3), 'xx'): 'a',
            #   ((2, 3), 'xx'): 'a',
            #   ((1, 2), 'xy'): 'b',
            # }
            symbols = []
            for symb in self.net_topology.values():
                if symb not in symbols:
                    symbols.append(str(symb))
            symbols.sort()

            factors = []
            for symb in symbols:
                factors.append(np.zeros_like(self.hs_factors[0]))
                for pair, label in self.net_topology.items():
                    if str(label) == symb:
                        factors[-1] += self.tuple_to_xs_factor(pair)
            return T.tensordot(self.J, factors, axes=1)

    def J_index_to_interaction(self, index):
        if self.net_topology is None:
            # if the index represents a self interaction...
            if 0 <= index < self.num_self_interactions:
                count = 0
                for qb, dirs in self.active_hs.items():
                    if index < count + len(dirs):
                        # index points to a dir of this qubit
                        return (qb, dirs[index - count])
                    else:
                        count += len(dirs)
            # if the index represents a pairwise interaction...
            elif index < self.num_self_interactions + self.num_interactions:
                index = index - self.num_self_interactions
                count = 0
                for pair, dirs in self.active_Js.items():
                    if index < count + len(dirs):
                        return (pair, dirs[index - count])
                    else:
                        count += len(dirs)
            else:
                raise ValueError('`index` has an invalid value.')
        else:
            raise NotImplementedError()

    def get_current_gate(self):
        """Returns the currently produced unitary, in complex form."""
        if self.net_topology is None:
            gate = np.concatenate((self.hs_factors, self.Js_factors), axis=0)
            gate = np.tensordot(self.J.get_value(), gate, axes=1)
            gate = scipy.linalg.expm(gate)
            gate = bigreal2complex(gate)
            return gate
        else:
            raise NotImplementedError()

    def fidelity_1s(self, state, target_state):
        # this builds the Hamiltonian of the system (in big real matrix
        # form), already multiplied with the 1j factor and ready for
        # exponentiation.
        H = self.build_custom_H_factors()
        # expH is the unitary evolution of the system
        expH = T.slinalg.expm(H)
        Uxpsi = T.dot(expH, state).reshape((state.shape[0], 1))
        Uxpsi_real = Uxpsi[:Uxpsi.shape[0] // 2]
        Uxpsi_imag = Uxpsi[Uxpsi.shape[0] // 2:]
        dm_real = Uxpsi_real * Uxpsi_real.T + Uxpsi_imag * Uxpsi_imag.T
        dm_imag = Uxpsi_imag * Uxpsi_real.T - Uxpsi_real * Uxpsi_imag.T

        # *col_fn* and *row_fn* are used inside *build_density_matrices*
        # to compute the partial traces
        def col_fn(col_idx, row_idx, matrix):
            subm_dim = 2 ** self.num_ancillae
            return T.nlinalg.trace(
                matrix[row_idx * subm_dim:(row_idx + 1) * subm_dim,
                       col_idx * subm_dim:(col_idx + 1) * subm_dim])

        def row_fn(row_idx, matrix):
            results, _ = theano.scan(
                fn=col_fn,
                sequences=T.arange(matrix.shape[1] // 2 ** self.num_ancillae),
                non_sequences=[row_idx, matrix]
            )
            return results

        dm_real_traced, _ = theano.scan(
            fn=row_fn,
            sequences=T.arange(dm_real.shape[0] // 2 ** self.num_ancillae),
            non_sequences=[dm_real]
        )
        dm_imag_traced, _ = theano.scan(
            fn=row_fn,
            sequences=T.arange(dm_imag.shape[0] // 2 ** self.num_ancillae),
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

        fid = T.dot(target_state, T.dot(dm_traced, target_state))

        return fid

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

        # this builds the Hamiltonian of the system (in big real matrix form),
        # already multiplied with the -1j factor and ready for exponentiation
        H = self.build_custom_H_factors()
        # expH is the unitary evolution of the system
        expH = T.slinalg.expm(H)

        # expH_times_state is the full output state given by the qubit
        # network.
        # `state` in general is a matrix (array of state vectors) so
        # that `expH_times_state` is also a matrix with a number of
        # rows equal to the number of training vectors. Every row
        # of this matrix is a state evolved according to the gate
        # implemented by the network with the current interactions
        # parameters.
        expH_times_state = T.tensordot(expH, states, axes=([1], [1])).T

        # `col_fn` and `row_fn` are used inside `build_density_matrices`
        # to compute the partial traces
        def col_fn(col_idx, row_idx, matrix):
            subm_dim = 2 ** self.num_ancillae
            return T.nlinalg.trace(
                matrix[row_idx * subm_dim:(row_idx + 1) * subm_dim,
                       col_idx * subm_dim:(col_idx + 1) * subm_dim])

        def row_fn(row_idx, matrix):
            results, _ = theano.scan(
                fn=col_fn,
                sequences=T.arange(matrix.shape[1] // 2 ** self.num_ancillae),
                non_sequences=[row_idx, matrix]
            )
            return results

        # `build_density_matrices` is to be called by the immediately
        # following theano.scan, and its output is given to `dm`.
        # Every call to `build_density_matrices` returns the density
        # matrix obtained after tracing out the ancillary degrees of
        # freedom. Overall `dm` will therefore be an array of such
        # density matrices.
        def compute_fidelities(i, matrix, target_states):
            # here matrix[i] is the i-th training state after evolution
            # through exp(-1j * H)
            Uxpsi = matrix[i].reshape((matrix[i].shape[0], 1))
            Uxpsi_real = Uxpsi[:Uxpsi.shape[0] // 2]
            Uxpsi_imag = Uxpsi[Uxpsi.shape[0] // 2:]
            dm_real = Uxpsi_real * Uxpsi_real.T + Uxpsi_imag * Uxpsi_imag.T
            dm_imag = Uxpsi_imag * Uxpsi_real.T - Uxpsi_real * Uxpsi_imag.T

            dm_real_traced, _ = theano.scan(
                fn=row_fn,
                sequences=T.arange(dm_real.shape[0] // 2 ** self.num_ancillae),
                non_sequences=[dm_real]
            )
            dm_imag_traced, _ = theano.scan(
                fn=row_fn,
                sequences=T.arange(dm_imag.shape[0] // 2 ** self.num_ancillae),
                non_sequences=[dm_imag]
            )

            target_state = target_states[i]
            target_state_real = target_state[:target_state.shape[0] // 2, None]
            target_state_imag = target_state[target_state.shape[0] // 2:, None]
            target_dm_real = (target_state_real * target_state_real.T +
                              target_state_imag * target_state_imag.T)
            target_dm_imag = (target_state_imag * target_state_real.T -
                              target_state_real * target_state_imag.T)

            prod_real = (T.dot(dm_real_traced, target_dm_real) -
                         T.dot(dm_imag_traced, target_dm_imag))
            tr_real = T.nlinalg.trace(prod_real)

            # prod_imag = (T.dot(dm_real_traced, target_dm_imag) +
            #              T.dot(dm_imag_traced, target_dm_real))
            # tr_imag = T.nlinalg.trace(prod_imag)

            # tr_abs = T.sqrt(tr_real ** 2 + tr_imag ** 2)

            # guess we should show why this is correct?
            return tr_real

        def fidelities_no_ptrace(i, states, target_states):
            state = states[i]
            target_state = target_states[i]
            state_real = state[:state.shape[0] // 2]
            state_imag = state[state.shape[0] // 2:]
            target_state_real = target_state[:target_state.shape[0] // 2]
            target_state_imag = target_state[target_state.shape[0] // 2:]

            fidelity_real = (T.dot(state_real, target_state_real) +
                             T.dot(state_imag, target_state_imag))
            fidelity_imag = (T.dot(state_real, target_state_imag) -
                             T.dot(state_imag, target_state_real))
            fidelity = fidelity_real ** 2 + fidelity_imag ** 2
            return fidelity

        # the function also supports the case in which there are no
        # ancillae over which to trace over.
        if self.num_ancillae == 0:
            fidelities, _ = theano.scan(
                fn=fidelities_no_ptrace,
                sequences=T.arange(expH_times_state.shape[0]),
                non_sequences=[expH_times_state, target_states]
            )
        else:
            fidelities, _ = theano.scan(
                fn=compute_fidelities,
                sequences=T.arange(expH_times_state.shape[0]),
                non_sequences=[expH_times_state, target_states]
            )

        # return fidelities
        return T.mean(fidelities)


def load_network_from_file(infile):
    """Returns a QubitNetwork object created from the file `infile`.

    The QubitNetwork objects should have been stored into the file in
    pickle format, using the appropriate `save_to_file` method.
    """
    import pickle
    with open(infile, 'rb') as file:
        data = pickle.load(file)
    net = QubitNetwork(
        num_qubits=data['num_qubits'],
        interactions=data['active_Js'],
        self_interactions=data['active_hs'],
        system_qubits=data['num_system_qubits'],
        J=data['J']
    )
    return net


def test_compiled_dm(net, state):
    state_for_net = theano.shared(complex2bigreal(state))
    cost = net.test_compiled_dm(state_for_net)

    dm = theano.function(
        inputs=[],
        outputs=cost
    )

    return dm()


def sgd_optimization(net=None, learning_rate=0.13, n_epochs=100,
                     batch_size=100,
                     backup_file=None,
                     saveafter_file=None,
                     training_dataset_size=1000,
                     test_dataset_size=1000,
                     target_gate=None,
                     decay_rate=0.1,
                     precompiled_functions=None,
                     print_fidelity=False):

    # parse the `net` parameter
    if net is None:
        _net = QubitNetwork(num_qubits=4,
                            interactions=('all', ['xx', 'yy', 'zz']),
                            self_interactions=('all', ['x', 'y', 'z']),
                            system_qubits=[0, 1, 2])
    elif type(net) == QubitNetwork:
        # everything fine, move along
        _net = net
    elif isinstance(net, str):
        # assume `net` is the path where the network was stored
        _net = load_network_from_file(net)
    else:
        raise ValueError('Invalid value for the argument `net`.')

    # parse `target_gate` parameter
    if target_gate is None:
        raise ValueError('`target_gate` must have a value.')

    # parse `backup_file` parameter
    if isinstance(backup_file, str):
        # we will assume that it is the path where to backup the net
        # BEFORE the training takes place, in case anything bad happens
        _net.save_to_file(backup_file)
        print('Network backup saved in {}'.format(backup_file))

    # definition of utility functions for later on
    def conditionally_save():
        if isinstance(net, str):
            _net.save_to_file(net)
            print('Network saved in {}'.format(net))
        elif saveafter_file is not None:
            _net.save_to_file(saveafter_file)
            print('Network saved in {}'.format(saveafter_file))

    print('Generating training data...')

    dataset = _net.generate_training_data(target_gate, training_dataset_size)
    states = theano.shared(
        np.asarray(dataset[0], dtype=theano.config.floatX)
    )
    target_states = theano.shared(
        np.asarray(dataset[1], dtype=theano.config.floatX)
    )

    test_dataset = _net.generate_training_data(target_gate, test_dataset_size)
    test_states = theano.shared(
        np.asarray(test_dataset[0], dtype=theano.config.floatX)
    )
    test_target_states = theano.shared(
        np.asarray(test_dataset[1], dtype=theano.config.floatX)
    )

    print('Building the model...')
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a minibatch

    # generate symbolic variables for input data and labels
    x = T.dmatrix('x')  # input state (data). Every row is a state vector
    y = T.dmatrix('y')  # output target state (label). As above

    _learning_rate = theano.shared(
        np.asarray(learning_rate, dtype=theano.config.floatX),
        name='learning_rate'
    )

    # define the cost function, that is, the fidelity. This is the
    # number we ought to maximize through the training.
    cost = _net.fidelity(x, y)

    # compute the gradient of the cost
    g_J = T.grad(cost=cost, wrt=_net.J)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = [(_net.J, _net.J + _learning_rate * g_J)]

    if precompiled_functions is None:
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

        test_model = theano.function(
            inputs=[],
            outputs=cost,
            updates=None,
            givens={
                x: test_states,
                y: test_target_states
            }
        )
    else:
        train_model, test_model = precompiled_functions

    print('Let\'s roll!')
    n_train_batches = states.get_value().shape[0] // batch_size
    fids_history = []
    fig, ax = plt.subplots(1, 1)

    try:
        for n_epoch in range(n_epochs):
            if print_fidelity:
                print('Epoch {}, '.format(n_epoch), end='')

            # compute fidelity and update parameters
            for minibatch_index in range(n_train_batches):
                minibatch_avg_cost = train_model(minibatch_index)

            # update fidelity history
            fids_history.append(test_model())
            if print_fidelity:
                print(fids_history[-1])

            # update plot
            ax.plot(fids_history, '-b')
            plt.suptitle('learning rate: {}\nfidelity: {}'.format(
                _learning_rate.get_value(), fids_history[-1]))
            fig.canvas.draw()

            # update learning rate
            _learning_rate.set_value(
                learning_rate / (1 + decay_rate * n_epoch))

            # generate a new set of training states
            dataset = _net.generate_training_data(
                target_gate, training_dataset_size)
            states.set_value(dataset[0])
            target_states.set_value(dataset[1])
    except KeyboardInterrupt:
        pass

    print('Finished training')

    # save results if appropriate parameters have been given
    conditionally_save()

    if precompiled_functions is None:
        return _net
    else:
        return _net, (train_model, test_model)
