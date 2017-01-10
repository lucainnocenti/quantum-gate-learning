import itertools
from collections import OrderedDict
import os
import qutip
# import matplotlib.pyplot as plt
import scipy.linalg
import numpy as np
import theano
import theano.tensor as T
import theano.tensor.slinalg  # for expm()
import theano.tensor.nlinalg  # for trace()
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

        if net_topology is None:
            self.net_topology = None
        else:
            self.net_topology = OrderedDict(net_topology)
        # `parse_interactions` fills the `self.interactions`,
        # `self.num_interactions` and `self.num_self_interactions`
        # variables
        self.parse_interactions(interactions)

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
                    value=np.random.randn(self.num_interactions),
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
                    raise ValueError('The number of specified parameters is '
                                     'not consistent with the value of `net'
                                     '_topology`.')
            self.J = theano.shared(
                value=np.asarray(J),
                name='J',
                borrow=True
            )

    def parse_interactions(self, interactions):
        """Sets the value of `self.interactions`. Returns None.

        The input-given value of `interactions` is parsed, and the value
        of `self.interactions` is accordingly set to describe the whole
        set of (self-)interactions in the network.

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
        None
        """
        outints = []

        if self.net_topology is not None:
            # the expected form of `self.net_topology` is a dictionary like
            # the following:
            # {
            #   ((1, 2), 'xx'): 'a',
            #   ((1, 3), 'xx'): 'a',
            #   ((2, 3), 'xx'): 'a',
            #   ((1, 2), 'xy'): 'b',
            # }
            outints = list(self.net_topology.keys())
        elif interactions == 'all':
            # create all self-interaction terms
            for qubit in range(self.num_qubits):
                for s in ['x', 'y', 'z']:
                    outints.append((qubit, s))
            # create all interactions terms
            for pair in self.pairs:
                for s1 in ['x', 'y', 'z']:
                    for s2 in ['x', 'y', 'z']:
                        outints.append((pair, s1 + s2))

        elif isinstance(interactions, tuple):
            if interactions[0] == 'all':
                # here we need to first iterate over the interaction
                # types because they can be either self- or pairwise
                # interactions, and depending on this they must be
                # associated to single or pairs of qubits, respectively.
                for d in interactions[1]:
                    if len(d) == 1:
                        for qubit in range(self.num_qubits):
                            outints.append((qubit, d))
                    elif len(d) == 2:
                        for pair in self.pairs:
                            outints.append((pair, d))
        elif isinstance(interactions, list):
            outints = interactions
        else:
            raise ValueError(
                'Invalid value given for interactions.',
                interactions)

        num_self_interactions = 0
        for q, d in outints:
            if len(d) == 1:
                num_self_interactions += 1

        self.interactions = outints
        self.num_interactions = len(outints)
        self.num_self_interactions = num_self_interactions

    # def decode_self_interactions(self, self_interactions):
    #     """OBSOLETE FUNCTION"""
    #     raise DeprecationWarning()

    #     if self_interactions == 'all':
    #         return OrderedDict(
    #             [(idx, ['x', 'y', 'z']) for idx in range(self.num_qubits)])
    #     elif isinstance(self_interactions, tuple):
    #         if self_interactions[0] == 'all':
    #             d = [(idx, self_interactions[1])
    #                  for idx in range(self.num_qubits)]
    #             return OrderedDict(d)
    #         else:
    #             raise ValueError('Invalid value for self_interactions.')
    #     elif (isinstance(self_interactions, dict) and
    #           all(isinstance(k, int) for k in self_interactions.keys())):
    #         return OrderedDict(self_interactions)
    #     else:
    #         raise ValueError('Invalid value of self_interactions.')

    # def count_interactions(self):
    #     count = 0
    #     for k, v in self.active_Js.items():
    #         if isinstance(v, str):
    #             count += 1
    #         else:
    #             count += len(v)
    #     return count

    # def count_self_interactions(self):
    #     count = 0
    #     for k, v in self.interactions:
    #         if len(v) == 1:
    #             count += 1
    #     return count

    def build_H_factor(self, pair):
        term = [qutip.qeye(2) for _ in range(self.num_qubits)]

        sigmas = [qutip.qeye(2),
                  qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]

        target, d = pair
        # if `d` indicates a self-interaction..
        if len(d) == 1:
            term[target] = sigmas[chars2pair(d)[0]]
        # if `d` indicates a pairwise interaction..
        elif len(d) == 2:
            term[target[0]] = sigmas[chars2pair(d)[0]]
            term[target[1]] = sigmas[chars2pair(d)[1]]

        return complex2bigreal(-1j * qutip.tensor(term).data.toarray())

    # def build_H_components(self):
    #     """Builds the list of factors to be multiplied by the parameters.

    #     Every element in the output numpy array is the factor to which a
    #     corresponding network parameters will have to be multiplied.
    #     More specifically, a 2-element tuple is returned, the first
    #     element of which containing the pairwise interactions term and the
    #     second element of which containing the self-interaction terms.

    #     All the terms are already multiplied by the imaginary unit 1j and
    #     converted into big real form with complex2bigreal, so that all
    #     that remains after is to multiply by the parameters and the matrix
    #     exponential, with something like:

    #     >>> J = T.dvector('J')
    #     >>> H = T.tensordot(J, terms, axes=1)
    #     >>> expH = T.slinalg.expm(H)

    #     where terms is the output of this function (mind the tuple though).
    #     """
    #     terms_template = [qutip.qeye(2) for _ in range(self.num_qubits)]
    #     factors = []
    #     # Js_factors = []
    #     # hs_factors = []

    #     sigmas = [qutip.qeye(2),
    #               qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]

    #     for target, d in self.interactions:
    #         term = terms_template[:]
    #         # if `d` indicates a self-interaction..
    #         if len(d) == 1:
    #             term[target] = sigmas[chars2pair(d)[0]]
    #         # if `d` indicates a pairwise interaction..
    #         elif len(d) == 2:
    #             term[target[0]] = sigmas[chars2pair(d)[0]]
    #             term[target[1]] = sigmas[chars2pair(d)[1]]

    #         term = complex2bigreal(-1j * qutip.tensor(term).data.toarray())
    #         factors.append(term)

    #     return np.asarray(factors)

    def build_H_factors(self, symbolic_result=True):
        dim_real_space = 2 * 2 ** self.num_qubits
        if self.net_topology is None:

            factors = np.zeros(
                (self.num_interactions, dim_real_space, dim_real_space),
                dtype=np.float64
            )
            for idx, pair in enumerate(self.interactions):
                factors[idx] = self.build_H_factor(pair)

            if symbolic_result:
                # return the dot product between `self.J` and `factors`,
                # amounting to the sum over `i` of `self.J[i] * factors[i]`
                return T.tensordot(self.J, factors, axes=1)
            else:
                return np.tensordot(self.J.get_value(), factors, axes=1)
        else:
            # the expected form of `self.net_topology` is a dictionary like
            # the following:
            # {
            #   ((1, 2), 'xx'): 'a',
            #   ((1, 3), 'xx'): 'a',
            #   ((2, 3), 'xx'): 'a',
            #   ((1, 2), 'xy'): 'b',
            # }

            # symbols = []
            # for symb in self.net_topology.values():
            #     if symb not in symbols:
            #         symbols.append(str(symb))
            # symbols.sort()
            symbols = sorted(set(self.net_topology.values()))

            factors = np.zeros(
                (self.num_interactions, dim_real_space, dim_real_space),
                dtype=np.float64
            )

            # The number of elements in `symbols` should be equal to
            # `self.num_interactions`, computed by `parse_interactions`.
            # Note that is the code below that determines to what
            # interaction does the i-th element of `self.J` correspond
            # to.
            # The i-th element of `self.J` will correspond to the
            # interactions terms associated to the i-th symbol listed
            # in `symbols` (after sorting).
            for idx, symb in enumerate(symbols):
                for pair, label in self.net_topology.items():
                    if str(label) == symb:
                        factors[idx] += self.build_H_factor(pair)

            if symbolic_result:
                return T.tensordot(self.J, factors, axes=1)
            else:
                return np.tensordot(self.J.get_value(), factors, axes=1)

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
            'interactions': self.interactions,
            'J': self.J.get_value()
        }
        if not os.path.isabs(outfile):
            outfile = os.path.join(os.path.dirname(__file__), outfile)
        with open(outfile, 'wb') as file:
            pickle.dump(data, file)

    def save_gate_to_file(self, outfile):
        np.savetxt(outfile, self.get_current_gate(), delimiter=',')

    def tuple_to_interaction_index(self, pair):
        self.interactions.index(pair)

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

    def tuple_to_J_index(self, pair):
        if self.net_topology is None:
            # if `pair` is a self-interaction
            if isinstance(pair[0], int):
                return self.tuple_to_xs_index(pair)
            # else we assume an interaction term
            elif len(pair[0]) == 2:
                return (self.num_self_interactions +
                        self.tuple_to_xs_index(pair))
            else:
                raise ValueError('Invalid value for pair[0].')
        else:
            raise NotImplementedError()

    def J_index_to_interaction(self, index):
        """
        Gives the tuple representing the interaction `self.J[index]`.

        The set of (self-)interaction parameters of a qubit network is
        stored in the `self.J` variable of the `QubitNetwork` instance.
        This function is a utility to easily recover which interaction
        corresponds to the given index.

        If `self.net_topology` has not been given, this is done by
        simply looking at `self.interactions`, which lists all (and
        only) the active interactions in the network.
        If a custom `self.net_topology` was given, then its value is
        used to recover the (self-)interaction corresponding to the `J`
        element. The output will therefore in this case be a list of
        tuples, each one representing a single interaction.
        """
        if self.net_topology is None:
            self.interactions[index]
        else:
            symbols = sorted(set(self.net_topology.values()))
            interactions = []
            for interaction, symb in self.net_topology.items():
                if symb == symbols[index]:
                    interactions.append(interaction)
            return interactions

    # def get_all_interactions(self):
    #     """DEPRECATED
    #     Returns a list of tuples representing all the interactions.
    #     """
    #     return self.interactions

    def remove_interaction(self, interaction_tuple):
        if self.net_topology is None:
            idx = self.interactions.index(interaction_tuple)
            Js = self.J.get_value()
            del self.interactions[idx]
            del Js[idx]
            self.J.set_value(Js)
        else:
            # idx = list(self.net_topology.keys()).index(interaction_tuple)
            symbol = self.net_topology[interaction_tuple]
            all_interactions = [k for k, v in self.net_topology.items()
                                if v == symbol]
            # if there are interactions associated to the same symbol..
            if len(all_interactions) > 1:
                del self.net_topology[interaction_tuple]
            elif len(all_interactions) == 1:
                symbols = sorted(set(self.net_topology.values()))
                Js = self.J.get_value()
                del Js[symbols.index(symbol)]
                self.J.set_value(Js)

    def get_current_gate(self):
        """Returns the currently produced unitary, in complex form."""
        gate = self.build_H_factors(symbolic_result=False)
        gate = scipy.linalg.expm(gate)
        gate = bigreal2complex(gate)
        return gate

    def fidelity_1s(self, state, target_state):
        """UNTESTED, UNFINISHED"""
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
        H = self.build_H_factors()
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

        # If no ancilla is present in the network, there is no need
        # to partial trace everything, so that the fidelity is simply
        # computed by the projecting the evolution of every element of
        # `states` over the corresponding element of `target_states`,
        # and taking the squared modulus of this number.
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

        # return the mean of the fidelities
        return T.mean(fidelities)
