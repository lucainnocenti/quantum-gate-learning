"""
Main class implementing the qubit network.
"""
import itertools
from collections import OrderedDict
import os
import numbers
import re
import warnings

import scipy.linalg
import numpy as np
import pandas as pd

import theano
import theano.tensor as T
import theano.tensor.slinalg  # for expm()
import theano.tensor.nlinalg  # for trace()

import qutip

from .utils import (chars2pair, complex2bigreal, bigreal2complex, chop,
                    custom_dataframe_sort)
from ._QubitNetwork import _find_suitable_name
from .hamiltonian import QubitNetworkHamiltonian

# from IPython.core.debugger import set_trace


class QubitNetwork(QubitNetworkHamiltonian):
    """
    Main object representing the qubit network.
    """
    def __init__(self,
                 num_qubits=None,
                 num_system_qubits=None,
                 interactions=None,
                 ancillae_state=None,
                 target_gate=None,
                 net_topology=None,
                 sympy_expr=None,
                 initial_values=None):
        # parameters initialization
        self.system_qubits = None
        self.target_gate = None
        self.num_system_qubits = None
        self.ancillae_state = None  # initial values for ancillae (if any)
        # Initialize QubitNetworkHamiltonian parent. This does two things:
        # 1. Computes `self.matrices` and `self.free_parameters`, to be
        #    later used to build the full Hamiltonian matrix (and thus
        #    the computational graph for the training).
        # 2. Actually, just the thing above.
        super().__init__(num_qubits=num_qubits,
                         expr=sympy_expr,
                         interactions=interactions,
                         net_topology=net_topology)
        # Initialize values of parameter in graph. This is stored in the
        # attribute `self.J` inherited from `QubitNetworkHamiltonian`,
        # which is shared theano tensor.
        self.set_initial_values(initial_values)
        # `self.target_gate` is given a value when the net is being
        # trained, for example by `sgd_optimization`. It is used simply
        # to keep track of what the network was trained to reproduce.
        # If no explicit value for `target_gate` is given, it is set
        # during the training.
        self.target_gate = target_gate
        # Build the initial state of the ancillae, if there are any
        if num_system_qubits is None:
            self.num_system_qubits = self.num_qubits
        else:
            self.num_system_qubits = num_system_qubits
        if self.num_system_qubits < self.num_qubits:
            self._initialize_ancillae(ancillae_state)
        # if self.num_ancillae > 0:
        #     if ancillae_state is None:
        #         self.ancillae_state = self.build_ancilla_state()
        #     else:
        #         self.ancillae_state = ancillae_state
        # else:
        #     self.ancillae_state = None
        

    def _initialize_ancillae(self, ancillae_state):
        """Returns an initial ancilla state, as a qutip.Qobj object.

        The generated state has every ancillary qubit in the up state.
        """
        raise NotImplementedError('To be done.')
        state = qutip.tensor([qutip.basis(2, 0)
                              for _ in range(self.num_ancillae)])
        return state


    def save_to_file(self, outfile, fmt='pickle', overwrite=False):
        """ Saves the currently trained net in a file.

        Parameters
        ----------
        fmt : Format of output file.
            Possible values are 'pickle' and 'json'.
        """
        if not overwrite:
            # change name if file already exists
            _outfile = _find_suitable_name(outfile)
            if _outfile != outfile:
                warnings.warn('File already existing, saving instead in'
                            ' {}.'.format(_outfile))
                outfile = _outfile
        # check format
        if fmt == 'pickle':
            import pickle
            data = {
                'num_qubits': self.num_qubits,
                'num_system_qubits': self.num_system_qubits,
                'ancillae_state': self.ancillae_state,
                'interactions': self.interactions,
                'target_gate': self.target_gate,
                'net_topology': self.net_topology,
                'J': self.J.get_value(),
                'initial_values': self.initial_values
            }
            if not os.path.isabs(outfile):
                outfile = os.path.join(os.getcwd(), outfile)
            with open(outfile, 'wb') as file:
                pickle.dump(data, file)
        elif fmt == 'json':
            import json

            data = dict()

            current_gate = self.get_current_gate().data.toarray()
            data['full_unitary.real'] = current_gate.real.tolist()
            data['full_unitary.imag'] = current_gate.imag.tolist()

            if self.num_system_qubits  < self.num_qubits:
                ancillae_state = self.ancillae_state.data.toarray()
                data['ancillae_state.real'] = ancillae_state.real.tolist()
                data['ancillae_state.imag'] = ancillae_state.imag.tolist()

            # old saved nets did not have the target_gate attribute, so we have
            # to check for its existence to avoid errors later. Also, it is
            # messy to reload complex numbers from the json dumped data, so
            # we save real and imaginary parts separately.
            if self.target_gate:
                if self.target_gate.issuper:
                    data['target_gate.issuper'] = 1
                else:
                    data['target_gate.issuper'] = 0

                target_gate = self.target_gate.data.toarray()
                data['target_gate.real'] = target_gate.real.tolist()
                data['target_gate.imag'] = target_gate.imag.tolist()
            else:
                data['target_gate.real'] = None
                data['target_gate.imag'] = None

            data['num_qubits'] = self.num_qubits
            data['num_system_qubits'] = self.num_system_qubits

            # json.dump cannot serialize dictionaries with tuples as keys, so
            # we have to stringify them. To restore the original keys we can
            # later use `eval`.
            if self.net_topology:
                net_topology = OrderedDict(
                    (str(k), v) for k, v in self.net_topology.items())
            else:
                net_topology = self.net_topology
            data['net_topology'] = net_topology

            data['interactions'] = self.interactions
            data['J'] = self.J.get_value().tolist()
            data['initial_values'] = self.initial_values.tolist()

            with open(outfile, 'w') as fp:
                json.dump(data, fp, indent=4)


    def tuple_to_J_index(self, interaction):
        if self.net_topology is None:
            # if `pair` is a self-interaction
            return self.interactions.index(interaction)
        else:
            raise NotImplementedError('I didn\'t implement this yet, sorry!')

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
            return self.interactions[index]
        else:
            symbols = sorted(set(self.net_topology.values()))
            interactions = []
            for interaction, symb in self.net_topology.items():
                if symb == symbols[index]:
                    interactions.append(interaction)
            return interactions

    def remove_interaction(self, interaction_tuple):
        """Removes the specified interaction from the network."""
        if self.net_topology is None:
            # in this case we just remove the specified interaction from
            # the `self.interactions` list, and the corresponding entry
            # in `self.J`
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
                # then we just remove the corresponding entry in the
                # `self.net_topology` variable (no need to change the
                # value of `self.J`)
                del self.net_topology[interaction_tuple]
            # if there are interactions associated to the same parameter
            elif len(all_interactions) == 1:
                # then we also remove the corresponding entry of
                # `self.J`, in addition to removing the entry in
                # `self.net_topology`
                symbols = self.net_topology_symbols
                Js = self.J.get_value()
                del Js[symbols.index(symbol)]
                self.J.set_value(Js)
                del self.net_topology_symbols[symbols.index(symbol)]

    def get_current_gate(self, return_qobj=True, chop_eps=None):
        """Returns the currently produced unitary, in complex form."""
        gate = self.build_H_factors(symbolic_result=False)
        gate = scipy.linalg.expm(gate)
        gate = bigreal2complex(gate)
        if chop_eps is not None:
            gate = chop(gate, chop_eps)

        if return_qobj:
            return qutip.Qobj(gate, dims=[[2] * self.num_qubits] * 2)
        else:
            return gate

    def get_grouped_interactions(self):
        """
        Return list of interactions, taking the topology into account.
        """
        if self.net_topology is None:
            return self.interactions
        else:
            outlist = []
            for symbol in self.net_topology_symbols:
                matching_interactions = []
                for interaction, label in self.net_topology.items():
                    if label == symbol:
                        matching_interactions.append(interaction)
                outlist.append(matching_interactions)
            return outlist

    def get_interactions_with_Js(self, renormalize_parameters=False):
        """
        Gives a dict associating each interaction to the correspoding J.

        Parameters
        ----------
        renormalize_parameters : bool
            If True, multiplies the pairwise interactions by 4 and the
            self-interactions by 2.
        """
        if self.net_topology is None:
            outdict = OrderedDict()
            Js = self.J.get_value()
            for interaction, J in zip(self.interactions, Js):
                # if `renormalize_parameter` is set to True the results
                # are printed using the convention in which the terms
                # in the Hamiltonian are 1/4 for the pairwise
                # interactions and 1/2 for the self-interactions.
                if renormalize_parameters:
                    if len(interaction[1]) == 2:
                        outdict[interaction] = 4 * J
                    elif len(interaction[1]) == 1:
                        outdict[interaction] = 2 * J
                else:
                    outdict[interaction] = J
            return outdict
        else:
            symbols = self.net_topology_symbols
            outdict = OrderedDict()
            Js = self.J.get_value()
            for idx, symbol in enumerate(symbols):
                interactions = []
                renormalize = 0
                for interaction, label in self.net_topology.items():
                    if label == symbol:
                        interactions.append(interaction)
                        # the following mess is to check that it makes
                        # sense to renormalize the parameters associated
                        # with a symbol: if a symbol is associated to
                        # both pairwise and self- interactions then we
                        # cannot consistently renormalize the Js.
                        if renormalize_parameters:
                            if len(interaction[1]) == 2 and renormalize == 2:
                                renormalize = 0
                                print('Can\' properly renormalize the paramete'
                                      'rs, reverting to non-renormalized form.'
                                     )
                            elif len(interaction[1]) == 1 and renormalize == 4:
                                renormalize = 0
                                print('Can\' properly renormalize the paramete'
                                      'rs, reverting to non-renormalized form.'
                                     )
                            elif renormalize == 0:
                                if len(interaction[1]) == 2:
                                    renormalize = 4
                                elif len(interaction[1]) == 1:
                                    renormalize = 2

                if not renormalize_parameters:
                    renormalize = 1
                outdict[tuple(interactions)] = Js[idx] * renormalize

            return outdict

    def test_fidelity(self,
                      states=None, target_states=None,
                      target_gate=None,
                      n_samples=10):
        """Computes an average fidelity with the current values of J."""
        if target_gate is None:
            if self.target_gate is None:
                raise ValueError('No target gate has been specified')
            else:
                target_gate = self.target_gate

        if states is None or target_states is None:
            states, target_states = self.generate_training_data(
                target_gate, n_samples)

        fidelity = theano.function(
            inputs=[],
            outputs=self.fidelity(states, target_states)
        )
        return fidelity()

    def test_fidelity_without_theano(self, target_gate=None, n_samples=10):
        """
        Computes the fidelity with random states, using only `qutip` functions.

        This function can be used to double check the output given in the
        optimization process. The whole computation is here done using
        high level `qutip` functions.
        """
        gate = self.get_current_gate()
        if target_gate is None:
            if self.target_gate is None:
                raise ValueError('No target gate has been specified')
            else:
                target_gate = self.target_gate
        # each element of `fidelities` will contain the fidelity obtained with
        # a single randomly generated input state
        fidelities = np.zeros(n_samples)
        for idx in range(fidelities.shape[0]):
            # generate random input state (over system qubits only)
            psi_in = qutip.rand_ket_haar(2 ** self.num_system_qubits)
            psi_in.dims = [
                [2] * self.num_system_qubits, [1] * self.num_system_qubits]
            # embed it into the bigger system+ancilla space (if necessary)
            if self.num_system_qubits < self.num_qubits:
                Psi_in = qutip.tensor(psi_in, self.ancillae_state)
            else:
                Psi_in = psi_in
            # evolve input state
            Psi_out = gate * Psi_in
            # trace out ancilla (if there is an ancilla to trace)
            if self.num_system_qubits < self.num_qubits:
                dm_out = Psi_out.ptrace(range(self.num_system_qubits))
            else:
                dm_out = qutip.ket2dm(Psi_out)
            # compute fidelity
            fidelity = (psi_in.dag() * target_gate.dag() *
                        dm_out * target_gate * psi_in)
            fidelities[idx] = fidelity[0, 0].real
        return fidelities.mean()

    # `fidelity_1s` computes the fidelity over a single pair
    # of state and target state, as opposite as the computation of the
    # mini-batch averages (using `theano.scan`) done by `fidelity`.
    def fidelity_1s(self, state, target_state, J=None):
        """Compute the fidelity from a single pair of states."""
        # this builds the Hamiltonian of the system (in big real matrix
        # form), already multiplied with the 1j factor and ready for
        # exponentiation.
        H = self.build_H_factors(symbolic_result=True, J=J)
        # expH is the unitary evolution of the system
        expH = T.slinalg.expm(H)
        Uxpsi = T.dot(expH, state).reshape((state.shape[0], 1))
        Uxpsi_real = Uxpsi[:Uxpsi.shape[0] // 2]
        Uxpsi_imag = Uxpsi[Uxpsi.shape[0] // 2:]
        dm_real = Uxpsi_real * Uxpsi_real.T + Uxpsi_imag * Uxpsi_imag.T
        dm_imag = Uxpsi_imag * Uxpsi_real.T - Uxpsi_real * Uxpsi_imag.T

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

    def fidelity(self, states, target_states, return_mean=True,
                 return_var=False):
        """The cost function of the model.

        The states given in `states` are evolved through the network
        with interactions parameters `self.J`, then traced over the
        ancilla degrees of freedom. Every state obtained in this way is
        then projected over the corresponding target state
        (corresponding element of `target_states`), thus computing the
        fidelity for every state. The average of these fidelities is the
        returned value.
        A properly formatted set of `states` and `target_states` can be
        generated by `self.generate_training_data`.

        Note: This function does not compute the values of the fidelity,
            it only build the corresponding theano computational graph.

        Parameters
        ----------
        states : numpy array, shape (n_states, dim_states)
            The initial states to be evolved into the network.
            Typically every element is the tensor product between a
            randomly generated ket state, and the ancillae states.
            The states are expected to be in "big real form", that is,
            every state is actually an array generated by concatenating
            the real and imaginary parts of the original ket vector
            (in this order).
            Every state is expected to span the *whole* qubit network,
            thus having a length of 2 * (2 ** self.num_qubits).
        target_states : numpy array, shape (n_states, dim_target_states)
            The ideal result to which to compare the elements of
            `states`.
            Every state in `target_states` is expected to be the result
            of evolving the corresponding state in `states` with the
            target gate that the training is trying to make the network
            implement.
        return_mean : bool, optional
            If True, returns the average values of the fidelities.
            If False, returns an array with all the fidelities.
        return_var : bool, optional
            If True, together with the mean fidelity also the max - min
            of the computed fidelities is returned.

        Returns
        -------
        A theano function, to be used for the MSGD algorithm
        """
        # Use free parameteres and matrix coefficients to build the
        # computational graph, using `theano.tensor.tensordot`. The
        # result is a theano.tensor version of the parametrized
        # Hamiltonian model, multiplied by -1j, in big real form.
        H = self.build_theano_graph()
        # expH is the unitary evolution of the system
        expH = T.slinalg.expm(H)

        # `expH_times_state` is the full output state given by the qubit
        # network.
        # `state` is in general a matrix (array of state vectors) so
        # that `expH_times_state` is also a matrix with a number of
        # rows equal to the number of training vectors. Every row
        # of this matrix is a state evolved according to the gate
        # implemented by the network with the current interactions
        # parameters.
        expH_times_state = T.tensordot(expH, states, axes=([1], [1])).T

        # If no ancilla is present in the network, there is no need
        # to partial trace anything, so that the fidelity can be simply
        # computed projecting the evolution of every element of
        # `states` over the corresponding element of `target_states`,
        # and taking the squared modulus of this number.
        num_ancillae = self.num_qubits - self.num_system_qubits
        if num_ancillae == 0:
            fidelities, _ = theano.scan(
                fn=_compute_fidelities_no_ptrace,
                sequences=T.arange(expH_times_state.shape[0]),
                non_sequences=[expH_times_state, target_states]
            )
        else:
            fidelities, _ = theano.scan(
                fn=_compute_fidelities,
                sequences=T.arange(expH_times_state.shape[0]),
                non_sequences=[expH_times_state, target_states, num_ancillae]
            )

        # the default behaviour is to return the mean computed fidelity,
        # but it is also possible to return all the computed values of
        # the fidelity:
        if return_mean:
            # return the mean of the fidelities
            if return_var:
                max_fid = T.max(fidelities)
                min_fid = T.min(fidelities)
                return T.mean(fidelities), max_fid - min_fid
            else:
                return T.mean(fidelities)
        else:
            # return the array with all of the computed fidelities
            # (one for each training state in the batch)
            return fidelities
    
    def net_parameters_to_dataframe(self, stringify_index=False):
        """
        Take parameters from a QubitNetwork object and put it in DataFrame.

        Parameters
        ----------
        stringify_index : bool
            If True, instead of a MultiIndex the output DataFrame will have
            a single index of strings, built applying `df.index.map(str)` to
            the original index structure.

        Returns
        -------
        A `pandas.DataFrame` with the interaction parameters ordered by
        qubits on which they act and type (interaction direction).
        """
        parameters = self.get_interactions_with_Js()
        qubits = []
        directions = []
        values = []
        for key, value in parameters.items():
            try:
                qubits.append(tuple(key[0]))
            except TypeError:
                qubits.append((key[0], ))
            directions.append(key[1])
            values.append(value)

        pars_df = pd.DataFrame({
            'qubits': qubits,
            'directions': directions,
            'values': values
        }).set_index(['qubits', 'directions']).sort_index()
        if stringify_index:
            pars_df.index = pars_df.index.map(str)
        return pars_df

    def plot_net_parameters(self, sort_index=True, plotly_online=False,
                            mode='lines+markers+text',
                            overlay_hlines=None,
                            asFigure=False, **kwargs):
        """Plot the current values of the parameters of the network."""
        import cufflinks
        df = self.net_parameters_to_dataframe()
        # optionally sort the index, grouping together self-interactions
        if sort_index:
            def sorter(elem):
                return len(elem[0][0])
            sorted_data = sorted(list(df.iloc[:, 0].to_dict().items()),
                                 key=sorter)
            x, y = tuple(zip(*sorted_data))
            df = pd.DataFrame({'x': x, 'y': y}).set_index('x')
            df.index = df.index.map(str)
        # decide online/offline
        if plotly_online:
            cufflinks.go_online()
        else:
            cufflinks.go_offline()
        # draw overlapping horizontal lines for reference if asked
        if overlay_hlines is None:
            overlay_hlines = np.arange(-np.pi, np.pi, np.pi / 2)
            # return df.iplot(kind='scatter', mode=mode, size=6,
            #                 title='Values of parameters',
            #                 asFigure=asFigure, **kwargs)
        from .plotly_utils import hline
        fig = df.iplot(kind='scatter', mode=mode, size=6,
                       title='Values of parameters',
                       text=df.index.tolist(),
                       asFigure=True, **kwargs)
        fig.layout.shapes = hline(0, len(self.interactions),
                                    overlay_hlines, dash='dash')
        fig.data[0].textposition = 'top'
        fig.data[0].textfont = dict(color='white', size=13)
        if asFigure:
            return fig
        else:
            return plotly.offline.iplot(fig)
            