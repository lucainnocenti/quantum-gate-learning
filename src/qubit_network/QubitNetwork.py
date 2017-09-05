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
    """Implement distinction between system and ancillae.
    """
    def __init__(self,
                 num_qubits=None,
                 num_system_qubits=None,
                 interactions=None,
                 ancillae_state=None,
                 net_topology=None,
                 sympy_expr=None):
        # parameters initialization
        self.ancillae_state = None  # initial values for ancillae (if any)

        # Initialize QubitNetworkHamiltonian parent. This computes
        # `self.matrices` and `self.free_parameters`.
        super().__init__(num_qubits=num_qubits,
                         expr=sympy_expr,
                         interactions=interactions,
                         net_topology=net_topology)
        # Build the initial state of the ancillae, if there are any
        if num_system_qubits is None:
            self.num_system_qubits = self.num_qubits
        else:
            self.num_system_qubits = num_system_qubits
        if self.num_system_qubits < self.num_qubits:
            self._initialize_ancillae(ancillae_state)        

    def _initialize_ancillae(self, ancillae_state):
        """Returns an initial ancilla state, as a qutip.Qobj object.

        The generated state has every ancillary qubit in the up state.
        """
        raise NotImplementedError('To be done.')
        state = qutip.tensor([qutip.basis(2, 0)
                              for _ in range(self.num_ancillae)])
        return state

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
            