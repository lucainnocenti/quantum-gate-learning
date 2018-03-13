import os
import glob
import fnmatch
import pprint
import pickle
import collections

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks

import qutip

from .QubitNetwork import QubitNetwork
from .model import QubitNetworkModel, QubitNetworkGateModel
from .utils import chop, getext, normalize_phase


def group_similar_elements(numbers, eps=1e-3):
    indices_left = list(range(len(numbers)))
    outlist = []
    for idx, num in enumerate(numbers):
        if idx not in indices_left:
            continue
        outlist.append([idx])
        to_remove = []
        for idxidx, idx2 in enumerate(indices_left):
            if np.abs(num - numbers[idx2]) < eps and idx != idx2:
                outlist[-1].append(idx2)
                to_remove.append(idxidx)
        for ir in sorted(to_remove, reverse=True):
            del indices_left[ir]

    return outlist


def group_similar_interactions(net, eps=1e-3):
    similar_indices = group_similar_elements(net.J.get_value())
    groups = []
    for indices_group in similar_indices:
        group = [net.J_index_to_interaction(idx) for idx in indices_group]
        groups.append(group)
    return groups


def vanishing_elements(net, eps=1e-4):
    """Return the elements corresponding to very small interactions."""
    Jvalues = net.J.get_value()
    small_elems = np.where(np.abs(Jvalues) < eps)[0]
    return [net.J_index_to_interaction(idx) for idx in small_elems]


def trace_ancillae_and_normalize(net, num_system_qubits=None, eps=1e-4):
    # if net is a QubitNetwork object
    if hasattr(net, 'get_current_gate'):
        gate = net.get_current_gate(return_qobj=True)
        gate = gate.ptrace(list(range(net.num_system_qubits)))
        gate = gate * np.exp(-1j * np.angle(gate[0, 0]))
        return chop(gate)
    elif isinstance(net, qutip.Qobj):
        # we otherwise assume `net` is a qutip.Qobj object
        if num_system_qubits is None:
            raise ValueError('`num_system_qubits` must be given.')

        gate = net.ptrace(list(range(num_system_qubits)))
        gate = gate * np.exp(-1j * np.angle(gate[0, 0]))
        return chop(gate)


def project_ancillae(net, ancillae_state):
    """Project the ancillae over the specified state."""
    gate = net.get_current_gate(return_qobj=True)
    ancillae_proj = qutip.ket2dm(ancillae_state)
    identity_over_system = qutip.tensor(
        [qutip.qeye(2) for _ in range(net.num_system_qubits)])
    proj = qutip.tensor(identity_over_system, ancillae_proj)
    return proj * gate * proj

# ----------------------------------------------------------------
# Get info and organize saved nets
# ----------------------------------------------------------------


def resave_all_pickle_as_json(path=None):
    """Take all `.pickle` files in `path` and resave them as `json` files."""
    import glob
    import qubit_network as qn

    if path is None:
        path = r'../data/nets/'

    all_nets = glob.glob(path + '*')
    for net_path in all_nets:
        net_name, net_ext = os.path.splitext(net_path)
        if (net_ext == '.pickle') and (net_name + '.json' not in all_nets):
            try:
                net = qn.load_network_from_file(net_path)
                net.save_to_file(net_name + '.json', fmt='json')
            except:
                print('Error while handling {}'.format(net_path))
                raise


def print_saved_nets_info(path=None):
    """Create table summarizing all the `.pickle` net files in `path`"""
    import glob
    import qubit_network as qn
    import pandas as pd

    if path is None:
        path = r'../data/nets/'

    all_nets = glob.glob(path + '*.pickle')
    data = []

    for net_file in all_nets:
        net = qn.load_network_from_file(net_file)

        try:
            data.append({})
            data[-1]['name'] = os.path.splitext(os.path.basename(net_file))[0]
            data[-1]['num_qubits'] = net.num_qubits
            data[-1]['num_ancillae'] = net.num_ancillae
            data[-1]['fid'] = net.fidelity_test(n_samples=100)
        except:
            print('An error was raised during processing of {}'.format(
                net_file))
            continue

    return pd.DataFrame(data)[['name', 'num_qubits', 'num_ancillae', 'fid']]


# ----------------------------------------------------------------
# Display gate matrices.
# ----------------------------------------------------------------


def plot_gate(net,
              norm_phase=True, permutation=None, func='abs',
              fmt='1.2f', annot=True, cbar=False,
              hvlines=None):
    """Pretty-print the matrix of the currently implemented gate.

    Parameters
    ----------
    net : QubitNetwork or matrix_like, optional
        If `net` is a `QubitNetwork` instance, than `net.get_current_gate` is
        used to extract the matrix of the implemented gate.
        In instead `net` is given dircetly as a matrix, we only plot it with
        a nice formatting.
    """
    try:
        gate = net.get_current_gate(return_qobj=True)
    except AttributeError:
        # if `net` does not have the `get_current_gate` method it is assumed
        # to be the matrix to be plotted.
        gate = net

    if permutation is not None:
        gate = gate.permute(permutation)
        gate = normalize_phase(gate)

    gate = gate.data.toarray()

    if func == 'abs':
        gate = np.abs(gate)
    elif func == 'real':
        gate = np.real(gate)
    elif func == 'imag':
        gate = np.imag(gate)
    else:
        raise ValueError('The possible values are abs, real, imag.')

    f, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(gate,
                     square=True, annot=annot, fmt=fmt,
                     linewidth=1, cbar=cbar)

    if hvlines is not None:
        ax.hlines(hvlines, *ax.get_xlim())
        ax.vlines(hvlines, *ax.get_ylim())


# ----------------------------------------------------------------
# Functions ot plot the fidelity vs J parameters for various random states.
# ----------------------------------------------------------------

def plot_fidelity_vs_J_live(net, xs, index_to_vary,
                            states=None, target_states=None,
                            n_states=5):
    """Plot the variation of the fidelity with an interaction parameter.

    Given an input `QubitNetwork` object, a sample of random input states is
    generated, and on each of them the fidelity is computed as a function of
    one of the interaction parameters.
    The resulting plot is updated every time the graph of a state is completed.

    Examples
    --------
    Load a pre-trained network from file, and plot the fidelity for a number
    of random input states as a function of the fifth interaction parameter
    `net.J[4]`, testing its values from -20 to 20 at intervals of 0.05:
    >>> import qubit_network as qn
    >>> import net_analysis_tools as nat
    >>> net = qn.load_network_from_file('path/to/net.pickle')
    >>> nat.plot_fidelity_vs_J_live(net, np.arange(-20, 20, 0.05), 4)
    <output graphics object>
    """
    import copy
    import matplotlib.pyplot as plt
    import theano

    if states is None or target_states is None:
        states, target_states = net.generate_training_data(size=n_states)

    _net = copy.deepcopy(net)
    Js = _net.J.get_value()

    fig, ax = plt.subplots(1, 1)
    fidelities = np.zeros(shape=(len(states), len(xs)))
    for state_idx, (state, target_state) in enumerate(
            zip(states, target_states)):
        compute_fidelity = theano.function(
            inputs=[], outputs=_net.fidelity_1s(state, target_state))

        for idx, x in enumerate(xs):
            new_Js = Js[:]
            new_Js[index_to_vary] = x
            _net.J.set_value(new_Js)

            fidelities[state_idx, idx] = compute_fidelity()

        ax.plot(xs, fidelities[state_idx])
        fig.canvas.draw()


def fidelity_vs_J(net):
    """Return a theano function that generates a fidelity vs interaction plot.

    This function differs from `plot_fidelity_vs_J_live` in that it does not
    directly compute values of the fidelity. Instead, it compiles and returns
    a `theano.function` object that, given as input a set of input states and
    corresponding target states, which interaction parameter to vary and the
    variation range, returns the set of values of the fidelities to plot.

    It is also worth noting that this function does not handle at all the
    actual drawing of the output plot. It only compiles a function to be used
    for such a plot.

    Examples
    --------
    Load a network from file, use `fidelity_vs_J` to compile the function to
    compute the fidelities for various states, and use it to plot the fidelity
    for various states when varying the fifth interaction parameter `net.J[4]`
    in the range `np.arange(-40, 40, 0.05)`.
    >>> import qubit_network as qn
    >>> import net_analysis_tools as nat
    >>> net = qn.load_network_from_file('path/to/net.pickle')
    >>> plots_generator = nat.fidelity_vs_J(net)
    >>> states, target_states = net.generate_training_data(net.target_gate, 10)
    >>> xs = np.arange(-40, 40, 0.05)
    >>> fidelities = plots_generator(states, target_states, xs, 4)
    >>> fig, ax = plt.subplots(1, 1)
    >>> for fids in fidelities:
    >>>     ax.plot(xs, fids)
    >>>     fig.canvas.draw()
    >>> <output graphics object>
    """
    import copy
    import theano
    import theano.tensor as T

    _net = copy.deepcopy(net)
    xs = T.dvector('xs')
    states = T.dmatrix('states')
    target_states = T.dmatrix('target_states')
    index_to_vary = T.iscalar('index_to_vary')

    def foreach_x(x, index_to_vary, states, target_states):
        _net.J = T.set_subtensor(_net.J[index_to_vary], x)
        return _net.fidelity(states, target_states, return_mean=False)

    results, _ = theano.scan(
        fn=foreach_x,
        sequences=xs,
        non_sequences=[index_to_vary, states, target_states]
    )
    fidelities = results.T
    return theano.function(
        inputs=[states, target_states, xs, index_to_vary],
        outputs=fidelities
    )


# ----------------------------------------------------------------
# Plotting and handling visualization of net parameters
# ----------------------------------------------------------------

# def dataframe_parameters_to_net(df, column_index, net=None):
#     """Load back the parameters to the net.

#     The DataFrame is expected to have the structure producd
#     by `net_parameters_to_dataframe` with the parameter
#     `stringify_index=True`.
#     """
#     # if the index is not a MultiIndex, it probably means it was
#     # stringified. Convert it back into a MultiIndex which more closely
#     # resembles the format we want.
#     if isinstance(df.index, pd.Index):
#         keys = df.index.map(eval).values
#     else:
#         keys = df.index.values
#     # in the QubitNetwork object the self-interaction
#     # qubit numbers are integeres, not tuples with a
#     # single integer.
#     for idx in range(len(keys)):
#         keys[idx] = list(keys[idx])
#         if len(keys[idx][0]) == 1:
#             keys[idx][0] = keys[idx][0][0]
#     keys = [tuple(item) for item in keys]
#     # get the interaction values we are interested in
#     interactions_values = df.iloc[:, column_index].values
#     # now we can effectively load the interactions
#     # and corresponding values into the net (and hope
#     # for the best).
#     if net is None:
#         # get maximum qubit index
#         num_qubits = max(
#             max(key[0]) if isinstance(key[0], tuple) else key[0]
#             for key in keys)
#         num_qubits += 1
#         net = QubitNetwork(
#             num_qubits,
#             system_qubits=num_qubits,
#             interactions=keys,
#             J=interactions_values)
#     else:
#         net.interactions = keys
#         net.J.set_value(interactions_values)

#     return net


# ----------------------------------------------------------------
# Loading nets from file
# ----------------------------------------------------------------
def _load_network_from_pickle_old(data):
    """Rebuild QubitNetworkModel from old style saved data.
    
    Parameters
    ----------
    data : dict
        Data loaded (usually) from a pickle file. It is expected to contain
        at least the 'J' key.
    Returns
    -------
    Pair of dicts `net_data` and `opt_data`.
    - `net_data` is to contain the model, typically in the form of a
        QubitNetworkGateModel object.
    - `opt_data` contains data concerning the optimization of the network. This
        is actually returned as a None because the optimization data was not
        saved in old nets.
    """
    # from IPython.core.debugger import set_trace; set_trace()
    topology = data.get('net_topology', None)
    interactions = data.get('interactions', None)
    if isinstance(interactions, list):
        # nets saved in the past used notation ((0, 1), 'xx'), as opposite
        # to the currently supported (1, 1). Here we do the conversion
        # from old to new style
        new_ints = []
        translation_rule = {'x': 1, 'y': 2, 'z': 3}
        for targets, types in interactions:
            new_int = [0] * data['num_qubits']
            if not isinstance(targets, tuple):
                targets = (targets,)
            for target, type_ in zip(targets, list(types)):
                new_int[target] = translation_rule[type_]
            new_ints.append(new_int)
        interactions = new_ints

    ints_values = data.get('J')
    model = QubitNetworkGateModel(
        num_qubits=data['num_qubits'],
        num_system_qubits=data['num_system_qubits'],
        interactions=interactions,
        net_topology=topology,
        target_gate=data['target_gate'],
        initial_values=ints_values)

    return model, None


def _load_network_from_pickle(filename):
    """
    Rebuild `QubitNetwork` from pickled data in `filename`.

    The QubitNetwork objects should have been stored into the file in
    pickle format, using the appropriate `save_to_file` method.
    The returned object is QubitNetworkGateModel.
    """

    with open(filename, 'rb') as file:
        data = pickle.load(file)
    if 'J' in data:
        return _load_network_from_pickle_old(data)
    # otherwise we can just use `sympy_model`:
    net_data = data['net_data']
    opt_data = data['optimization_data']
    model = QubitNetworkGateModel(
        sympy_expr=net_data['sympy_model'],
        target_gate=opt_data['target_gate'],
        free_parameters_order=net_data['free_parameters'],
        initial_values=opt_data['final_interactions']
    )
    return model, opt_data


def _load_network_from_json(filename):
    raise NotImplementedError('Not implemented yet, load from pickle.')


def load_network_from_file(filename, fmt=None):
    """
    Rebuild `QubitNetwork` object from data in `filename`.
    """
    # if no format has been given, get it from the file name
    if fmt is None:
        _, fmt = os.path.splitext(filename)
        fmt = fmt[1:]
    # decide which function to call to load the data
    if fmt == 'pickle':
        return _load_network_from_pickle(filename)
    elif fmt == 'json':
        return _load_network_from_json(filename)
    else:
        raise ValueError('Only pickle or json formats are supported.')


class NetDataFile:
    """
    Represent a single data file containing a saved net.

    Parameters
    ----------
    path : string
        Path of the data file.
    Attributes
    ----------
    path : string
        Full path of the file.
    data : QubitNetworkGateModel object
        This is loaded via `_load` from file. The actual content is in
        the internal `_data` attribute, this one can be used to load
        the content if it wasn't already loaded. Same for `opt_data`
        and `_opt_data`.
    opt_data : dict
    """
    def __init__(self, path):
        self.path = path
        self.name = os.path.splitext(os.path.split(self.path)[1])[0]
        # the actual `QubitNetwork` object is only loaded when required
        self._data = None
        self._opt_data = None

    def __repr__(self):
        return self.name + ' (' + getext(self.path) + ')'

    def __getattr__(self, value):
        return getattr(self.data, value)

    def _load(self):
        """
        Read data from the stored path and save it into `self._data`.

        If the data was already loaded, it is loaded again.
        """
        self._data, self._opt_data = load_network_from_file(self.path)

    def get_target_gate(self):
        """
        Return the name of the target gate according to the file name.

        This function assumes that the file name follows the naming
        convention 'gatename_blabla_otherinfo.pickle'.
        """
        if '_' not in self.name:
            return self.name
        else:
            return self.name.split('_')[0]

    def get_fidelity(self, n_samples=40):
        return self.data.fidelity_test(n_samples=n_samples)

    @property
    def data(self):
        """
        The dict stored in file, loaded on demand.
        """
        if self._data is None:
            self._load()
        return self._data

    @property
    def opt_data(self):
        if self._data is None:
            self._load()
        return self._opt_data

    def _get_interactions_old_style(self):
        data = self._data
        topology = data.get('net_topology', None)
        interactions = data.get('interactions', None)
        ints_values = data.get('J')
        if topology is not None:
            ints_dict = collections.OrderedDict()
            for interaction, symb in topology.items():
                try:
                    ints_dict[symb].append(interaction)
                except KeyError:
                    ints_dict[symb] = [interaction]
            ints_out = list(zip(ints_dict.values(), ints_values))
        else:
            ints_out = list(zip(interactions, ints_values))
        return ints_out

    @property
    def interactions(self):
        """
        Gives the trained interactions in a nicely formatted DataFrame.
        """
        interactions, values = self.free_parameters, self.parameters.get_value()
        # now put everything in dataframe
        return pd.DataFrame({
            'interaction': interactions,
            'value': values
        }).set_index('interaction')


class NetsDataFolder:
    """
    Class representing a folder containing nets data files.

    This function assumes that all the `.json` and `.pickle` files in
    the given directory are files containing a `QubitNetwork` object in
    appropriate format.

    Parameters
    ----------
    path : string, optional
        The directory where the net files are searched for.
        If not given, it defaults to '../data/nets'.

    Attributes
    ----------
    path : string
    files : list of strings
    nets : list of NetDataFile objects
    """

    def __init__(self, path_or_files='../data/nets/'):
        if isinstance(path_or_files, str):
            self._load_from_dir(path_or_files)
        elif isinstance(path_or_files, (list, tuple)):
            self._load_from_file_list(path_or_files)

    def _load_from_dir(self, path):
        # ensure tha path is of the form 'whatever/'
        if path[-1] != '/':
            path += '/'
        # raise error if path is not a directory
        if not os.path.isdir(path):
            raise ValueError('path must be a valid directory.')
        self.path = path
        # load pickle files in path
        self.files = glob.glob(path + '*.pickle')
        # raise error if no json and pickle files are found
        if len(self.files) == 0:
            raise FileNotFoundError('No valid data files found in '
                                    '{}.'.format(path))
        # for each data file associate a `NetDataFile` object, and store
        # the collection of such objects in `self.nets`.
        self.nets = []
        for file_ in self.files:
            self.nets.append(NetDataFile(file_))

    def _load_from_file_list(self, files_list):
        self.path = None
        self.files = list(files_list)
        self.nets = [NetDataFile(file_) for file_ in files_list]

    def __repr__(self):
        return self._repr_dataframe().__repr__()

    def _repr_html_(self):
        return self._repr_dataframe()._repr_html_()

    def _repr_dataframe(self):
        names = [net.name for net in self.nets]
        target_gates = [net.get_target_gate() for net in self.nets]
        # load sorted data in pandas DataFrame
        df = pd.DataFrame({
            'target gates': target_gates,
            'names': names
        })[['target gates', 'names']]
        # return formatted string
        return df.sort_values(by=['names']).reset_index(drop=True)

    def __getitem__(self, key):
        try:
            if isinstance(key, slice):
                self.nets = self.nets[key]
                self.files = self.files[key]
                return self
            elif isinstance(key, (list, tuple)):
                self.nets = [self.nets[idx] for idx in key]
                return self
            else:
                return self.nets[key]
        # if numbered indexing didn't work, we try assuming  `key` is
        # a string, and look for matching net names.
        except TypeError:
            # if `key` contains a wildcard, use is to match using filter
            if '*' in key:
                matching_nets = list(self.filter(key))
            # otherwise assume it just denotes the beginning of the name
            else:
                matching_nets = list(self.filter(key + '*'))

            return matching_nets

    def short(self):
        """
        Return a shortened version of the list of saved nets.
        """
        nets_in_df = self._repr_dataframe()
        counts = collections.Counter(nets_in_df['target gates'])
        unique_gates = nets_in_df['target gates'].unique()
        return pd.DataFrame({
            'target gate': unique_gates,
            'number of saved nets': list(counts.values())
        })[['target gate', 'number of saved nets']]

    def filter(self, pat):
        """
        Return a subset of the nets in `self.nets` satisfying condition.

        Simple wildcard matching provided by `fnmatch.filter` is used.
        """
        new_data = NetsDataFolder(self.path)
        new_data.files = fnmatch.filter(self.files, '*/' + pat)
        new_data.nets = [net for net in self.nets
                         if fnmatch.fnmatch(net.name, pat)]
        return new_data
        # names = fnmatch.filter([net.name for net in self.nets], pat)
        # for net in self.nets:
        #     if net.name in names:
        #         yield net

    def get_filenames(self):
        return [os.path.splitext(name)[0] for name in self.files]

    def reload(self):
        self = NetsDataFolder(self.path)
        return self

    def view_fidelities(self, n_samples=40):
        data = self._repr_dataframe()
        fids = [net.fidelity_test(n_samples=n_samples)
                for net in self.nets]
        data = pd.concat((
            data,
            pd.Series(fids, name='fidelity')
        ), axis=1)
        return data

    def view_parameters(self, n_samples=40):
        """
        Return a dataframe showing the parameters for every net.
        """
        data = None
        for net in self.nets:
            # compute fidelity for net
            fid = net.fidelity_test(n_samples=n_samples)
            # get data for net
            new_df = net.interactions.rename(columns={'value': fid})
            if data is None:
                data = new_df
                continue
            data = pd.concat((data, new_df), axis=1)
        return data

    def plot_parameters(self, joined=True, hlines=None, return_fig=False):
        """
        Plot an overlay scatter plot of all the nets.
        """
        data = self.view_parameters()
        fids = data.columns
        data.columns = np.arange(len(fids))
        # stringify indices for the legend later
        data.index = data.index.map(str)
        fig = data.iplot(mode='lines+markers', size=6, asFigure=True)
        # readd legend labels (this is necessary because cufflinks
        # seems to make a mess when multiple columns have the same name)
        for trace_idx in range(len(fig.data)):
            fig.data[trace_idx].name = fids[trace_idx]
        for trace in fig.data:
            if joined:
                trace.update({'connectgaps': True})
            else:
                trace.update({'connectgaps': False,
                              'mode': 'markers'})
        # put overlay hlines
        from .plotly_utils import hline
        if hlines is None:
            hlines = np.arange(-np.pi, np.pi, np.pi / 2)
        fig.layout.shapes = hline(0, len(data) - 1,
                                  hlines, dash='dash')
        # finally draw the damn thing
        if return_fig:
            return fig
        import plotly.offline
        plotly.offline.iplot(fig)
