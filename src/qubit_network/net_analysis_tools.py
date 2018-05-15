import os
import glob
import logging
import progressbar
import fnmatch
import pprint
import pickle
import collections

import numpy as np
import pandas as pd
import sympy

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

    if path is None:
        path = r'../data/nets/'

    all_nets = glob.glob(path + '*')
    for net_path in all_nets:
        net_name, net_ext = os.path.splitext(net_path)
        if (net_ext == '.pickle') and (net_name + '.json' not in all_nets):
            try:
                net = load_network_from_file(net_path)
                net.save_to_file(net_name + '.json', fmt='json')
            except:
                print('Error while handling {}'.format(net_path))
                raise


# ----------------------------------------------------------------
# Display gate matrices.
# ----------------------------------------------------------------


def plot_gate(net, ptrace=None, norm_phase=True, permutation=None, func='abs',
              fmt='1.2f', annot=True, cbar=False, hvlines=None, ax=None):
    """Pretty-print the matrix of the currently implemented gate.

    Parameters
    ----------
    net : QubitNetworkGateModel or matrix_like, optional
        If `net` is a `QubitNetwork` instance, than `net.get_current_gate` is
        used to extract the matrix of the implemented gate.
        In instead `net` is given dircetly as a matrix, we only plot it with
        a nice formatting.
    ptrace : list of ints
        If not None, the output gate is partial traced before plotting.
        The value is directly given to `qutip.ptrace`.
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
    if ptrace is not None:
        gate = gate.ptrace(ptrace)

    gate = gate.data.toarray()

    if func == 'abs':
        gate = np.abs(gate)
    elif func == 'real':
        gate = np.real(gate)
    elif func == 'imag':
        gate = np.imag(gate)
    else:
        raise ValueError('The possible values are abs, real, imag.')

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(gate,
                     square=True, annot=annot, fmt=fmt,
                     linewidth=1, cbar=cbar)

    if hvlines is not None:
        ax.hlines(hvlines, *ax.get_xlim())
        ax.vlines(hvlines, *ax.get_ylim())


# ----------------------------------------------------------------
# Functions ot plot the fidelity vs J parameters for various random states.
# ----------------------------------------------------------------

def plot_fidelity_vs_J_qutip(net, xs, index_to_vary,
                             states=None, target_states=None,
                             n_states=5, ax=None):
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
    >>> import net_analysis_tools as nat
    >>> net = nat.load_network_from_file('path/to/net.pickle')
    >>> nat.plot_fidelity_vs_J_live(net, np.arange(-20, 20, 0.05), 4)
    <output graphics object>
    """
    import copy
    import matplotlib.pyplot as plt
    import theano
    # from IPython.core.debugger import set_trace; set_trace()
    if states is None or target_states is None:
        # states, target_states = net.generate_training_states(n_states)
        hs_dims = 2**net.num_system_qubits
        dims = [[2] * net.num_system_qubits, [1] * net.num_system_qubits]
        states = [qutip.rand_ket_haar(hs_dims, dims=dims) for _ in range(n_states)]
        # states = np.asarray(states)
        target_gate = net.target_gate
        target_states = [target_gate * ket for ket in states]
        # if there are ancillae, they are added to the inputs
        if net.num_qubits > net.num_system_qubits:
            if net.ancillae_state is not None:
                ancillae = net.ancillae_state
            else:
                num_ancillae = net.num_qubits - net.num_system_qubits
                ancillae = qutip.tensor(*(qutip.basis(2, 0) for _ in range(num_ancillae)))
            states = [qutip.tensor(state, ancillae) for state in states]
        # target_states = np.einsum('ij,kj->ki', target_gate, states)
    # create another instance of model to avoid changing the original one
    _net = copy.deepcopy(net)
    # extract parameters
    try:
        pars_ref = _net.parameters
        pars_values = _net.parameters.get_value()
    except AttributeError:
        pars_ref = _net.J
        pars_values = _net.J.get_value()
    # initialise figure object, if the use does not want to plot on his own axis
    if ax is None:
        _, ax = plt.subplots(1, 1)
    # initialise array of fidelities (for all states)
    fidelities = np.zeros(shape=(len(states), len(xs)))
    # for state_idx, (state, target_state) in enumerate(zip(states, target_states)):
    import progressbar
    bar = progressbar.ProgressBar()
    for idx, x in enumerate(bar(xs)):
        # we need to copy the array here, otherwise we change the original
        new_pars = np.array(pars_values)
        if isinstance(index_to_vary, str) and index_to_vary == 'all':   
            # in this case the range is intended as a percentage change
            new_pars *= x
        else:
            new_pars[index_to_vary] = x
        pars_ref.set_value(new_pars)
        current_gate = _net.get_current_gate()
        fids = []
        for state, target_state in zip(states, target_states):
            out_state = (current_gate * state).ptrace(range(_net.num_system_qubits))
            fids.append(qutip.fidelity(out_state, target_state))
        fidelities[:, idx] = fids

    ax.plot(xs, fidelities.T)


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
        try:
            _net.parameters = T.set_subtensor(_net.parameters[index_to_vary], x)
        except AttributeError:
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
    logging.debug('Trying to load net from pickled file "{}"'.format(filename))
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    if 'J' in data:
        logging.debug('Old format detected. Trying to load using old rules')
        return _load_network_from_pickle_old(data)
    # otherwise we can just use `sympy_model`:
    net_data = data['net_data']
    opt_data = data['optimization_data']
    # load ancillae state, if any
    ancillae_state = net_data.get('ancillae_state', None)
    if ancillae_state is not None:
        num_ancillae = int(np.log2(ancillae_state.shape[0]))
    else:
        num_ancillae = 0
    logging.info('Number of ancillae: {}'.format(num_ancillae))
    # Deduce number of qubits from saved model. The model is always saved in
    # `sympy_model`, but it may NOT be a sympy object. This is because for more
    # than 4-5 qubits converting into a corresponding sympy matrix gets very
    # expensive. Instead, alternative equivalent representations are used.
    # If not a sympy.Matrix object, the model should be a tuple.
    if isinstance(net_data['sympy_model'], sympy.Matrix):
        logging.debug('Model saved using sympy.Matrix object')
        num_qubits = int(np.log2(net_data['sympy_model'].shape[0]))
    # else, we assume the content of 'sympy_model' is a tuple with structure
    # (parameters, matrices)
    else:
        logging.debug('Model saved using efficient sympy style')
        num_qubits = int(np.log2(net_data['sympy_model'][1][0].shape[0]))

    model = QubitNetworkGateModel(
        sympy_expr=net_data['sympy_model'],
        target_gate=opt_data['target_gate'],
        free_parameters_order=net_data['free_parameters'],
        initial_values=opt_data['final_interactions'],
        num_system_qubits=num_qubits - num_ancillae
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
        if self._data is None:
            self._load()
        if hasattr(self._data, value):
            return getattr(self._data, value)
        else:
            raise AttributeError('No attribute with name "{}" in NetDataFile'
                                 ', nor in its data dict.'.format(value))

    def _load(self):
        """
        Read data from the stored path and save it into `self._data`.

        If the data was already loaded, it is loaded again.
        """
        self._data, self._opt_data = load_network_from_file(self.path)
        if self._data is None:
            raise ValueError('Something went wrong, unable to load data from'
                              ' the file "{}"'.format(self.path))

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
        # give warning if _opt_data is None, which probably means this is an
        # old style net, for which no optimization data has been saved.
        if self._opt_data is None:
            import warnings
            warnings.warn('You tried to access `_opt_data`, but its value has '
                          'not been set. This can mean that for this net no '
                          'optimization data was saved.')
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
        interactions_names, values = self.free_parameters, self.parameters.get_value()
        # now put everything in dataframe
        return pd.DataFrame({
            'interaction': [symb.name for symb in interactions_names],
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
        self.files = []
        self.nets = []
        self.paths = []
        if isinstance(path_or_files, str):
            self._load_nets_from_list([path_or_files])
        elif isinstance(path_or_files, (list, tuple)):
            self._load_nets_from_list(path_or_files)

    def _add_net_from_file(self, file_):
        # check that `file` is an actual file
        if not os.path.isfile(file_):
            raise ValueError('"{}" is not a valid path.'.format(file))
        logging.info('Loading net from "{}".'.format(file_))
        self.files.append(file_)
        self.nets.append(NetDataFile(file_))

    def _load_from_dir(self, path):
        # ensure tha path is of the form 'whatever/'
        if path[-1] != '/':
            path += '/'
        # raise error if path is not a directory
        if not os.path.isdir(path):
            raise ValueError('path must be a valid directory.')
        self.paths.append(path)
        # for each data file associate a `NetDataFile` object
        for file_ in glob.glob(path + '*.pickle'):
            self._add_net_from_file(file_)

    def _load_nets_from_list(self, paths_list):
        for path in paths_list:
            if os.path.isfile(path):
                self._add_net_from_file(path)
            else:
                self._load_from_dir(path)

    def __repr__(self):
        return self._repr_dataframe().__repr__()

    def _repr_html_(self):
        return self._repr_dataframe()._repr_html_()

    def _repr_dataframe(self, sort=True):
        # names = [net.name for net in self.nets]
        # target_gates = [net.get_target_gate() for net in self.nets]
        # load sorted data in pandas DataFrame
        pd.set_option('display.max_colwidth', -1)
        df = pd.DataFrame({
            'paths': self.files
            # 'target gates': target_gates,
            # 'names': names
        })
        # return formatted string
        if sort:
            return df.sort_values(by=['paths']).reset_index(drop=True)
        else:
            return df

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
        
        The returned object is a new `NetsDataFolder` instance.
        Simple wildcard matching is provided by `fnmatch.filter`.
        """
        logging.info('Filtering using pattern: "{}".'.format(pat))
        filtered_files = fnmatch.filter(self.files, '*/' + pat)
        new_data = NetsDataFolder(filtered_files)
        if len(new_data.files) == 0:
            print('No matching files.')
        return new_data

    def get_filenames(self):
        return [os.path.splitext(name)[0] for name in self.files]

    def reload(self):
        self = NetsDataFolder(self.path)
        return self

    def view_fidelities(self, n_samples=40, num_iterations=True,
                        num_system_qubits=True, num_ancillae=True):
        """Display nets in folder, each with an estimate of the fidelity.
        """
        data = self._repr_dataframe(sort=False)
        logging.info('Computing fidelities:')
        # compute fidelities
        fids = []
        bar = progressbar.ProgressBar()
        for net in bar(self.nets):
            fids.append(net.fidelity_test(n_samples=n_samples))
        # add other information on top of fidelities, if requested
        other_fields = []
        if num_iterations:
            if net._opt_data is not None:
                iterations = [net.opt_data['log']['fidelities'].shape[0]
                              for net in self.nets]
                iterations = pd.Series(iterations, name='num iterations')
            else:
                iterations = pd.Series(['???' for _ in self.nets],
                                       name='num iterations')
            other_fields.append(iterations)
        if num_system_qubits:
            other_fields.append(pd.Series(
                [net.num_system_qubits for net in self.nets],
                name='num system qubits'
            ))
        if num_ancillae:
            other_fields.append(pd.Series(
                [net.num_qubits - net.num_system_qubits for net in self.nets],
                name='num ancillae'
            ))
        # produce final dataframe to return
        data = pd.concat((
            data,
            pd.Series(fids, name='fidelity'),
            *other_fields
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

    def plot_parameters(self, connectgaps=True, hlines=[], return_fig=False,
                        marker_size=6):
        """
        Plot an overlay scatter plot of all the nets using plotly.
        """
        import plotly.graph_objs as go
        # retrieve data to plot
        data = self.view_parameters()
        fids = data.columns
        data_cols = data.values.T
        # make trace object
        traces = []
        for trace_idx, data_col in enumerate(data_cols):
            trace = go.Scatter(
                x = data.index,
                y = data_col,
                mode='lines+markers',
                marker=dict(size=marker_size),
                connectgaps=True,
                name=fids[trace_idx]
            )
            if not connectgaps:
                trace.update({'connectgaps': False})
            traces.append(trace)
        # fig = data.iplot(mode='lines+markers', size=6, asFigure=True)
        # put overlay hlines
        if len(hlines) > 0:
            from .plotly_utils import hline
            fig.layout.shapes = hline(0, len(data) - 1,
                                      hlines, dash='dash')
        # finally draw the damn thing
        if return_fig:
            return traces
        import plotly.offline
        plotly.offline.iplot(traces)
