"""Functions to process and modify content of `QubitNetwork` objects.

This module contains a list of functions specifically aimed to work with
`QubitNetwork` objects. It differs from `net_analysis_tools` in that the
methods in this module write or modify data saved in `QubitNetwork` objects,
rather than just reading and analysing it.
"""

import collections
import os

import numpy as np
import pandas as pd

import qutip
import theano
import theano.tensor as T

# package imports
from .QubitNetwork import QubitNetwork
from .net_analysis_tools import load_network_from_file
from .model import FidelityGraph, _gradient_updates_momentum, Optimizer
from IPython.core.debugger import set_trace


def transfer_J_values(source_net, target_net):
    """
    Transfer the values of the interactions from source to target net.

    All the interactions corresponding to the `J` values of `source_net`
    are checked, and those interactions that are also active in
    `target_net` are copied into `target_net`.
    """
    source_J = source_net.J.get_value()
    target_J = target_net.J.get_value()
    target_interactions = target_net.interactions

    for idx, J in enumerate(source_J):
        interaction = source_net.J_index_to_interaction(idx)
        # print(interaction)
        # if `interaction` is active in `target_net`, then we transfer
        # its value from `source_net` to `target_net`.
        if interaction in target_interactions:
            target_idx = target_net.tuple_to_J_index(interaction)
            target_J[target_idx] = J

    target_net.J.set_value(target_J)


def sgd_optimization(
        net=None,
        learning_rate=0.13,
        n_epochs=100,
        batch_size=100,
        backup_file=None,
        saveafter_file=None,
        training_dataset_size=1000,
        test_dataset_size=1000,
        target_gate=None,
        decay_rate=0.1,
        # plot_errors=False,
        truncate_fidelity_history=200,
        SGD_method='momentum'):
    """Start the MBSGD training on the net.

    Parameters
    ----------
    net : QubitNetwork object or str
        The qubit network to be trained.

        If a string is given, it is assumed to be the path of some
        pre-saved net. When this is the case, the net is loaded from the
        given path, and the training started from the interaction
        parameters of the loaded net. At the end of the training, the
        resulting net is also automatically saved in the same file.

        If no value for `net` is given, a default 3 qubits + 1 ancilla
        network with all interactions on is assumed.
    learning_rate : float
        Specifies the rate of change of the parameters at each iteration
        of gradient descent. This rate is usually not fixed anyway, so
        that the `learning_rate` parameter only gives the initial value
        of the actual learning rate that will be used during training.
    n_epochs : int
        For every epoch, the number of gradient descent iterations is
        given by the number of training states divided by `batch_size`.
        At every such iteration the fidelity is computed taking the mean
        fidelity over `batch_size` training states.
        At the end of every epoch is also when a new point is added to
        the dynamical plot recording the progression of the training,
        and a new set of training states is generated.
    batch_size : int
        The number of training states to use at every iteration.
    backup_file : str
        If given, it is assumed to be a valid path, and the net will be
        saved on this path *before* the training procedure takes place.
        Equivalent to just save the net manually with `save_to_file`
        before calling `sgd_optimization`.
    saveafter_file : str
        If given, it is assumed to be a valid path, where to save the
        trained net at the end of the training. It doesn't do anything
        if `net` is given as a string, as in that case the default
        behaviour is to save the net at the end of the training in the
        same file from which it was loaded.

        Note that the net is saved in `saveafter_file` even if the
        training is manually aborted before its natural end.
    training_dataset_size : int
        blablalba

    plot_errors : bool (YET TO IMPLEMENT)
        If True, at every epoch the difference between max and min
        fidelities is reported.
    """
    # parse `backup_file` parameter
    if isinstance(backup_file, str):
        # we will assume that it is the path where to backup the net
        # BEFORE the training takes place, in case anything bad happens
        _net.save_to_file(backup_file)
        print('Network backup saved in {}'.format(backup_file))

    # build model
    model = FidelityGraph(_net.num_qubits, _net.num_system_qubits,
                          *_net.build_theano_graph(), _net.target_gate)
    # initialize optimizer
    optimizer = Optimizer(model, learning_rate=1.,
                          training_dataset_size=100,
                          test_dataset_size=100,
                          batch_size=10)

    # -------- DO THE ACTUAL MAXIMIZATION --------

    print('Let\'s roll!')
    # The try-except block allows to stop the computation with ctrl-C
    # without losing all the computed data. This effectively makes it
    # possible to stop the computation at any moment saving all the data
    try:
        _run_optimization(
            optimizer, n_epochs,
            truncate_fidelity_history=truncate_fidelity_history,
            decay_rate=decay_rate
        )
    except KeyboardInterrupt:
        pass

    print('Finished training')
    print('Final fidelity: ', end='')
    print(optimizer.test_model())

    # save results if appropriate parameters have been given
    if isinstance(net, str):
        _net.save_to_file(net)
        print('Network saved in {}'.format(net))
    elif saveafter_file is not None:
        _net.save_to_file(saveafter_file)
        print('Network saved in {}'.format(saveafter_file))


    # if precompiled_functions is None:
    #     return _net
    # else:
    #     return _net, (train_model, test_model)
