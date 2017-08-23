"""Functions to process and modify content of `QubitNetwork` objects.

This module contains a list of functions specifically aimed to work with
`QubitNetwork` objects. It differs from `net_analysis_tools` in that the
methods in this module write or modify data saved in `QubitNetwork` objects,
rather than just reading and analysing it.
"""

import pickle
import collections

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import qutip
import theano
import theano.tensor as T

# package imports
from .QubitNetwork import QubitNetwork


def load_network_from_file(infile):
    """Returns a QubitNetwork object created from the file `infile`.

    The QubitNetwork objects should have been stored into the file in
    pickle format, using the appropriate `save_to_file` method.
    """

    with open(infile, 'rb') as file:
        data = pickle.load(file)

    if 'target_gate' not in data.keys():
        data['target_gate'] = None

    if 'net_topology' not in data.keys():
        data['net_topology'] = None

    if 'ancillae_state' not in data.keys():
        num_ancillae = data['num_qubits'] - data['num_system_qubits']
        data['ancillae_state'] = qutip.tensor(
            [qutip.basis(2, 0) for _ in range(num_ancillae)])

    net = QubitNetwork(
        num_qubits=data['num_qubits'],
        interactions=data['interactions'],
        system_qubits=data['num_system_qubits'],
        ancillae_state=data['ancillae_state'],
        target_gate=data['target_gate'],
        net_topology=data['net_topology'],
        J=data['J'])
    return net


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


def _gradient_updates_momentum(params, grad, learning_rate, momentum):
    '''
    Compute updates for gradient descent with momentum

    :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of theano.tensor.var.TensorVariable
            Parameters to compute gradient against
        - learning_rate : float
            Gradient descent learning rate
        - momentum : float
            Momentum parameter, should be at least 0 (standard gradient
            descent) and less than 1

    :returns:
        updates : list
            List of updates, one for each parameter
    '''
    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0
    # List of update steps for each parameter
    updates = []
    if not isinstance(params, list):
        params = [params]
    # Just gradient descent on cost
    for param in params:
        # For each parameter, we'll create a previous_step shared variable.
        # This variable keeps track of the parameter's update step
        # across iterations. We initialize it to 0
        previous_step = theano.shared(
            param.get_value() * 0., broadcastable=param.broadcastable)
        step = momentum * previous_step + learning_rate * grad
        # Add an update to store the previous step value
        updates.append((previous_step, step))
        # Add an update to apply the gradient descent step to the
        # parameter itself
        updates.append((param, param + step))
    return updates


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
        precompiled_functions=None,
        print_fidelity=False,
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

    # -------- OPTIONS PARSING --------

    # Parse the `net` parameter.
    # `net` is the argument obtained from the interface,
    # `_net` is the variable used in the function (usually derived from `net`)
    _net = None
    if net is None:
        _net = QubitNetwork(num_qubits=4, system_qubits=3, interactions='all')
    elif isinstance(net, str):
        # assume `net` is the path where the network was stored
        _net = load_network_from_file(net)

    if _net is None:
        _net = net

    # parse `target_gate` parameter
    if target_gate is None:
        if _net.target_gate is None:
            raise ValueError('`target_gate` must have a value.')
        else:
            target_gate = _net.target_gate
    else:
        _net.target_gate = target_gate

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

    # -------- DATA GENERATION AND PREPARATION --------

    print('Generating training data...')

    # `generate_training_data` outputs a pair, the first element of which
    # is a list of states spanning the system qubits, while the second
    # element is a list of states spanning only
    dataset = _net.generate_training_data(target_gate, training_dataset_size)
    states = theano.shared(np.asarray(dataset[0], dtype=theano.config.floatX))
    target_states = theano.shared(
        np.asarray(dataset[1], dtype=theano.config.floatX))

    test_dataset = _net.generate_training_data(target_gate, test_dataset_size)
    test_states = theano.shared(
        np.asarray(test_dataset[0], dtype=theano.config.floatX))
    test_target_states = theano.shared(
        np.asarray(test_dataset[1], dtype=theano.config.floatX))

    # -------- BUILD COMPUTATIONAL GRAPH FOR THE MBSGD --------

    print('Building the model...')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a minibatch
    # generate symbolic variables for input data and labels
    x = T.dmatrix('x')  # input state (data). Every row is a state vector
    y = T.dmatrix('y')  # output target state (label). As above

    _learning_rate = theano.shared(
        np.asarray(learning_rate, dtype=theano.config.floatX),
        name='learning_rate')

    # Define the cost function, that is, the fidelity. This is the
    # number we ought to maximize through the training.
    cost = _net.fidelity(x, y)
    # all_fidelities = _net.fidelity(x, y, return_mean=True)

    # compute the gradient of the cost
    g_J = T.grad(cost=cost, wrt=_net.J)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    if SGD_method == 'momentum':
        updates = _gradient_updates_momentum(_net.J, g_J, _learning_rate, 0.5)
    else:
        raise ValueError('SGD_method has an invalid value.')
    # updates = [(_net.J, _net.J + _learning_rate * g_J)]

    # Theoretically it should be possible to reuse already compiled
    # computational graph, but I didn't really test this functionality yet.
    if precompiled_functions is None:
        # compile the training function `train_model`, that while computing
        # the cost at every iteration (batch), also updates the weights of
        # the network based on the rules defined in `updates`.
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: states[index * batch_size:(index + 1) * batch_size],
                y: target_states[index * batch_size:(index + 1) * batch_size]
            })

        # `test_model` is used to test the fidelity given by the currently
        # trained parameters. It's called at regular intervals during
        # the computation, and is the value shown in the dynamically
        # update plot that is shown when the training is ongoing.
        test_model = theano.function(
            inputs=[],
            outputs=cost,
            updates=None,
            givens={x: test_states,
                    y: test_target_states})
    else:
        train_model, test_model = precompiled_functions

    # -------- DO THE ACTUAL MAXIMIZATION --------

    print('Let\'s roll!')
    n_train_batches = states.get_value().shape[0] // batch_size
    # fids_history = np.array([])
    if truncate_fidelity_history is None:
        fids_history = []
    else:
        fids_history = collections.deque(maxlen=truncate_fidelity_history)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # The try-except block allows to stop the computation with ctrl-C
    # without losing all the computed data. This effectively makes it
    # possible to stop the computation at any moment saving all the data
    # like it would have happened were the computation ended on its own.
    try:
        for n_epoch in range(n_epochs):
            if print_fidelity:
                print('Epoch {}, '.format(n_epoch), end='')

            # compute fidelity and update parameters
            for minibatch_index in range(n_train_batches):
                train_model(minibatch_index)

            # update fidelity history
            fids_history.append(test_model())
            if fids_history[-1] == 1:
                print('Fidelity 1 obtained, stopping.')
                break

            # new_fidelities = np.array(test_model())
            # new_fidelities = new_fidelities.reshape(
            #     [new_fidelities.shape[0], 1])
            # if n_epoch == 0:
            #     fids_history = new_fidelities
            # else:
            #     fids_history = np.concatenate(
            #         (fids_history, new_fidelities), axis=1)
            # print(new_variance)
            if print_fidelity:
                print(fids_history[-1])

            # if n_epoch > 0:
            #     # update plot
            #     sns.tsplot(fids_history, ci=100)
            #     # ax.plot(fids_history, '-b')
            #     plt.suptitle(('learning rate: {}\nfidelity: {}'
            #                   '\nmax - min: {}').format(
            #         _learning_rate.get_value(),
            #         np.mean(fids_history[:, -1]),
            #         np.ptp(fids_history[:, -1]))
            #     )
            #     fig.canvas.draw()
            if truncate_fidelity_history is None:
                ax.plot(fids_history, '-b')
            else:
                if len(fids_history) == truncate_fidelity_history:
                    x_coords = np.arange(
                        n_epoch - truncate_fidelity_history + 1, n_epoch + 1)
                else:
                    x_coords = np.arange(len(fids_history))

                ax.clear()
                ax.plot(x_coords, fids_history, '-b')
            plt.suptitle('learning rate: {}\nfidelity: {}'.format(
                _learning_rate.get_value(), fids_history[-1]))
            fig.canvas.draw()

            # update learning rate
            _learning_rate.set_value(learning_rate /
                                     (1 + decay_rate * n_epoch))

            # generate a new set of training states
            dataset = _net.generate_training_data(target_gate,
                                                  training_dataset_size)
            states.set_value(dataset[0])
            target_states.set_value(dataset[1])
    except KeyboardInterrupt:
        pass

    print('Finished training')
    print('Final fidelity: ', end='')
    print(_net.test_fidelity())

    # save results if appropriate parameters have been given
    conditionally_save()

    # if precompiled_functions is None:
    #     return _net
    # else:
    #     return _net, (train_model, test_model)
