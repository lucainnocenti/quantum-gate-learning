import numpy as np
import matplotlib.pyplot as plt
# import qutip
import theano
import theano.tensor as T
from QubitNetwork import QubitNetwork


def load_network_from_file(infile):
    """Returns a QubitNetwork object created from the file `infile`.

    The QubitNetwork objects should have been stored into the file in
    pickle format, using the appropriate `save_to_file` method.
    """
    import pickle
    with open(infile, 'rb') as file:
        data = pickle.load(file)

    # the interface was recently changed and `active_hs` and `active_Js`
    # aren't class attributes anymore. Instead the interactions are all
    # stored into the `QubitNetwork.interactions` attribute.
    # Nevertheless some networks were saved with the old interace so
    # we need to handle these cases appropriately
    if 'active_Js' in data.keys():
        interactions = []
        for qubit, dirs in data['active_hs'].items():
            for d in dirs:
                interactions.append((qubit, d))

        for pair, dirs in data['active_Js'].items():
            for d in dirs:
                interactions.append((pair, d))
    else:
        interactions = data['interactions']

    net = QubitNetwork(
        num_qubits=data['num_qubits'],
        interactions=interactions,
        system_qubits=data['num_system_qubits'],
        J=data['J']
    )
    return net


def transfer_J_values(source_net, target_net):
    source_J = source_net.J.get_value()
    target_J = target_net.J.get_value()
    target_interactions = target_net.get_all_interactions()

    for idx, J in enumerate(source_J):
        interaction = source_net.J_index_to_interaction(idx)
        # print(interaction)
        # if `interaction` is active in `target_net`, then we transfer
        # its value from `source_net` to `target_net`.
        if interaction in target_interactions:
            target_idx = target_net.tuple_to_J_index(interaction)
            target_J[target_idx] = J

    target_net.J.set_value(target_J)


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
    print(isinstance(net, QubitNetwork))
    if net is None:
        _net = QubitNetwork(num_qubits=4,
                            interactions=('all', ['xx', 'yy', 'zz']),
                            self_interactions=('all', ['x', 'y', 'z']),
                            system_qubits=[0, 1, 2])
    elif isinstance(net, QubitNetwork):
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
