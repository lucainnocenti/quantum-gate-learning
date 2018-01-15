import os
import pandas as pd
import numpy as np
import qutip

import theano
import theano.tensor as T
import theano.tensor.slinalg

import matplotlib.pyplot as plt
import seaborn as sns

from .model import QubitNetworkGateModel


def _gradient_updates_momentum(params, grad, learning_rate, momentum):
    """
    Compute updates for gradient descent with momentum

    Parameters
    ----------
    cost : theano.tensor.var.TensorVariable
        Theano cost function to minimize
    params : list of theano.tensor.var.TensorVariable
        Parameters to compute gradient against
    learning_rate : float
        Gradient descent learning rate
    momentum : float
        Momentum parameter, should be at least 0 (standard gradient
        descent) and less than 1

    Returns
    -------
    updates : list
        List of updates, one for each parameter
    """
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


def _gradient_updates_adadelta(params, grads):
    eps = 1e-6
    rho = 0.95
    # initialize needed shared variables
    def shared_from_var(p):
        return theano.shared(
            p.get_value() * np.asarray(0, dtype=theano.config.floatX))
    zgrads = shared_from_var(params)
    rparams2 = shared_from_var(params)
    rgrads2 = shared_from_var(params)

    zgrads_update = (zgrads, grads)
    rgrads2_update = (rgrads2, rho * rgrads2 + (1 - rho) * grads**2)

    params_step = -T.sqrt(rparams2 + eps) / T.sqrt(rgrads2 + eps) * zgrads
    rparams2_update = (rparams2, rho * rparams2 + (1 - rho) * params_step**2)
    params_update = (params, params - params_step)

    updates = (zgrads_update, rgrads2_update, rparams2_update, params_update)
    return updates


class Optimizer:
    """
    Main object handling the optimization of a `QubitNetwork` instance.

    Parameters
    ----------
    net : object or string
        Object representing the qubit network to be trained. If a string
        is given the object is loaded from fileusing `Optimizer._load_net`.
    learning_rate : float
        Initial learning rate for the training. The value of the learning
        rate will usually (depending on the training method) be adapted
        during training.
    decay_rate : float
        Determines the rate at which the learning rate decreases for
        each epoch.
    training_dataset_size
    test_dataset_size
    batch_size
    n_epochs
    target_gate
    sgd_method

    Attributes
    ----------
    net : some subclass of QubitNetworkModel
    hyperpars : dict
        Contains all the hyperparameters of the model.
    vars : dict of theano objects
        Contains the theano objects used in the fidelity graph.
    cost
    grad
    train_model
    test_model
    updates
    log
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, net,
                 learning_rate=None, decay_rate=None,
                 training_dataset_size=10,
                 test_dataset_size=10,
                 batch_size=None,
                 n_epochs=None,
                 target_gate=None,
                 sgd_method='momentum'):
        # the net parameter can be a QubitNetwork object or a str
        self.net = Optimizer._load_net(net)
        self.net.target_gate = target_gate
        self.hyperpars = dict(
            train_dataset_size=training_dataset_size,
            test_dataset_size=test_dataset_size,
            batch_size=batch_size,
            n_epochs=n_epochs,
            sgd_method=sgd_method,
            initial_learning_rate=learning_rate,
            decay_rate=decay_rate
        )
        # self.vars stores the shared variables for the computation
        def _sharedfloat(arr, name):
            return theano.shared(np.asarray(
                arr, dtype=theano.config.floatX), name=name)
        inputs_length = 2 * 2**self.net.num_qubits
        outputs_length = 2 * self.net.outputs_size
        self.vars = dict(
            index=T.lscalar('minibatch index'),
            learning_rate=_sharedfloat(learning_rate, 'learning rate'),
            train_inputs=_sharedfloat(
                np.zeros((training_dataset_size, inputs_length)),
                'training inputs'),
            train_outputs=_sharedfloat(
                np.zeros((training_dataset_size, outputs_length)),
                'training outputs'),
            test_inputs=_sharedfloat(
                np.zeros((test_dataset_size, inputs_length)),
                'test inputs'),
            test_outputs=_sharedfloat(
                np.zeros((test_dataset_size, outputs_length)),
                'test outputs'),
            parameters=self.net.parameters
        )
        self.cost = self.net.fidelity()
        self.cost.name = 'mean fidelity'
        self.grad = T.grad(cost=self.cost, wrt=self.vars['parameters'])
        self.train_model = None  # to be assigned in `compile_model`
        self.test_model = None  # assigned in `compile_model`
        # define updates, to be performed at every call of `train_XXX`
        self.updates = self._make_updates(sgd_method)
        # initialize log to be filled with the history later
        self.log = {'fidelities': None, 'parameters': None}
        # create figure object
        self._fig = None
        self._ax = None

    @classmethod
    def load(cls, file):
        """Load from saved file."""
        import pickle
        _, ext = os.path.splitext(file)
        if ext != '.pickle':
            raise NotImplementedError('Only pickle files for now!')
        with open(file, 'rb') as f:
            data = pickle.load(f)
        net_data = data['net_data']
        opt_data = data['optimization_data']
        # create QubitNetwork instance
        num_qubits = np.log2(net_data['sympy_model'].shape[0]).astype(int)
        if net_data['ancillae_state'] is None:
            num_system_qubits = num_qubits
        else:
            raise NotImplementedError('WIP')
        net = QubitNetworkGateModel(
            num_qubits=num_qubits,
            num_system_qubits=num_system_qubits,
            free_parameters_order=net_data['free_parameters'],
            sympy_expr=net_data['sympy_model'],
            initial_values=opt_data['final_interactions'])
        # call __init__ to create `Optimizer` instance
        hyperpars = opt_data['hyperparameters']
        optimizer = cls(
            net,
            learning_rate=hyperpars['initial_learning_rate'],
            decay_rate=hyperpars['decay_rate'],
            training_dataset_size=hyperpars['train_dataset_size'],
            test_dataset_size=hyperpars['test_dataset_size'],
            batch_size=hyperpars['batch_size'],
            n_epochs=hyperpars['n_epochs'],
            sgd_method=hyperpars['sgd_method'],
            target_gate=opt_data['target_gate'])
        optimizer.log = opt_data['log']
        optimizer.initial_parameters_values = opt_data['initial_interactions']
        return optimizer

    @staticmethod
    def _load_net(net):
        """
        Parse the `net` parameter given during init of `Optimizer`.
        """
        if isinstance(net, str):
            raise NotImplementedError('To be reimplemented')
        return net

    def _get_meaningful_history(self):
        fids = self.log['fidelities']
        # we cut from the history the last contiguous block of
        # values that are closer to 1 than `eps`
        eps = 1e-10
        try:
            end_useful_log = np.diff(np.abs(1 - fids) < eps).nonzero()[0][-1]
        # if the fidelity didn't converge to 1 the above raises an
        # IndexError. We then look to remove all the trailing zeros
        except IndexError:
            try:
                end_useful_log = np.diff(fids == 0).nonzero()[0][-1]
            # if also the above doesn't work, we just return the whole thing
            except IndexError:
                end_useful_log = len(fids)
        saved_log = dict()
        saved_log['fidelities'] = fids[:end_useful_log]
        if self.log['parameters'] is not None:
            saved_log['parameters'] = self.log['parameters'][:end_useful_log]
        return saved_log

    def save_results(self, file, overwrite=False):
        """Save optimization results.

        The idea is here to save all the information required to
        reproduce a given training session.
        """
        if os.path.isfile(file) and not overwrite:
            raise FileExistsError('File already exists. Use the `overwrite`'
                                  ' switch to overwrite existing file.')
        net_data = dict(
            sympy_model=self.net.get_matrix(),
            free_parameters=self.net.free_parameters,
            ancillae_state=self.net.ancillae_state
        )
        optimization_data = dict(
            target_gate=self.net.target_gate,
            hyperparameters=self.hyperpars,
            initial_interactions=self.net.initial_values,
            final_interactions=self._get_meaningful_history()['parameters'][-1]
        )
        # cut redundant log history
        optimization_data['log'] = self._get_meaningful_history()
        # prepare and finally save to file
        data_to_save = dict(
            net_data=net_data, optimization_data=optimization_data)
        _, ext = os.path.splitext(file)
        if ext == '.pickle':
            import pickle
            with open(file, 'wb') as fp:
                pickle.dump(data_to_save, fp)
            print('Successfully saved to {}'.format(file))
        else:
            raise ValueError('Only saving to pickle is supported.')


    def _make_updates(self, sgd_method):
        """Return updates, for `train_model` and `test_model`."""
        assert isinstance(sgd_method, str)
        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs
        if sgd_method == 'momentum':
            momentum = 0.5
            learn_rate = self.vars['learning_rate']
            updates = _gradient_updates_momentum(
                self.vars['parameters'], self.grad,
                learn_rate, momentum)
        elif sgd_method == 'adadelta':
            updates = _gradient_updates_adadelta(
                self.vars['parameters'], self.grad)
        else:
            new_pars = self.vars['parameters']
            new_pars += self.vars['learning_rate'] * self.grad
            updates = [(self.vars['parameters'], new_pars)]
        return updates

    def _update_fig(self, len_shown_history):
        # retrieve or create figure object
        if self._fig is None:
            self._fig, self._ax = plt.subplots(1, 1, figsize=(10, 5))
        fig, ax = self._fig, self._ax
        ax.clear()
        # plot new fidelities
        n_epoch = self.log['n_epoch']
        fids = self.log['fidelities']
        if len_shown_history is None:
            ax.plot(fids[:n_epoch], '-b', linewidth=1)
        else:
            if n_epoch + 1 == len_shown_history:
                x_coords = np.arange(
                    n_epoch - len_shown_history + 1, n_epoch + 1)
            else:
                x_coords = np.arange(n_epoch + 1)
            ax.plot(x_coords, fids[x_coords], '-b', linewidth=1)
        plt.suptitle('learning rate: {}\nfidelity: {}'.format(
            self.vars['learning_rate'].get_value(), fids[n_epoch]))
        fig.canvas.draw()

    def refill_test_data(self):
        """Generate new test data and put them in shared variable.
        """
        inputs, outputs = self.net.generate_training_states(
            self.hyperpars['test_dataset_size'])
        self.vars['test_inputs'].set_value(inputs)
        self.vars['test_outputs'].set_value(outputs)

    def refill_training_data(self):
        """Generate new training data and put them in shared variable.
        """
        inputs, outputs = self.net.generate_training_states(
            self.hyperpars['train_dataset_size'])
        self.vars['train_inputs'].set_value(inputs)
        self.vars['train_outputs'].set_value(outputs)

    def train_epoch(self):
        """Generate training states and train for an epoch."""
        self.refill_training_data()
        n_train_batches = (self.hyperpars['train_dataset_size'] //
                           self.hyperpars['batch_size'])
        for minibatch_index in range(n_train_batches):
            self.train_model(minibatch_index)

    def test_epoch(self, save_parameters=True):
        """Compute fidelity, and store fidelity and parameters."""
        fidelity = self.test_model()
        n_epoch = self.log['n_epoch']
        if save_parameters:
            self.log['parameters'][n_epoch] = (
                self.vars['parameters'].get_value())
        self.log['fidelities'][n_epoch] = fidelity

    def _compile_model(self):
        """Compile train and test models.

        Compile the training function `train_model`, that while computing
        the cost at every iteration (batch), also updates the weights of
        the network based on the rules defined in `updates`.
        """
        batch_size = self.hyperpars['batch_size']
        batch_start = self.vars['index'] * batch_size
        batch_end = (self.vars['index'] + 1) * batch_size
        train_inputs_batch = self.vars['train_inputs'][batch_start: batch_end]
        train_outputs_batch = self.vars['train_outputs'][batch_start: batch_end]
        print('Compiling model ...', end='')
        self.train_model = theano.function(
            inputs=[self.vars['index']],
            outputs=self.cost,
            updates=self.updates,
            givens={
                self.net.inputs: train_inputs_batch,
                self.net.outputs: train_outputs_batch
            })
        # `test_model` is used to test the fidelity given by the currently
        # trained parameters. It's called at regular intervals during
        # the computation, and is the value shown in the dynamically
        # updated plot that is shown when the training is ongoing.
        self.test_model = theano.function(
            inputs=[],
            outputs=self.cost,
            updates=None,
            givens={self.net.inputs: self.vars['test_inputs'],
                    self.net.outputs: self.vars['test_outputs']})
        print(' done.')

    def _run(self, save_parameters=True, len_shown_history=200):
        # generate testing states
        self.refill_test_data()
        # now let's prepare the theano graph
        self._compile_model()
        # at the end of each epoch, the current (estimated) fidelity is
        # displayed, and a new set of training samples is generated
        n_epochs = self.hyperpars['n_epochs']
        # initialize log. Note that this initialisation means that if the
        # training stops before all the epochs are processed, then the
        # 'fidelities' and 'parameters' arrays will contain tails of
        # not meaningful zeros
        self.log['fidelities'] = np.zeros(n_epochs)
        if save_parameters:
            self.log['parameters'] = np.zeros((
                n_epochs, len(self.vars['parameters'].get_value())))
        # run epochs
        for n_epoch in range(n_epochs):
            self.log['n_epoch'] = n_epoch
            self.train_epoch()
            self.test_epoch(save_parameters=save_parameters)
            self._update_fig(len_shown_history)
            # stop if fidelity 1 is obtained
            if self.log['fidelities'][n_epoch] == 1:
                print('Fidelity 1 obtained, stopping.')
                break
            # update learning rate
            self.vars['learning_rate'].set_value(
                self.hyperpars['initial_learning_rate'] / (
                    1 + self.hyperpars['decay_rate'] * n_epoch))

    def run(self, save_parameters=True, len_shown_history=200,
            save_after=None):
        """
        Start the optimization.

        Parameters
        ----------
        save_parameters : bool, optional
            If True, the entire history of the parameters is stored.
        len_shown_history : int, optional
            If not None, the figure showing the fidelity for every epoch
            only shows the last `len_shown_history` epochs.
        save_after : str, optional
            If not None, it is used to save the results to file.
        """
        args = locals()
        # catch abort to stop training at will
        try:
            self._run(args)
        except KeyboardInterrupt:
            pass

        if save_after is not None:
            self._save_results()

    def plot_parameters_history(self, return_fig=False, return_df=False):
        import cufflinks
        names = [par.name for par in self.net.free_parameters]
        df = pd.DataFrame(self._get_meaningful_history()['parameters'])
        initial_values = pd.DataFrame([self.initial_parameters_values])
        df = pd.concat([initial_values, df], ignore_index=True)
        new_col_names = dict(zip(range(df.shape[1]), names))
        df.rename(columns=new_col_names, inplace=True)
        if return_df:
            return df

        fig = df.iplot(asFigure=True)
        if return_fig:
            return fig
        import plotly
        # just plotting offline at the moment
        plotly.offline.iplot(fig)
