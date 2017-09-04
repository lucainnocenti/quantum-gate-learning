import os
import numpy as np
import qutip

import theano
import theano.tensor as T

import matplotlib.pyplot as plt
import seaborn as sns

from .utils import complex2bigreal


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


def _split_bigreal_ket(ket):
    """Splits in half a real vector of length 2N

    Given an input ket vector in big real form, returns a pair of real
    vectors, the first containing the first N elements, and the second
    containing the last N elements.
    """
    ket_real = ket[:ket.shape[0] // 2]
    ket_imag = ket[ket.shape[0] // 2:]
    return ket_real, ket_imag


def _ket_to_dm(ket):
    """Builds theano function to convert kets in dms in big real form.

    The input is expected to be a 1d real array, storing the state
    vector as `(psi_real, psi_imag)`, where `psi` is the complex vector.
    The outputs are real and imaginary part of the corresponding density
    matrix.
    """
    ket_real, ket_imag = _split_bigreal_ket(ket)[:, None]

    dm_real = ket_real * ket_real.T + ket_imag * ket_imag.T
    dm_imag = ket_imag * ket_real.T - ket_real * ket_imag.T
    return dm_real, dm_imag


def _compute_fidelities_col_fn(col_idx, row_idx, matrix, num_ancillae):
    """
    `_compute_fidelities_col_fn` and `(...)row_fn` are the functions that
    handle the computation of the partial traces. The latter is run on
    each block of rows of the matrix to partial trace, with each block
     containing `2 ** num_ancillae` rows.
    For each block of rows, the former scans through the corresponding
    blocks of columns, taking the trace of each resulting submatrix.
    """
    subm_dim = 2**num_ancillae
    return T.nlinalg.trace(matrix[row_idx * subm_dim:(row_idx + 1) * subm_dim,
                                  col_idx * subm_dim:(col_idx + 1) * subm_dim])


def _compute_fidelities_row_fn(row_idx, matrix, num_ancillae):
    """See `_compute_fidelities_col_fn`."""
    results, _ = theano.scan(
        fn=_compute_fidelities_col_fn,
        sequences=T.arange(matrix.shape[1] // 2**num_ancillae),
        non_sequences=[row_idx, matrix, num_ancillae])
    return results



def _fidelity_with_ptrace(i, matrix, target_states, num_ancillae):
    """
    Compute fidelity between target and obtained states.

    This function is intended to be called in `theano.scan` from
    `QubitNetwork.fidelity`, and it operates with `theano` "symbolic"
    tensor objects. It *does not* operate with numbers.

    Parameters
    ----------
    i : int
        Denotes the element to take from `matrix`. This is necessary
        because of how `theano.scan` works (it is not possible to just
        pass the corresponding matrix[i] element to the function).
    matrix : theano 2d array
        An array of training states. `matrix[i]` is the `i`-th training
        state after evolution through `exp(-1j * H)`, *in big real ket
        form*.
        In other words, `matrix` is the set of states obtained after
        the evolution through the net, to be compared with the
        corresponding set of training target states.
        `matrix[i]` has length `2 * (2 ** num_qubits)`, with
    target_states : theano 2d array
        The set of target states. `target_states[i]` is the state that
        we would like `matrix[i]` to be equal to.
    num_ancillae : int
        The number of ancillae in the network.
    """
    # - `dm_real` and `dm_imag` will be square matrices of length
    #   `2 ** num_qubits`.
    dm_real, dm_imag = _ket_to_dm(matrix[i])
    # `dm_real_traced` and `dm_imag_traced` are square matrices
    # of length `2 ** num_system_qubits`.
    dm_real_traced, _ = theano.scan(
        fn=_compute_fidelities_row_fn,
        sequences=T.arange(dm_real.shape[0] // 2**num_ancillae),
        non_sequences=[dm_real, num_ancillae])
    dm_imag_traced, _ = theano.scan(
        fn=_compute_fidelities_row_fn,
        sequences=T.arange(dm_imag.shape[0] // 2**num_ancillae),
        non_sequences=[dm_imag, num_ancillae])

    #  ---- Old method to compute trace of product of dms: ----
    # target_dm_real, target_dm_imag = _ket_to_dm(target_states[i])

    # prod_real = (T.dot(dm_real_traced, target_dm_real) -
    #              T.dot(dm_imag_traced, target_dm_imag))
    # tr_real = T.nlinalg.trace(prod_real)

    # # we need only take the trace of the real part of the product,
    # # as if \rho and \rho' are two Hermitian matrices, then
    # # Tr(\rho_R \rho'_I) = Tr(\rho_I \rho'_R) = 0.
    # return tr_real

    # ---- New method: ----
    target_real, target_imag = _split_bigreal_ket(target_states[i])

    # `psi` and `psi_tilde` have length 2 * (2 ** numSystemQubits)
    psi = target_states[i][:, None]
    psi_tilde = T.concatenate((-target_imag, target_real))[:, None]
    # `big_dm` is a square matrix with same length
    big_dm = T.concatenate((
        T.concatenate((dm_imag_traced, dm_real_traced), axis=1),
        T.concatenate((-dm_real_traced, dm_imag_traced), axis=1)
    ), axis=0)
    out_fidelity = psi.T.dot(big_dm).dot(psi_tilde)
    return out_fidelity


def _fidelity_no_ptrace(i, states, target_states):
    """
    Compute symbolic fidelity between `states[i]` and `target_states[i]`.

    Both `states[i]` and `target_states[i]` are real vectors of same
    length.
    """
    state = states[i]
    target_state = target_states[i]

    # state_real = state[:state.shape[0] // 2]
    # state_imag = state[state.shape[0] // 2:]
    # target_state_real = target_state[:target_state.shape[0] // 2]
    # target_state_imag = target_state[target_state.shape[0] // 2:]
    state_real, state_imag = _split_bigreal_ket(state)
    target_state_real, target_state_imag = _split_bigreal_ket(target_state)

    fidelity_real = (T.dot(state_real, target_state_real) +
                     T.dot(state_imag, target_state_imag))
    fidelity_imag = (T.dot(state_real, target_state_imag) -
                     T.dot(state_imag, target_state_real))
    fidelity = fidelity_real ** 2 + fidelity_imag ** 2
    return fidelity


class FidelityGraph:
    def __init__(self, num_qubits, num_system_qubits, parameters,
                 hamiltonian_model, target_gate, ancillae_state=None):
        # TODO: reduce attributes duplication, probably via binding the
        #       `QubitNetwork` instance here and using values from there
        self.num_qubits = num_qubits
        self.num_system_qubits = num_system_qubits
        self.parameters = parameters  # shared variable for parameters
        self.inputs = T.dmatrix('inputs')
        self.outputs = T.dmatrix('outputs')
        self.hamiltonian_model = hamiltonian_model
        if target_gate is not None:
            assert isinstance(target_gate, qutip.Qobj)
        self.target_gate = target_gate
        self.ancillae_state = ancillae_state

    @staticmethod
    def _fidelities_with_ptrace(output_states, target_states, num_ancillae):
        """Compute fidelities in the case of ancillary qubits.

        This function handles the case of the fidelity when the output
        states are *larger* than the target states. In this case the
        fidelity is computed taking the partial trace with respect to
        the ancillary degrees of freedom of the output, and taking the
        fidelity of the resulting density matrix with the target
        (pure) state.
        """
        num_states = output_states.shape[0]
        fidelities, _ = theano.scan(
            fn=_fidelity_with_ptrace,
            sequences=T.arange(num_states),
            non_sequences=[output_states, target_states, num_ancillae]
        )
        return fidelities

    @staticmethod
    def _fidelities_no_ptrace(output_states, target_states):
        """Compute fidelities when there are no ancillary qubits.
        """
        num_states = output_states.shape[0]
        fidelities, _ = theano.scan(
            fn=_fidelity_no_ptrace,
            sequences=T.arange(num_states),
            non_sequences=[output_states, target_states]
        )
        return fidelities

    def compute_evolution_matrix(self):
        """Compute matrix exponential of iH."""
        return T.slinalg.expm(self.hamiltonian_model)

    def _target_outputs_from_inputs_open_map(self, input_states):
        raise NotImplementedError('Not implemented yet')
        # Note that in case of an open map target, all target states are
        # density matrices, instead of just kets like they would when the
        # target is a unitary gate.
        target_states = []
        for psi in input_states:
            # the open evolution is implemented vectorizing density
            # matrices and maps: `A * rho * B` becomes
            # `unvec(vec(tensor(A, B.T)) * vec(rho))`.
            vec_dm_ket = qutip.operator_to_vector(qutip.ket2dm(psi))
            evolved_ket = self.target_gate * vec_dm_ket
            evolved_ket = qutip.vector_to_operator(evolved_ket)
            target_states.append(evolved_ket)
        return target_states

    def _target_outputs_from_inputs(self, input_states):
        # defer operation to other method for open maps
        if self.target_gate.issuper:
            return self._target_outputs_from_inputs_open_map(input_states)
        # unitary evolution of input states. `target_gate` is qutip obj
        return [self.target_gate * psi for psi in input_states]

    def generate_training_states(self, num_states):
        """Create training states for the training.

        This function generates every time it is called a set of
        input and corresponding target output states, to be used during
        training. These values will be used during the computation
        through the `givens` parameter of `theano.function`.

        Returns
        -------
        A tuple with two elements: training vectors and labels.
        NOTE: The training and target vectors have different lengths!
              The latter span the whole space while the former only the
              system one.

        training_states: an array of vectors.
            Each vector represents a state in the full system+ancilla space,
            in big real form. These states span the whole space simply
            out of convenience, but are obtained as tensor product of
            the target states over the system qubits with the initial
            states of the ancillary qubits.
        target_states: an array of vectors.
            Each vector represents a state spanning only the system qubits,
            in big real form. Every such state is generated by evolving
            the corresponding `training_state` through the matrix
            `target_unitary`.

        This generation method is highly non-optimal. However, it takes
        about ~250ms to generate a (standard) training set of 100 states,
        which amounts to ~5 minutes over 1000 epochs with a training dataset
        size of 100, making this factor not particularly important.
        """
        assert self.target_gate is not None, 'target_gate not set'

        # 1) Generate random input states over system qubits
        # `rand_ket_haar` seems to be sligtly faster than `rand_ket`
        length_inputs = 2 ** self.num_system_qubits
        qutip_dims = [[2 for _ in range(self.num_system_qubits)],
                      [1 for _ in range(self.num_system_qubits)]]
        training_inputs = [
            qutip.rand_ket_haar(length_inputs, dims=qutip_dims)
            for _ in range(num_states)
        ]
        # 2) Compute corresponding output states
        target_outputs = self._target_outputs_from_inputs(training_inputs)
        # 3) Tensor product of training input states with ancillae
        for idx, ket in enumerate(training_inputs):
            if self.num_system_qubits < self.num_qubits:
                ket = qutip.tensor(ket, self.ancillae_state)
            training_inputs[idx] = complex2bigreal(ket)
        training_inputs = np.asarray(training_inputs)
        # 4) Convert target outputs in big real form.
        # NOTE: the target states are kets if the target gate is unitary,
        #       and density matrices for target open maps.
        target_outputs = np.asarray(
            [complex2bigreal(st) for st in target_outputs])
        # return results as matrices
        _, len_inputs, _ = training_inputs.shape
        _, len_outputs, _ = target_outputs.shape
        training_inputs = training_inputs.reshape((num_states, len_inputs))
        target_outputs = target_outputs.reshape((num_states, len_outputs))
        return training_inputs, target_outputs

    def fidelity(self, return_mean=True):
        """Return theano graph for fidelity given training states.

        In the output theano expression `fidelities`, the tensors
        `output_states` and `target_states` are left "hanging", and will
        be replaced during the training through the `givens` parameter
        of `theano.function`.
        """
        output_states = T.tensordot(
            self.compute_evolution_matrix(), self.inputs, axes=([1], [1])).T
        num_ancillae = self.num_qubits - self.num_system_qubits
        if num_ancillae > 0:
            fidelities = self._fidelities_with_ptrace(
                output_states, self.outputs, num_ancillae)
        else:
            fidelities = self._fidelities_no_ptrace(output_states,
                                                    self.outputs)
        if return_mean:
            return T.mean(fidelities)
        else:
            return fidelities


def _sharedfloat(arr, name):
    return theano.shared(np.asarray(arr, dtype=theano.config.floatX), name=name)


class Optimizer:
    """
    Main object handling the optimization of a `QubitNetwork` instance.
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, net,
                 learning_rate=None, decay_rate=None,
                 training_dataset_size=None, test_dataset_size=None,
                 batch_size=None,
                 n_epochs=None,
                 target_gate=None,
                 sgd_method='momentum'):
        if batch_size is None:
            raise ValueError('Missing batch size value.')
        if learning_rate is None:
            raise ValueError('Missing value for `learning_rate`.')
        # the net parameter can be a QubitNetwork object or a str
        self.net = Optimizer._load_net(net)
        self.target_gate = target_gate
        self.model = FidelityGraph(
            self.net.num_qubits, self.net.num_system_qubits,
            *self.net.build_theano_graph(), self.target_gate
        )
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
        inputs_length = 2 * 2**self.net.num_qubits
        outputs_length = 2 * 2**self.net.num_system_qubits
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
            parameters=self.model.parameters
        )
        self.cost = self.model.fidelity()
        self.cost.name = 'mean fidelity'
        self.grad = T.grad(cost=self.cost, wrt=self.vars['parameters'])
        # define updates, to be performed at every call of `train_XXX`
        self.updates = self._make_updates(sgd_method)
        # initialize log to be filled with the history later
        self.log = {'fidelities': None, 'parameters': None}
        # create figure object
        self._fig = None
        self._ax = None
        # compile the training function `train_model`, that while computing
        # the cost at every iteration (batch), also updates the weights of
        # the network based on the rules defined in `updates`.
        # from IPython.core.debugger import set_trace; set_trace()
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
                self.model.inputs: train_inputs_batch,
                self.model.outputs: train_outputs_batch
            })

        # `test_model` is used to test the fidelity given by the currently
        # trained parameters. It's called at regular intervals during
        # the computation, and is the value shown in the dynamically
        # updated plot that is shown when the training is ongoing.
        self.test_model = theano.function(
            inputs=[],
            outputs=self.cost,
            updates=None,
            givens={self.model.inputs: self.vars['test_inputs'],
                    self.model.outputs: self.vars['test_outputs']})
        print(' done.')

    @staticmethod
    def _load_net(net):
        """
        Parse the `net` parameter given during init of `Optimizer`.
        """
        if isinstance(net, str):
            raise NotImplementedError('To be reimplemented')
        return net

    def save_results(self, file):
        """Save optimization results.

        The idea is here to save all the information required to
        reproduce a given training session.
        """
        net_data = dict(
            sympy_model=self.net.get_matrix(),
            ancillae_state=self.net.ancillae_state,
            initial_interactions=self.net.initial_values
        )
        optimization_data = dict(
            target_gate=self.target_gate,
            hyperparameters=self.hyperpars
        )
        # cut redundant log history
        fids = self.log['fidelities']
        # we cut from the history the last contiguous block of
        # values that are closer to 1 than `eps`
        eps = 1e-10
        end_useful_log = np.diff(np.abs(1 - fids) < eps).nonzero()[0][-1]
        saved_log = dict()
        saved_log['fidelities'] = fids[:end_useful_log]
        if self.log['parameters'] is not None:
            saved_log['parameters'] = self.log['parameters'][:end_useful_log]
        optimization_data['log'] = saved_log

        data_to_save = dict(
            net_data=net_data, optimization_data=optimization_data)
        filename, ext = os.path.splitext(file)
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
            ax.plot(fids[:n_epoch], '-b')
        else:
            if n_epoch + 1 == len_shown_history:
                x_coords = np.arange(
                    n_epoch - len_shown_history + 1, n_epoch + 1)
            else:
                x_coords = np.arange(n_epoch + 1)
            ax.plot(x_coords, fids[x_coords], '-b')
        plt.suptitle('learning rate: {}\nfidelity: {}'.format(
            self.vars['learning_rate'].get_value(), fids[n_epoch]))
        fig.canvas.draw()

    def refill_test_data(self):
        """Generate new test data and put them in shared variable.
        """
        inputs, outputs = self.model.generate_training_states(
            self.hyperpars['test_dataset_size'])
        self.vars['test_inputs'].set_value(inputs)
        self.vars['test_outputs'].set_value(outputs)

    def refill_training_data(self):
        """Generate new training data and put them in shared variable.
        """
        inputs, outputs = self.model.generate_training_states(
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

    def _run(self, save_parameters=True, len_shown_history=200):
        # generate testing states
        self.refill_test_data()

        n_epochs = self.hyperpars['n_epochs']
        # initialize log
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
            if self.log['fidelities'][-1] == 1:
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
