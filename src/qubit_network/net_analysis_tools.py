import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import qutip

from utils import chop


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


def normalize_phase(gate):
    """Change the global phase to make the top-left element real."""
    return gate * np.exp(-1j * np.angle(gate[0, 0]))


def trace_ancillae_and_normalize(net, num_system_qubits=None, eps=1e-4):
    # if net is a QubitNetwork object
    if hasattr(net, 'get_current_gate'):
        gate = net.get_current_gate(return_Qobj=True)
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
    gate = net.get_current_gate(return_Qobj=True)
    ancillae_proj = qutip.ket2dm(ancillae_state)
    identity_over_system = qutip.tensor(
        [qutip.qeye(2) for _ in range(net.num_system_qubits)])
    proj = qutip.tensor(identity_over_system, ancillae_proj)
    return proj * gate * proj


def plot_gate(net, norm_phase=True, permutation=None):
    gate = net.get_current_gate(return_Qobj=True)
    if permutation is not None:
        gate = gate.permute(permutation)
        gate = normalize_phase(gate)

    gate = abs(gate.data.toarray())

    f, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(gate,
                     square=True, annot=True, fmt='1.2f',
                     linewidth=1, cbar=False)
    ax.hlines([8], *ax.get_xlim())
    ax.vlines([8], *ax.get_ylim())


# ----------------------------------------------------------------
# Functions ot plot the fidelity vs J parameters for various random states.
# ----------------------------------------------------------------

def plot_fidelity_vs_J_live(net, xs, index_to_vary,
                            states=None, target_states=None,
                            n_states=5):
    import matplotlib.pyplot as plt
    import theano

    if states is None or target_states is None:
        states, target_states = net.generate_training_data(size=n_states)

    Js = net.J.get_value()

    fig, ax = plt.subplots(1, 1)
    fidelities = np.zeros(shape=(len(states), len(xs)))
    for state_idx, (state, target_state) in enumerate(
            zip(states, target_states)):
        compute_fidelity = theano.function(
            inputs=[], outputs=net.fidelity_1s(state, target_state))

        for idx, x in enumerate(xs):
            new_Js = Js[:]
            new_Js[index_to_vary] = x
            net.J.set_value(new_Js)

            fidelities[state_idx, idx] = compute_fidelity()

        ax.plot(xs, fidelities[state_idx])
        fig.canvas.draw()


def fidelity_vs_J(net):
    import copy
    import theano
    import theano.tensor as T

    _net = copy.copy(net)
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
