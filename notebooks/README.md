# Guide to notebooks

- [fredkin_paper.ipynb](./fredkin_paper.ipynb): Notebook with reproduction of results of [Banchi et al.](http://localhost:8888/notebooks/fredkin_paper.ipynb).
The training functions as used here may be obsolete and not running with the current version of the code.
- [toffoli_only_diagonal_from_reduced_expression.ipynb](./toffoli_only_diagonal_from_reduced_expression.ipynb): Contains the code used to train the model for the Toffoli gate with only diagonal pairwise interactions, starting from the reduced expression given by the conditions in the paper. This is also where the plots for the Toffoli in the paper have been generated.


# Results stored in net files
## 3-qubit gates without ancillae, all interactions

All of the following nets have been successfully trained with the following code (appropriately changing the `target_gate` parameter):

```
net = QubitNetwork(
    num_qubits=3,
    system_qubits=3,
    interactions='all'
)
qn.sgd_optimization(
    net=net,
    learning_rate=2,
    n_epochs=1000,
    batch_size=10,
    target_gate=gate,
    training_dataset_size=50,
    test_dataset_size=100,
    decay_rate=.01
)
```
where `gate` is generated using standard `qutip` functions (e.g. `qutip.toffoli()`, `qutip.fredkin()`, and so on).

| Target gate | Obtained fidelity |
| ---- | -------- |
| [Toffoli (CC-X)][toff3qb_1] | 1. |
| [Toffoli (CC-X)][toff3qb_2] | 0.9999 |
| [Fredkin (C-SWAP)][fredkin3qb_1] | 0.99998 |
| [Fredkin (C-SWAP)][fredkin3qb_2] | 0.99999 |
| [CC-Z][ccz3qb] | 1. |
| [CC-S][ccs3qb] | 1. |
| [CC-Hadamard][ccH3qb] | 1. |

The above target gates are generated with the following `qutip` functions:

```
toffoli = qutip.toffoli()

fredkin = qutip.fredkin()

ccZ = (qutip.tensor(qutip.projection(2, 0, 0), qutip.qeye(2), qutip.qeye(2)) +
       qutip.tensor(qutip.projection(2, 1, 1), qutip.cphase(np.pi)))

ccS = (qutip.tensor(qutip.projection(2, 0, 0), qutip.qeye(2), qutip.qeye(2)) +
       qutip.tensor(qutip.projection(2, 1, 1), qutip.cphase(np.pi / 2)))

ccHadamard = (qutip.tensor(qutip.projection(2, 0, 0), qutip.qeye(2), qutip.qeye(2)) +
              qutip.tensor(qutip.projection(2, 1, 1), qutip.qip.gates.controlled_gate(qutip.hadamard_transform())))
```

[toff3qb_1]: ../data/nets/toffoli_3q_all_1fid.pickle
[toff3qb_2]: ../data/nets/toffoli_3q_all_0.9999fid.pickle
[fredkin3qb_1]: ../data/nets/fredkin_3q_all_0.9999fid.pickle
[fredkin3qb_2]: ../data/nets/fredkin_3q_all_0.99999fid.pickle
[ccz3qb]: ../data/nets/ccZ_3q_all_1fid.pickle
[ccS3qb]: ../data/nets/ccS_3q_all_1fid.pickle
[ccH3qb]: ../data/nets/ccH_3q_all_1fid.pickle

## 3-qubit Toffoli gates, using topology obtained after conditions

Generating code, also found in [toffoli_only_diagonal_from_reduced_expression.ipynb](toffoli_only_diagonal_from_reduced_expression.ipynb) ([view on nbviewer](https://nbviewer.jupyter.org/github/lucainnocenti/quantum-gate-learning/blob/0879f34de99ffc55dcc344c198e7c1e3c64c699a/notebooks/toffoli_only_diagonal_from_reduced_expression.ipynb)):

    def J(*args):
    return sympy.Symbol('J' + ''.join(str(arg) for arg in args))
    def pauli(*args):
    return pauli_product(*args)
    toffoli_diagonal = J(3, 0, 0) * pauli(3, 0, 0)
    toffoli_diagonal += J(0, 3, 0) * pauli(0, 3, 0)
    toffoli_diagonal += J(0, 0, 1) * pauli(0, 0, 1)
    toffoli_diagonal += J(3, 0, 3) * (pauli(0, 0, 3) + pauli(3, 0, 3))
    toffoli_diagonal += J(0, 3, 3) * (pauli(0, 0, 3) + pauli(0, 3, 3))
    toffoli_diagonal += (J(1, 0, 1) * pauli(1, 0, 0) + J(0, 1, 1) * pauli(0, 1, 0)) * (pauli(0, 0, 0) + pauli(0, 0, 1))
    toffoli_diagonal += J(2, 2, 0) * (pauli(1, 1, 0) + pauli(2, 2, 0))
    toffoli_diagonal += J(3, 3, 0) * pauli(3, 3, 0)
    toffoli_diagonal

    net = QubitNetworkModel(sympy_expr=toffoli_diagonal, initial_values=1)
    optimizer = Optimizer(
        net=net,
        learning_rate=1.,
        n_epochs=500,
        batch_size=2,
        target_gate=qutip.toffoli(),
        training_dataset_size=200,
        test_dataset_size=100,
        decay_rate=.005,
        sgd_method='momentum'
    )
    optimizer.run()
Corresponding parameters historie*s* [here](https://nbviewer.jupyter.org/github/lucainnocenti/quantum-gate-learning/tree/0879f34de99ffc55dcc344c198e7c1e3c64c699a/data/parameters_histories/).

### Toffoli gate with only diagonal interactions
Starting with the initial "Ansatz" provided by Abdullah, we easily find a solution for the Toffoli using only interactions of type `xx`, `yy`, `zz`, `x`, `y`, `z`.

## 3 qubits + 1 ancilla networks, regular topology, only z selfinteractions

| Target gate | Obtained fidelity |
| ----------- | ----------------- |
| [Fredkin][fredkin3qb+1a_1] | 0.996 |
| [Fredkin][fredkin3qb+1a_2] | 0.998 |
| [Fredkin][fredkin3qb+1a_3] | 0.99999 |
| [Fredkin][fredkin3qb+1a_4] | 0.999999 |
| [Toffoli][toffoli3qb+1a] | 0.989 |

[fredkin3qb+1a_1]: ../data/nets/fredkin_3q+1a_allpairs_onlyz_0.996fid.pickle
[fredkin3qb+1a_2]: ../data/nets/fredkin_3q+1a_allpairs_onlyz_0.998fid.pickle
[fredkin3qb+1a_3]: ../data/nets/fredkin_3q+1a_allpairs_onlyz_0.99999fid.pickle
[fredkin3qb+1a_4]: ../data/nets/fredkin_3q+1a_allpairs_onlyz_0.999999fid.pickle
[toffoli3qb+1a]: ../data/nets/toffoli_3q+1a_all_0.989fid.pickle


## 3 qubits + 1 ancilla networks, regular topology, all interactions

| Target gate | Obtained fidelity |
| ----------- | ----------------- |
| [Toffredkin][toffredkin3qb+1a] | 0.99998 (possibly improvable) |

[toffredkin3qb+1a]: ../data/nets/toffredkin_3q+1a_0.9999fid.pickle
