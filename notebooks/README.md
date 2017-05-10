## 3-qubit gates reachable without ancillae

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
| [Toffoli (CC-X)][toff3qb] | 1 (up to numerical precision) |
| [Fredkin (C-SWAP)][fredkin3qb_1] | 0.99998 |
| [Fredkin (C-SWAP)][fredkin3qb_2] | 0.99999 |
| [CC-Z][ccz3qb] | 1 (up to numerical precision) |
| [CC-S][ccs3qb] | 1 (up to numerical precision) |
| [CC-Hadamard][ccH3qb] | 1 (up to numerical precision)

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

[toff3qb]: ../data/nets/toffoli_3q_all_1fid.pickle
[fredkin3qb_1]: ../data/nets/fredkin_3q_all_0.9999fid.pickle
[fredkin3qb_2]: ../data/nets/fredkin_3q_all_0.99999fid.pickle
[ccz3qb]: ../data/nets/ccZ_3q_all_1fid.pickle
[ccS3qb]: ../data/nets/ccS_3q_all_1fid.pickle
[ccH3qb]: ../data/nets/ccH_3q_all_1fid.pickle


## 3 qubits + 1 ancilla networks, regular topology, only z selfinteractions

| Target gate | Obtained fidelity |
| ----------- | ----------------- |
| [Fredkin][fredkin3qb+1a_1] | 0.996 |
| [Fredkin][fredkin3qb+1a_2] | 0.998 |
| [Fredkin][fredkin3qb+1a_3] | 0.99999 |
| [Fredkin][fredkin3qb+1a_4] | 0.999999 |
| [Toffoli][toffoli3qb+1a] | ?? |

[fredkin3qb+1a_1]: ../data/nets/fredkin_3q+1a_allpairs_onlyz_0.996fid.pickle
[fredkin3qb+1a_2]: ../data/nets/fredkin_3q+1a_allpairs_onlyz_0.998fid.pickle
[fredkin3qb+1a_3]: ../data/nets/fredkin_3q+1a_allpairs_onlyz_0.99999fid.pickle
[fredkin3qb+1a_4]: ../data/nets/fredkin_3q+1a_allpairs_onlyz_0.999999fid.pickle
[toffoli3qb+1a]: ../data/nets/toffoli_3q+1a_all_?????.pickle


## 3 qubits + 1 ancilla networks, regular topology, all interactions

| Target gate | Obtained fidelity |
| ----------- | ----------------- |
| [Toffredkin][toffredkin3qb+1a] | 0.99998 (possibly improvable) |

[toffredkin3qb+1a]: ../data/nets/toffredkin_3q+1a_0.9999fid.pickle
