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
| [Fredkin (C-SWAP)][fredkin3qb] | 0.99998 (didn't manage to get higher fidelities) |
| [CC-Z][ccz3qb] | 1 (up to numerical precision) |
| [CC-S][ccs3qb] | 1 (up to numerical precision) |
| [CC-Hadamard][ccH3qb] | 1 (up to numerical precision)


[toff3qb]: ../data/nets/toffoli_3q_all_1fid.pickle
[fredkin3qb]: ../data/nets/fredkin_3q_all_0.99999fid.pickle
[ccz3qb]: ../data/nets/ccZ_3q_all_1fid.pickle
[ccS3qb]: ../data/nets/ccS_3q_all_1fid.pickle
[ccH3qb]: ../data/nets/ccH_3q_all_1fid.pickle


## 3-qubit gates obtained after tracing one ancilla

| Target gate | Obtained fidelity |
| ----------- | ----------------- |
| [Fredkin][fredkin3qb+1a] | 0.999999998 (possibly improvable, values from Banchi et al.) |
| [Toffredkin][toffredkin3qb+1a] | 0.99998 (possibly improvable) |

[fredkin3qb+1a]: ../data/nets/fredkin_Banchietal.pickle
[toffredkin3qb+1a]: ../data/nets/toffredkin_3q+1a_0.9999fid.pickle
