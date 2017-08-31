import numpy as np

import theano
import theano.tensor as T


class FidelityGraph:
    def __init__(self,
                 num_qubits,
                 num_system_qubits,
                 parameters,
                 hamiltonian_model):
        self.num_qubits = num_qubits
        self.num_system_qubits = num_system_qubits
        self.parameters = parameters
        self.hamiltonian_model = hamiltonian_model

    def compute_evolution_matrix(self):
        return T.slinalg.expm(self.hamiltonian_model)
