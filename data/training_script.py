import glob
from collections import OrderedDict
import itertools
import os
import sys
import pickle
import numpy as np
import scipy
import sympy
import pandas as pd
import argparse
import logging

import qutip
import qutip.qip.algorithms.qft

import theano
import theano.tensor as T

import qubit_network.net_analysis_tools as nat
import qubit_network.utils
from qubit_network.utils import chop, complex2bigreal, bigreal2complex, bigreal2qobj
from qubit_network.QubitNetwork import pauli_product
from qubit_network.model import QubitNetworkGateModel
from qubit_network.Optimizer import Optimizer
from qubit_network.net_analysis_tools import NetDataFile, NetsDataFolder
from qubit_network.analytical_conditions import commuting_generator


def make_optimizer_sympy(sympy_model, target_gate, initial_values, n_epochs,
                         training_dataset_size, batch_size, sgd_method):
    model = QubitNetworkGateModel(
        sympy_expr=sympy_model, initial_values=initial_values)
    optimizer = Optimizer(
        net=model,
        learning_rate=1.,
        decay_rate=.005,
        n_epochs=n_epochs,
        batch_size=batch_size,
        target_gate=target_gate,
        training_dataset_size=training_dataset_size,
        test_dataset_size=100,
        sgd_method=sgd_method,
        headless=True
    )
    return model, optimizer


def make_optimizer_allints(num_ancillae, target_gate, initial_values, n_epochs,
                           training_dataset_size, batch_size, sgd_method):
    num_system_qubits = int(scipy.log2(target_gate.shape[0]))
    num_qubits = num_system_qubits + num_ancillae
    model = QubitNetworkGateModel(
        num_qubits=num_qubits, num_system_qubits=num_system_qubits,
        interactions='all', initial_values=initial_values
    )
    optimizer = Optimizer(
        net=model,
        learning_rate=1.,
        decay_rate=.005,
        n_epochs=n_epochs,
        batch_size=batch_size,
        target_gate=target_gate,
        training_dataset_size=training_dataset_size,
        test_dataset_size=100,
        sgd_method=sgd_method,
        headless=True
    )
    return model, optimizer


def sympy_models_from_string(model_name):
    if model_name == 'doublefredkin_diagonal':
        antifredkin = qutip.tensor(qutip.projection(2, 0, 0), qutip.swap())
        antifredkin += qutip.tensor(qutip.projection(2, 1, 1), qutip.qeye([2, 2]))
        antifredkin
        ff = qutip.tensor(qutip.projection(2, 0, 0), qutip.fredkin())
        ff += qutip.tensor(qutip.projection(2, 1, 1), antifredkin)
        ff_diagonal = commuting_generator(ff, interactions='diagonal')
        return ff_diagonal, ff
    else:
        raise ValueError('{} is not a known model name'.format(model_name))


def qutip_gate_from_path(path):
    _, fmt = os.path.splitext(path)
    if fmt is None:
        raise ValueError('An extension must be used.')

    if fmt is not '.pickle':
        raise ValueError('Only pickle extension is supported ATM.')

    with open(path, 'rb') as file:
        data = pickle.load(file)
    if not isinstance(data, qutip.Qobj):
        raise ValueError('This should be a qutip object, not a "{}"'.format(
            data.__class__))
    return data


def qutip_gate_from_string(gate_name, num_qubits):
    if os.path.exists(gate_name):
        logging.info('Gate specified from file "{}".'.format(gate_name))
        if num_qubits is not None:
            logging.warning('The number of qubits cannot be specified when loa'
                            'ding a gate from file. I\'m ignoring it')
        return qutip_gate_from_path(gate_name)

    if gate_name == 'qft':
        if num_qubits is None:
            logging.info('Assuming num_qubits=3 for the QFT gate.')
            num_qubits = 3
        return qutip.qip.algorithms.qft.qft(num_qubits)
    elif gate_name == 'cnot':
        if num_qubits is not None:
            logging.warning('The input num_qubits is not used with the CNOT ga'
                            'te (which always has two qubits). I\'m ignoring i'
                            'ts value.')
        return qutip.cnot()
    else:
        raise ValueError('{} is not a known gate name.'.format(gate_name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial_values', type=int, default=None)
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--n_attempts', type=int)
    parser.add_argument('--folder', type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--model_type', type=str, default=None,
                        help='Either "all_interactions" or "sympy".')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--num_qubits', type=int, default=None)
    parser.add_argument('--num_ancillae', type=int, default=None)
    parser.add_argument('--training_dataset_size', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--sgd_method', type=str, default='momentum')
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    # FORMAT = logging.Formatter("%(levelname)s - %(message)s")
    FORMAT = "[%(asctime)s %(filename)18s:%(lineno)3s - %(funcName)25s()] %(message)s"
    formatter = logging.Formatter(FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if args.model_type == 'all_interactions':
        logging.info('Using "all interactions" model')
        # if args.num_qubits is None:
        #     num_qubits = 3
        # else:
        #     num_qubits = args.num_qubits
        if args.num_ancillae is None:
            num_ancillae = 1
        else:
            num_ancillae = args.num_ancillae
        if args.model_name is None:
            raise ValueError('The model_name option must be given')
        target_gate = qutip_gate_from_string(args.model_name,
                                             num_qubits=args.num_qubits)
        def optimizer_initializer():
            return make_optimizer_allints(
                num_ancillae=num_ancillae,
                target_gate=target_gate,
                initial_values=args.initial_values,
                n_epochs=args.n_epochs,
                training_dataset_size=args.training_dataset_size,
                batch_size=args.batch_size,
                sgd_method=args.sgd_method
            )
    elif args.model_type == 'sympy':
        logging.info('Using sympy model')
        sympy_model, target_gate = sympy_models_from_string(args.model_name)
        def optimizer_initializer():
            return make_optimizer_sympy(
                sympy_model,
                target_gate=target_gate,
                initial_values=args.initial_values,
                n_epochs=args.n_epochs,
                training_dataset_size=args.training_dataset_size,
                batch_size=args.batch_size,
                sgd_method=args.sgd_method
            )

    for i in range(args.n_attempts):
        logging.info('Starting training no.{}'.format(str(i + 1)))
        model, optimizer = optimizer_initializer()
        optimizer.run()
        if args.folder[0] == '/' or args.folder[0] == '.':
            file_ = ''
        else:
            file_ = './'
        file_ += args.folder + '/training_no_' + str(i + 1) + '.pickle'
        optimizer.save_results(file_, overwrite=args.overwrite)
        logging.info('Fidelity obtained: {}'.format(model.fidelity_test()))


if __name__ == '__main__':
    main()

# run as:
# nohup bash -c "(python ./training_script.py --n_attempts 5 --n_epochs 5 --initial_values 0 --folder aigis12 >> ./aigis12/log.txt)" &