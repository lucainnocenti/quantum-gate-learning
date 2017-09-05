"""Functions to process and modify content of `QubitNetwork` objects.

This module contains a list of functions specifically aimed to work with
`QubitNetwork` objects. It differs from `net_analysis_tools` in that the
methods in this module write or modify data saved in `QubitNetwork` objects,
rather than just reading and analysing it.
"""

import collections
import os

import numpy as np
import pandas as pd

import qutip
import theano
import theano.tensor as T

# package imports
from .QubitNetwork import QubitNetwork
from .net_analysis_tools import load_network_from_file
from .model import _gradient_updates_momentum, Optimizer
from IPython.core.debugger import set_trace


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
