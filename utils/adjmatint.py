#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def adjmat_basic(p=8, fo=2, n=8):
    """
    Structured sparsity : fixed fi, fo.
    Returns an nxp adjacency matrix adjmat where each row is an output neuron
    with fi 1s and each column is an input neuron with fo 1s.
    """
    fi = p * fo // n
    adjmat = np.zeros((n, p))
    # Count the number of 1s in each column.
    fo_counter = np.zeros(p)
    # Start with uniform probabilities to get a 1 anywhere.
    fo_probs = np.asarray(p * [1.0/p])

    for ni in range(n):
        # Usual case.
        if ni < (n - fo):
            # Nonzero elements should have uniform prob.
            fo_probs[fo_probs!=0] = np.count_nonzero(fo_probs) * [1.0 / np.count_nonzero(fo_probs)]
            fi_pattern = np.random.choice(p,size=fi, replace=False, p=fo_probs)
        else:
            # Last fo output neurons.
            # Input neurons with low fanout must be chosen now.
            fi_pattern = np.where(fo_counter <= ni - (n - fo))[0]
            # If equal, all required positions are in fi_pattern, no need for choice any more.
            if len(fi_pattern) != fi:
                temp_fo_probs = np.copy(fo_probs)
                # Since elements that must be chosen are already chosen, make their fo_probs 0 temporarily.
                temp_fo_probs[fi_pattern] = 0
                temp_fo_probs[temp_fo_probs!=0] = np.count_nonzero(temp_fo_probs) * [1.0 / np.count_nonzero(temp_fo_probs)]
                # Get remaining 1s.
                fi_pattern = np.concatenate((fi_pattern, np.random.choice(p, size=fi - len(fi_pattern), replace=False, p=temp_fo_probs)))

        adjmat[ni][fi_pattern] = 1
        fo_counter[fi_pattern] += 1
        # If a column has fo 1s, it has reached max and should never be chosen again.
        fo_probs[fo_counter==fo] = 0

    return adjmat
