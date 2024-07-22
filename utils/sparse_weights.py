#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tf_keras.callbacks


class SparseWeights(tf_keras.callbacks.Callback):
    """
    Class to enforce sparsity in weights.
    """
    def __init__(self, adjmats, numlayers):
        # adjmats still holds the non-transposed matrices, so need to transpose them numlayers, exluding input.
        # For example, for [784, 100, 100, 10], this is 3.
        self.adjmats = []
        self.numlayers = numlayers

        for i in range(numlayers):
            self.adjmats.append(adjmats['adjmat{0}{1}'.format(i, i + 1)].T)

    def on_batch_end(self, batch, logs={}):
        wb = self.model.get_weights()

        for i in range(self.numlayers):
            wb[-2 * (self.numlayers - i)] *= self.adjmats[i]

        self.model.set_weights(wb)
