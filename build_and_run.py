#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle

import numpy as np

from tf_keras.regularizers import l2
from tf_keras.models import load_model
from tf_keras.optimizers.legacy import Adam
from tf_keras.initializers import Constant

import models.keras_nets as nets
import utils.adjmatint as adjmatint
from data.dataloader import load_data
from utils.sparse_weights import SparseWeights


def merge_dicts(a, b):
    """
    If a and b have any common keys, their values must be lists.
    This function extends the lists in a by those in b.
    If a has keys which b doesn't, their values need not be lists and are left intact.
    If b has keys which a doesn't, their values need not be lists and are added to a.
    Modifies a in-place, no need to return.
    """
    for k in b:
        if k in a:
            a[k].extend(b[k])
        else:
            a[k] = b[k]


def build_and_run_model(config, fo, dataset_path, l2_val=8e-5, loss='categorical_crossentropy',
                        optimizer=Adam(decay=1e-5), metrics=['accuracy'], batch_size=256,
                        total_epochs=50, epoch_step=-1, input_pad=800, preprocesses=()):
    """
    Parameters description:
    model: output from any net function.
    loss, optimizer, metrics: needed for model compilation.
    config, fo: config and fanout. These are required only for the MLP portion of the net.
    z: Degree of parallelism. Only required for clash-free adjmats. Should be nparray same size as fo.
    epoch_step: Save model and accs after this many epochs.
        Default: -1, then epoch_step=total_epochs.
        Otherwise: must be <= total_epochs.
    dataset_path: the dataset path.
     input_pad, output_pad: If not -1, xdata is padded with 0s to reach input_pad, likewise for ydata.
        Example: MNIST 784, 10 can have input_pad=1024, output_pad=16.
        Do not use input_pad when the 1st layer is a ConvNet.
    preprocesses: any input preprocessing function from data_processing. Enter as tuple, example (dp.gaussian_normalize).
    results_filepath: Path to store model and results.
    """

    model = nets.any_cl_only(config, bias_initializer=Constant(0.1))

    xtr, ytr, xva, yva, xte, yte = load_data(dataset_path)
    for preprocess in preprocesses:
        xtr, xva, xte = preprocess(xtr, xva, xte)

    if epoch_step == -1:
        epoch_step = total_epochs

    callbacks = []
    results_filepath = './results/net{0}_fo{1}_l2{2}'.format(config, fo, l2_val) \
        .replace('   ', ' ').replace('  ', ' ').replace('[ ', '[').replace(' ', '_')

    # Pad xdata and ydata with 0s as applicable.
    if len(xtr.shape) == 2:
        if input_pad > xtr.shape[1]:
            print('Padding xdata from {0} to {1}'.format(xtr.shape[1], input_pad))
            xtr = np.concatenate((xtr, np.zeros((xtr.shape[0], input_pad-xtr.shape[1]))), axis=1)
            xva = np.concatenate((xva, np.zeros((xva.shape[0], input_pad-xva.shape[1]))), axis=1)
            xte = np.concatenate((xte, np.zeros((xte.shape[0], input_pad-xte.shape[1]))), axis=1)

    xtr = np.concatenate((xtr, xva), axis=0)
    ytr = np.concatenate((ytr, yva), axis=0)
    validation_data = None

    # Adjmats.
    # Adjmats are stored as a dictionary 'adjmats'.
    # Alternatively they can be loaded from pre-saved npz file 'adjmats' which is also like a dict.
    # We size adjmats as (output_dim, input_dim), but tf_keras works as (input_dim, output_dim).
    # Hence, needs to be transpose.
    numlayers = len(config) - 1
    wb = model.get_weights()

    # Otherwise, create it from scratch.
    adjmats = {}
    for i in range(numlayers):
        # For basic or random.
        adjmats['adjmat{0}{1}'.format(i, i + 1)] = adjmatint.adjmat_basic(config[i], fo[i], config[i + 1])
    spwt = SparseWeights(adjmats, numlayers)

    # Set initial weights to be sparse.
    for i in range(numlayers):
        wb[-2 * (numlayers-i)] *= adjmats['adjmat{0}{1}'.format(i, i + 1)].T
    model.set_weights(wb)

    # Model compile and run.
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.save(os.path.join(results_filepath, 'model.h5'))

    # Keeps all records.
    recs = {}
    with open(os.path.join(results_filepath, 'train_records.pkl'), 'wb') as f:
        pickle.dump(recs, f)

    for k in range(total_epochs // epoch_step):
        model = load_model(os.path.join(results_filepath, 'model.h5'))
        with open(os.path.join(results_filepath, 'train_records.pkl'), 'rb') as f:
            recs = pickle.load(f)

        # Any sparse case: sparsity is enforced ONLY IF callbacks=[spwt] is included in arguments to model.fit().
        if np.array_equal(config[1:], fo) == False:
            callbacks.append(spwt)
        history = model.fit(xtr, ytr, batch_size=batch_size, epochs=epoch_step,
                            validation_data=validation_data, callbacks=callbacks)

        # Save model and records for resuming training.
        model.save(os.path.join(results_filepath, 'model.h5'))
        merge_dicts(recs,history.history)

        # Don't save for the last iteration because testing follows.
        if k != (total_epochs // epoch_step-1):
            with open(os.path.join(results_filepath, 'train_records.pkl'), 'wb') as f:
                pickle.dump(recs, f)

    # Testing.
    score = model.evaluate(xte, yte, batch_size=batch_size)
    test_dict = {'test_loss' : score[0]}

    for i in range(len(metrics)):
        test_dict['test_' + metrics[i]] = score[i + 1]
        print('\nTest {0} = {1}'.format(metrics[i], score[i + 1]))

    merge_dicts(recs,test_dict)

    with open(os.path.join(results_filepath, 'test_records.pkl'), 'wb') as f:
        pickle.dump(recs, f)
