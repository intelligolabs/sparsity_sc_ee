#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tf_keras.layers import Dense
from tf_keras.regularizers import l2
from tf_keras.models import Sequential


def add_cls(model, config, activation='relu', output_activation='softmax',
            kernel_initializer='he_normal', bias_initializer='zeros', kernel_regularizer=l2(0.)):
    """
    Helper function to add CLs to any EXISTING model
    Inputs:
        model: existing model.
        config: must know what the exact shape of CL portion is.
        kernel_initializer, bias_initializer, kernel_regularizer: use the same for all layers.
        Use activation for all hidden layers, output_activation for output layer.
    Output:
        Model with CLs attached.
    Possible improvement:
        Add dropout as as ndarray input with size = len(config)-1.
    """
    for i in range(1,len(config)):
        # Use output_activation for output layer and name it 'output'.
        if i == len(config)-1:
            model.add(Dense(config[i], input_shape=(config[i-1],), activation=output_activation,
                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                            kernel_regularizer=kernel_regularizer, name='output'))
        else:
            # Standard hidden layers.
            model.add(Dense(config[i], input_shape=(config[i-1],), activation=activation,
                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                            kernel_regularizer=kernel_regularizer))

    return model



def any_cl_only(config, activation='relu', output_activation='softmax',
                kernel_initializer='he_normal', bias_initializer='zeros',
                kernel_regularizer=l2(0.)):
    """
    Any MLP network.
    lr and decay are set to defaults for Adam as in tf_keras.
    """
    model = Sequential()
    model = add_cls(model, config, activation=activation, output_activation=output_activation,
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer)

    return model