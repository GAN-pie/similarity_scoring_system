#!/usr/bin/env python3
# coding: utf-8

from keras.layers import Input, Dense, Dropout, Activation
# from keras.initializers import glorot_normal, lecun_normal
from keras.models import Model
from keras.regularizers import l1, l2


def make_feedforward(options):
        input_shape, hidden_size = options.input_shape, options.hidden_size
        dropout_rate = 0.25
        weight_init = "glorot_normal"
        weight_constraint = None
        bias_init = "zeros"
        bias_constraint = None
        if options.weight_decay:
            weight_regularizer = l1(options.weight_decay)
            bias_regularizer = l1(options.weight_decay)
        else:
            weight_regularizer = None
            bias_regularizer = None

        input_layer = Input(shape=input_shape)

        core = Dense(
            hidden_size,
            kernel_initializer=weight_init,
            kernel_regularizer=weight_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=weight_constraint,
            bias_initializer=bias_init,
            bias_constraint=bias_constraint)(input_layer)

        core = Activation("tanh")(core)
        core = Dropout(dropout_rate)(core)

        # core = Dense(
        #     hidden_size,
        #     kernel_initializer=weight_init,
        #     kernel_regularizer=weight_regularizer,
        #     bias_regularizer=bias_regularizer,
        #     kernel_constraint=weight_constraint,
        #     bias_initializer=bias_init,
        #     bias_constraint=bias_constraint)(core)

        core = Activation("tanh")(core)
        core = Dropout(dropout_rate)(core)

        return Model(inputs=input_layer, outputs=core, name="feedforward")
