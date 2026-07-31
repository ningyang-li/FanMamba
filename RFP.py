# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 09:17:48 2025

@author: fes_m
"""

from tensorflow.keras.layers import Layer, Concatenate, Reshape, AveragePooling3D, MaxPooling3D

pi = 3.141593
pool_band = 2

class RFP(Layer):
    '''
    '''
    def __init__(self, redian=pi/4, n_circle=3, n_centralized_sequence=8, sequence_length=7, steps=[1, 2, 3], mode="avg", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redian = redian
        self.n_circle = n_circle
        self.n_centralized_sequence = n_centralized_sequence
        self.sequence_length = sequence_length
        self.steps = steps
        self.mode = mode
        
        # in the spatial dimension
        # pool size is equal to the number of pixels of the last circle plus 1 (this is ensure the last circle can be reduced), while strides is 1
        # in the spectral dimension
        # regular pool size and stride are adopted
        # self.pool_size = (1, 1 + ((2 * self.n_circle + 1) ** 2 - (2 * (self.n_circle - 1) + 1) ** 2) // self.n_centralized_sequence, 2)
        self.pool_size = (1, self.steps[-1] + 1, pool_band)
        self.strides = (1, 1, pool_band)
        # after FSP, reduce 1 circle and half of bands

    
    def call(self, x):
        # input (bs, model_internal_channel, 1, self.centralized_sequence*self.sequence_length, n_band)
        _, n_band = x.shape[1], x.shape[-1]
        x = Reshape((_, self.n_centralized_sequence, self.sequence_length, n_band))(x)
        # extract the centers
        centers = x[:, :, :, 0:1, :]
        if self.mode == "avg":
            # pool the spectral dimension of centers
            centers = AveragePooling3D(pool_size=(1, 1, pool_band), strides=(1, 1, pool_band), data_format="channels_first", padding="valid")(centers)
            # pool spectral and spatial dimensions of circles
            y = AveragePooling3D(pool_size=self.pool_size, strides=self.strides, data_format="channels_first", padding="valid")(x[:, :, :, 1:, :])
        else:
            centers = MaxPooling3D(pool_size=(1, 1, pool_band), strides=(1, 1, pool_band), data_format="channels_first", padding="valid")(centers)
            y = MaxPooling3D(pool_size=self.pool_size, strides=self.strides, data_format="channels_first", padding="valid")(x[:, :, :, 1:, :])
        # merge
        y = Concatenate(axis=-2)([centers, y])
        
        reduce_sequence_length = y.shape[-2]
        reduced_band = y.shape[-1]
        y = Reshape((_, 1, self.n_centralized_sequence * reduce_sequence_length, reduced_band))(y)

        return y

















