# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:09:26 2025

@author: fes_m
"""

# keras-2.3.1 python-3.6.13 tensorflow-2.0.0

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import LayerNormalization
import keras
from tensorflow.keras.layers import Layer, Lambda, Conv1D, Conv3D, Dense, Flatten, Activation, Input, Dropout, Reshape, GlobalAveragePooling3D, Concatenate, AveragePooling3D
from tensorflow.keras.models import Model
# from keras_layer_normalization import LayerNormalization
from einops import rearrange, repeat
from dataclasses import dataclass, field
import numpy as np
from typing import Union
import math
from typing import Dict

from FS import FS
from RFP import RFP
from Adaptive_Sum import Adaptive_Sum

pi = 3.141593
pool_band = 2

@dataclass
class ModelArgs:
    '''
    parameters of model
    '''
    input_shape: tuple = (1, 3, 3, 10)
    model_input_channel: int = 8
    model_state: int = 8
    projection_expand_factor: int = 2
    redians: Dict[str, float] = field(default_factory=lambda: {"pi/4": pi/4, "pi/2": pi/2, "pi": pi, "2*pi": 2*pi})
    conv_kernel_size: int = 5
    delta_t_min: float = 0.001
    delta_t_max: float = 0.1
    delta_t_scale: float = 0.1
    delta_t_init_floor: float = 1e-4
    conv_use_bias: bool = True
    dense_use_bias: bool = False
    num_blocks: int = 4
    dropout_rate: float = 0.2
    cls_dim: int = 128
    num_classes: int = None
    loss:Union[str, keras.losses.Loss] = None
    metrics = ['accuracy']

    def __post_init__(self):
        self.width = int(self.input_shape[1])
        self.n_band = self.input_shape[-1]
        self.n_pixel = int(self.width * self.width)
        self.n_circle = int(np.floor(self.width / 2.))
        
        self.n_centralized_sequence_dict = {}
        self.sequence_length_dict = {}
        self.steps_dict = {}
        for r in self.redians:
            self.n_centralized_sequence_dict[r] = int((2 * pi) / self.redians[r])
            self.sequence_length_dict[r] = int((self.n_pixel - 1) / self.n_centralized_sequence_dict[r]) + 1
            self.steps_dict[r] = np.arange(1, self.n_circle + 1, 1, dtype="int32") * int(self.redians[r] / (pi/4))
        
        self.model_internal_channel: int = self.model_input_channel * self.projection_expand_factor
        self.delta_t_rank = math.ceil(float(self.model_input_channel) / 2)

        if self.loss == None:
            raise ValueError(f"loss cannot be {self.loss}")


def _swish(x):
    return x * K.sigmoid(x)
    # return Activation("relu")(x)

    
def relevant_selective_scan(xc, xs, delta, A, B, C, D, E):
    # xc shape:    (bs, model_internal_channel, 1, 1, n_band)
    #              (b   c                       w  u  d)
    # xs shape:    (bs, model_internal_channel, 1, squence_length, n_band)
    #              (b   c                       w  v                 d)
    # delta shape: (bs, model_internal_channel, 1, squence_length, n_band)
    # A shape:     (model_internal_channel, model_state)
    #              (c                       s)
    # B shape:     (bs, model_state, 1, squence_length, n_band)
    # C shape:     (bs, model_state, 1, squence_length, n_band)
    # D shape:     (model_internal_channel,)
    # E shape:     (bs, model_state, 1, 1, n_band)
    
    # broadcast xc and E
    xc = K.repeat_elements(xc, rep=xs.shape[-2], axis=-2)
    E = K.repeat_elements(E, rep=xs.shape[-2], axis=-2)

    xc = rearrange(xc, "b c w v d -> b w v d c")
    xs = rearrange(xs, "b c w v d -> b w v d c")
    delta = rearrange(delta, "b c w v d -> b w v d c")
    B = rearrange(B, "b s w v d -> b w v d s")
    C = rearrange(C, "b s w v d -> b w v d s")
    E = rearrange(E, "b s w v d -> b w v d s")
    
    # first step of A_bar = exp(dA), i.e., dA
    dA = tf.einsum("bwvdc,cs->bwvdcs", delta, A)
    dB_xs = tf.einsum("bwvdc,bwvdc,bwvds->bwvdcs", delta, xs, B)
    dE_xc = tf.einsum("bwvdc,bwvdc,bwvds->bwvdcs", delta, xc, E)
    
    dA_cumsum = tf.pad(dA[:, :, 1:, 1:], [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0], [0, 0]])[:, :, 1:, 1:, :, :]
    dA_cumsum = tf.reverse(dA_cumsum, axis=[1, 2, 3])
    
    # Cumulative sum along all the input tokens, parallel prefix sum
    # calculates dA for all the input tokens parallely
    dA_cumsum = tf.math.cumsum(dA_cumsum, axis=3)
    dA_cumsum = tf.math.cumsum(dA_cumsum, axis=2)
    
    # second step of A_bar = exp(dA), i.e., exp(dA)
    dA_cumsum = tf.exp(dA_cumsum)
    dA_cumsum = tf.reverse(dA_cumsum, axis=[2, 3])   # flip back along axis 2，3
    
    # h(t) = A_bar * h(t-1) + B_bar * X(t) + E_bar * Xc
    # h = x * B_bar * dA_cumsum + xc * E_bar * dA_cumsum
    # (bwvdcs, bwvdcs, bwvdcs)->(bwvdcs)
    h = (dB_xs + dE_xc) * dA_cumsum
    # 1e-12 to avoid division by 0
    h = tf.math.cumsum(h, axis=3)
    h = tf.math.cumsum(h, axis=2)
    h = h/(dA_cumsum + 1e-12)
    
    y = tf.einsum("bwvdcs,bwvds->bwvdc", h, C)
    y = y + xs * D
    y = rearrange(y, "b w v d c -> b c w v d")
    
    return y


class CLGMambaBlock(Layer):
    '''
    input: (Xc, x) -> (n_channel, 1, n_pixel, n_band) # n_pixel = width * width + (n_central_sequence - 1) = n_central_sequence * (sequence_length + 1)
           width = 7 (n_original_pixel = width * width = 49)
           circle = floor(width / 2) = 3
               
           for redian = pi/4:
               n_pixel_cicle_1 = (1 + 2 * 1) - 1 = 8
               scan_setp_in_first_circle = n_pixel_cicle_1 / (pi / redian) = 1
               scan_step_in_second_circle = scan_setp_in_first_circle * 2 = 2
               scan_step_in_third_circle = scan_setp_in_first_circle * 3 = 3
               sequence = 1 + scan_setp_in_first_circle + scan_step_in_second_circle + scan_step_in_third_circle = 1 + 1 + 2 + 3 = 7
               n_central_sequence = (width * width - 1) / (sequence - 1) = 2pi/r = 8
               n_pixel = n_central_sequence * sequence_length = 8 * 7 = 56
               
               [[c, s1-1, s1-2, ..., s1-6],
                [c, s2-1, s2-2, ..., s2-6],
                ..., 
                [c, s8-1, s8-2, ..., s8-6]]
               
            for redian = pi/2:
               n_pixel_cicle_1 = (1 + 2 * 1) - 1 = 8
               scan_setp_in_first_circle = n_pixel_cicle_1 / (pi / redian) = 2
               scan_step_in_second_circle = scan_setp_in_first_circle * 2 = 4
               scan_step_in_third_circle = scan_setp_in_first_circle * 3 = 6
               sequence = 1 + scan_setp_in_first_circle + scan_step_in_second_circle + scan_step_in_third_circle = 1 + 2 + 4 + 6 = 13
               n_central_sequence = (width * width - 1) / (sequence - 1) = 2pi/r = 4
               n_pixel = n_central_sequence * sequence_length = 8 * 7 = 52
               
               [[c, s1-1, s1-2, ..., s1-12],
                [c, s2-1, s2-2, ..., s2-12],
                ...,
                [c, s4-1, s4-2, ..., s4-12]]
               
            for redian = pi:
               n_pixel_cicle_1 = (1 + 2 * 1) - 1 = 8
               scan_setp_in_first_circle = n_pixel_cicle_1 / (pi / redian) = 4
               scan_step_in_second_circle = scan_setp_in_first_circle * 2 = 8
               scan_step_in_third_circle = scan_setp_in_first_circle * 3 = 12
               sequence = 1 + scan_setp_in_first_circle + scan_step_in_second_circle + scan_step_in_third_circle = 1 + 4 + 8 + 12 = 25
               n_central_sequence = (width * width - 1) / (sequence - 1) = 2pi/r = 2
               n_pixel = n_central_sequence * sequence_length = 8 * 7 = 50
               
               [[c, s1-1, s1-2, ..., s1-24],
                [c, s2-1, s2-2, ..., s2-24]]
               
            for redian = 2pi:
               n_pixel_cicle_1 = (1 + 2 * 1) - 1 = 8
               scan_setp_in_first_circle = n_pixel_cicle_1 / (pi / redian) = 8
               scan_step_in_second_circle = scan_setp_in_first_circle * 2 = 16
               scan_step_in_third_circle = scan_setp_in_first_circle * 3 = 24
               sequence = 1 + scan_setp_in_first_circle + scan_step_in_second_circle + scan_step_in_third_circle = 1 + 8 + 16 + 24 = 49
               n_central_sequence = (width * width - 1) / (sequence - 1) = 2pi/r = 1
               n_pixel = n_central_sequence * sequence_length = 8 * 7 = 49
               
               [[c, s1-1, s1-2, ..., s1-48]]
    '''
    
    def __init__(self, modelargs, cur_redian: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("\nThis is CLGMamba Block")
        self.args = modelargs
        self.cur_redian = cur_redian
        self.redian = self.args.redians[cur_redian]
        self.n_centralized_sequence_dict = self.args.n_centralized_sequence_dict[self.cur_redian]
        self.sequence_length = self.args.sequence_length_dict[self.cur_redian]
        self.steps = self.args.steps_dict[self.cur_redian]
                
        # x and res
        self.in_projection = Conv3D(filters=self.args.model_internal_channel*2, kernel_size=1, strides=1, padding="same", data_format="channels_first", use_bias=self.args.conv_use_bias)
        
        # (bs, channel, 1, n_pixel, n_band) => (bs, channel, 1*n_pixel*n_band), flat spatial and spectral dimensions to conduct causal convolution
        # self.conv1d = Conv1D(filters=self.args.model_internal_channel, kernel_size=self.args.conv_kernel_size, strides=1, use_bias=self.args.conv_use_bias, data_format="channels_last", padding="causal")
        self.conv1d_s = Conv3D(filters=self.args.model_internal_channel, kernel_size=(1, 1, 7), strides=1, use_bias=self.args.conv_use_bias, data_format="channels_first", padding="same")
        self.conv1d_c = Conv3D(filters=self.args.model_internal_channel, kernel_size=(1, 1, 7), strides=1, use_bias=self.args.conv_use_bias, data_format="channels_first", padding="same")
        # these layers take in center pixel 'xc' and current token 'x' and output the input-specific delta, B, C, E (S6)
        self.x_projection = Conv3D(filters=self.args.delta_t_rank + self.args.model_state * 2, kernel_size=1, strides=1, data_format="channels_first", padding="same", use_bias=False)
        self.xc_projection = Conv3D(filters=self.args.model_state, kernel_size=1, strides=1, data_format="channels_first", padding="same", use_bias=False)    
        
        # this layer projects delta from delta_t_rank to the mamba internal
        self.delta_t_projection = Conv3D(filters=self.args.model_internal_channel, kernel_size=1, strides=1, padding="same", data_format="channels_first", use_bias=True)
        
        self.A = repeat(tf.range(1, self.args.model_state+1, dtype=tf.float32), "s -> c s", c=self.args.model_internal_channel)
        
        self.A_log = tf.Variable(tf.math.log(self.A), trainable=True, dtype=tf.float32)
        
        self.D = tf.Variable(np.ones(self.args.model_internal_channel), trainable=True, dtype=tf.float32)
        
        self.out_projection = Conv3D(filters=self.args.model_input_channel, kernel_size=1, strides=1, padding="same", data_format="channels_first", use_bias=self.args.dense_use_bias)
        
        
    def call(self, x):
        '''
        Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba page.
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
        '''
        (bs, n_channel, _, n_pixel, n_band) = x.shape
        
        # input_projection
        x_and_res = self.in_projection(x) # (batch, 2*model_internal_channel, 1, n_pixel-1, n_band)
        (x, res) = tf.split(x_and_res, [self.args.model_internal_channel, self.args.model_internal_channel], axis=1)

        # select the first pixel of first sequence (center pixel)
        xc = x[:, :, :, 0:1, :] # xc -> (bs, n_channel, 1, 1, n_band)
        xc = self.conv1d_c(xc)
        # sequence-wise rssm
        # y = tf.zeros(x.shape, dtype=tf.float32) # create y
        separated_y = []
        for i in range(self.n_centralized_sequence_dict):
            # (batch, model_internal_channel, 1, squence_length, n_band)
            xs = x[:, :, :, i*self.sequence_length:(i+1)*self.sequence_length]

            # (batch, (1*n_squence_length*n_band), model_internal_channel, )
            # xs = rearrange(xs, "b c w v d -> b (w v d) c")
            xs = self.conv1d_s(xs)
            # x = rearrange(x, "b (w v d) c -> b c w v d", w=1, v=width, d=n_band)
            # xs = tf.transpose(xs, perm=[0, 2, 1]) # (batch, model_internal_channel, (1*squence_length*n_band))
            # xs = Reshape((self.args.model_internal_channel, 1, self.sequence_length, n_band))(xs) # (batch, model_internal_channel, 1, n_squence_length, n_band)
            xs = tf.nn.swish(xs)
            # rssm
            result = self.rssm(xc, xs) # (batch, model_internal_channel, 1, squence_length, n_band)
            separated_y.append(result)
        
        y = separated_y[0]
        for i in range(1, self.n_centralized_sequence_dict):
            y = Concatenate(axis=-2)([y, separated_y[i]])

        y = y * tf.nn.swish(res)
        
        return self.out_projection(y)
    
    def rssm(self, xc, xs):
        '''
        Relevant State Space Module
        
        h(t) = A_bar*h(t-1) + B_bar*x(t) + E_bar*xc
        y(t) = C*h(t)
        
        x(t) -> (bs, 1, 1, n_band, model_internal_dim)
        xc   -> (bs, 1, 1, n_band, model_internal_dim)
        
        Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper
            - run_SSM(A, B, C, u) in the Annotated S4
            Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
        '''
        (model_internal_channel, model_state) = self.A_log.shape
        
        # Compute delta, A, B, C, D, E, the state space parameters
        #   A, D are input independent (see Mamba paper Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #   delta, B, C, E are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                    and is why Mamba is called **selective** state spaces)
        A = -K.exp(K.cast(self.A_log, "float32")) # shape -> (model_internal_channel, model_state)
        D = K.cast(self.D, "float32") # shape -> (model_internal_channel,)
        
        x_dbc = self.x_projection(xs) # shape -> (bs, delta_t_rank + model_state * 2, 1, sequence_length, n_band)
        # delta.shape -> (bs, delta_t_rank, 1, sequence_length, n_band)
        # B, C, E shape -> (bs, model_state, 1, sequence_length, n_band)
        (delta, B, C) = tf.split(x_dbc, num_or_size_splits=[self.args.delta_t_rank, self.args.model_state, self.args.model_state], axis=1)
        E = self.xc_projection(xc)
        
        delta = tf.nn.softplus(self.delta_t_projection(delta)) # shape -> (bs, model_internal_channel, 1, sequence_length, n_band)
        
        # x shape:     (bs, model_internal_channel, 1, sequence_length, n_band)
        # delta shape: (bs, model_internal_channel, 1, sequence_length, n_band)
        # A shape:     (model_internal_channel, model_state)
        # B shape:     (bs, model_state, 1, sequence_length, n_band)
        # C shape:     (bs, model_state, 1, sequence_length, n_band)
        # D shape:     (model_internal_channel,)
        # E shape:     (bs, model_state, 1, sequence_length, n_band)

        return relevant_selective_scan(xc, xs, delta, A, B, C, D, E)    


class ResidualBlock(Layer):
    def __init__(self, modelargs: ModelArgs, cur_redian: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = modelargs
        self.cur_redian = cur_redian
        self.mixer = FanMambaBlock(modelargs, cur_redian)
        # layer normalization
        # self.norm = LayerNormalization(axis=(1, 2, 3, 4), epsilon=1e-5)
        self.norm = LayerNormalization(epsilon=1e-5)
    
    def call(self, x):
        '''
        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
        '''
        y = self.norm(x)
        y = self.mixer(y)
        # if x.shape[1] != y.shape[1]:
        #     print("adjust")
        #     x = Conv3D(filters=y.shape[1], kernel_size=1, strides=1, padding="same", data_format="channels_first")(x)
        return  y + x


def init_model(args:ModelArgs, optimizer):
    # input (bs, 1, width, width, n_band)
    I = Input(shape=(args.input_shape), name="input_ids")
    
    # embedding (bs, 1, width, width, n_band)
    I_E = Conv3D(filters=args.model_input_channel, kernel_size=(1, 1, 7), strides=1, padding="same", data_format="channels_first", name="embedding")(I)

    # four-way FanMamba
    # r1
    # FanMamba block
    Fs = []
    original_n_circle = args.n_circle
    original_n_band = args.n_band
    F = I_E
    # copy args, record and update the number of circle, sequence_length, number of band after FSP
    args.n_circle = original_n_circle
    args.n_band = original_n_band
    n_reduced_pixels = args.n_pixel
    for r in args.redians:
        args.sequence_length_dict[r] = int((args.n_pixel - 1) / args.n_centralized_sequence_dict[r]) + 1
        # cs (bs, 1, 1, n_centralized_sequence*sequence_length, n_band)
        F = FS(redian=args.redians[r],
                  n_circle=args.n_circle,
                  n_centralized_sequence=args.n_centralized_sequence_dict[r],
                  sequence_length=args.sequence_length_dict[r],
                  steps=args.steps_dict[r][:args.n_circle],
                  recover_mode=False)(F)
        for i in range(args.num_blocks):
            print("\n redian", r, "- block", i)
            # FanMamba block (bs, 1, 1, n_centralized_sequence*sequence_length, n_band)
            F = ResidualBlock(args, r)(F)
            
            # conduct FSP when number of circle > 1, or only conduct avg pool in spectral dimension
            if args.n_circle > 1:
                # fan-shaped pooling in centralized sequences (bs, 1, 1, n_centralized_sequence*reduced_sequence_length, n_reduced_band)
                F = RFP(args.redians[r],
                        args.n_circle,
                        args.n_centralized_sequence_dict[r],
                        args.sequence_length_dict[r],
                        steps=args.steps_dict[r][:args.n_circle])(F)
                # update
                args.n_circle -= 1
                args.width -= 2
                args.n_pixel = args.width**2
                bs, model_internal_channel, _, n_reduced_pixels, args.n_band = F.shape
                args.sequence_length_dict[r] = n_reduced_pixels // args.n_centralized_sequence_dict[r]
            else:
                # do not pool in spatial dimension, but pool in spectral dimension
                if args.n_band > 2:
                    F = AveragePooling3D(pool_size=(1, 1, pool_band),
                                          strides=(1, 1, pool_band),
                                          data_format="channels_first",
                                          padding="valid")(F)
                    args.n_band = F.shape[-1]
            
            # for regularization
            F = Dropout(args.dropout_rate)(F)
            
        # cs recover (bs, 1, width, width, n_band)
        F = FS(redian=args.redians[r],
                  n_circle=args.n_circle,
                  n_centralized_sequence=args.n_centralized_sequence_dict[r],
                  sequence_length=args.sequence_length_dict[r],
                  steps=args.steps_dict[r][:args.n_circle],
                  recover_mode=True)(F)
        Fs.append(F)
    
    F4 = Fs[-1]
    F4 = LayerNormalization(epsilon=1e-5)(F4)
    F4 = GlobalAveragePooling3D(data_format="channels_first")(F4)
    P = Dense(args.num_classes, activation="softmax")(F4)
    
    model = Model(inputs=I, outputs=P, name="FanMamba")
    
    model.compile(loss=args.loss, optimizer=optimizer, metrics=args.metrics)
    
    return model


if __name__ == "__main__":
    from tensorflow.keras.optimizers import Adam
    adam = Adam(learning_rate=0.001)
    args = ModelArgs(input_shape=(1, 7, 7, 200),
                    model_input_channel=16,
                    model_state=16,
                    redians={"pi/4": pi/4, "pi/2": pi/2, "pi": pi, "2*pi": 2*pi},
                    num_blocks=1,
                    dropout_rate=0.2,
                    num_classes=3,
                    loss="categorical_crossentropy")
    model = init_model(args, adam)
    model.summary()
    
    pass
    

