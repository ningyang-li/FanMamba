# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 09:17:29 2025

@author: fes_m
"""

from tensorflow.keras.layers import Layer, Concatenate, Reshape
import tensorflow as tf
import numpy as np

pi = 3.141593

class FS(Layer):
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
    
    def __init__(self, redian=pi/4, n_circle=3, n_centralized_sequence=8, sequence_length=7, steps=[1, 2, 3], recover_mode=False, *args, **kwargs):
        '''
        default setting is for the case that width is 7 and redian is pi/4.
        '''
        super().__init__(*args, **kwargs)
        self.redian = redian
        self.n_circle = n_circle
        self.n_centralized_sequence = n_centralized_sequence
        self.sequence_length = sequence_length
        self.steps = steps
        self.recover_mode = recover_mode
        
        self.width = 2 * self.n_circle + 1
        self.center = (self.n_circle, self.n_circle)
        
    
    def call(self, x):
        '''
        x -> (bs, model_internal_channel, width, width, n_band)
        0-0   0-1   0-2   0-3   0-4   0-5   0-6
        1-0   1-1   1-2   1-3   1-4   1-5   1-6
        2-0   2-1   2-2   2-3   2-4   2-5   2-6
        3-0   3-1   3-2   3-3   3-4   3-5   3-6
        4-0   4-1   4-2   4-3   4-4   4-5   4-6
        5-0   5-1   5-2   5-3   5-4   5-5   5-6
        6-0   6-1   6-2   6-3   6-4   6-5   6-6
        
        fan_shaped_scan
        3-21  3-22  3-23  3-0   3-1   3-2   3-3
        3-20  2-14  2-15  2-0   2-1   2-2   3-4
        3-19  2-13  1-7   1-0   1-1   2-3   3-5
        3-18  2-12  1-6    c    1-2   2-4   3-6
        3-17  2-11  1-5   1-4   1-3   2-5   3-7
        3-16  2-10  2-9   2-8   2-7   2-6   3-8
        3-15  3-14  3-13  3-12  3-11  3-10  3-9
        
        y -> (bs, model_internal_channel, 1, n_central_sequence*sequence_length, n_band)
        if redian is pi/4,
        [[c, c1-p0, c2-p0,  c2-p1,  c3-p0,  c3-p1,  c3-p2],
         [c, c1-p1, c2-p3,  c2-p2,  c3-p5,  c3-p4,  c3-p3],
         [c, c1-p2, c2-p4,  c2-p5,  c3-p6,  c3-p7,  c3-p8],
         [c, c1-p3, c2-p7,  c2-p6,  c3-p11, c3-p10, c3-p9],
         [c, c1-p4, c2-p8,  c2-p9,  c3-p12, c3-p13, c3-p14],
         [c, c1-p5, c2-p11, c2-p10, c3-p17, c3-p16, c3-p15],
         [c, c1-p6, c2-p12, c2-p13, c3-p18, c3-p19, c3-p20],
         [c, c1-p7, c2-p15, c2-p14, c3-p23, c3-p22, c3-p21]]
        
        '''
        bs, model_internal_channel, width, width, n_band = x.shape
        # extract the center pixel (bs, model_internal_channel, 1, 1, n_band)
        xc = self.extract_center(x)
        
        # map to centralized space and sequence space
        self.generate_scan_direction()
        self.map_2_centralized_space()
        self.map_2_sequence_space()

        if self.recover_mode == False:
            # map x to centralized_sequences
            centralized_sequences = []
            for c in range(self.n_centralized_sequence):
                # add the center pixel at the first position of each sequence
                centralized_sequences.append(xc)
                for p in range(1, self.sequence_length):
                    key_y2ss = str(c) + "-" + str(p)
                    key_ss2cs = str(self.ss_map[key_y2ss][0]) + "-" + str(self.ss_map[key_y2ss][1])
                    i, j = self.cs_map[key_ss2cs]
                    centralized_sequences.append(x[:, :, i:i+1, j:j+1, :])
            
            t = centralized_sequences[0]
            for i in range(1, len(centralized_sequences)):
                t = Concatenate(axis=-2)([t, centralized_sequences[i]])
            
            self.centralized_sequences = Reshape((model_internal_channel, 1, self.n_centralized_sequence*self.sequence_length, n_band))(t)

            return self.centralized_sequences
        else:
            # save the center pixel, the mean of all the center pixels in centralized sequences
            xc = x[:, :, :, 0:1, :]
            xc = tf.reduce_sum(xc, axis=2, keepdims=True)
            
            # recover centralized_sequences to 1-d centralized sequences
            x = Reshape((model_internal_channel, 1, self.n_centralized_sequence*(self.sequence_length), n_band))(x)

            recover_x = []
            for i in range(self.width):
                for j in range(self.width):
                    if i == self.n_circle and j == self.n_circle: # add the center pixel in self.n_centralized_sequence times
                        recover_x.append(xc)
                    else: # add circular pixels
                        pos_in_cs = (self.m_circle[i, j], self.m_position[i, j])
                        c, p = pos_in_ss = np.where((self.centralized_sequences_circle == pos_in_cs[0]) & (self.centralized_sequences_position == pos_in_cs[1]) == True)
                        c = int(c[0])
                        p = int(p[0])
                        recover_x.append(x[:, :, :, self.sequence_length*c+p:self.sequence_length*c+p+1, :])

            t = recover_x[0]
            for i in range(1, len(recover_x)):
                t = Concatenate(axis=-2)([t, recover_x[i]])

            self.recover_x = Reshape((model_internal_channel, self.width, self.width, n_band))(t)

            return self.recover_x
                
    
    def extract_center(self, x):
        return x[:, :, self.center[0]:self.center[0]+1, self.center[1]:self.center[1]+1, :]
    
    
    def map_2_centralized_space(self):
        '''
        map
        
        0-0   0-1   0-2   0-3   0-4   0-5   0-6
        1-0   1-1   1-2   1-3   1-4   1-5   1-6
        2-0   2-1   2-2   2-3   2-4   2-5   2-6
        3-0   3-1   3-2   3-3   3-4   3-5   3-6
        4-0   4-1   4-2   4-3   4-4   4-5   4-6
        5-0   5-1   5-2   5-3   5-4   5-5   5-6
        6-0   6-1   6-2   6-3   6-4   6-5   6-6
        
        to
        
        3-21  3-22  3-23  3-0   3-1   3-2   3-3
        3-20  2-14  2-15  2-0   2-1   2-2   3-4
        3-19  2-13  1-7   1-0   1-1   2-3   3-5
        3-18  2-12  1-6    c    1-2   2-4   3-6
        3-17  2-11  1-5   1-4   1-3   2-5   3-7
        3-16  2-10  2-9   2-8   2-7   2-6   3-8
        3-15  3-14  3-13  3-12  3-11  3-10  3-9
        
        with circular traverse.
        
        taking the second circle as an example, from start_point_c2 (n_circle - 2, n_circle)=(1, 3),
        traverse toward right, down, left, up, right directions to label pixels,
        traverse direction will change when current pixel is the corner of circle,
        traverse process ends when return to the start point.
        '''
        start_points = []
        pixel_each_circle = []
        for i in range(self.n_circle+1):
            start_points.append([self.n_circle - i, self.n_circle])
            pixel_each_circle.append((1 + i * 2) * (1 + i * 2) - (1+ (i - 1) * 2) * (1+ (i - 1) * 2))
                
        def is_corner(i, j, center, id_circle):
            if abs(i - center[0]) == id_circle and abs(j - center[1]) == id_circle:
                return True
            else:
                return False
        
        # circular traverse
        self.m_circle = np.zeros((self.width, self.width), dtype="int32")
        self.m_position = np.zeros((self.width, self.width), dtype="int32")
        
        # label the center pixel
        self.m_circle[self.n_circle, self.n_circle] = 0
        self.m_position[self.n_circle, self.n_circle] = 0
        
        # label circles
        for c in range(1, self.n_circle+1):
            start = start_points[c]
            i = start[0]
            j = start[1]
            # "right": (0, 1), "down": (1, 0), "left": (0, -1), "up": (-1, 0), "right": (0, 1)
            traverse_step = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0), 4: (0, 1)}
            traverse_direction = 0
            step = traverse_step[traverse_direction]
            
            for p in range(pixel_each_circle[c]):
                self.m_circle[i, j] = c
                self.m_position[i, j] = p
                
                if is_corner(i, j, self.center, c):
                    # adjust direction
                    traverse_direction += 1
                    step = traverse_step[traverse_direction]
                    
                i += step[0]
                j += step[1]
        
        # map
        self.cs_map = {}
        
        for i in range(self.width):
            for j in range(self.width):
                key = str(self.m_circle[i, j]) + "-" + str(self.m_position[i, j])
                self.cs_map[key] = (i, j)
                
        # recover
        self.cs_recover = {}
        
        for i in range(self.width):
            for j in range(self.width):
                key = str(self.m_circle[i, j]) + "-" + str(self.m_position[i, j])
                self.cs_map[key] = (i, j)
            
    
    def generate_scan_direction(self):
        '''
        for sample width 9,
        
        redian = pi/4
        direction -> (n_centralized_sequence, n_circle) = (8, 4)
        [[ 1,  1,  1,  1],
         [-1, -1, -1, -1],
         [ 1,  1,  1,  1],
         [-1, -1, -1, -1],
         [ 1,  1,  1,  1],
         [-1, -1, -1, -1],
         [ 1,  1,  1,  1],
         [-1, -1, -1, -1]]
        note: the first directions of 2nd, 4-th, 6-th, and 8-th sequences can also be 1 because the step in first circel is 1.
        
        redian = pi/2
        direction -> (n_centralized_sequence, n_circle) = (4, 4)
        [[1, -1, 1, -1],
         [1, -1, 1, -1],
         [1, -1, 1, -1],
         [1, -1, 1, -1]]
        
        redian = pi/4
        direction -> (n_centralized_sequence, n_circle) = (2, 4)
        [[1, -1, 1, -1],
         [1, -1, 1, -1]]
        
        redian = pi/4
        direction -> (n_centralized_sequence, n_circle) = (1, 4)
        [[1, 1, 1, 1]]
        '''
        # expand
        if self.redian == pi/4:
            direction_kernel = np.array([[1], [-1]], dtype="int32")
            dir_0 = np.tile(direction_kernel, reps=int(self.n_centralized_sequence/2))
            dir_0 = np.reshape(dir_0, (1, self.n_centralized_sequence))
            dir_0 = np.tile(dir_0, reps=self.n_circle)
            dir_0 = np.reshape(dir_0, (self.n_centralized_sequence, self.n_circle))
            self.direction = dir_0
            
        elif self.redian == pi/2 or self.redian == pi:
            # both share the same process as follow
            direction_kernel = np.array([[1, -1]], dtype="int32")
            if self.n_circle > direction_kernel.shape[1]:
                rate = int(np.floor(self.n_circle / direction_kernel.shape[1]))
                dir_1 = np.tile(direction_kernel, reps=rate)
                if self.n_circle % direction_kernel.shape[1] != 0:
                    dir_1 = np.concatenate((dir_1, direction_kernel[:, 0:1]), axis=-1)
            else:
                dir_1 = direction_kernel[:, 0:self.n_circle]
                
            dir_1 = np.tile(dir_1, reps=self.n_centralized_sequence)
            dir_1 = np.reshape(dir_1, (self.n_centralized_sequence, self.n_circle))
            self.direction = dir_1
            
        elif self.redian == 2*pi:
            direction_kernel = np.array([[1]], dtype="int32")
            dir_3 = np.tile(direction_kernel, reps=self.n_circle)
            dir_3 = np.tile(dir_3, reps=self.n_centralized_sequence)
            dir_3 = np.reshape(dir_3, (self.n_centralized_sequence, self.n_circle))
            self.direction = dir_3
        else:
            raise ValueError("redian should be one of (pi/4, pi/2, pi, 2pi)")
    
    
    def map_2_sequence_space(self):
        '''
        for sample width 9,
        
        redian = pi/4
            direction -> (n_central_sequence, n_circle) = (8, 4)
            [[ 1,  1,  1,  1],
             [-1, -1, -1, -1],
             [ 1,  1,  1,  1],
             [-1, -1, -1, -1],
             [ 1,  1,  1,  1],
             [-1, -1, -1, -1],
             [ 1,  1,  1,  1],
             [-1, -1, -1, -1]]
            note: the first directions of 2nd, 4-th, 6-th, and 8-th sequences can also be 1 because the step in first circel is 1.
            
            centralized_sequences (8, 11)
            [[c, c1-p0, c2-p0,  c2-p1,  c3-p0,  c3-p1,  c3-p2,  c4-p0,  c4-p1,  c4-p2,  c4-p3],
             [c, c1-p1, c2-p3,  c2-p2,  c3-p5,  c3-p4,  c3-p3,  c4-p7,  c4-p6,  c4-p5,  c4-p4],
             [c, c1-p2, c2-p4,  c2-p5,  c3-p6,  c3-p7,  c3-p8,  c4-p8,  c4-p9,  c4-p10, c4-p11],
             [c, c1-p3, c2-p7,  c2-p6,  c3-p11, c3-p10, c3-p9,  c4-p15, c4-p14, c4-p13, c4-p12],
             [c, c1-p4, c2-p8,  c2-p9,  c3-p12, c3-p13, c3-p14, c4-p16, c4-p17, c4-p18, c4-p19],
             [c, c1-p5, c2-p11, c2-p10, c3-p17, c3-p16, c3-p15, c4-p23, c4-p22, c4-p21, c4-p20],
             [c, c1-p6, c2-p12, c2-p13, c3-p18, c3-p19, c3-p20, c4-p24, c4-p25, c4-p26, c4-p27],
             [c, c1-p7, c2-p15, c2-p14, c3-p23, c3-p22, c3-p21, c4-p31, c4-p30, c4-p29, c4-p28]]
        
        redian = pi/2
            direction -> (n_central_sequence, n_circle) = (4, 4)
            [[1, -1, 1, -1],
             [1, -1, 1, -1],
             [1, -1, 1, -1],
             [1, -1, 1, -1]]
            
            centralized_sequences (4, 21)
            [[c, c1-p0, c1-p1, c2-p3,  c2-p2,  c2-p1,  c2-p0,  c3-p0,  c3-p1,  c3-p2,  c3-p3,  c3-p4,  c3-p5,  c4-p7,  c4-p6,  c4-p5,  c4-p4,  c4-p3,  c4-p2,  c4-p1,  c4-p0],
             [c, c1-p2, c1-p3, c2-p7,  c2-p6,  c2-p5,  c2-p4,  c3-p6,  c3-p7,  c3-p8,  c3-p9,  c3-p10, c3-p11, c4-p15, c4-p14, c4-p13, c4-p12, c4-p11, c4-p10, c4-p9,  c4-p8],
             [c, c1-p4, c1-p5, c2-p11, c2-p10, c2-p9,  c2-p8,  c3-p12, c3-p13, c3-p14, c3-p15, c3-p16, c3-p17, c4-p23, c4-p22, c4-p21, c4-p20, c4-p19, c4-p18, c4-p17, c4-p16],
             [c, c1-p6, c1-p7, c2-p15, c2-p14, c2-p13, c2-p12, c3-p18, c3-p19, c3-p20, c3-p21, c3-p22, c3-p23, c4-p31, c4-p30, c4-p29, c4-p28, c4-p27, c4-p26, c4-p25, c4-p24]]
        
        redian = pi/4
            direction -> (n_central_sequence, n_circle) = (2, 4)
            [[1, -1, 1, -1],
             [1, -1, 1, -1]]
            
            centralized_sequences (2, 41)
            [[c, c1-p0, c1-p1, c1-p2, c1-p3, c2-p7,  c2-p6,  c2-p5,  c2-p4,  c2-p3,  c2-p2,  c2-p1,  c2-p0,  c3-p0,  c3-p1,  c3-p2,  c3-p3,  c3-p4,  c3-p5,  c3-p6,  c3-p7,  c3-p8,  c3-p9,  c3-p10, c3-p11, c4-p15, c4-p14, c4-p13, c4-p12, c4-p11, c4-p10, c4-p9,  c4-p8,  c4-p7,  c4-p6,  c4-p5,  c4-p4,  c4-p3,  c4-p2,  c4-p1,  c4-p0],
             [c, c1-p4, c1-p5, c1-p6, c1-p7, c2-p15, c2-p14, c2-p13, c2-p12, c2-p11, c2-p10, c2-p9,  c2-p8,  c3-p12, c3-p13, c3-p14, c3-p15, c3-p16, c3-p17, c3-p18, c3-p19, c3-p20, c3-p21, c3-p22, c3-p23, c4-p31, c4-p30, c4-p29, c4-p28, c4-p27, c4-p26, c4-p25, c4-p24, c4-p23, c4-p22, c4-p21, c4-p20, c4-p19, c4-p18, c4-p17, c4-p16]]
        
        redian = pi/4
            direction -> (n_central_sequence, n_circle) = (1, 4)
            [[1, 1, 1, 1]]
        
            centralized_sequences (1, 81)
            [[c,
              c1-p0, c1-p1, c1-p2, c1-p3, c1-p4, c1-p5, c1-p6, c1-p7,
              c2-p0, c2-p1, c2-p2, c2-p3, c2-p4, c2-p5, c2-p6, c2-p7, c2-p8, c2-p9, c2-p10, c2-p11, c2-p12, c2-p13, c2-p14, c2-p15,
              c3-p0, c3-p1, c3-p2, c3-p3, c3-p4, c3-p5, c3-p6, c3-p7, c3-p8, c3-p9, c3-p10, c3-p11, c3-p12, c3-p13, c3-p14, c3-p15, c3-p16, c3-p17, c3-p18, c3-p19, c3-p20, c3-p21, c3-p22, c3-p23,
              c4-p0, c4-p1, c4-p2, c4-p3, c4-p4, c4-p5, c4-p6, c4-p7, c4-p8, c4-p9, c4-p10, c4-p11, c4-p12, c4-p13, c4-p14, c4-p15, c4-p16, c4-p17, c4-p18, c4-p19, c4-p20, c4-p21, c4-p22, c4-p23, c4-p24, c4-p25, c4-p26, c4-p27, c4-p28, c4-p29, c4-p30, c4-p31]
        '''
        self.centralized_sequences_circle = np.zeros((self.n_centralized_sequence, self.sequence_length), dtype="int32")
        self.centralized_sequences_position = np.zeros((self.n_centralized_sequence, self.sequence_length), dtype="int32")
        
        start_position_per_circle = np.zeros((self.n_circle,), dtype="int32")
        for cs in range(self.n_centralized_sequence):
            # first add the center pixel
            self.centralized_sequences_circle[cs, 0] = 0
            self.centralized_sequences_position[cs, 0] = 0
            
            # then add circles
            steps_cur_sequence = 1
            for c in range(self.n_circle):
                # how many pixels have been scanned
                temp_circle = []
                temp_position = []
                for s in range(self.steps[c]):
                    temp_circle.append(c + 1)
                    temp_position.append(start_position_per_circle[c] + s)
                    
                # flip, if direction is -1
                if self.direction[cs, c] == -1:
                    temp_position.reverse()

                # save
                self.centralized_sequences_circle[cs, steps_cur_sequence:steps_cur_sequence + len(temp_circle)] = np.array(temp_circle)
                self.centralized_sequences_position[cs, steps_cur_sequence:steps_cur_sequence + len(temp_position)] = np.array(temp_position)
                
                # update steps and start position
                steps_cur_sequence += len(temp_position)
                start_position_per_circle[c] += len(temp_position)
        
        # if redian is 2pi, deviate the positions of the second, third, ... , circles, to right 1, 2, ..., pixels
        if self.redian == 2*pi and self.n_circle > 1:
            for i in range(2, self.n_circle+1):
                # copy undeviated positions
                old_positions = self.centralized_sequences_position[:, (1 + 2 * (i - 1)) ** 2:(1 + 2 * i) ** 2]
                positions_move_to_front = old_positions[:, -(i-1):].copy()
                positions_move_to_back = old_positions[:, :-(i-1)].copy()
                new_positions = old_positions.copy()
                new_positions[:, :(i-1)] = positions_move_to_front
                new_positions[:, (i-1):] = positions_move_to_back
                self.centralized_sequences_position[:, (1 + 2 * (i - 1)) ** 2:(1 + 2 * i) ** 2] = new_positions
        
        self.ss_map = {}
        
        for i in range(self.n_centralized_sequence):
            for j in range(self.sequence_length):
                key = str(i) + "-" + str(j)
                self.ss_map[key] = (self.centralized_sequences_circle[i, j], self.centralized_sequences_position[i, j])
                
        
if __name__ == "__main__":
    print("Centralized Scan Test")
    
    f = np.ones((1, 1, 8, 11, 200))
    x = FS(redian=pi/4, n_circle=4, n_centralized_sequence=8, sequence_length=11, steps=[1, 2, 3, 4], recover_mode=True)(f)
    
        
