#!/usr/bin/env python
# coding: utf-8

import keras
from keras.layers import Dropout, Activation, BatchNormalization, Dense, average, Lambda, Concatenate
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Dropout, AveragePooling2D
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model
# import pydot
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def conv2d_block(input_tensor, n_filters, kernel_size, strides = (2,2)):
    x = Conv2D(n_filters,(kernel_size,kernel_size), strides = strides, padding = 'same', kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization()(x)    
    x = Activation('relu')(x)
    return x


def inception_block_with_avgpool(inputs, num1x1, num3x3r, num3x3, num3x3dblr, num3x3dbl, numPool):
    
    #1*1 Conv
    b1x1 = conv2d_block(inputs, num1x1, 1, (1,1))
    
    #3*3 Conv
    b3x3reduce = conv2d_block(inputs, num3x3r, 1, (1,1))
    b3x3 = conv2d_block(b3x3reduce, num3x3r, 3, (1,1))
    
    #Second 3*3 Conv
    b3x3dreduce = conv2d_block(inputs, num3x3dblr, 1, (1,1))
    b3x3d_a = conv2d_block(b3x3dreduce, num3x3dbl, 3, (1,1))
    b3x3d = conv2d_block(b3x3d_a, num3x3dbl, 3, (1,1))
    
    #Average Pooling
    avg_pool = AveragePooling2D(pool_size=(3, 3),strides = (1,1), padding = 'same')(inputs)
    bpool = conv2d_block(avg_pool, numPool, 1, (1,1))
    
    #Concatenate
    out = concatenate([b1x1, b3x3, b3x3d, bpool], axis=3)
    return out


def inception_block_with_maxpool(inputs, num1x1, num3x3r, num3x3, num3x3dblr, num3x3dbl, numPool):
    
    #1*1 Conv
    b1x1 = conv2d_block(inputs, num1x1, 1, (1,1))
    
    #3*3 Conv
    b3x3reduce = conv2d_block(inputs, num3x3r, 1, (1,1))
    b3x3 = conv2d_block(b3x3reduce, num3x3r, 3, (1,1))
    
    #Second 3*3 Conv
    b3x3dreduce = conv2d_block(inputs, num3x3dblr, 1, (1,1))
    b3x3d_a = conv2d_block(b3x3dreduce, num3x3dbl, 3, (1,1))
    b3x3d = conv2d_block(b3x3d_a, num3x3dbl, 3, (1,1))
    
    #Max Pooling
    max_pool = MaxPooling2D(pool_size=(3, 3),strides = (1,1), padding = 'same')(inputs)
    bpool = conv2d_block(max_pool, numPool, 1, (1,1))
    
    #Concatenate
    out = concatenate([b1x1, b3x3, b3x3d, bpool], axis=3)
    return out


def inception_block_pass_through(inputs, num1x1, num3x3r, num3x3, num3x3dblr, num3x3dbl, numPool):
    #3*3 Conv
    b3x3reduce = conv2d_block(inputs, num3x3r, 1, (1,1))
    b3x3 = conv2d_block(b3x3reduce, num3x3r, 3, (2,2))
    
    #Second 3*3 Conv
    b3x3dreduce = conv2d_block(inputs, num3x3dblr, 1, (1,1))
    b3x3d_a = conv2d_block(b3x3dreduce, num3x3dbl, 3, (1,1))
    b3x3d = conv2d_block(b3x3d_a, num3x3dbl, 3, (2,2))
    
    #Max Pooling
    bpool = MaxPooling2D(pool_size=(3, 3),strides = (2,2), padding = 'same')(inputs)
    
    #Concatenate
    out = concatenate([b3x3, b3x3d, bpool], axis=3)
    return out


def squeeze(layer):
    return K.squeeze(K.squeeze(layer,axis = 1), axis = 1)


def bn_inception1(inputs, classes = 101):
    c1 = conv2d_block(inputs, 64, 7)
    p1 = MaxPooling2D(pool_size=(2, 2), padding = 'same')(c1)
    c2a = conv2d_block(p1,64, 1, (1,1))
    c2b = conv2d_block(p1,192, 3, (1,1))
    p2 = MaxPooling2D(pool_size=(2, 2), padding = 'same')(c2b)
    
    #Inception Blocks
    i3a = inception_block_with_avgpool(p2, 64, 64, 64, 64, 96, 32)
    i3b = inception_block_with_avgpool(i3a, 64, 64, 96, 64, 96, 64)
    i3c = inception_block_pass_through(i3b, 0, 128, 160, 64, 96, 0)
    i4a = inception_block_with_avgpool(i3c, 224, 64, 96, 96, 128, 128)
    i4b = inception_block_with_avgpool(i4a, 192, 96, 128, 96, 128, 128)
    i4c = inception_block_with_avgpool(i4b, 160, 128, 160, 128, 160, 128)
    i4d = inception_block_with_avgpool(i4c, 96, 128, 192, 160, 192, 128)
    i4e = inception_block_pass_through(i4d, 0, 128, 192, 192, 256, 0)
    i5a = inception_block_with_avgpool(i4e, 352, 192, 320, 160, 224, 128)
    i5b = inception_block_with_maxpool(i5a, 352, 192, 320, 192, 224, 128) 
    
    # Global Average
    p3 = AveragePooling2D(pool_size=(7,7))(i5b)
    
    res = Dense(classes, kernel_initializer='he_normal')(p3)
    out = Lambda(squeeze)(res)
    
    return out

def bn_inception2(inputs, classes = 101):
    
    c1 = conv2d_block(inputs, 64, 7)
    p1 = MaxPooling2D(pool_size=(2, 2), padding = 'same')(c1)
    c2a = conv2d_block(p1,64, 1, (1,1))
    c2b = conv2d_block(p1,192, 3, (1,1))
    p2 = MaxPooling2D(pool_size=(2, 2), padding = 'same')(c2b)
        
    inception3a = inception_block_with_avgpool(c2b, 32, 32, 32, 32, 48, 16)
    inception3b = inception_block_pass_through(inception3a, 0, 64, 80, 32, 48, 0) 
    inception4a = inception_block_with_avgpool(inception3b, 96, 48, 64, 48, 64, 64) 
    inception4b = inception_block_with_avgpool(inception4a, 48, 64, 96, 80, 96, 64) 
    inception4c = inception_block_pass_through(inception4b, 0, 128, 192, 192, 256, 0)
    inception5a = inception_block_with_maxpool(inception4c, 176, 96, 160, 96, 112, 64) 

    # Global Average
    p3 = AveragePooling2D(pool_size=(8,8))(inception5a)
    
    res = Dense(classes, kernel_initializer='he_normal')(p3)
    out = Lambda(squeeze)(res)
    
    return out


def Network(i1,i2,i3):
    o1 = bn_inception2(i1)
    o2 = bn_inception2(i2)
    o3 = bn_inception2(i3)
    out = average([o1, o2, o3])
    
#     model = Model(inputs = [i1,i2,i3], outputs =[out])
    return out

def TSN():
    i1 = Input(shape = (224,224,1), name = "input_1")
    i2 = Input(shape = (224,224,1), name = "input_2")
    i3 = Input(shape = (224,224,1), name = "input_3")
    outSpatial = Network(i1,i2,i3)


    i4 = Input(shape = (224,224,2), name = "input_4")
    i5 = Input(shape = (224,224,2), name = "input_5")
    i6 = Input(shape = (224,224,2), name = "input_6")
    outTemporal = Network(i4,i5,i6)

    res = average([outSpatial, outTemporal], name = 'output')
    model = Model(inputs = [i1,i2,i3,i4,i5,i6], outputs =[res])
    
    return model


# model = TSN()


# plot_model(model, to_file='model2.png', show_shapes = True)


# print(model.summary())

