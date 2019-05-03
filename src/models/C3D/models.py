#!/usr/bin/env python
# coding: utf-8

from keras.layers import Dropout, Dense, Flatten
from keras.layers import Conv3D, MaxPooling3D, Convolution3D, ZeroPadding3D
from keras.models import Model, Sequential


def C3D_V2(summary=False, backend='tf'):
    model = Sequential()
    if backend == 'tf':
        input_shape=(16, 112, 112, 3) # l, h, w, c
    else:
        input_shape=(3, 16, 112, 112)
    # 1st layer group
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv1',
                            subsample=(1, 1, 1), 
                            input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv2',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv3a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv3b',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool3'))
    # 4th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv4a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv4b',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool4'))
    # 5th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv5a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv5b',
                            subsample=(1, 1, 1)))
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool5'))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))

    # since we use Sports1M for pre-training we have 487 classes here.
    model.add(Dense(487, activation='relu', name='fc8'))
    if summary:
        print(model.summary())
    return model

def finalC3D():
    # Base model where we will use pre-trained weights
    base_model = C3D_V2()
    
    # adding a Dense layer at the bottom of base model
    x = base_model.output
    x = Dropout(.5)(x)
    
    # 101 classes for UCF, hence this layer.
    res = Dense(101,activation='softmax')(x)
    
    return base_model, Model(inputs=[base_model.input], outputs=[res])
