#!/usr/bin/env python
# coding: utf-8

import keras
from keras.layers import Dropout, Activation, BatchNormalization, Dense, average, Lambda, Concatenate, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Dropout, AveragePooling2D, ConvLSTM2D, Conv3D, MaxPooling3D, GlobalAveragePooling3D, MaxPool3D, Convolution3D, ZeroPadding3D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.utils import plot_model
# import pydot
import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def conv2d_block(input_tensor, n_filters, kernel_size, strides = (2,2)):
    x = Conv2D(n_filters,(kernel_size,kernel_size), strides = strides, padding = 'same', kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization()(x)    
    x = Activation('relu')(x)
    return x

def squeeze3d(layer):
    print(layer.shape)
    return K.squeeze(layer,axis = 1)


def conv3d_block(input_tensor, n_filters = 16, kernel_size = 3, strides = (1,1,1)):
    x = Conv3D(n_filters,(kernel_size, kernel_size, kernel_size), strides = strides, padding = 'same', kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization()(x)    
    x = Activation('relu')(x)
    return x


def convlstm2d_block(input_tensor, n_filters = 64, kernel_size = 3, strides = (1, 1)):
    x = ConvLSTM2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), strides=strides, padding='same', kernel_initializer='he_normal', recurrent_initializer='he_normal', return_sequences=True)(input_tensor)
    return x


def NetworkC3D(inp, num_classes = 101):
    c1 = conv3d_block(inp, n_filters = 16)
    mp1 = MaxPooling3D(pool_size=(2,2,2), padding = 'same')(c1)
    c2 = conv3d_block(mp1, n_filters = 32)
    c3 = conv3d_block(c2, n_filters = 32)
    mp2 = MaxPooling3D(pool_size=(1,2,2), padding = 'same')(c3)
    c4 = conv3d_block(mp2, n_filters = 64)
    c5 = conv3d_block(c4, n_filters = 64)
    c6 = conv3d_block(c5, n_filters = 64)
    mp3 = MaxPooling3D(pool_size=(1,2,2), padding = 'same')(c6)
    c7 = conv3d_block(mp3, n_filters = 64)
    c8 = conv3d_block(c7, n_filters = 64)
    c9 = conv3d_block(c8, n_filters = 64)
    mp4 = MaxPooling3D(pool_size=(1,2,2), padding = 'same')(c9)
    
    cl1 = convlstm2d_block(mp4)
    
    g1 = GlobalAveragePooling3D()(cl1)
    d1 = Dropout(rate = 0.5)(g1)
    res = Dense(num_classes, kernel_initializer='he_normal')(d1)
#    out = Lambda(squeeze3d)(res)
    out = res
    return out


def C3D_f():
    inp = Input(shape = (16, 224,224,3), name = "input_1")
    out = NetworkC3D(inp)
    model = Model(inputs = [inp], outputs =[out])
    return model

def C3D():
    # Define model
    l2=keras.regularizers.l2
    nb_classes = 101
    weight_decay = 0.00005
    patch_size, img_cols, img_rows = 16, 224, 224
    model = Sequential()
    model.add(Conv3D(16,(3,3,3),
                            input_shape=(patch_size, img_cols, img_rows, 3),
                            activation='relu'))
    model.add(Conv3D(16,(3,3,3), strides=(1,1,1),padding='same', 
                        dilation_rate=(1,1,1), kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay), use_bias=False, 
                        name='Conv3D_2a_a', activation = 'relu'))
    model.add(MaxPooling3D(pool_size=(2,2,2)))


    model.add(Conv3D(32,(3,3,3), strides=(1,1,1),padding='same', 
                        dilation_rate=(1,1,1), kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay), use_bias=False, 
                        name='Conv3D_2b_a', activation = 'relu'))
    model.add(Conv3D(32,(3,3,3), strides=(1,1,1),padding='same', 
                        dilation_rate=(1,1,1), kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay), use_bias=False, 
                        name='Conv3D_2b_b', activation = 'relu'))
    model.add(MaxPooling3D(pool_size=(1, 2,2)))


    model.add(Conv3D(64,(3,3,3), strides=(1,1,1),padding='same', 
                        dilation_rate=(1,1,1), kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay), use_bias=False, 
                        name='Conv3D_2c_a', activation = 'relu'))
    model.add(Conv3D(64,(3,3,3), strides=(1,1,1),padding='same', 
                        dilation_rate=(1,1,1), kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay), use_bias=False, 
                        name='Conv3D_2c_b', activation = 'relu'))
    model.add(Conv3D(64,(3,3,3), strides=(1,1,1),padding='same', 
                        dilation_rate=(1,1,1), kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay), use_bias=False, 
                        name='Conv3D_2c_c', activation = 'relu'))
    model.add(MaxPooling3D(pool_size=(1, 2,2)))


    model.add(Conv3D(128,(3,3,3), strides=(1,1,1),padding='same', 
                        dilation_rate=(1,1,1), kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay), use_bias=False, 
                        name='Conv3D_2d_a', activation = 'relu'))
    model.add(Conv3D(128,(3,3,3), strides=(1,1,1),padding='same', 
                        dilation_rate=(1,1,1), kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay), use_bias=False, 
                        name='Conv3D_2d_b', activation = 'relu'))
    model.add(Conv3D(128,(3,3,3), strides=(1,1,1),padding='same', 
                        dilation_rate=(1,1,1), kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay), use_bias=False, 
                        name='Conv3D_2d_c', activation = 'relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))



    model.add(ConvLSTM2D(filters=64, kernel_size=(3,3),
                      strides=(1,1),padding='same',
                          kernel_initializer='he_normal', recurrent_initializer='he_normal',
                          kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                          return_sequences=True, name='gatedclstm2d_2'))

    model.add(ConvLSTM2D(filters=64, kernel_size=(3,3),
                      strides=(1,1),padding='same',
                          kernel_initializer='he_normal', recurrent_initializer='he_normal',
                          kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                          return_sequences=True, name='gatedclstm2d_3'))

    model.add(ConvLSTM2D(filters=64, kernel_size=(3,3),
                      strides=(1,1),padding='same',
                          kernel_initializer='he_normal', recurrent_initializer='he_normal',
                          kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                          return_sequences=True, name='gatedclstm2d_4'))


    #model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))
    #model.add(Flatten())
    model.add(GlobalAveragePooling3D())
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes,kernel_initializer='normal'))

    model.add(Activation('softmax'))
    
    return model

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
    model.add(Dense(487, activation='relu', name='fc8'))
    if summary:
        print(model.summary())
    return model

def finalC3D():
    base_model = C3D_V2()
    for layer in base_model.layers[:-3]:
        layer.trainable = False
    x = base_model.output
    x = Dropout(.5)(x)
    res = Dense(101,activation='softmax')(x)
    
    return base_model, Model(inputs=[base_model.input], outputs=[res])
    

def c3d_model():
    input_shape = (16,112,112,3)
    weight_decay = 0.005
    nb_classes = 101

    inputs = Input(input_shape)
    x = Conv3D(64,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(inputs)
    x = MaxPool3D((2,2,1),strides=(2,2,1),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes,kernel_regularizer=l2(weight_decay))(x)
    x = Activation('softmax')(x)

    model = Model(inputs, x)
    return model
