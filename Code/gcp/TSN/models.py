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
    p3 = Dropout(rate = 0.5)(p3)
    res = Dense(classes, kernel_initializer='he_normal')(p3)
    out = Lambda(squeeze)(res)
    
    return out

def bn_inception2(inputs, classes = 101):
    
    c1 = conv2d_block(inputs, 64, 7)
    p1 = MaxPooling2D(pool_size=(2, 2), padding = 'same')(c1)
    c2a = conv2d_block(p1,64, 1, (1,1))
    c2b = conv2d_block(p1,192, 3, (1,1))
        
    inception3a = inception_block_with_avgpool(c2b, 32, 32, 32, 32, 48, 16)
    inception3b = inception_block_pass_through(inception3a, 0, 64, 80, 32, 48, 0) 
    inception4a = inception_block_with_avgpool(inception3b, 96, 48, 64, 48, 64, 64) 
    inception4b = inception_block_with_avgpool(inception4a, 48, 64, 96, 80, 96, 64) 
    inception4c = inception_block_pass_through(inception4b, 0, 128, 192, 192, 256, 0)
    inception5a = inception_block_with_maxpool(inception4c, 176, 96, 160, 96, 112, 64) 

    # Global Average
    p3 = AveragePooling2D(pool_size=(8,8))(inception5a)
    p3 = Dropout(rate = 0.5)(p3)
    res = Dense(classes, kernel_initializer='he_normal')(p3)
    out = Lambda(squeeze)(res)
    
    return out


def Network(i1,i2,i3):
    o1 = bn_inception2(i1)
    o2 = bn_inception2(i2)
    o3 = bn_inception2(i3)
    out = average([o1, o2, o3])
    
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
    x = base_model.output
    res = Dense(101,activation='relu')(x)
    
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
model = C3D_V2()
model.summary()