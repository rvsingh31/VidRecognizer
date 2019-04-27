#!/usr/bin/env python
# coding: utf-8
import keras
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, average
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils import plot_model
from inception import InceptionResNetV2
# import pydot
import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'



def Network(i1,i2,i3):
    bm1,fm1 = inception(i1)
    bm2,fm2 = inception(i2)
    bm3,fm3 = inception(i3)
    out = average([fm1.output, fm2.output, fm3.output])

    return out


def TSN():

    #Spatial Layers
    with K.name_scope('SpatialLayer1'):
        i1 = Input(shape = (224,224,3), name = "input_1")
        bm1 = InceptionResNetV2(input_tensor = i1, weights='imagenet', include_top=False)
        x = bm1.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(rate = 0.2)(x)
        sp1 = Dense(101, activation='softmax')(x)
        for layer in bm1.layers:
            layer.trainable = False

    with K.name_scope('SpatialLayer2'):
        i2 = Input(shape = (224,224,3), name = "input_2")
        bm2 = InceptionResNetV2(input_tensor = i2, weights='imagenet', include_top=False)
        x = bm2.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(rate = 0.2)(x)
        sp2 = Dense(101, activation='softmax')(x)
        for layer in bm2.layers:
            layer.trainable = False

    with K.name_scope('SpatialLayer3'):
        i3 = Input(shape = (224,224,3), name = "input_3")
        bm3 = InceptionResNetV2(input_tensor = i3, weights='imagenet', include_top=False)
        x = bm3.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(rate = 0.2)(x)
        sp3 = Dense(101, activation='softmax')(x)
        for layer in bm3.layers:
            layer.trainable = False

    out1 = average([sp1, sp2, sp3])


    #Temporal Layers
    with K.name_scope('TemporalLayer1'):
        i4 = Input(shape = (224,224,3), name = "input_4")
        bm4 = InceptionResNetV2(input_tensor = i4, weights='imagenet', include_top=False)
        x = bm4.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(rate = 0.2)(x)
        sp4 = Dense(101, activation='softmax')(x)
        for layer in bm4.layers:
            layer.trainable = False

    with K.name_scope('TemporalLayer2'):
        i5 = Input(shape = (224,224,3), name = "input_5")
        bm5 = InceptionResNetV2(input_tensor = i5, weights='imagenet', include_top=False)
        x = bm5.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(rate = 0.2)(x)
        sp5 = Dense(101, activation='softmax')(x)
        for layer in bm5.layers:
            layer.trainable = False

    with K.name_scope('TemporalLayer3'):   
        i6 = Input(shape = (224,224,3), name = "input_6")
        bm6 = InceptionResNetV2(input_tensor = i6, weights='imagenet', include_top=False)
        x = bm6.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(rate = 0.2)(x)
        sp6 = Dense(101, activation='softmax')(x)
        for layer in bm6.layers:
            layer.trainable = False

    out2 = average([sp4, sp5, sp6])

    #Combined
    res = average([out1, out2], name = 'output')
    model = Model(inputs = [i1,i2,i3,i4,i5,i6], outputs =[res])

    return model

# model = TSN()
# plot_model(model, to_file='pretrained_model_iv2.png', show_shapes = True)
# print(model.summary())