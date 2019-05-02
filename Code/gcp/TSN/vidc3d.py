import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from collections import defaultdict
import keras
from skimage.transform import resize
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from models import finalC3D
from keras.optimizers import SGD
from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score


class dataGenerator(keras.utils.Sequence):

    def __init__(self, filepath, batch_size, ffpath, segments = 3, test=False):
        np.random.seed(0)
        self.filenames = list()
        self.labels = list()
        self.batch_size = batch_size
        self.filepath = filepath
        self.ffpath = ffpath
        self.segments = segments
        self.DS_FACTOR = 2
        
        with open(self.filepath,"r") as f:
            for line in f.readlines():
                arr = line.split(" ")
                self.filenames.append(arr[0])
                self.labels.append(int(arr[1].strip()))
        
        length = len(self.filenames)
        self.idxs = np.random.permutation(length)
        
    def __len__(self):
        return len(self.filenames)//self.batch_size

    def __getitem__(self, idx):
        this_batch = self.idxs[idx * self.batch_size:min((idx + 1) * self.batch_size, len(self.filenames))]
        batch_x = list()
        batch_y = list()
        for each in this_batch:
            batch_x.append(self.filenames[each])
            batch_y.append(self.labels[each])          
        # batch_x = self.filenames[idx * self.batch_size:min((idx + 1) * self.batch_size, len(self.filenames))]
        # batch_y = self.labels[idx * self.batch_size:min((idx + 1) * self.batch_size, len(self.filenames))]

        Xframes = list()
        Y = list()
        for index,each in enumerate(batch_x):
            infopath = os.path.join(self.ffpath,each,"info.txt")
            imgpath = os.path.join(self.ffpath,each,"frames")
            f = open(infopath,"r")
            total_frames = int(f.readlines()[0].strip().split(':')[1])
            f.close()
            idxs = sorted(np.random.randint(0, total_frames, 16))
            Xframes.append(self.getFrames(idxs, imgpath))
            Y.append(batch_y[index])

        finalX, finalY = np.array(Xframes)[:, :, ::self.DS_FACTOR, ::self.DS_FACTOR], self.one_hot_encode(np.array(Y))
        return (finalX,finalY)
    
    def getTestData(self):
        batch_x = list()
        batch_y = list()
        for each in self.idxs:
            batch_x.append(self.filenames[each])
            batch_y.append(self.labels[each])
        
        Xframes = list()
        Y = list()
        for index,each in enumerate(batch_x):
            infopath = os.path.join(self.ffpath,each,"info.txt")
            imgpath = os.path.join(self.ffpath,each,"frames")
            f = open(infopath,"r")
            total_frames = int(f.readlines()[0].strip().split(':')[1])
            f.close()
            idxs = sorted(np.random.randint(0, total_frames, 16))
            Xframes.append(self.getFrames(idxs, imgpath))
            Y.append(batch_y[index])

        finalX, finalY = np.array(Xframes)[:, :, ::self.DS_FACTOR, ::self.DS_FACTOR], self.one_hot_encode(np.array(Y))
        return (finalX,finalY)
        
    
    def one_hot_encode(self,data, classes = 101):
        """
        :param data: data to be one hot encoded
        :return: np array with one hot encoding
        """
        labels = np.zeros((data.size, classes))
        labels[np.arange(data.size), data - 1] = 1
        return labels
    
    def readImg(self,path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return img
    
    def getFrames(self,idxs, imgpath):

        stack = list()
        for i in idxs:
            framename = "frame_"+str(i)+".jpg"
            stack.append(self.readImg(os.path.join(imgpath,framename)))
            
        return np.array(stack)



def readImg(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def getFrames(idxs, imgpath):

    stack = list()
    for i in idxs:
        framename = "frame_"+str(i)+".jpg"
        stack.append(readImg(os.path.join(imgpath,framename)))
        
    return np.array(stack)

def getTestDataForPrediction(filename, batch, ffpath):
    infopath = os.path.join(ffpath,filename,"info.txt")
    imgpath = os.path.join(ffpath,filename,"frames")
    f = open(infopath,"r")
    lines = f.readlines()
    total_frames = int(lines[0].strip().split(':')[1])
    ground_truth = int(lines[1].strip().split(':')[1])
    f.close()
    Xframes = list()
    for idx in range(math.ceil(total_frames/batch)):
        idxs = list(range(idx*batch, min((idx+1)*batch,total_frames)))
        Xframes.append(getFrames(idxs,imgpath))
        
    finalX = np.array(Xframes)[:, :, ::2, ::2]
    return finalX, ground_truth


def test():
    # filenameTest = "custom3Test.txt"
    # ffpath = "FramesFlows/custom3"
    # dgTest = dataGenerator(filenameTest, 16, ffpath)
    filenameTest = "GolfSwing/v_GolfSwing_g17_c01.avi"
    # filenameTest = "TableTennisShot/v_TableTennisShot_g21_c01.avi"
    ffpath = "FramesFlows/custom3"
    X,y = getTestDataForPrediction(filenameTest, 16, ffpath)

    #Create and Compile model
    saved_model_file = "weights-improvement-{}-{}.hdf5".format(str(20), str(0.81))
    K.clear_session()
    _, model = finalC3D()
    
    model.load_weights("weights/c3d/{}".format(saved_model_file))
        
    y_pred = K.argmax(model.predict(X), 1)
    y_pred = K.eval(y_pred)
    
    print (y)
    print (y_pred)

    
def main():
    filenameTrain = "custom3Train.txt"
    filenameVal = "custom3Val.txt"
    ffpath = "FramesFlows/custom3"
    dgTrain = dataGenerator(filenameTrain, 16, ffpath)

    ffpath = os.path.join("FramesFlows/custom3")
    dgVal = dataGenerator(filenameVal, 16, ffpath)


    #Create and Compile model
    K.clear_session()
    base_model, model = finalC3D()

    model_file = "/mnt/disks/disk1/project/weights/c3d/sports1M_weights_tf.h5"

    np.random.seed(0)
    create_new = True
    lr = 0.0001
    if not create_new and os.path.exists('c3d_model_checkpoints') and len(os.listdir("c3d_model_checkpoints")) > 0:
        print ("Found saved models")
        mydict = dict()
        sorted_files = sorted(os.listdir("c3d_model_checkpoints"), key = lambda x: int(x.split('-')[2]), reverse = True)
        saved_model = sorted_files[0]
        initial_epoch = int(saved_model.split('-')[2])
        base_model.load_weights(os.path.join("c3d_model_checkpoints",saved_model))

        sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        csv_logger = CSVLogger('c3d_training.log')
        #Checkpointing
        filepath="c3d_model_checkpoints/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        # es = EarlyStopping(monitor='val_acc', mode='max', min_delta=1, patience = 10)
        # callbacks_list = [checkpoint,es]
        callbacks_list = [checkpoint, csv_logger]

        print (initial_epoch)
        print (saved_model)

        #Fit generator
        history = model.fit_generator(dgTrain,initial_epoch = initial_epoch, epochs = 100,validation_data = dgVal, callbacks = callbacks_list)

    else:

        sgd = SGD(lr=lr, momentum=0.9, nesterov=True)

        base_model.load_weights(model_file)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary() 
        csv_logger = CSVLogger('c3d_training.log')
        #Checkpointing
        filepath="c3d_model_checkpoints/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only = True,  mode='max')
        # es = EarlyStopping(monitor='val_acc', mode='max', min_delta=1, patience = 10)
        # callbacks_list = [checkpoint,es]
        callbacks_list = [checkpoint, csv_logger]

        history = model.fit_generator(dgTrain, epochs = 50,validation_data = dgVal, callbacks=callbacks_list)

if __name__ == "__main__":
    do_test = True
    if do_test:
        test()
    else:
        main()
