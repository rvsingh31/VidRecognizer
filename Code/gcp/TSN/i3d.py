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
from i3d_inception import finalI3D
from keras.optimizers import SGD
from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score


class dataGenerator(keras.utils.Sequence):

    def __init__(self, filepath, batch_size, ffpath, segments = 16, test=False, data_type="FRAME"):
        np.random.seed(0)
        self.filenames = list()
        self.labels = list()
        self.batch_size = batch_size
        self.filepath = filepath
        self.ffpath = ffpath
        self.segments = segments
        self.DS_FACTOR = 1
        self.DATA_TYPE = data_type
        
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

        X = list()
        Y = list()
        for index,each in enumerate(batch_x):
            infopath = os.path.join(self.ffpath,each,"info.txt")
            f = open(infopath,"r")
            total_frames = int(f.readlines()[0].strip().split(':')[1])
            f.close()
            idxs = sorted(np.random.randint(1, total_frames, 16))
            if self.DATA_TYPE == "FRAME":
                imgpath = os.path.join(self.ffpath,each,"frames")
                X.append(self.getFrames(idxs, imgpath))
            else:
                imgpath = os.path.join(self.ffpath,each,"flows")
                X.append(self.getFlows(idxs, imgpath))
            Y.append(batch_y[index])

        finalX, finalY = np.array(X), self.one_hot_encode(np.array(Y))
        return (finalX,finalY)

    
    def getFlows(self,idxs, flowspath):

        stack = list()
        for i in idxs:
            f1 = "flow_x_"+str(i)+".jpg"
            f2 = "flow_y_"+str(i)+".jpg"
            grayx = self.readImg(os.path.join(flowspath,f1))
            grayy = self.readImg(os.path.join(flowspath,f2))
            img = np.stack((grayx, grayx, grayy),axis = 2)
            img = np.squeeze(img,axis = 3)
            stack.append(img)
            
        return np.array(stack)
    
    def getTestData(self):
        batch_x = list()
        batch_y = list()
        for each in self.idxs:
            batch_x.append(self.filenames[each])
            batch_y.append(self.labels[each])
        
        X = list()
        Y = list()
        for index,each in enumerate(batch_x):
            infopath = os.path.join(self.ffpath,each,"info.txt")
            imgpath = os.path.join(self.ffpath,each,"frames")
            f = open(infopath,"r")
            total_frames = int(f.readlines()[0].strip().split(':')[1])
            f.close()
            idxs = sorted(np.random.randint(0, total_frames, 16))
            if self.DATA_TYPE == "FRAME":
                X.append(self.getFrames(idxs, imgpath))
            else:
                X.append(self.getFlows(idxs, imgpath))
            Y.append(batch_y[index])

        finalX, finalY = np.array(X), self.one_hot_encode(np.array(Y))
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
        if self.DATA_TYPE != "FRAME":
            img = resize(img,(img.shape[0],img.shape[1],1))
        return img
    
    def getFrames(self,idxs, imgpath):

        stack = list()
        for i in idxs:
            framename = "frame_"+str(i)+".jpg"
            stack.append(self.readImg(os.path.join(imgpath,framename)))
            
        return np.array(stack)

def test():
    filenameTest = "custom3Test.txt"
    ffpath = "FramesFlows/custom3"
    dgTestFrame = dataGenerator(filenameTest, 16, ffpath, data_type="FRAME")
    dgTestFlow = dataGenerator(filenameTest, 16, ffpath, data_type="FLOW")

    #Create and Compile model
    saved_model_frame = "weights-improvement-{}-{}.hdf5".format('', '')
    saved_model_flow = "weights-improvement-{}-{}.hdf5".format('', '')
    K.clear_session()
    _, modelFrame = finalI3D(input_shape=(16, 224, 224, 3))
    _, modelFlow = finalI3D(input_shape=(16, 224, 224, 2))
    
    modelFrame.load_weights("i3d_model_checkpoints/frame/{}".format(saved_model_frame))
    modelFlow.load_weights("i3d_model_checkpoints/flow/{}".format(saved_model_flow))
    
    XFrame, y = dgTest.dgTestFrame()
    y_pred_frame = K.argmax(modelFrame.predict(XFrame), 1)
    y_pred_frame = K.eval(y_pred_frame)
    
    y_true = K.eval(K.argmax(y, 1))
    
    XFlow, _ = dgTest.dgTestFrame()
    y_pred_flow = K.argmax(modelFlow.predict(XFlow), 1)
    y_pred_flow = K.eval(y_pred_flow)

    del XFlow
    gc.collect()
    
    print(confusion_matrix(y_true, (y_pred_flow + y_pred_frame)/2.))
    print(accuracy_score(y_true, (y_pred_flow + y_pred_frame)/2.))
    

    
def main(dtype="FRAME"):
    filenameTrain = "custom3Train.txt"
    filenameVal = "custom3Val.txt"
    ffpath = "FramesFlows/custom3"
    dgTrain = dataGenerator(filenameTrain, 16, ffpath, data_type=dtype)

    ffpath = os.path.join("FramesFlows/custom3")
    dgVal = dataGenerator(filenameVal, 16, ffpath, data_type=dtype)


    #Create and Compile model
    K.clear_session()
    base_model, model = finalI3D(input_shape=(16, 224, 224, 3))


    np.random.seed(0)
    create_new = True
    lr = 0.0001
    if not create_new and os.path.exists('i3d_model_checkpoints') and len(os.listdir("i3d_model_checkpoints")) > 0:
        print ("Found saved models")
        mydict = dict()
        sorted_files = sorted(os.listdir("i3d_model_checkpoints"), key = lambda x: int(x.split('-')[2]), reverse = True)
        saved_model = sorted_files[0]
        initial_epoch = int(saved_model.split('-')[2])

        sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        csv_logger = CSVLogger('i3d_training.log')
        #Checkpointing
        filepath="i3d_model_checkpoints/"+dtype.lower()+"/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
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

        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary() 
        csv_logger = CSVLogger('i3d_training.log')
        #Checkpointing
        filepath="i3d_model_checkpoints/"+dtype.lower()+"/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only = True,  mode='max')
        # es = EarlyStopping(monitor='val_acc', mode='max', min_delta=1, patience = 10)
        # callbacks_list = [checkpoint,es]
        callbacks_list = [checkpoint, csv_logger]

        history = model.fit_generator(dgTrain, epochs = 50,validation_data = dgVal, callbacks=callbacks_list)

if __name__ == "__main__":
    do_test = False
    if do_test:
        test(dtype="FRAME")
    else:
        main(dtype="FRAME")

