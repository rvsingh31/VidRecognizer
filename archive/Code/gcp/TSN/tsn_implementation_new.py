import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from collections import defaultdict
import keras
from skimage.transform import resize
from pretrained_models import TSN
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import accuracy_score


class dataGenerator(keras.utils.Sequence):

    def __init__(self, filepath, batch_size, ffpath, segments = 3, test=False):
        np.random.seed(0)
        self.filenames = list()
        self.labels = list()
        self.batch_size = batch_size
        self.filepath = filepath
        self.ffpath = ffpath
        self.segments = segments
        
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

        Xframes = defaultdict(list)
        Y = list()
        Xflows = defaultdict(list)
        for index,each in enumerate(batch_x):
            infopath = os.path.join(self.ffpath,each,"info.txt")
            imgpath = os.path.join(self.ffpath,each,"frames")
            flowspath = os.path.join(self.ffpath,each,"flows")
            f = open(infopath,"r")
            total_frames = int(f.readlines()[0].strip().split(':')[1])
            f.close()
            idxs = []
            base = total_frames//self.segments
            low = 1
            num_frames = 1
            for _ in range(self.segments):
                high = min(low + base, total_frames)
                idxs.extend(np.random.randint(low, high, num_frames))
                low = high + 1 
            frames = self.getFrames(idxs, imgpath)
            flows = self.getFlows(idxs, flowspath)
            
            for i in range(len(frames)):
                Xframes[i%self.segments].append(frames[i])
                        
            for i in range(len(flows)):
                Xflows[i%self.segments].append(flows[i])
                
            Y.extend([batch_y[index]]*(num_frames))
            
        finalX = dict()
        i = 1
        for key in Xframes.keys():
            finalX['input_'+str(i)] = np.array(Xframes[key])
            i += 1
        for key in Xflows.keys():
            finalX['input_'+str(i)] = np.array(Xflows[key])
            i += 1
            
        finalY = {'output':self.one_hot_encode(np.array(Y))}

        return (finalX,finalY)
    
    def getTestData(self):
        batch_x = list()
        batch_y = list()
        for each in self.idxs:
            batch_x.append(self.filenames[each])
            batch_y.append(self.labels[each])
        
        Xframes = defaultdict(list)
        Y = list()
        Xflows = defaultdict(list)
        for index,each in enumerate(batch_x):
            infopath = os.path.join(self.ffpath,each,"info.txt")
            imgpath = os.path.join(self.ffpath,each,"frames")
            flowspath = os.path.join(self.ffpath,each,"flows")
            f = open(infopath,"r")
            total_frames = int(f.readlines()[0].strip().split(':')[1])
            f.close()
            idxs = []
            base = total_frames//self.segments
            low = 1
            num_frames = 1
            for _ in range(self.segments):
                high = min(low + base, total_frames)
                idxs.extend(np.random.randint(low, high, num_frames))
                low = high + 1 
            frames = self.getFrames(idxs, imgpath)
            flows = self.getFlows(idxs, flowspath)
            
            for i in range(len(frames)):
                Xframes[i%self.segments].append(frames[i])
                        
            for i in range(len(flows)):
                Xflows[i%self.segments].append(flows[i])
                
            Y.extend([batch_y[index]]*(num_frames))
            
        finalX = dict()
        i = 1
        for key in Xframes.keys():
            finalX['input_'+str(i)] = np.array(Xframes[key])
            i += 1
        for key in Xflows.keys():
            finalX['input_'+str(i)] = np.array(Xflows[key])
            i += 1
            
        finalY = {'output':self.one_hot_encode(np.array(Y))}

        return (finalX,finalY)


    def one_hot_encode(self,data, classes = 101):
        """
        :param data: data to be one hot encoded
        :return: np array with one hot encoding
        """
        labels = np.zeros((data.size, classes))
        labels[np.arange(data.size), data - 1] = 1
        return labels


    def getFlows(self,idxs, flowspath):

        stack = list()
        for i in idxs:
            f1 = "flow_x_"+str(i)+".jpg"
            f2 = "flow_y_"+str(i)+".jpg"
            grayx = self.readImg(os.path.join(flowspath,f1), "flows")
            grayy = self.readImg(os.path.join(flowspath,f2), "flows")
            img = np.stack((grayx,grayx,grayy),axis = 2)
            img = np.squeeze(img,axis = 3)
            stack.append(img)
            
        return np.array(stack)
    
    def readImg(self,path, type = "frames"):
        img = cv2.imread(path)
        if type == "frames":
            return img
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # grayimg = np.expand_dims(grayimg,axis = 2)
        img = resize(img,(img.shape[0],img.shape[1],1))
        return img
    
    
    def getFrames(self,idxs, imgpath):

        stack = list()
        for i in idxs:
            framename = "frame_"+str(i)+".jpg"
            stack.append(self.readImg(os.path.join(imgpath,framename)))
            
        return np.array(stack)


np.random.seed(0)

if len(sys.argv) > 1:

    filenameTrain = "custom3Train.txt"
    filenameVal = "custom3Val.txt"
    filenameTest = "custom3Test.txt"
    ffpath = "FramesFlows/custom3"

    dgTrain = dataGenerator(filenameTrain,16,ffpath)
    dgVal = dataGenerator(filenameVal,16,ffpath)
    dgTest = dataGenerator(filenameTest, 1, ffpath)

    #Create and Compile model
    K.clear_session()
    model = TSN()

    #Test or Train
    if sys.argv[1] == "train":
        if os.path.exists('tsn_model_checkpoints') and len(os.listdir("tsn_model_checkpoints")) > 0:
            print ("Found saved models")
            mydict = dict()
            sorted_files = sorted(os.listdir("tsn_model_checkpoints"), key = lambda x: int(x.split('-')[2]), reverse = True)
            saved_model = sorted_files[0]
            initial_epoch = int(saved_model.split('-')[2])
            model.load_weights(os.path.join("tsn_model_checkpoints",saved_model))
            
            model.compile(optimizer= keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
            csv_logger = CSVLogger('tsn_training.log')
            #Checkpointing
            filepath="tsn_model_checkpoints/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            # es = EarlyStopping(monitor='val_acc', mode='max', min_delta=1, patience = 10)
            # callbacks_list = [checkpoint,es]
            callbacks_list = [checkpoint, csv_logger]

            print (initial_epoch)
            print (saved_model)
            
            #Fit generatoro
            history = model.fit_generator(dgTrain,initial_epoch = initial_epoch, epochs = 100,validation_data = dgVal, callbacks = callbacks_list)

        else:

            model.compile(optimizer= keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
            csv_logger = CSVLogger('tsn_training.log')
            #Checkpointing
            filepath="tsn_model_checkpoints/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only = True,  mode='max')
            # es = EarlyStopping(monitor='val_acc', mode='max', min_delta=1, patience = 10)
            # callbacks_list = [checkpoint,es]
            callbacks_list = [checkpoint, csv_logger]

            history = model.fit_generator(dgTrain, epochs = 50,validation_data = dgVal, callbacks = callbacks_list)

    else:

        if os.path.exists('tsn_model_checkpoints') and len(os.listdir("tsn_model_checkpoints")) > 0:
            print ("Found saved models")
            mydict = dict()
            sorted_files = sorted(os.listdir("tsn_model_checkpoints"), key = lambda x: int(x.split('-')[2]), reverse = True)
            saved_model = sorted_files[0]
            initial_epoch = int(saved_model.split('-')[2])
            model.load_weights(os.path.join("tsn_model_checkpoints",saved_model))
            Xtest, ytest = dgTest.getTestData()
            ytrue = K.argmax(ytest['output'],1)
            ypred = model.predict(Xtest)
            ypred = K.argmax(ypred,1)
            print ("Testing Accuracy:",accuracy_score(K.eval(ytrue), K.eval(ypred)))
        else:
            print ("No saved models. Please train first")

else:
    print ("Please specify whether to train or test")
