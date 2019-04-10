import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from collections import defaultdict
import keras
from bn_model import TSN
from keras import backend as K

#Create folder for storing the frames
 
splitfiledir = r"..\ucfTrainTestlist"
splitfile = "trainlist01.txt"
splitname = splitfile.split('.')[0]


def storeFramesAndFlows(framesdir,splitfiledir,splitfile,splitname):
    
    #Read splitfile
    lines = open(os.path.join(splitfiledir,splitfile),"r")
    for line in lines:
        arr = line.split(" ")
        vidclass = arr[1] 
        line = arr[0].split("/")
        action = line[0]
        filename = line[1]
        if filename.split("_")[2] not in ('g08','g09'):
            continue
        actionpath = os.path.join(framesdir,splitname,action)
        framepath = os.path.join(actionpath,filename,"frames")
        flowpath = os.path.join(actionpath,filename,"flows")
        
        #Create folder for Action
        if not os.path.exists(actionpath):
            os.mkdir(actionpath)
        
        #Create folder for videofile
        if not os.path.exists(os.path.join(actionpath,filename)):
            os.mkdir(os.path.join(actionpath,filename))
        
        #Create folder for frames
        if not os.path.exists(framepath):
            os.mkdir(framepath)
            
        #Create folder for flows
        if not os.path.exists(flowpath):
            os.mkdir(flowpath)

        #Read video and collect frames, flows
        vidcap = cv2.VideoCapture(os.path.join(path,action,filename))
        count = 0 
        prevFrame = None
        nextFrame = None
        while True:
            success,image = vidcap.read()
            #Resize image to remain consistent with BN Inception model
            if not success:
                break
            image = cv2.resize(image,(224,224))
            frame = "frame_%d.jpg"%count
            flow_x = "flow_x_%d.jpg"%count
            flow_y = "flow_y_%d.jpg"%count
            framename = os.path.join(framepath,frame)
            flowname_x = os.path.join(flowpath,flow_x)
            flowname_y = os.path.join(flowpath,flow_y)
            if count == 0:
                prevFrame = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                cv2.imwrite(framename,image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                count += 1
                continue
            
            nextFrame = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            cv2.imwrite(framename,image)
            optical_flow = cv2.opt.DualTVL1OpticalFlow_create()
            flow = optical_flow.calc(prevFrame, nextFrame, None)
            prevFrame = nextFrame
            flow[...,0] = cv2.normalize(flow[...,0],None,0,255,cv2.NORM_MINMAX)
            flow[...,1] = cv2.normalize(flow[...,1],None,0,255,cv2.NORM_MINMAX)
            cv2.imwrite(flowname_x,flow[...,0])
            cv2.imwrite(flowname_y,flow[...,1])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            count += 1
            
        filename = os.path.join(actionpath,filename,"info.txt")
        #Store the frames count in txt file
        rate = open(filename,"w")
        rate.write("frames:"+str(count))
        rate.write("\n")
        rate.write("class:"+vidclass)
        rate.close()

        #Close the video object
        vidcap.release()

    print ("Stored")




#Store Frames
framesdir = r"../FramesFlows"

path = r"../../UCF-101"
# path = r"E:\capstone_adbi_data\UCF-101"

print ("Extracting images")

if not os.path.exists(os.path.join(framesdir,splitname)):
    os.mkdir(os.path.join(framesdir,splitname))
    storeFramesAndFlows(framesdir,splitfiledir,splitfile,splitname)
else:
    print ("Frames stored already!")

