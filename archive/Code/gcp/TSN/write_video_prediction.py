import cv2
import numpy as np

f = open("classInd.txt","r")
lines = [line.strip().split(" ") for line in f.readlines()]
f.close()

actions_map = {(int(line[0])-1):line[1] for line in lines}

actions = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
confidence = [0.3218086, 0.30393335, 0.50292057, 0.3637636, 0.6533337, 0.51918936, 0.24666283, 0.23461409, 0.43313158, 0.32986924, 0.42131943, 0.35405362, 0.3364, 0.40841392, 0.4567551, 0.27676046, 0.3317784, 0.6036323, 0.757768, 0.5641499, 0.59122604, 0.68902344, 0.4566407, 0.5313914, 0.54710686]
confidence = [round(c*100,2) for c in confidence]
videofile = r'E:\capstone_adbi_data\UCF-101\PlayingSitar\v_PlayingSitar_g05_c01.avi'

cap = cv2.VideoCapture(videofile)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(3))
height = int(cap.get(4))

out = cv2.VideoWriter('test1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width,height))

if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
itr = 0
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    text = actions_map[actions[(itr//16)%len(actions)]] + ":" + str(confidence[(itr//16)%len(confidence)]) 
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
    itr += 1
    out.write(frame)
    cv2.imshow('Frame',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break 
  else: 
    break
cap.release()
out.release()
cv2.destroyAllWindows()