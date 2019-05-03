import cv2
import numpy as np

f = open("classInd.txt","r")
lines = [line.strip().split(" ") for line in f.readlines()]
f.close()

actions_map = {(int(line[0])-1):line[1] for line in lines}

actions = [42,61,42,42,42,42]
confidence = [0.68814725, 0.34163043, 0.43209386, 0.7391411, 0.8304755, 0.6209355] 
confidence = [round(c*100,2) for c in confidence]
videofile = r'E:\capstone_adbi_data\UCF-101\HulaHoop\v_HulaHoop_g12_c03.avi'

cap = cv2.VideoCapture(videofile)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(3))
height = int(cap.get(4))

out = cv2.VideoWriter('vid1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width,height))

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