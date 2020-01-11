# Extract N frames from videos's subfolder and save then Frames's respective folder, then resize all frames
import os
import math
import cv2
import re
listing = os.listdir("./videos/")
count = 1

for file in listing:
    train_val_dir  = os.listdir("./videos/" + file + "/" )   
    for file_2 in train_val_dir:
       class_dir = os.listdir("./videos/" + file + "/" + file_2 + "/" ) 
       for videos in class_dir:
           cap = cv2.VideoCapture("./videos/" + file + "/" + file_2 + "/" + videos)
           os.makedirs("./activity_data/" + file + "/" + file_2 + "/" + videos)
           fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
           frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
           duration = frame_count/fps
           def getFrame(sec):
               cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
               hasFrames,image = cap.read()
               if hasFrames:
                   e=cv2.imwrite("./activity_data/"+ file +"/" + file_2 + "/" + videos +"/" + videos + "_" +str(count)+".jpg", image)
               return hasFrames
           sec = 0
           N = 48 # How many frames will be extracted
           interwall = duration/N   #//it will capture image in each 'interwall' second
           count = 1
           success = getFrame(sec)
           while success:
               count = count + 1
               sec = sec + interwall
               success = getFrame(sec)


os.system('sh resize.sh')

