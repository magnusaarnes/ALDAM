import numpy as np
import cv2
import time
import datetime as dt
import matplotlib.pyplot as plt
import os
import commons

## Set variables
# Camera image downsample scaling
scale = 1
# Camera resolution
hi = 720
wi = 1280

# Data directory path
directory = 'C:/Users/sindr/Documents/UniversiTales/V22/Python Messus/Løst og fast/OpenCV Saliency Test 2/data'
photo_capture_name = 'last_camcap.png'
salmap_name = 'saliency_map.png'

print("Folder content at start: ", os.listdir(directory))

## Set up webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, wi)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, hi)

## fps counting
last_time = dt.datetime.today().timestamp()
diffs = []
time_sec0 = dt.datetime.today().timestamp()
time_sec1 = dt.datetime.today().timestamp()

while (True):
    # Capture the video frame by frame
    _ret, frame = cam.read()

    # Image preprocessing
    frame = cv2.resize(frame,(frame.shape[1]//scale, frame.shape[0]//scale))
    frame = cv2.flip(frame,1)

    # Perform saliency
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(frame)
    saliencyMap = (saliencyMap*255).astype("uint8")
  
    # Image post-processing
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _ret, thresh1 = cv2.threshold(saliencyMap,80,255,cv2.THRESH_BINARY)
    smoother = cv2.blur(thresh1, (5,5))
    _ret, thresh2 = cv2.threshold(smoother,80,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawn_im = grayframe.copy()
    cv2.drawContours(drawn_im, contours, -1, (0,255,0), 3)

    # Image Merging
    hcon1 = cv2.hconcat([grayframe, saliencyMap, thresh1])
    hcon2 = cv2.hconcat([smoother,thresh2, drawn_im])
    displ = cv2.vconcat([hcon1, hcon2])

    # Display the resulting frame
    cv2.imshow('frame', displ)
      
    # the 'q' button is set as the quitting button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ## Keep track of fps
    # Add new time diff to list
    new_time = dt.datetime.today().timestamp()
    diffs.append(new_time - last_time)
    last_time = new_time
    time_sec1 = dt.datetime.today().timestamp()

    # Clip the list
    if len(diffs) > 10:
        diffs = diffs[-10:]
    # Display fps every 3 seconds
    if (int(time_sec1 - time_sec0) >= 3):
        time_sec0 = dt.datetime.today().timestamp()
        print("Last 10 frame average fps: ",int (len(diffs) / sum(diffs)))

## Close webcam
cam.release()
cv2.destroyAllWindows()